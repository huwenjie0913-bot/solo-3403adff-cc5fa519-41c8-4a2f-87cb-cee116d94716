"""端到端 API 测试。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def make_job(cool_rate=3.0, locked_cool=False, max_total=None, kiln_max=900.0):
    return {
        "name": "熔合作业-1",
        "pieces": [
            {"name": "A-薄", "layers": 2, "max_thickness_mm": 6.0,
             "thermal_diffusivity_m2s": 4.5e-7,
             "anneal_point_c": 516.0, "strain_point_c": 470.0,
             "allowed_delta_t_c": 10.0},
            {"name": "B-厚", "layers": 4, "max_thickness_mm": 12.0,
             "thermal_diffusivity_m2s": 4.5e-7,
             "anneal_point_c": 516.0, "strain_point_c": 470.0,
             "allowed_delta_t_c": 8.0},
        ],
        "program": {"start_c": 25.0, "segments": [
            {"kind": "ramp", "target_c": 500.0, "rate_c_per_min": 3.0},
            {"kind": "ramp", "target_c": 810.0, "rate_c_per_min": 3.0},
            {"kind": "hold", "duration_min": 30.0},
            {"kind": "cool", "target_c": 516.0, "rate_c_per_min": 5.0},
            {"kind": "hold", "duration_min": 60.0},
            {"kind": "cool", "target_c": 470.0, "rate_c_per_min": cool_rate,
             "locked": locked_cool},
            {"kind": "cool", "target_c": 25.0, "rate_c_per_min": 5.0},
        ]},
        "constraints": {"kiln_max_temp_c": kiln_max, "min_step_min": 5.0,
                        "max_total_duration_min": max_total},
    }


def create(client, payload):
    r = client.post("/jobs", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ---------- 名义分析 ----------

def test_gentle_program_passes(client):
    out = create(client, make_job())
    a = out["analysis"]
    assert a["verdict"] == "pass"
    assert a["violations"] == []
    by_name = {p["name"]: p for p in a["pieces"]}
    assert by_name["A-薄"]["effective_hold_min"] > 30
    assert by_name["B-厚"]["max_allowed_rate_c_per_min"] == pytest.approx(12.0, abs=0.01)
    assert by_name["B-厚"]["anneal_range_crossing_rate_c_per_min"] > 0


def test_fast_cooling_flagged_with_piece_and_time(client):
    out = create(client, make_job(cool_rate=30.0))
    a = out["analysis"]
    assert a["verdict"] == "fail"
    fast = [v for v in a["violations"] if v["type"] == "fast_cooling"]
    assert fast, "应检出过快降温"
    assert any(v["piece"] == "B-厚" for v in fast)
    assert all("time_min" in v and v["time_min"] > 0 for v in fast)
    assert all(v["segment_index"] == 5 for v in fast)


def test_program_jump_flagged(client):
    payload = make_job()
    payload["program"]["segments"][2] = {"kind": "hold", "duration_min": 30.0, "target_c": 850.0}
    out = create(client, payload)
    jumps = [v for v in out["analysis"]["violations"] if v["type"] == "program_jump"]
    assert jumps and jumps[0]["segment_index"] == 2


def test_kiln_max_temp_flagged(client):
    out = create(client, make_job(kiln_max=700.0))
    types = {v["type"] for v in out["analysis"]["violations"]}
    assert "kiln_max_temp" in types


# ---------- 排程 ----------

def test_schedule_fixes_fast_cooling(client):
    out = create(client, make_job(cool_rate=30.0))
    r = client.post(f"/jobs/{out['job_id']}/schedule", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    seg = body["program"]["segments"][5]
    assert seg["rate_c_per_min"] <= 12.0 * 0.9 + 1e-6
    assert body["analysis"]["verdict"] == "pass"
    # 段时长应对齐最小控温步长（5 min 的整数倍）
    dur_min = (516.0 - 470.0) / seg["rate_c_per_min"]
    assert dur_min % 5.0 == pytest.approx(0.0, abs=1e-6)
    r2 = client.get(f"/jobs/{out['job_id']}/export/program.csv?source=scheduled")
    assert r2.status_code == 200


def test_schedule_reports_locked_segment_conflict(client):
    out = create(client, make_job(cool_rate=30.0, locked_cool=True))
    body = client.post(f"/jobs/{out['job_id']}/schedule", json={}).json()
    assert body["status"] == "infeasible"
    locked = [c for c in body["conflicts"] if c["type"] == "locked_segment"]
    assert locked, "应报告锁定段冲突"
    assert locked[0]["pieces"] == ["B-厚"]  # A 允许 60°C/min，不受限


def test_schedule_reports_duration_limit(client):
    out = create(client, make_job(max_total=200.0))
    body = client.post(f"/jobs/{out['job_id']}/schedule", json={}).json()
    assert body["status"] == "infeasible"
    dur = [c for c in body["conflicts"] if c["type"] == "duration_limit"]
    assert dur and dur[0]["required_min"] > 200.0
    assert dur[0]["pieces"] == ["B-厚"]


# ---------- 实测复核 ----------

def test_measured_gap_stays_unknown(client):
    out = create(client, make_job())
    samples = [{"time_min": t, "temp_c": 516.0} for t in range(0, 61, 5)]
    samples += [{"time_min": t, "temp_c": 500.0} for t in range(120, 181, 5)]
    r = client.post(f"/jobs/{out['job_id']}/measurements",
                    json={"samples": samples, "max_gap_min": 6.0})
    a = r.json()["analysis"]
    assert a["verdict"] == "unknown"  # 缺测段不得插值判定安全
    assert a["unknown_intervals"] == [{"start_min": 60.0, "end_min": 120.0}]
    assert a["assessed_until_min"] <= 60.0


def test_measured_override_detects_fast_cooling(client):
    out = create(client, make_job())
    samples = [{"time_min": float(t), "temp_c": 516.0} for t in range(0, 61, 2)]
    for k in range(1, 24):  # 46°C / 44 min ≈ 1.05°C/min，安全段
        samples.append({"time_min": 60.0 + 2 * k, "temp_c": 516.0})
    # 快速降温：516 → 470 用 2 min（23°C/min，超过 B 的 12°C/min）
    samples.append({"time_min": 108.0, "temp_c": 470.0})
    samples += [{"time_min": float(t), "temp_c": 470.0} for t in range(110, 151, 2)]
    r = client.post(f"/jobs/{out['job_id']}/measurements",
                    json={"samples": samples, "max_gap_min": 5.0})
    a = r.json()["analysis"]
    assert a["verdict"] == "fail"
    fast = [v for v in a["violations"] if v["type"] == "fast_cooling"]
    assert any(v["piece"] == "B-厚" for v in fast)


# ---------- 导出 ----------

def test_exports(client):
    out = create(client, make_job())
    jid = out["job_id"]
    r = client.get(f"/jobs/{jid}/export/timeseries.json")
    assert r.status_code == 200
    body = r.json()
    assert body["t_min"] and body["kiln_c"]
    assert set(body["pieces"]) == {"A-薄", "B-厚"}
    n = len(body["t_min"])
    assert len(body["pieces"]["A-薄"]["center_c"]) == n

    r = client.get(f"/jobs/{jid}/export/program.csv")
    assert r.status_code == 200
    lines = r.text.strip().splitlines()
    assert lines[0] == "index,kind,start_c,target_c,rate_c_per_min,duration_min,locked"
    assert len(lines) == 8  # 表头 + 7 段


def test_job_detail_and_analysis_endpoints(client):
    out = create(client, make_job())
    jid = out["job_id"]
    detail = client.get(f"/jobs/{jid}").json()
    assert detail["payload"]["pieces"][0]["name"] == "A-薄"  # 材料快照
    assert any(a["kind"] == "nominal" for a in detail["analyses"])
    rec = client.get(f"/jobs/{jid}/analysis?kind=nominal").json()
    assert rec["result"]["verdict"] == "pass"
    assert client.get("/jobs/nonexistent").status_code == 404
