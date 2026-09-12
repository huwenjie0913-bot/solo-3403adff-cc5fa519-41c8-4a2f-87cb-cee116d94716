"""窑炉热响应辨识与曲线跟随预测模块测试。"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.identification import simulate_fopdt


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 合成数据工具
# ---------------------------------------------------------------------------

CAL_POINTS = [(0, 25), (60, 400), (120, 400), (200, 810), (280, 810),
              (340, 516), (420, 516), (480, 470), (560, 470), (640, 200), (700, 200)]
DT = 5.0


def calibration_timeline():
    ts = np.arange(0, 700 + DT, DT)
    u = np.interp(ts, *[np.array(x) for x in zip(*CAL_POINTS)])
    return ts, u


def make_profile_body(channels, *, mass_min=20.0, mass_max=120.0,
                      setpoints=None, excitation=20.0, long_gap=None):
    ts, u = calibration_timeline()
    sp = setpoints or [{"time_min": float(t), "temp_c": float(v)} for t, v in zip(ts, u)]
    return {
        "name": "窑炉-1",
        "load_mass_min_kg": mass_min,
        "load_mass_max_kg": mass_max,
        "setpoints": sp,
        "channels": channels,
        "excitation_min_c": excitation,
        "long_gap_min": long_gap,
    }


def fopdt_channel(cid, zone, theta, tau, K=1.05, b=0.0, *,
                  position="", noise=0.0, drop=(), ts=None, u=None):
    if ts is None or u is None:
        ts, u = calibration_timeline()
    y = simulate_fopdt(ts, u, theta, tau, K, b)
    if noise:
        y = y + np.random.default_rng(42).normal(0, noise, y.size)
    samples = [
        {"time_min": float(t), "temp_c": float(v)}
        for t, v in zip(ts, y)
        if not any(a < t < b_ for a, b_ in drop)
    ]
    ch = {"channel_id": cid, "zone": zone, "position": position,
          "sample_interval_min": DT, "samples": samples}
    return ch


def create_profile(client, body):
    r = client.post("/profiles", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def job_payload(profile_id=None, *, load_mass=50.0, zones=("top", "bottom"),
                cool_rate=3.0, kiln_max=900.0, max_total=None, start=25.0):
    return {
        "name": "熔合-档案作业",
        "load_mass_kg": load_mass if profile_id else None,
        "profile_id": profile_id,
        "pieces": [
            {"name": "A-薄", "layers": 2, "max_thickness_mm": 6.0,
             "thermal_diffusivity_m2s": 4.5e-7,
             "anneal_point_c": 516.0, "strain_point_c": 470.0,
             "allowed_delta_t_c": 10.0, "zone": zones[0]},
            {"name": "B-厚", "layers": 4, "max_thickness_mm": 12.0,
             "thermal_diffusivity_m2s": 4.5e-7,
             "anneal_point_c": 516.0, "strain_point_c": 470.0,
             "allowed_delta_t_c": 8.0, "zone": zones[1]},
        ],
        "program": {"start_c": start, "segments": [
            {"kind": "ramp", "target_c": 500.0, "rate_c_per_min": 4.0},
            {"kind": "ramp", "target_c": 810.0, "rate_c_per_min": 4.0},
            {"kind": "hold", "duration_min": 120.0},
            {"kind": "cool", "target_c": 516.0, "rate_c_per_min": 5.0},
            {"kind": "hold", "duration_min": 60.0},
            {"kind": "cool", "target_c": 470.0, "rate_c_per_min": cool_rate},
            {"kind": "cool", "target_c": 200.0, "rate_c_per_min": 5.0},
        ]},
        "constraints": {"kiln_max_temp_c": kiln_max, "min_step_min": 5.0,
                        "max_total_duration_min": max_total},
    }


# ---------------------------------------------------------------------------
# 辨识
# ---------------------------------------------------------------------------

def test_fopdt_parameters_recovered(client):
    body = make_profile_body([
        fopdt_channel("T1", "top", 10.0, 30.0, position="顶部中心"),
        fopdt_channel("B1", "bottom", 15.0, 40.0),
    ])
    out = create_profile(client, body)
    prof = out["profile"]
    assert prof["status"] == "active"
    by_id = {c["channel_id"]: c for c in prof["channels"]}
    p = by_id["T1"]["parameters"]
    assert p["delay_min"] == pytest.approx(10.0, abs=1.0)
    assert p["time_constant_min"] == pytest.approx(30.0, abs=2.0)
    assert p["gain"] == pytest.approx(1.05, abs=0.02)
    assert abs(p["bias_c"]) < 1.0
    assert p["r_squared"] > 0.999
    assert p["rmse_c"] < 0.1
    # 残差与适用温区
    assert "max_abs_residual_c" in p
    rng = by_id["T1"]["applicable_range_c"]
    assert rng["min_c"] == pytest.approx(25.0, abs=1.0)
    assert rng["max_c"] == pytest.approx(810.0, abs=1.0)
    assert by_id["T1"]["position"] == "顶部中心"
    assert by_id["B1"]["parameters"]["delay_min"] == pytest.approx(15.0, abs=1.0)


def test_unsorted_samples_reported_with_channel_and_interval(client):
    ch = fopdt_channel("T1", "top", 10.0, 30.0)
    samples = ch["samples"]
    samples[12], samples[13] = samples[13], samples[12]  # 制造乱序
    out = create_profile(client, make_profile_body([ch]))
    issues = out["profile"]["channels"][0]["issues"]
    unsorted = [i for i in issues if i["type"] == "unsorted_samples"]
    assert len(unsorted) == 1
    assert unsorted[0]["channel_id"] == "T1"
    assert unsorted[0]["start_min"] > unsorted[0]["end_min"]
    # 乱序自动排序后仍能辨识
    assert out["profile"]["channels"][0]["status"] == "identified"


def test_long_gap_excluded_from_fit(client):
    ch = fopdt_channel("T1", "top", 10.0, 30.0, drop=((200.0, 320.0),))
    out = create_profile(client, make_profile_body([ch]))
    c = out["profile"]["channels"][0]
    gaps = [i for i in c["issues"] if i["type"] == "long_gap"]
    assert gaps and gaps[0]["start_min"] == 200.0 and gaps[0]["end_min"] == 320.0
    assert c["excluded_gap_intervals"] == [{"start_min": 200.0, "end_min": 320.0}]
    # 缺口不参与拟合，但其余数据仍恢复参数
    p = c["parameters"]
    assert p["delay_min"] == pytest.approx(10.0, abs=2.0)
    assert p["time_constant_min"] == pytest.approx(30.0, abs=4.0)
    full = len(calibration_timeline()[0])
    assert c["used_samples"] < full - 20  # 缺口段确实被排除


def test_insufficient_excitation_reports_channel(client):
    flat = {"channel_id": "DEAD", "zone": "dead", "sample_interval_min": 5.0,
            "samples": [{"time_min": 0.0, "temp_c": 500.0},
                        {"time_min": 60.0, "temp_c": 500.4},
                        {"time_min": 120.0, "temp_c": 500.2}]}
    body = make_profile_body([flat], setpoints=[
        {"time_min": 0.0, "temp_c": 500.0},
        {"time_min": 120.0, "temp_c": 500.3},
    ])
    out = create_profile(client, body)
    c = out["profile"]["channels"][0]
    assert c["status"] == "insufficient_excitation"
    assert any(i["type"] == "insufficient_excitation" and i["channel_id"] == "DEAD"
               for i in c["issues"])
    assert out["profile"]["status"] == "inactive"


def test_time_constant_below_sample_resolution_unidentifiable(client):
    ts, u = calibration_timeline()
    y = simulate_fopdt(ts, u, 0.0, 0.4, 1.0, 0.0)  # τ=0.4 min ≪ 5 min 采样
    ch = {"channel_id": "FAST", "zone": "z", "sample_interval_min": 5.0,
          "samples": [{"time_min": float(t), "temp_c": float(v)} for t, v in zip(ts, y)]}
    out = create_profile(client, make_profile_body([ch]))
    c = out["profile"]["channels"][0]
    assert c["status"] == "unidentifiable"
    assert any(i["type"] == "unidentifiable" for i in c["issues"])


def test_setpoint_axis_must_be_increasing(client):
    body = make_profile_body([fopdt_channel("T1", "top", 10.0, 30.0)])
    body["setpoints"][5] = body["setpoints"][3]
    r = client.post("/profiles", json=body)
    assert r.status_code == 422
    assert "单调" in r.text


def test_partial_channel_failure_keeps_profile_active(client):
    good = fopdt_channel("T1", "top", 10.0, 30.0)
    # DEAD 位于设定轴上 60–120 min 的 400°C 保温平台：实测无变化、激励不足
    bad = {"channel_id": "DEAD", "zone": "dead", "sample_interval_min": 5.0,
           "samples": [{"time_min": 60.0, "temp_c": 400.1},
                       {"time_min": 90.0, "temp_c": 400.0},
                       {"time_min": 120.0, "temp_c": 400.2}]}
    out = create_profile(client, make_profile_body([good, bad]))
    statuses = {c["channel_id"]: c["status"] for c in out["profile"]["channels"]}
    assert statuses == {"T1": "identified", "DEAD": "insufficient_excitation"}
    assert out["profile"]["status"] == "active"


def test_duplicate_zone_noted(client):
    out = create_profile(client, make_profile_body([
        fopdt_channel("T1", "top", 10.0, 30.0),
        fopdt_channel("T2", "top", 12.0, 33.0),
    ]))
    assert any(i["type"] == "duplicate_zone" for i in out["profile"]["issues"])


def test_out_of_setpoint_range_reported(client):
    ts, u = calibration_timeline()
    y = simulate_fopdt(ts, u, 10.0, 30.0, 1.05, 0.0)
    samples = [{"time_min": float(t), "temp_c": float(v)} for t, v in zip(ts, y)]
    samples.append({"time_min": 5000.0, "temp_c": 100.0})
    ch = {"channel_id": "T1", "zone": "top", "sample_interval_min": 5.0,
          "samples": samples}
    out = create_profile(client, make_profile_body([ch]))
    issues = out["profile"]["channels"][0]["issues"]
    assert any(i["type"] == "out_of_setpoint_range" for i in issues)


# ---------------------------------------------------------------------------
# 持久化与版本
# ---------------------------------------------------------------------------

def test_profile_persisted_with_raw_samples(client):
    body = make_profile_body([fopdt_channel("T1", "top", 10.0, 30.0)])
    out = create_profile(client, body)
    pid = out["profile_id"]
    assert out["version"] == 1

    detail = client.get(f"/profiles/{pid}").json()
    assert detail["channels"][0]["parameters"]["time_constant_min"] == pytest.approx(30.0, abs=2)

    raw = client.get(f"/profiles/{pid}/samples").json()["samples"]
    n_set = sum(1 for r in raw if r["kind"] == "setpoint")
    n_ch = sum(1 for r in raw if r["kind"] == "channel")
    assert n_set == len(body["setpoints"])
    assert n_ch == len(body["channels"][0]["samples"])
    # seq 保留原始上传顺序
    ch_rows = sorted((r for r in raw if r["kind"] == "channel"), key=lambda r: r["seq"])
    assert [r["time_min"] for r in ch_rows[:3]] == [
        s["time_min"] for s in body["channels"][0]["samples"][:3]
    ]


def test_profile_versions_are_immutable(client):
    body1 = make_profile_body([fopdt_channel("T1", "top", 10.0, 30.0)])
    out1 = create_profile(client, body1)
    pid = out1["profile_id"]
    body2 = make_profile_body([fopdt_channel("T1", "top", 14.0, 45.0)])
    r = client.post(f"/profiles/{pid}/versions", json=body2)
    assert r.status_code == 201 and r.json()["version"] == 2
    # 旧版本保持不变，最新为 v2
    v1 = client.get(f"/profiles/{pid}?version=1").json()
    vlatest = client.get(f"/profiles/{pid}").json()
    assert v1["channels"][0]["parameters"]["delay_min"] == pytest.approx(10.0, abs=1.5)
    assert vlatest["version"] == 2
    assert vlatest["channels"][0]["parameters"]["delay_min"] == pytest.approx(14.0, abs=1.5)
    # 不存在的档案
    assert client.post("/profiles/nope/versions", json=body2).status_code == 404


# ---------------------------------------------------------------------------
# 作业 × 档案：预测分析
# ---------------------------------------------------------------------------

@pytest.fixture()
def profile(client):
    body = make_profile_body([
        fopdt_channel("T1", "top", 10.0, 30.0),
        fopdt_channel("B1", "bottom", 15.0, 40.0),
    ])
    out = create_profile(client, body)
    return out["profile_id"]


def test_predicted_analysis_lists_ideal_local_following_overshoot_worst(client, profile):
    r = client.post("/jobs", json=job_payload(profile))
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    assert r.json()["profile"] == {"id": profile, "version": 1}

    pred = a["prediction"]
    assert pred["profile_id"] == profile and pred["profile_version"] == 1
    zm = pred["zones"]
    assert set(zm) == {"top", "bottom"}
    # 升降温跟随误差都为正且可量化；底部惯性更大，滞后更严重
    assert zm["top"]["max_heating_lag_c"] > 1.0
    assert zm["bottom"]["max_cooling_lag_c"] > zm["top"]["max_cooling_lag_c"]
    assert zm["top"]["max_abs_follow_error_c"] > 0.0
    # K=1.05：保温稳态预测高于设定 → 非零超调（应在保温段附近，量级 ~40°C）
    assert zm["top"]["max_overshoot_c"] == pytest.approx(0.05 * 810, abs=8.0)
    assert zm["top"]["overshoot_time_min"] is not None

    # 每件工件摘要带区域跟随指标，且时序含理想设定与区域预测
    for piece in a["pieces"]:
        assert piece["following"]["channel_id"] in ("T1", "B1")
        assert "max_heating_lag_c" in piece["following"]
    ts = a["timeseries"]
    n = len(ts["t_min"])
    assert len(ts["setpoint_c"]) == n
    assert len(ts["zones"]["top"]["predicted_c"]) == n
    assert len(ts["pieces"]["A-薄"]["kiln_c"]) == n  # 该工件所在区域的局部预测

    # 最不利工件：底部厚件（惯性大、有未均温违规）
    worst = pred["worst_piece"]
    assert worst["name"] == "B-厚" and worst["zone"] == "bottom"


def test_load_mass_out_of_range_rejects_safety(client, profile):
    body = job_payload(profile, load_mass=5.0)  # 校准范围 20–120 kg
    r = client.post("/jobs", json=body)
    assert r.status_code == 201
    a = r.json()["analysis"]
    assert a["verdict"] == "rejected"
    types = {x["type"] for x in a["rejections"]}
    assert "load_mass_out_of_range" in types
    assert not a["violations"] and not a["pieces"]  # 拒绝时不做安全判定


def test_temperature_range_out_of_calibration_rejects(client, profile):
    body = job_payload(profile)
    body["program"]["segments"][1]["target_c"] = 850.0  # 档案仅校准到 810°C
    r = client.post("/jobs", json=body)
    a = r.json()["analysis"]
    assert a["verdict"] == "rejected"
    rej = [x for x in a["rejections"] if x["type"] == "temperature_out_of_calibrated_range"]
    assert rej and {x["piece"] for x in rej} == {"A-薄", "B-厚"}


def test_missing_zone_or_mass_is_422(client, profile):
    body = job_payload(profile)
    body["load_mass_kg"] = None
    assert client.post("/jobs", json=body).status_code == 422
    body = job_payload(profile)
    body["pieces"][0]["zone"] = None
    assert client.post("/jobs", json=body).status_code == 422
    # 未知区域
    body = job_payload(profile, zones=("side", "bottom"))
    assert client.post("/jobs", json=body).status_code == 422
    # 不存在 / inactive 档案
    body = job_payload("nonexistent_id")
    assert client.post("/jobs", json=body).status_code == 404


def test_inactive_profile_rejected_for_jobs(client):
    flat = {"channel_id": "DEAD", "zone": "dead", "sample_interval_min": 5.0,
            "samples": [{"time_min": 0.0, "temp_c": 500.0},
                        {"time_min": 60.0, "temp_c": 500.2}]}
    out = create_profile(client, make_profile_body(
        [flat], setpoints=[{"time_min": 0.0, "temp_c": 500.0},
                           {"time_min": 60.0, "temp_c": 500.1}]))
    assert out["profile"]["status"] == "inactive"
    body = job_payload(out["profile_id"], zones=("dead", "dead"))
    assert client.post("/jobs", json=body).status_code == 409


def test_schedule_uses_profile_and_version_snapshot(client, profile):
    # 降温 30°C/min：名义下厚件会 fast_cooling，排程需降速
    body = job_payload(profile, cool_rate=30.0)
    created = client.post("/jobs", json=body).json()
    jid = created["job_id"]
    r = client.post(f"/jobs/{jid}/schedule", json={})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "ok"
    seg = out["program"]["segments"][5]
    assert seg["rate_c_per_min"] <= 12.0 * 0.9 + 1e-6
    # 排程分析同样走区域预测
    assert out["analysis"]["prediction"]["profile_id"] == profile
    assert out["analysis"]["verdict"] == "pass"

    # 档案出新版本：旧作业仍按 v1 快照判定
    body2 = make_profile_body([
        fopdt_channel("T1", "top", 10.0, 30.0),
        fopdt_channel("B1", "bottom", 15.0, 40.0),
    ])
    client.post(f"/profiles/{profile}/versions", json=body2)
    detail = client.get(f"/jobs/{jid}").json()
    assert detail["payload"]["profile_version"] == 1
    rec = client.get(f"/jobs/{jid}/analysis?kind=scheduled").json()
    assert rec["result"]["analysis"]["prediction"]["profile_version"] == 1


def test_schedule_rejected_when_out_of_range(client, profile):
    body = job_payload(profile)
    body["program"]["segments"][1]["target_c"] = 850.0
    jid = client.post("/jobs", json=body).json()["job_id"]
    out = client.post(f"/jobs/{jid}/schedule", json={}).json()
    assert out["status"] == "rejected"
    assert any(x["type"] == "temperature_out_of_calibrated_range" for x in out["rejections"])
    # 拒绝结论留痕但不产生可执行排程
    rec = client.get(f"/jobs/{jid}").json()
    assert any(a["kind"] == "scheduled_rejected" for a in rec["analyses"])
    assert client.get(f"/jobs/{jid}/export/program.csv?source=scheduled").status_code == 404


def test_existing_jobs_and_analyses_not_rewritten(client, profile):
    # 同一作业多次排程：每次追加分析，历史记录保留
    body = job_payload(profile, cool_rate=30.0)
    jid = client.post("/jobs", json=body).json()["job_id"]
    client.post(f"/jobs/{jid}/schedule", json={"safety_factor": 0.9})
    client.post(f"/jobs/{jid}/schedule", json={"safety_factor": 0.8})
    detail = client.get(f"/jobs/{jid}").json()
    scheduled = [a for a in detail["analyses"] if a["kind"] == "scheduled"]
    assert len(scheduled) == 2  # 追加而非改写


def test_no_profile_job_keeps_legacy_behavior(client):
    from tests.test_api import make_job
    out = client.post("/jobs", json=make_job(cool_rate=30.0))
    assert out.status_code == 201
    a = out.json()["analysis"]
    assert a["verdict"] == "fail"
    assert "prediction" not in a
