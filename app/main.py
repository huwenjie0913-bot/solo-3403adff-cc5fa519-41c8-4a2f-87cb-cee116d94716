"""玻璃热加工工作室退火曲线校核服务（FastAPI 入口）。

本机运行：uvicorn app.main:app --reload
"""
from __future__ import annotations

import csv
import io
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from . import db
from .analysis import analyze
from .identification import build_prediction, check_applicability
from .models import (
    JobCreate,
    MeasurementBatch,
    ProfileCreate,
    ScheduleRequest,
)
from .scheduler import optimize


REJECT_VERDICT = "rejected"


def _analyze_with_profile(body: JobCreate, profile: dict) -> dict:
    """选择有效档案：先校验装载质量/温区适用性，再把设定转换为区域预测炉温。"""
    piece_zones = {p.name: p.zone for p in body.pieces}
    prediction = build_prediction(body.program, body.options, profile, piece_zones)
    rejections = check_applicability(
        profile, body.load_mass_kg, piece_zones, prediction["setpoint_c"]
    )
    if rejections:
        # 超出档案校准范围：拒绝判定安全，不进入导热与违规判定
        return {
            "verdict": REJECT_VERDICT,
            "rejections": rejections,
            "profile": {
                "id": profile["id"], "version": profile["version"], "name": profile["name"],
            },
            "program_range_c": [
                round(float(min(prediction["setpoint_c"])), 3),
                round(float(max(prediction["setpoint_c"])), 3),
            ],
            "pieces": [],
            "violations": [],
            "timeseries": {"t_min": [], "kiln_c": [], "pieces": {}},
            "total_duration_min": round(float(prediction["t_min"][-1]), 3),
        }
    result = analyze(
        body.pieces,
        prediction=prediction,
        constraints=body.constraints,
        options=body.options,
    )
    return result


def _resolve_profile_or_409(profile_id: str) -> dict:
    profile = db.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="响应档案不存在")
    if profile["status"] != "active":
        raise HTTPException(
            status_code=409,
            detail=f"档案 {profile_id} 无任何辨识成功的通道，不能用于安全判定",
        )
    return profile


def _validate_profile_ref(body: JobCreate, profile: dict) -> None:
    """请求层面的前置校验（作业结构与档案区域定义不匹配属 422）。"""
    if body.load_mass_kg is None:
        raise HTTPException(status_code=422, detail="指定 profile_id 时必须提供 load_mass_kg")
    for p in body.pieces:
        if not p.zone:
            raise HTTPException(status_code=422, detail=f"工件 {p.name} 未指定 zone，无法使用响应档案")
    known = {c["zone"] for c in profile["channels"] if c["status"] == "identified"}
    for p in body.pieces:
        if p.zone not in known:
            raise HTTPException(
                status_code=422,
                detail=f"工件 {p.name} 的区域 {p.zone} 在档案中没有辨识成功的通道",
            )


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="退火曲线校核服务", version="1.0.0", lifespan=lifespan)


def _load_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="作业不存在")
    return job


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", status_code=201)
def create_job(body: JobCreate) -> dict:
    """创建作业（材料快照随作业入库），并立即分析。

    指定 profile_id 时按有效档案把程序设定转换为各区域预测炉温后再判定；
    装载质量或温区超出档案校准范围时结论为 rejected（拒绝判定安全）。
    档案版本快照随作业固化，之后档案新建版本不会改写本作业。
    """
    payload = body.model_dump()
    profile = None
    if body.profile_id:
        profile = _resolve_profile_or_409(body.profile_id)
        _validate_profile_ref(body, profile)
        payload["profile_version"] = profile["version"]
        payload["profile_name"] = profile["name"]

    job_id = db.create_job(payload)
    if profile is not None:
        result = _analyze_with_profile(body, profile)
    else:
        result = analyze(
            body.pieces,
            program=body.program,
            constraints=body.constraints,
            options=body.options,
        )
    db.save_analysis(job_id, "nominal", result)
    return {"job_id": job_id, "analysis": result,
            "profile": ({"id": profile["id"], "version": profile["version"]}
                        if profile is not None else None)}


@app.get("/jobs")
def list_jobs() -> list:
    return db.list_jobs()


@app.get("/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    job = _load_job(job_id)
    return {**job, "analyses": db.list_analyses(job_id)}


@app.get("/jobs/{job_id}/analysis")
def job_analysis(job_id: str, kind: Optional[str] = Query(default=None)) -> dict:
    """取最近一次分析结果；kind 可选 nominal / measured / scheduled。"""
    _load_job(job_id)
    rec = db.get_analysis(job_id, kind)
    if not rec:
        raise HTTPException(status_code=404, detail="没有分析结果")
    return rec


@app.post("/jobs/{job_id}/measurements")
def upload_measurements(job_id: str, body: MeasurementBatch) -> dict:
    """上传实测炉温并复核。缺测段保持未知，不插值判定安全。"""
    job = _load_job(job_id)
    samples = [s.model_dump() for s in body.samples]
    if body.replace:
        db.replace_measurements(job_id, samples)
    else:
        db.replace_measurements(job_id, db.get_measurements(job_id) + samples)
    stored = db.get_measurements(job_id)
    model = JobCreate(**job["payload"])
    measured = {
        "time_min": [s["time_min"] for s in stored],
        "temp_c": [s["temp_c"] for s in stored],
        "max_gap_min": body.max_gap_min,
    }
    result = analyze(
        model.pieces,
        measured=measured,
        constraints=model.constraints,
        options=model.options,
    )
    db.save_analysis(job_id, "measured", result)
    return {"job_id": job_id, "analysis": result}


@app.post("/jobs/{job_id}/schedule")
def schedule(job_id: str, body: Optional[ScheduleRequest] = None) -> dict:
    """在锁定段、窑炉上限、最小步长与总时长约束下调整斜率与保温时长。

    成功时返回全部工件均合规的新程序（并存为 scheduled 分析）；
    失败时列出相互冲突的材料或时限。
    """
    job = _load_job(job_id)
    req = body or ScheduleRequest()
    profile = None
    if job["payload"].get("profile_id"):
        # 排程沿用作业创建时的档案版本快照，不用最新版本
        prof_id = job["payload"]["profile_id"]
        prof_ver = job["payload"].get("profile_version")
        profile = db.get_profile(prof_id, prof_ver)
        if not profile or profile["status"] != "active":
            raise HTTPException(status_code=409, detail="作业绑定的档案版本无效，不能判定安全")
    outcome = optimize(job["payload"], req.safety_factor, req.max_iterations, profile=profile)
    if outcome["status"] == "ok":
        db.save_analysis(
            job_id, "scheduled",
            {"program": outcome["program"], "analysis": outcome["analysis"]},
        )
    elif outcome["status"] == "rejected":
        # 档案不适用：记录拒绝结论（不可改写已有分析，只追加审计记录）
        db.save_analysis(job_id, "scheduled_rejected", outcome)
    return outcome


# ---------------------------------------------------------------------------
# 窑炉热响应档案（FOPDT 辨识）
# ---------------------------------------------------------------------------

@app.post("/profiles", status_code=201)
def create_profile(body: ProfileCreate) -> dict:
    """上传设定温度与一个或多个热电偶实测值，按探头拟合 FOPDT 模型。

    原始采样、拟合参数与档案版本存入 SQLite；档案一经写入不可变。
    """
    from .identification import identify_profile

    request = body.model_dump()
    result = identify_profile(request)
    ref = db.save_profile(request, result)
    return {"profile_id": ref["id"], "version": ref["version"], "profile": result}


@app.get("/profiles")
def list_profiles() -> list:
    return db.list_profiles()


@app.get("/profiles/{profile_id}")
def profile_detail(profile_id: str, version: Optional[int] = Query(default=None)) -> dict:
    profile = db.get_profile(profile_id, version)
    if not profile:
        raise HTTPException(status_code=404, detail="响应档案不存在")
    return profile


@app.post("/profiles/{profile_id}/versions", status_code=201)
def new_profile_version(profile_id: str, body: ProfileCreate) -> dict:
    """为已有档案追加不可变新版本（前一版本不被改写，旧作业继续引用旧版本）。"""
    from .identification import identify_profile

    if not db.get_profile(profile_id):
        raise HTTPException(status_code=404, detail="响应档案不存在")
    request = body.model_dump()
    result = identify_profile(request)
    ref = db.save_profile(request, result, supersedes_id=profile_id)
    return {"profile_id": ref["id"], "version": ref["version"], "profile": result}


@app.get("/profiles/{profile_id}/samples")
def profile_samples(profile_id: str, version: Optional[int] = Query(default=None)) -> dict:
    """按原始上传顺序回读原始采样（含乱序取证）。"""
    profile = db.get_profile(profile_id, version)
    if not profile:
        raise HTTPException(status_code=404, detail="响应档案不存在")
    return {
        "profile_id": profile_id,
        "version": profile["version"],
        "samples": db.get_profile_samples(profile_id, profile["version"]),
    }


@app.get("/jobs/{job_id}/export/timeseries.json")
def export_timeseries(job_id: str, kind: Optional[str] = Query(default=None)) -> dict:
    """导出逐时刻 JSON（炉温、各工件中心/表面温度与温差）。"""
    _load_job(job_id)
    rec = db.get_analysis(job_id, kind)
    if not rec:
        raise HTTPException(status_code=404, detail="没有分析结果")
    result = rec["result"]
    analysis = result.get("analysis", result)  # scheduled 结果多包一层
    return {
        "job_id": job_id,
        "kind": rec["kind"],
        "verdict": analysis.get("verdict"),
        **analysis.get("timeseries", {}),
    }


@app.get("/jobs/{job_id}/export/program.csv")
def export_program(job_id: str, source: str = Query(default="nominal")) -> PlainTextResponse:
    """导出窑炉程序 CSV；source=nominal|scheduled。"""
    job = _load_job(job_id)
    if source == "scheduled":
        rec = db.get_analysis(job_id, "scheduled")
        if not rec:
            raise HTTPException(status_code=404, detail="没有排程结果")
        program = rec["result"]["program"]
    else:
        program = job["payload"]["program"]

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["index", "kind", "start_c", "target_c", "rate_c_per_min", "duration_min", "locked"])
    cur = program["start_c"]
    for i, seg in enumerate(program["segments"]):
        if seg["kind"] == "hold":
            target = seg["target_c"] if seg["target_c"] is not None else cur
            rate = ""
            duration = round(seg["duration_min"], 3)
        else:
            target = seg["target_c"]
            rate = round(seg["rate_c_per_min"], 4)
            duration = round(abs(target - cur) / seg["rate_c_per_min"], 3) if seg["rate_c_per_min"] else ""
        w.writerow([i, seg["kind"], round(cur, 3), target, rate, duration, seg["locked"]])
        cur = target
    return PlainTextResponse(out.getvalue(), media_type="text/csv")
