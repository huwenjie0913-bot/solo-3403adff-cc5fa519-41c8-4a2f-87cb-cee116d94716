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
from .models import JobCreate, MeasurementBatch, ScheduleRequest
from .scheduler import optimize


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
    """创建作业（材料快照随作业入库），并立即按名义程序分析。"""
    payload = body.model_dump()
    job_id = db.create_job(payload)
    result = analyze(
        body.pieces,
        program=body.program,
        constraints=body.constraints,
        options=body.options,
    )
    db.save_analysis(job_id, "nominal", result)
    return {"job_id": job_id, "analysis": result}


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
    outcome = optimize(job["payload"], req.safety_factor, req.max_iterations)
    if outcome["status"] == "ok":
        db.save_analysis(
            job_id, "scheduled",
            {"program": outcome["program"], "analysis": outcome["analysis"]},
        )
    return outcome


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
