"""SQLite 持久化：作业（含材料快照）、实测炉温、分析结果。"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(
    os.environ.get("ANNEALING_DB", Path(__file__).resolve().parent.parent / "annealing.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    job_id TEXT NOT NULL,
    time_min REAL NOT NULL,
    temp_c REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_measurements_job ON measurements(job_id);
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_job ON analyses(job_id);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(payload: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs VALUES (?,?,?,?)",
            (job_id, payload["name"], _now(), json.dumps(payload, ensure_ascii=False)),
        )
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "payload": json.loads(row["payload"]),
    }


def list_jobs() -> list:
    with _conn() as c:
        rows = c.execute("SELECT id, name, created_at FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def save_analysis(job_id: str, kind: str, result: dict) -> str:
    analysis_id = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute(
            "INSERT INTO analyses VALUES (?,?,?,?,?)",
            (analysis_id, job_id, kind, _now(), json.dumps(result, ensure_ascii=False)),
        )
    return analysis_id


def list_analyses(job_id: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, kind, created_at FROM analyses WHERE job_id=? ORDER BY created_at DESC",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analysis(job_id: str, kind: Optional[str] = None) -> Optional[dict]:
    sql = "SELECT * FROM analyses WHERE job_id=?"
    args: list = [job_id]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    sql += " ORDER BY created_at DESC LIMIT 1"
    with _conn() as c:
        row = c.execute(sql, args).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "kind": row["kind"],
        "created_at": row["created_at"],
        "result": json.loads(row["result"]),
    }


def replace_measurements(job_id: str, samples: list) -> None:
    with _conn() as c:
        c.execute("DELETE FROM measurements WHERE job_id=?", (job_id,))
        c.executemany(
            "INSERT INTO measurements VALUES (?,?,?)",
            [(job_id, s["time_min"], s["temp_c"]) for s in samples],
        )


def get_measurements(job_id: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT time_min, temp_c FROM measurements WHERE job_id=? ORDER BY time_min",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]
