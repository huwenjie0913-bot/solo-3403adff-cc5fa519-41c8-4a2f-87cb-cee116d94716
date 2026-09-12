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
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    supersedes_id TEXT,
    created_at TEXT NOT NULL,
    result TEXT NOT NULL,
    request TEXT NOT NULL,
    PRIMARY KEY (id, version)
);
CREATE TABLE IF NOT EXISTS profile_samples (
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,           -- setpoint / channel
    channel_id TEXT,
    seq INTEGER NOT NULL,         -- 原始上传顺序（乱序取证用）
    time_min REAL NOT NULL,
    temp_c REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_samples ON profile_samples(profile_id, version);
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


# ---------------------------------------------------------------------------
# 窑炉热响应档案
#
# 档案一经写入即不可变；同名/同 id 重新上传产生新版本（version + 1，
# supersedes 指向前一版本）。作业 payload 内固化 profile_id/version 快照，
# 因此档案的新版本不会改写已有作业与分析。
# ---------------------------------------------------------------------------

def save_profile(request: dict, result: dict, supersedes_id: Optional[str] = None) -> dict:
    """创建档案或新版本，原始采样按上传顺序（seq）固化。"""
    profile_id = supersedes_id or uuid.uuid4().hex[:12]
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(MAX(version), 0) FROM profiles WHERE id=?",
            (profile_id,),
        ).fetchone()
        version = row[0] + 1
        c.execute(
            "INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?)",
            (profile_id, version, result["name"], result["status"],
             supersedes_id, _now(),
             json.dumps(result, ensure_ascii=False),
             json.dumps(request, ensure_ascii=False)),
        )
        rows = []
        for i, s in enumerate(request["setpoints"]):
            rows.append((profile_id, version, "setpoint", None, i,
                         s["time_min"], s["temp_c"]))
        for ch in request["channels"]:
            for i, s in enumerate(ch["samples"]):
                rows.append((profile_id, version, "channel", ch["channel_id"], i,
                             s["time_min"], s["temp_c"]))
        c.executemany(
            "INSERT INTO profile_samples VALUES (?,?,?,?,?,?,?)", rows
        )
    return {"id": profile_id, "version": version}


def get_profile(profile_id: str, version: Optional[int] = None) -> Optional[dict]:
    with _conn() as c:
        if version is None:
            row = c.execute(
                "SELECT * FROM profiles WHERE id=? ORDER BY version DESC LIMIT 1",
                (profile_id,),
            ).fetchone()
        else:
            row = c.execute(
                "SELECT * FROM profiles WHERE id=? AND version=?",
                (profile_id, version),
            ).fetchone()
    if not row:
        return None
    result = json.loads(row["result"])
    return {
        "id": row["id"],
        "version": row["version"],
        "name": row["name"],
        "status": row["status"],
        "supersedes_id": row["supersedes_id"],
        "created_at": row["created_at"],
        "request": json.loads(row["request"]),
        **result,
    }


def list_profiles() -> list:
    with _conn() as c:
        rows = c.execute(
            """SELECT p.id, p.version, p.name, p.status, p.created_at, p.supersedes_id,
                      (SELECT MAX(version) FROM profiles WHERE id=p.id) AS latest
               FROM profiles p
               ORDER BY p.created_at DESC, p.version DESC""",
        ).fetchall()
    return [dict(r) for r in rows]


def get_profile_samples(profile_id: str, version: int) -> list:
    """按原始上传顺序（seq）返回原始采样，含设定轴与各通道。"""
    with _conn() as c:
        rows = c.execute(
            """SELECT kind, channel_id, seq, time_min, temp_c
               FROM profile_samples
               WHERE profile_id=? AND version=?
               ORDER BY kind, channel_id, seq""",
            (profile_id, version),
        ).fetchall()
    return [dict(r) for r in rows]
