"""Persistent job state, backed by SQLite so job status survives a worker/API
restart (the old implementation kept jobs in an in-memory dict that vanished on
every restart, which doesn't fly for a real queue-backed system).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "jobs.sqlite3"

# A "running" job whose updated_at hasn't moved in this long is assumed to
# belong to a worker that crashed mid-job (e.g. CUDA OOM) without reaching
# core/tasks.py's except/finally - there is no heartbeat thread; this is
# checked lazily on every get_job() call using the timestamp on_progress()
# already bumps, so no extra polling infrastructure is needed.
STALE_RUNNING_SECONDS = 5 * 60

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    # WAL lets the /status polling reader proceed concurrently with the
    # worker's update_job writes instead of blocking on SQLite's default
    # rollback-journal exclusive lock - matches the mode huey's own
    # SqliteStorage already uses for huey.sqlite3 (core/queue_app.py).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT,
            percent REAL DEFAULT 0,
            error TEXT,
            result_path TEXT,
            download_name TEXT,
            config_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    return conn


def create_job(job_id: str, config_json: str) -> None:
    now = time.time()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, stage, percent, config_json, created_at, updated_at) "
            "VALUES (?, 'queued', 'Queued', 0, ?, ?, ?)",
            (job_id, config_json, now, now),
        )


def update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with _lock, _connect() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", values)


def get_job(job_id: str) -> dict | None:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        job = dict(row)
        if job["status"] == "running" and time.time() - job["updated_at"] > STALE_RUNNING_SECONDS:
            timeout_message = (
                f"Islem {STALE_RUNNING_SECONDS // 60} dakikadan uzun suredir ilerlemedi "
                f"(worker sureci muhtemelen coktu). Lutfen yeniden deneyin."
            )
            with _lock, _connect() as write_conn:
                write_conn.execute(
                    "UPDATE jobs SET status = 'error', stage = 'Zaman asimi', error = ?, updated_at = ? "
                    "WHERE job_id = ? AND status = 'running'",
                    (timeout_message, time.time(), job_id),
                )
            job["status"] = "error"
            job["stage"] = "Zaman asimi"
            job["error"] = timeout_message
        return job


def delete_stale_jobs(ttl_seconds: int) -> list[str]:
    cutoff = time.time() - ttl_seconds
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT job_id, result_path FROM jobs WHERE created_at < ?", (cutoff,)).fetchall()
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
        return [r[0] for r in rows]
