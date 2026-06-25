"""Huey task queue, backed by SQLite (no external Redis server required - this
runs entirely on Windows with no extra services). The Flask API enqueues jobs
here; a separate `worker.py` consumer process executes them, persistently
holding AI models in memory/VRAM across jobs.
"""
from __future__ import annotations

from pathlib import Path

from huey import SqliteHuey

HUEY_DB_PATH = Path(__file__).resolve().parent.parent / "huey.sqlite3"

huey = SqliteHuey("framescript", filename=str(HUEY_DB_PATH))
