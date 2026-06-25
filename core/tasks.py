"""Huey task that runs a full pipeline job. Executed by the worker.py consumer
process, never by the Flask request thread - this is what lets models stay
persistently loaded across jobs instead of reloading per request.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from core.job_store import update_job
from core.models import ExtractionRequest, PipelineConfig
from core.pipeline import run_pipeline
from core.queue_app import huey


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except ImportError:
        return False


@huey.task()
def process_video_job(job_id: str, input_path_str: str, work_dir_str: str, extraction_json: str, config_json: str) -> None:
    input_path = Path(input_path_str)
    work_dir = Path(work_dir_str)
    cuda_oom = False

    def on_progress(stage: str, percent: float) -> None:
        update_job(job_id, status="running", stage=stage, percent=percent)

    try:
        update_job(job_id, status="running", stage="Basliyor", percent=0)
        extraction_request = ExtractionRequest(**json.loads(extraction_json))
        config = PipelineConfig(**json.loads(config_json))

        result = run_pipeline(input_path, work_dir, extraction_request, config, on_progress=on_progress)

        update_job(
            job_id,
            status="done",
            stage="Tamamlandi",
            percent=100,
            result_path=str(result["zip_path"]),
            download_name=f"{input_path.stem}_frames.zip",
        )
    except Exception as exc:  # noqa: BLE001 - report any failure to the job store
        update_job(job_id, status="error", stage="Hata", error=str(exc))
        cuda_oom = _is_cuda_oom(exc)
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except OSError:
            pass
        raw_dir = work_dir / "raw"
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
        processed_dir = work_dir / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir, ignore_errors=True)

    if cuda_oom:
        # A CUDA OOM can leave the GPU context unusable for the rest of this
        # process's life (observed: identical OOM errors on every subsequent
        # job, regardless of how much VRAM is actually freed). Exit non-zero
        # so worker.py's supervisor loop restarts us as a fresh process with
        # a clean GPU context, instead of every future job failing the same
        # way until someone notices and restarts manually.
        print(
            f"[tasks.py] job {job_id} hit a CUDA OOM; exiting this consumer "
            f"process so the supervisor restarts it cleanly.",
            file=sys.stderr,
        )
        sys.stderr.flush()
        os._exit(1)
