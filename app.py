#!/usr/bin/env python3
"""Flask API for the AI video frame processing pipeline.

Thin by design: this process only accepts uploads, builds an ExtractionRequest +
PipelineConfig, enqueues a Huey job, and serves status/results/presets. All
actual frame extraction and AI processing happens in worker.py's consumer
process so models stay persistently loaded across jobs (see core/tasks.py).

Run:
  python worker.py     (in one terminal - the GPU worker)
  python app.py         (in another - the web API)
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from core.job_store import create_job, delete_stale_jobs, get_job, update_job
from core.models import ExtractionMode, ExtractionRequest, PipelineConfig, ProcessingMode
from core.singleton_lock import SingleInstanceError, acquire_singleton_lock
from core.tasks import process_video_job

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
JOB_TTL_SECONDS = 30 * 60
PRESETS_DIR = Path("presets")
PRESETS_DIR.mkdir(exist_ok=True)


def _parse_extraction_request(form) -> ExtractionRequest:
    mode = ExtractionMode(form.get("extraction_mode", ExtractionMode.N_SAMPLE.value))
    return ExtractionRequest(
        mode=mode,
        fps=form.get("fps", type=float),
        n_frames=form.get("n_frames", type=int),
        start_seconds=form.get("start_seconds", type=float),
        end_seconds=form.get("end_seconds", type=float),
        scene_threshold=form.get("scene_threshold", default=0.3, type=float),
    )


def _parse_pipeline_config(form) -> PipelineConfig:
    return PipelineConfig(
        mode=ProcessingMode(form.get("processing_mode", ProcessingMode.FAST.value)),
        target_width=form.get("target_width", type=int),
        target_height=form.get("target_height", type=int),
        enable_upscale=form.get("enable_upscale") == "on",
        enable_face_restore=form.get("enable_face_restore") == "on",
        enable_blur_restore=form.get("enable_blur_restore") == "on",
        enable_dedup=form.get("enable_dedup") == "on",
        enable_background_removal=form.get("enable_background_removal") == "on",
        enable_segmentation=form.get("enable_segmentation") == "on",
        enable_ocr=form.get("enable_ocr") == "on",
        enable_quality_scoring=form.get("enable_quality_scoring") == "on",
        export_formats=[form.get("export_format", "jpg")] + (
            ["contact_sheet"] if form.get("contact_sheet") == "on" else []
        ),
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    for stale_id in delete_stale_jobs(JOB_TTL_SECONDS):
        work_dir = Path("uploads") / stale_id
        if work_dir.exists():
            for f in work_dir.rglob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)

    video_file = request.files.get("video")
    if not video_file or video_file.filename == "":
        abort(400, "Video dosyasi secilmedi.")

    suffix = Path(video_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        abort(400, f"Desteklenmeyen dosya turu. Desteklenenler: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    try:
        extraction_request = _parse_extraction_request(request.form)
        config = _parse_pipeline_config(request.form)
    except ValueError as exc:
        abort(400, str(exc))

    job_id = uuid.uuid4().hex
    work_dir = Path("uploads") / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / f"input{suffix}"
    video_file.save(input_path)

    extraction_json = json.dumps({
        "mode": extraction_request.mode.value,
        "fps": extraction_request.fps,
        "n_frames": extraction_request.n_frames,
        "start_seconds": extraction_request.start_seconds,
        "end_seconds": extraction_request.end_seconds,
        "scene_threshold": extraction_request.scene_threshold,
    })
    config_json = json.dumps({
        "mode": config.mode.value,
        "target_width": config.target_width,
        "target_height": config.target_height,
        "enable_upscale": config.enable_upscale,
        "enable_face_restore": config.enable_face_restore,
        "enable_blur_restore": config.enable_blur_restore,
        "enable_dedup": config.enable_dedup,
        "enable_background_removal": config.enable_background_removal,
        "enable_segmentation": config.enable_segmentation,
        "enable_ocr": config.enable_ocr,
        "enable_quality_scoring": config.enable_quality_scoring,
        "export_formats": config.export_formats,
    })

    create_job(job_id, config_json)
    process_video_job(job_id, str(input_path), str(work_dir), extraction_json, config_json)

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, "Is bulunamadi.")
    return jsonify({
        "status": job.get("status"),
        "stage": job.get("stage"),
        "percent": job.get("percent", 0),
        "error": job.get("error"),
    })


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    job = get_job(job_id)
    if not job or job.get("status") != "done" or not job.get("result_path"):
        abort(404, "Dosya hazir degil.")
    result_path = Path(job["result_path"])
    if not result_path.exists():
        abort(404, "Dosya bulunamadi.")
    return send_file(
        result_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=job.get("download_name", "frames.zip"),
    )


@app.route("/presets", methods=["GET"])
def list_presets():
    presets = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        presets.append(f.stem)
    return jsonify(presets)


@app.route("/presets/<name>", methods=["GET"])
def get_preset(name):
    preset_path = PRESETS_DIR / f"{name}.json"
    if not preset_path.exists():
        abort(404, "Preset bulunamadi.")
    return jsonify(json.loads(preset_path.read_text(encoding="utf-8")))


@app.route("/presets/<name>", methods=["POST"])
def save_preset(name):
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
    if not safe_name:
        abort(400, "Gecersiz preset adi.")
    preset_path = PRESETS_DIR / f"{safe_name}.json"
    preset_path.write_text(json.dumps(request.get_json(force=True)), encoding="utf-8")
    return jsonify({"saved": safe_name})


if __name__ == "__main__":
    try:
        acquire_singleton_lock("app")
    except SingleInstanceError as exc:
        print(f"[app.py] {exc}", file=sys.stderr)
        sys.exit(1)
    Path("uploads").mkdir(exist_ok=True)
    # use_reloader=False: the debug reloader re-execs this process, which
    # would otherwise need to dodge its own singleton lock - simpler to just
    # disable it for this local single-developer tool.
    app.run(debug=True, threaded=True, use_reloader=False)
