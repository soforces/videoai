"""FFmpeg-only frame extraction.

Supports MP4/MOV/MKV/AVI/WEBM containers with H264/H265/AV1/VP9/ProRes/DNxHD
codecs (ffmpeg handles container/codec demuxing transparently - no per-codec
branching is needed here). Always extracts at the source resolution; any
resizing happens later in the upscale stage, never during extraction.

Five extraction modes, all driven by ffmpeg's own filters/flags:
  - all_frames:       every decoded frame, passthrough fps
  - fps:              fixed output fps via -vf fps=N
  - n_sample:         N frames evenly spaced across full duration
  - timestamp_range:  all frames between start/end seconds (-ss/-to)
  - scene_detect:     ffmpeg's scene-change filter (select='gt(scene,T)')
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from core.models import ExtractionMode, ExtractionRequest

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if base.exists():
        for exe in base.rglob(f"{name}.exe"):
            return str(exe)
    sys.exit(f"{name} not found on PATH.")


FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")


def probe_duration_seconds(input_path: Path) -> float:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(input_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return float(result.stdout.strip())


def probe_video_info(input_path: Path) -> dict:
    # ffprobe's csv writer emits fields in its own internal stream order, not the
    # order passed to -show_entries, so use -of default and parse key=value lines.
    result = subprocess.run(
        [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,r_frame_rate,nb_frames",
            "-of", "default=noprint_wrappers=1", str(input_path),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    info: dict = {}
    for line in result.stdout.strip().splitlines():
        key, _, value = line.partition("=")
        info[key] = value
    return {
        "width": int(info.get("width", 0)),
        "height": int(info.get("height", 0)),
        "codec": info.get("codec_name", ""),
        "nb_frames": info.get("nb_frames"),
    }


def _run_extract(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed: {proc.stderr[-2000:]}")


def extract_frames(input_path: Path, out_dir: Path, request: ExtractionRequest) -> list[Path]:
    """Extracts frames as lossless PNGs (frame_%06d.png) at source resolution into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"

    base_cmd = [FFMPEG, "-y"]

    if request.mode == ExtractionMode.TIMESTAMP_RANGE:
        if request.start_seconds is not None:
            base_cmd += ["-ss", str(request.start_seconds)]
        base_cmd += ["-i", str(input_path)]
        if request.end_seconds is not None:
            duration = request.end_seconds - (request.start_seconds or 0)
            base_cmd += ["-t", str(max(duration, 0))]
        base_cmd += ["-vsync", "0", str(pattern)]

    elif request.mode == ExtractionMode.ALL_FRAMES:
        base_cmd += ["-i", str(input_path), "-vsync", "0", str(pattern)]

    elif request.mode == ExtractionMode.FPS:
        fps = request.fps or 1.0
        base_cmd += ["-i", str(input_path), "-vf", f"fps={fps}", "-vsync", "0", str(pattern)]

    elif request.mode == ExtractionMode.N_SAMPLE:
        n = request.n_frames or 1
        duration = probe_duration_seconds(input_path)
        info = probe_video_info(input_path)
        nb_frames_str = info.get("nb_frames")
        total_frames = int(nb_frames_str) if nb_frames_str and nb_frames_str.isdigit() else None

        if total_frames and total_frames > 0:
            stride = max(total_frames // n, 1)
            base_cmd += ["-i", str(input_path), "-vf", f"select='not(mod(n\\,{stride}))'",
                         "-vsync", "0", str(pattern)]
        else:
            sampled_fps = n / duration if duration > 0 else 1.0
            base_cmd += ["-i", str(input_path), "-vf", f"fps={sampled_fps}", "-vsync", "0", str(pattern)]

    elif request.mode == ExtractionMode.SCENE_DETECT:
        threshold = request.scene_threshold
        base_cmd += [
            "-i", str(input_path),
            "-vf", f"select='gt(scene,{threshold})'", "-vsync", "0", str(pattern),
        ]
    else:
        raise ValueError(f"Unknown extraction mode: {request.mode}")

    _run_extract(base_cmd)

    frames = sorted(out_dir.glob("frame_*.png"))
    if request.mode == ExtractionMode.N_SAMPLE and request.n_frames and len(frames) > request.n_frames:
        step = len(frames) / request.n_frames
        keep_idx = {round(i * step) for i in range(request.n_frames)}
        for i, f in enumerate(frames):
            if i not in keep_idx:
                f.unlink(missing_ok=True)
        frames = sorted(out_dir.glob("frame_*.png"))
    return frames
