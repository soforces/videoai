"""Pipeline orchestrator: extraction -> analysis -> dedup -> decision -> AI
plugins -> quality scoring -> export, with persistent progress reporting.

Stage order is driven by PipelineConfig.stage_order so users can reorder/disable
stages without touching this file; this module just executes whichever stages
are enabled, in the configured order, and skips ones with no work to do.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2

from analysis.blur import analyze_blur
from analysis.dedup import filter_frames, is_low_value, phash
from analysis.face import detect_faces
from analysis.quality import score_quality
from analysis.scene import detect_scene_changes
from analysis.text_detect import has_probable_text
from core.decision_engine import build_decision_map
from core.models import ExtractionRequest, FrameAnalysis, FrameDecision, PipelineConfig
from extraction.ffmpeg_extract import extract_frames
from plugins.base import PluginNotInstalledError
from plugins.export.exporters import build_contact_sheet, build_zip
from plugins.registry import registry

ProgressCallback = Callable[[str, float], None]


def _noop_progress(stage: str, percent: float) -> None:
    pass


def run_pipeline(
    input_path: Path,
    work_dir: Path,
    extraction_request: ExtractionRequest,
    config: PipelineConfig,
    on_progress: ProgressCallback = _noop_progress,
) -> dict:
    """Returns {"frames": [Path...], "analyses": [FrameAnalysis...], "zip_path": Path, "contact_sheet_path": Path|None}."""
    raw_dir = work_dir / "raw"
    processed_dir = work_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    on_progress("Frames cikariliyor (ffmpeg)", 2)
    frame_paths = extract_frames(input_path, raw_dir, extraction_request)
    if not frame_paths:
        raise RuntimeError("Videodan hic kare cikarilamadi.")

    on_progress("Kareler analiz ediliyor", 10)
    images = [cv2.imread(str(p)) for p in frame_paths]
    hashes = [phash(img) for img in images]
    low_value_flags = [is_low_value(img) for img in images]
    scene_flags = detect_scene_changes(images)

    dedup_result = filter_frames(hashes, low_value_flags) if config.enable_dedup else None
    duplicate_indices = set(dedup_result.duplicate_of.keys()) | dedup_result.low_value_indices if dedup_result else set()

    total_frames = max(len(images), 1)
    # ~20 progress updates across this loop regardless of frame count, so a
    # large extraction doesn't hammer job_store.update_job once per frame
    # while still distinguishing "slow but healthy" from "stuck" in the UI.
    progress_every = max(1, total_frames // 20)

    analyses: list[FrameAnalysis] = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        blur_var, blur_level = analyze_blur(img)
        quality = score_quality(img)
        analyses.append(
            FrameAnalysis(
                index=i,
                path=frame_paths[i],
                width=w,
                height=h,
                blur_score=blur_var,
                blur_level=blur_level,
                faces=detect_faces(img) if config.enable_face_restore else [],
                is_duplicate=i in duplicate_indices,
                is_scene_change=scene_flags[i] if i < len(scene_flags) else False,
                has_text=has_probable_text(img) if config.enable_ocr else False,
                quality_score=quality.quality_score,
                sharpness_score=quality.sharpness_score,
                artifact_score=quality.artifact_score,
                perceptual_hash=str(hashes[i]),
            )
        )
        if (i + 1) % progress_every == 0 or (i + 1) == total_frames:
            pct = 10 + ((i + 1) / total_frames) * 10
            on_progress(f"Kareler analiz ediliyor ({i + 1}/{total_frames})", round(pct, 1))
    on_progress("Karar haritasi olusturuluyor", 20)
    decision_map = build_decision_map(analyses, config)

    # Free any GPU model a *previous* job left resident that this video's own
    # decision map never asks for - the whole point of per-frame decisions is
    # that only the AI this specific video actually needs should be running;
    # a worker that has, over its lifetime, touched every plugin shouldn't
    # keep all of them loaded just because this job doesn't need them.
    _unload_unneeded_gpu_plugins(_collect_needed_plugins(decision_map, config))

    kept_analyses = [a for a in analyses if not decision_map[a.index].skip]
    total_kept = max(len(kept_analyses), 1)

    output_paths: list[Path] = []
    mask_paths: list[Path] = []
    for processed_count, analysis in enumerate(kept_analyses, start=1):
        img = images[analysis.index]
        decision = decision_map[analysis.index]

        if decision.face_plugin:
            img = _apply_plugin("faces", decision.face_plugin, img)
        if decision.restoration_plugin:
            img = _apply_plugin("restoration", decision.restoration_plugin, img)
        if decision.upscale_plugin == "realesrgan_swinir_ensemble":
            img = _upscale_ensemble(img, scale=2)
        elif decision.upscale_plugin:
            img = _apply_plugin("upscale", decision.upscale_plugin, img, scale=2)
        # SAM2 only registers under "segmentation" (it returns a raw mask, not
        # a composited image like rembg) - max_quality's background_plugin
        # decision of "sam2" composites that same mask itself rather than
        # looking up a non-existent "background"-category "sam2" plugin, and
        # reuses one mask computation for both the background-removal and
        # segmentation-mask-export stages instead of running SAM2 twice.
        sam2_mask = None
        if decision.background_plugin == "sam2" or decision.segmentation_plugin == "sam2":
            sam2_mask = _apply_plugin("segmentation", "sam2", img)
            if sam2_mask is img:
                sam2_mask = None

        if decision.background_plugin == "sam2":
            if sam2_mask is not None:
                img = _composite_on_white(img, sam2_mask)
        elif decision.background_plugin:
            img = _apply_plugin("background", decision.background_plugin, img)

        if decision.segmentation_plugin == "sam2":
            if sam2_mask is not None:
                mask_path = processed_dir / f"frame_{analysis.index:06d}_mask.png"
                cv2.imwrite(str(mask_path), sam2_mask)
                mask_paths.append(mask_path)
        elif decision.segmentation_plugin:
            mask = _apply_plugin("segmentation", decision.segmentation_plugin, img)
            if mask is not img:
                mask_path = processed_dir / f"frame_{analysis.index:06d}_mask.png"
                cv2.imwrite(str(mask_path), mask)
                mask_paths.append(mask_path)
        if decision.ocr_plugin:
            try:
                plugin = registry.get("ocr", decision.ocr_plugin)
                plugin.ensure_loaded()
                analysis.ocr_text = plugin.process(img)
            except (KeyError, PluginNotInstalledError):
                pass
        if config.enable_quality_scoring:
            analysis.musiq_score = _score_optional("analysis_models", "musiq", img)
            analysis.anomaly_score = _score_optional("analysis_models", "clip_anomaly", img)

        _release_cached_gpu_memory()

        out_path = processed_dir / f"frame_{analysis.index:06d}.png"
        cv2.imwrite(str(out_path), img)
        output_paths.append(out_path)

        pct = 20 + (processed_count / total_kept) * 65
        on_progress(f"AI islem ({processed_count}/{total_kept})", round(pct, 1))

    on_progress("Disa aktariliyor", 90)
    export_format = config.export_formats[0] if config.export_formats else "jpg"
    exporter = registry.get("export", export_format)
    final_paths: list[Path] = []
    for out_path in output_paths:
        final_path = out_path.with_suffix(f".{export_format}")
        exporter.process_file(out_path, final_path)
        final_paths.append(final_path)

    extra_files: list[Path] = list(mask_paths)

    ocr_results = {a.index: a.ocr_text for a in analyses if a.ocr_text}
    if ocr_results:
        ocr_path = work_dir / "ocr_text.json"
        ocr_path.write_text(json.dumps(ocr_results, ensure_ascii=False, indent=2), encoding="utf-8")
        extra_files.append(ocr_path)

    if config.enable_quality_scoring:
        quality_report = {
            a.index: {
                "quality_score": a.quality_score,
                "musiq_score": a.musiq_score,
                "anomaly_score": a.anomaly_score,
                "blur_level": a.blur_level,
            }
            for a in analyses
            if not decision_map[a.index].skip
        }
        report_path = work_dir / "quality_report.json"
        report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        extra_files.append(report_path)

    zip_path = work_dir / "frames.zip"
    build_zip(final_paths, zip_path, extra_files=extra_files)

    contact_sheet_path = None
    if "contact_sheet" in config.export_formats:
        contact_sheet_path = work_dir / "contact_sheet.jpg"
        build_contact_sheet(final_paths, contact_sheet_path)

    on_progress("Tamamlandi", 100)
    return {
        "frames": final_paths,
        "analyses": analyses,
        "zip_path": zip_path,
        "contact_sheet_path": contact_sheet_path,
    }


def _collect_needed_plugins(decision_map: dict[int, FrameDecision], config: PipelineConfig) -> set[tuple[str, str]]:
    """The exact set of (category, name) plugins this job's own decisions
    will actually call - the per-video analysis already computed in
    build_decision_map(), just inverted into a lookup set instead of being
    consulted frame-by-frame."""
    needed: set[tuple[str, str]] = set()
    for decision in decision_map.values():
        if decision.skip:
            continue
        if decision.face_plugin:
            needed.add(("faces", decision.face_plugin))
        if decision.restoration_plugin:
            needed.add(("restoration", decision.restoration_plugin))
        if decision.upscale_plugin == "realesrgan_swinir_ensemble":
            needed.add(("upscale", "realesrgan"))
            needed.add(("upscale", "swinir"))
        elif decision.upscale_plugin:
            needed.add(("upscale", decision.upscale_plugin))
        if decision.background_plugin == "sam2" or decision.segmentation_plugin == "sam2":
            needed.add(("segmentation", "sam2"))
        elif decision.background_plugin:
            needed.add(("background", decision.background_plugin))
        if decision.ocr_plugin:
            needed.add(("ocr", decision.ocr_plugin))
    if config.enable_quality_scoring:
        needed.add(("analysis_models", "musiq"))
        needed.add(("analysis_models", "clip_anomaly"))
    return needed


def _unload_unneeded_gpu_plugins(needed: set[tuple[str, str]]) -> None:
    for category, plugins_in_category in registry.list_all().items():
        for name, plugin in plugins_in_category.items():
            if (category, name) not in needed and plugin._loaded and getattr(plugin, "device", None) == "cuda":
                plugin.unload()
    _release_cached_gpu_memory()


def _release_cached_gpu_memory() -> None:
    """Frees PyTorch's cached-but-unused CUDA blocks after each frame. A
    single job can touch several architecturally different models (e.g.
    SwinIR's transformer attention buffers vs. Real-ESRGAN's conv tiles) on
    this machine's 4GB GPU; without this, each model's peak-shaped cached
    blocks pile up unused but still reserved, fragmenting the small budget
    until even much smaller allocations fail."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _apply_plugin(category: str, name: str, image, **kwargs):
    from plugins.base import run_with_gpu_oom_retry

    try:
        plugin = registry.get(category, name)
        plugin.ensure_loaded()
        # The heaviest single allocations in this pipeline happen during
        # inference (e.g. SwinIR's tiled forward pass), not while loading
        # weights - wrapping the process() call itself, not just load(), is
        # what actually catches and recovers from those.
        return run_with_gpu_oom_retry(lambda: plugin.process(image, **kwargs), exclude=plugin)
    except (KeyError, PluginNotInstalledError):
        return image


def _composite_on_white(image, mask) -> "cv2.Mat":
    """Composites `image` over white using a uint8 0/255 mask, matching
    plugins/background/rembg_plugin.py's compositing convention."""
    alpha = mask.astype("float32")[:, :, None] / 255.0
    white = (255 * (1 - alpha))
    return (image.astype("float32") * alpha + white).astype("uint8")


def _score_optional(category: str, name: str, image) -> float | None:
    from plugins.base import run_with_gpu_oom_retry

    try:
        plugin = registry.get(category, name)
        plugin.ensure_loaded()
        return run_with_gpu_oom_retry(lambda: plugin.process(image), exclude=plugin)
    except (KeyError, PluginNotInstalledError):
        return None


def _upscale_ensemble(image, scale: int):
    """max_quality's Real-ESRGAN+SwinIR ensemble: run both x4 upscalers and
    pixel-average their output (simple, real ensembling - not a placeholder),
    falling back to whichever one ran if the other isn't available."""
    a = _apply_plugin("upscale", "realesrgan", image, scale=scale)
    b = _apply_plugin("upscale", "swinir", image, scale=scale)
    if a is image:
        return b
    if b is image:
        return a
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return cv2.addWeighted(a, 0.5, b, 0.5, 0)
