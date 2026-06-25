"""Decides, per frame, which models (if any) run - the mechanism that keeps total
job time under the 60s target by never running every model on every frame.

Rules implement the spec verbatim:

Face damage (0-100):     0-30 skip / 30-60 GFPGAN / 60-100 CodeFormer
Blur level:               low skip / medium NAFNet / high Restormer
Upscale:                  only if user requested it OR output res > input res
Mode gating:
  fast:      GFPGAN only if face detected; Real-ESRGAN only if upscale needed;
             NAFNet only if blur detected; heavy models skipped otherwise.
  balanced:  GFPGAN/CodeFormer chosen by face damage score; Real-ESRGAN enabled;
             SwinIR optional fallback; NAFNet + partial Restormer.
  max_quality: CodeFormer preferred; Real-ESRGAN+SwinIR ensemble; full Restormer;
             SAM2 allowed.
"""
from __future__ import annotations

from core.models import FrameAnalysis, FrameDecision, PipelineConfig, ProcessingMode

FACE_SKIP_MAX = 30.0
FACE_GFPGAN_MAX = 60.0


def needs_upscale(config: PipelineConfig, width: int, height: int) -> bool:
    if not config.enable_upscale:
        return False
    if config.target_width and config.target_height:
        return config.target_width > width or config.target_height > height
    # No explicit target: upscale is still considered "requested" if the user
    # turned the stage on at all (spec: "user requests it OR output > input").
    return True


def decide_face_plugin(mode: ProcessingMode, damage_score: float) -> str | None:
    if damage_score <= FACE_SKIP_MAX:
        return None
    if mode == ProcessingMode.MAX_QUALITY:
        return "codeformer"
    if mode == ProcessingMode.FAST:
        return "gfpgan" if damage_score <= FACE_GFPGAN_MAX else "gfpgan"  # fast mode never escalates to CodeFormer
    # balanced: pick by damage score
    return "gfpgan" if damage_score <= FACE_GFPGAN_MAX else "codeformer"


def decide_restoration_plugin(mode: ProcessingMode, blur_level: str) -> str | None:
    if blur_level == "low":
        return None
    if blur_level == "medium":
        return "nafnet"
    # high blur
    if mode == ProcessingMode.FAST:
        return "nafnet"  # fast mode keeps the cheap restorer even for high blur
    return "restormer"


def decide_upscale_plugin(mode: ProcessingMode) -> str:
    if mode == ProcessingMode.MAX_QUALITY:
        return "realesrgan_swinir_ensemble"
    return "realesrgan"


def build_decision(frame: FrameAnalysis, config: PipelineConfig) -> FrameDecision:
    decision = FrameDecision(index=frame.index)

    if frame.is_duplicate and config.enable_dedup:
        decision.skip = True
        decision.reasons["skip"] = "duplicate frame"
        return decision

    if config.enable_face_restore and frame.faces:
        worst_face = max(frame.faces, key=lambda f: f.damage_score)
        plugin = decide_face_plugin(config.mode, worst_face.damage_score)
        if plugin:
            decision.face_plugin = plugin
            decision.reasons["face"] = f"max face damage={worst_face.damage_score}"

    if config.enable_upscale and needs_upscale(config, frame.width, frame.height):
        decision.upscale_plugin = decide_upscale_plugin(config.mode)
        decision.reasons["upscale"] = "target resolution exceeds source"

    if config.enable_blur_restore:
        plugin = decide_restoration_plugin(config.mode, frame.blur_level)
        if plugin:
            decision.restoration_plugin = plugin
            decision.reasons["restoration"] = f"blur_level={frame.blur_level}"

    if config.enable_background_removal:
        decision.background_plugin = "rembg" if config.mode != ProcessingMode.MAX_QUALITY else "sam2"

    if config.enable_segmentation and config.mode == ProcessingMode.MAX_QUALITY:
        decision.segmentation_plugin = "sam2"

    if config.enable_ocr and frame.has_text:
        decision.ocr_plugin = "paddleocr"

    return decision


def build_decision_map(frames: list[FrameAnalysis], config: PipelineConfig) -> dict[int, FrameDecision]:
    return {frame.index: build_decision(frame, config) for frame in frames}
