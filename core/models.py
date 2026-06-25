"""Shared dataclasses used across extraction, analysis, decision and plugin layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ProcessingMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    MAX_QUALITY = "max_quality"


class ExtractionMode(str, Enum):
    ALL_FRAMES = "all_frames"
    FPS = "fps"
    N_SAMPLE = "n_sample"
    TIMESTAMP_RANGE = "timestamp_range"
    SCENE_DETECT = "scene_detect"


@dataclass
class ExtractionRequest:
    mode: ExtractionMode
    fps: float | None = None
    n_frames: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    scene_threshold: float = 0.3


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    damage_score: float = 0.0


@dataclass
class FrameAnalysis:
    index: int
    path: Path
    width: int = 0
    height: int = 0
    blur_score: float = 0.0
    blur_level: str = "low"  # low | medium | high
    faces: list[FaceBox] = field(default_factory=list)
    is_duplicate: bool = False
    is_scene_change: bool = False
    has_text: bool = False
    ocr_text: str | None = None
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    artifact_score: float = 0.0
    musiq_score: float | None = None
    anomaly_score: float | None = None
    perceptual_hash: str | None = None


@dataclass
class FrameDecision:
    index: int
    skip: bool = False
    upscale_plugin: str | None = None
    face_plugin: str | None = None
    restoration_plugin: str | None = None
    segmentation_plugin: str | None = None
    background_plugin: str | None = None
    ocr_plugin: str | None = None
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    mode: ProcessingMode = ProcessingMode.FAST
    target_width: int | None = None
    target_height: int | None = None
    enable_upscale: bool = True
    enable_face_restore: bool = True
    enable_blur_restore: bool = True
    enable_dedup: bool = True
    enable_background_removal: bool = False
    enable_segmentation: bool = False
    enable_ocr: bool = False
    enable_quality_scoring: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["jpg"])
    stage_order: list[str] = field(
        default_factory=lambda: [
            "extract", "analyze", "dedup", "decide", "face", "upscale",
            "restoration", "background", "segmentation", "ocr", "quality", "export",
        ]
    )
