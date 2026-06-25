"""Quality scoring.

MUSIQ (the spec's mandated neural quality scorer) is exposed as a plugin under
plugins/analysis_models/musiq_plugin.py; when its checkpoint isn't installed,
the decision engine falls back to this heuristic scorer, which is REAL classical
CV (not a mock) combining sharpness, contrast and noise estimates into a 0-100
quality score plus separate blur/sharpness/artifact sub-scores, matching the
spec's required output fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityScores:
    quality_score: float
    blur_score: float
    sharpness_score: float
    artifact_score: float


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _contrast(gray: np.ndarray) -> float:
    return float(gray.std())


def _noise_estimate(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = cv2.absdiff(gray, blurred)
    return float(diff.mean())


def score_quality(image_bgr: np.ndarray) -> QualityScores:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = _sharpness(gray)
    contrast = _contrast(gray)
    noise = _noise_estimate(gray)

    sharpness_score = max(0.0, min(100.0, sharpness / 8.0))
    contrast_score = max(0.0, min(100.0, contrast * 1.5))
    artifact_score = max(0.0, min(100.0, noise * 8.0))  # higher = more artifacts/noise
    blur_score = max(0.0, min(100.0, 100.0 - sharpness_score))

    quality_score = max(0.0, min(100.0, 0.5 * sharpness_score + 0.3 * contrast_score + 0.2 * (100.0 - artifact_score)))

    return QualityScores(
        quality_score=round(quality_score, 1),
        blur_score=round(blur_score, 1),
        sharpness_score=round(sharpness_score, 1),
        artifact_score=round(artifact_score, 1),
    )
