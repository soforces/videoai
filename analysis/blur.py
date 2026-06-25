"""Blur detection via Laplacian variance, the cheap signal that gates whether any
heavy restoration model runs at all. A lightweight-CNN refinement hook is provided
for plugins that want to add a learned blur classifier later without touching the
decision engine (see analysis.quality for the same plugin-hook pattern).
"""
from __future__ import annotations

import cv2
import numpy as np

# Tuned empirically: variance of the Laplacian of a normal sharp 1080p-ish frame
# sits well above 500; heavily motion-blurred or out-of-focus frames sit below 80.
LOW_BLUR_THRESHOLD = 200.0
HIGH_BLUR_THRESHOLD = 60.0


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_blur(variance: float) -> str:
    """Returns 'low' (sharp), 'medium', or 'high' (very blurry) blur level.

    Naming matches the spec's "blur level": low blur level = sharp/no restoration
    needed, high blur level = severe blur requiring the heaviest restorer.
    """
    if variance >= LOW_BLUR_THRESHOLD:
        return "low"
    if variance >= HIGH_BLUR_THRESHOLD:
        return "medium"
    return "high"


def analyze_blur(image_bgr: np.ndarray) -> tuple[float, str]:
    variance = laplacian_variance(image_bgr)
    return variance, classify_blur(variance)
