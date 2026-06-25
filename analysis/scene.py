"""Lightweight scene-change tagging for extraction modes other than scene_detect
(e.g. tagging which frames within an fps/all-frames extraction are scene boundaries,
useful for the decision engine and for temporal-consistency processing later).

Uses a simple normalized histogram-correlation diff between consecutive frames -
no extra ffmpeg pass needed since this runs on already-extracted frames in memory.
"""
from __future__ import annotations

import cv2
import numpy as np

SCENE_CHANGE_CORRELATION_THRESHOLD = 0.7


def _hist(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def detect_scene_changes(images_bgr: list[np.ndarray]) -> list[bool]:
    if not images_bgr:
        return []
    flags = [False]
    prev_hist = _hist(images_bgr[0])
    for img in images_bgr[1:]:
        hist = _hist(img)
        correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        flags.append(correlation < SCENE_CHANGE_CORRELATION_THRESHOLD)
        prev_hist = hist
    return flags
