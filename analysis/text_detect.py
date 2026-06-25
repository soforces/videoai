"""Cheap text-presence heuristic used to decide whether the (heavy) PaddleOCR
plugin is worth running on a given frame at all. Uses MSER region density, which
reliably flags frames containing dense high-contrast glyph-like regions (signage,
captions, UI overlays) without running any neural network.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_TEXT_REGIONS = 6


def has_probable_text(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setMinArea(30)
    mser.setMaxArea(2000)
    regions, _ = mser.detectRegions(gray)

    text_like = 0
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        if h == 0:
            continue
        aspect = w / h
        if 0.1 < aspect < 8 and 6 <= h <= 80:
            text_like += 1
        if text_like >= MIN_TEXT_REGIONS:
            return True
    return text_like >= MIN_TEXT_REGIONS
