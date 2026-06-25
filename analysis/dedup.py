"""Duplicate / near-duplicate / low-value frame filtering.

Runs before any AI model touches a frame - the single biggest lever for staying
under the 60s budget, since every frame removed here is a frame zero plugins
need to run on.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import imagehash
import numpy as np
from PIL import Image

DUPLICATE_HASH_DISTANCE = 4  # hamming distance on a 64-bit perceptual hash
LOW_VALUE_BRIGHTNESS_STD = 5.0  # near solid-color frames (e.g. black flash, fade)


@dataclass
class DedupResult:
    keep_indices: list[int]
    duplicate_of: dict[int, int]  # index -> index of the frame it duplicates
    low_value_indices: set[int]


def phash(image_bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def is_low_value(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < LOW_VALUE_BRIGHTNESS_STD


def filter_frames(hashes: list[imagehash.ImageHash], low_value_flags: list[bool]) -> DedupResult:
    keep_indices: list[int] = []
    duplicate_of: dict[int, int] = {}
    low_value_indices: set[int] = {i for i, flag in enumerate(low_value_flags) if flag}

    last_kept_idx: int | None = None
    last_kept_hash: imagehash.ImageHash | None = None

    for i, h in enumerate(hashes):
        if i in low_value_indices:
            continue
        if last_kept_hash is not None and (h - last_kept_hash) <= DUPLICATE_HASH_DISTANCE:
            duplicate_of[i] = last_kept_idx  # type: ignore[assignment]
            continue
        keep_indices.append(i)
        last_kept_idx = i
        last_kept_hash = h

    return DedupResult(keep_indices=keep_indices, duplicate_of=duplicate_of, low_value_indices=low_value_indices)
