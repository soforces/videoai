"""Face detection + damage scoring.

Detection uses OpenCV's bundled Haar cascade (zero extra downloads, fast on CPU,
good enough to gate whether the much heavier GFPGAN/CodeFormer models should run
at all). Damage scoring is a heuristic combining face-region blur and resolution -
it estimates how badly a face needs restoration on a 0-100 scale per the spec's
face processing rules (0-30 skip, 30-60 GFPGAN, 60-100 CodeFormer).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.models import FaceBox


def _load_cascade() -> cv2.CascadeClassifier:
    # cv2's CascadeClassifier::read uses fopen() under the hood, which chokes on
    # non-ASCII path components (e.g. this project's "Masaüstü" directory) on
    # Windows. Stage the XML under the ASCII-safe system temp dir once and load
    # from there instead of cv2.data.haarcascades directly.
    src = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    staged = Path(tempfile.gettempdir()) / "framescript_haarcascade_frontalface_default.xml"
    if not staged.exists():
        shutil.copyfile(src, staged)
    return cv2.CascadeClassifier(str(staged))


_face_cascade = _load_cascade()

# A face region needs ~120px width to look "clean" after restoration; below that,
# upscaling artifacts and compression dominate and damage score rises fast.
REFERENCE_FACE_WIDTH = 120.0


def detect_faces(image_bgr: np.ndarray) -> list[FaceBox]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes = _face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))

    faces: list[FaceBox] = []
    for (x, y, w, h) in boxes:
        face_region = image_bgr[y:y + h, x:x + w]
        damage = score_face_damage(face_region, w)
        faces.append(FaceBox(x=int(x), y=int(y), w=int(w), h=int(h), damage_score=damage))
    return faces


def score_face_damage(face_region_bgr: np.ndarray, face_width_px: int) -> float:
    if face_region_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Blur component: lower variance -> higher damage. Saturates at 0 blur_var -> 100.
    blur_damage = max(0.0, min(100.0, 100.0 - (blur_var / 4.0)))

    # Resolution component: small face crops carry more compression/upscale damage.
    res_ratio = min(face_width_px / REFERENCE_FACE_WIDTH, 1.0)
    res_damage = (1.0 - res_ratio) * 100.0

    damage = 0.6 * blur_damage + 0.4 * res_damage
    return round(max(0.0, min(100.0, damage)), 1)
