"""RIFE (hzwer/ECCV2022-RIFE) - frame interpolation, fully wired and GPU
accelerated. Uses the official v3.x HD checkpoint (flownet.pkl + its matching
IFNet_HDv3/RIFE_HDv3 model definitions, downloaded as a pair from the
upstream Google Drive release - the v3.x arch is checkpoint-specific, unlike
the older model/RIFE.py classes in vendor/ECCV2022-RIFE).

vendor/ECCV2022-RIFE provides `model/warplayer.py` + `model/loss.py` (no
PyPI package, no top-level `__init__.py`, but no other vendored repo defines
a conflicting global `model`/`train_log` package either, so these are loaded
as ordinary namespace packages via a scoped `sys.path` insertion in load()
rather than direct file-exec, since RIFE_HDv3.py itself does
`from model.warplayer import warp` and `from train_log.IFNet_HDv3 import *`.

process_pair() is the real entrypoint (interpolation needs two frames); the
required process() from BasePlugin is a thin pass-through since this plugin
isn't part of the per-frame image pipeline in core/pipeline.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from plugins.base import BasePlugin
from plugins.registry import registry

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "ECCV2022-RIFE"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache" / "train_log"


def _load_rife_model_class():
    vendor_str = str(VENDOR_DIR)
    checkpoint_parent_str = str(CHECKPOINT_DIR.parent)
    sys.path.insert(0, checkpoint_parent_str)
    sys.path.insert(0, vendor_str)
    try:
        from train_log.RIFE_HDv3 import Model
    finally:
        sys.path.remove(vendor_str)
        sys.path.remove(checkpoint_parent_str)
    return Model


@registry.register
class RIFEPlugin(BasePlugin):
    category = "interpolation"
    name = "rife"

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if not (CHECKPOINT_DIR / "flownet.pkl").exists():
            from plugins.base import PluginNotInstalledError

            raise PluginNotInstalledError(
                self.name,
                f"Missing {CHECKPOINT_DIR / 'flownet.pkl'}. Download the v3.x HD model zip from "
                "https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view and extract "
                "train_log/{flownet.pkl,IFNet_HDv3.py,RIFE_HDv3.py} into models_cache/train_log/.",
            )

        Model = _load_rife_model_class()
        model = Model()
        model.load_model(str(CHECKPOINT_DIR), -1)
        model.eval()
        model.device()
        self.model = model

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Pass-through: RIFE operates on frame *pairs*, see process_pair()."""
        return image

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
        super().unload()

    def process_pair(self, frame_a: np.ndarray, frame_b: np.ndarray, t: float = 0.5, **kwargs) -> np.ndarray:
        """Returns the interpolated frame at position `t` (0..1) between
        frame_a and frame_b. Only t=0.5 is a single direct network call;
        other ratios are approximated via the same bisection the upstream
        inference_img.py --ratio mode uses."""
        self.ensure_loaded()
        h, w = frame_a.shape[:2]
        img0 = self._to_tensor(frame_a)
        img1 = self._to_tensor(frame_b)

        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        with torch.no_grad():
            if abs(t - 0.5) < 1e-6:
                middle = self.model.inference(img0, img1)
            else:
                middle = self._bisect(img0, img1, t)

        out = (middle[0] * 255.0).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
        return np.ascontiguousarray(out[:h, :w])

    def _bisect(self, img0: torch.Tensor, img1: torch.Tensor, t: float, threshold: float = 0.02, max_cycles: int = 8):
        t0, t1 = 0.0, 1.0
        lo, hi = img0, img1
        middle = self.model.inference(lo, hi)
        for _ in range(max_cycles):
            mid_t = (t0 + t1) / 2
            if abs(t - mid_t) <= threshold / 2:
                break
            if t < mid_t:
                hi, t1 = middle, mid_t
            else:
                lo, t0 = middle, mid_t
            middle = self.model.inference(lo, hi)
        return middle

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        # Matches upstream inference_img.py: fed as whatever channel order
        # cv2.imread returns (BGR), with no RGB conversion - the network was
        # trained/evaluated the same way, so converting here would feed it
        # channel-swapped input and corrupt the flow estimate.
        t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().to(self.device) / 255.0
        return t.unsqueeze(0)
