"""GFPGAN - fast-mode face restoration, fully wired and GPU accelerated. Loaded
once and kept resident; runs its own internal face detector/aligner per call
since GFPGANer needs full-frame context to align faces before restoring them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import core.torch_compat  # noqa: F401 - must run before basicsr/gfpgan imports

from plugins.base import BasePlugin
from plugins.registry import registry

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
WEIGHT_NAME = "GFPGANv1.4.pth"


@registry.register
class GFPGANPlugin(BasePlugin):
    category = "faces"
    name = "gfpgan"

    def __init__(self):
        super().__init__()
        self.restorer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        from basicsr.utils.download_util import load_file_from_url
        from gfpgan import GFPGANer

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = MODELS_DIR / WEIGHT_NAME
        if not weight_path.exists():
            load_file_from_url(WEIGHT_URL, model_dir=str(MODELS_DIR), file_name=WEIGHT_NAME)

        self.restorer = GFPGANer(
            model_path=str(weight_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device,
        )

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        self.ensure_loaded()
        _, _, restored = self.restorer.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True,
        )
        return restored if restored is not None else image

    def unload(self) -> None:
        self.restorer = None
        torch.cuda.empty_cache()
        super().unload()
