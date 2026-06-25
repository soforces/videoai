"""Real-ESRGAN upscaler - the spec's primary upscale model, fully wired and GPU
accelerated. Model is loaded exactly once (persistent across jobs) and kept
resident on the GPU; `tile` is used to bound VRAM usage on this machine's
4GB laptop GPU rather than batching whole frames at full resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import core.torch_compat  # noqa: F401 - must run before basicsr/realesrgan imports

from plugins.base import BasePlugin
from plugins.registry import registry

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
WEIGHT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHT_NAME = "RealESRGAN_x4plus.pth"


@registry.register
class RealESRGANPlugin(BasePlugin):
    category = "upscale"
    name = "realesrgan"

    def __init__(self):
        super().__init__()
        self.upsampler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = MODELS_DIR / WEIGHT_NAME
        if not weight_path.exists():
            load_file_from_url(WEIGHT_URL, model_dir=str(MODELS_DIR), file_name=WEIGHT_NAME)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(weight_path),
            model=model,
            tile=400 if self.device == "cuda" else 0,
            tile_pad=10,
            pre_pad=0,
            half=self.device == "cuda",
            device=self.device,
        )

    def process(self, image: np.ndarray, scale: int = 2, **kwargs) -> np.ndarray:
        self.ensure_loaded()
        output, _ = self.upsampler.enhance(image, outscale=scale)
        return output

    def unload(self) -> None:
        self.upsampler = None
        torch.cuda.empty_cache()
        super().unload()
