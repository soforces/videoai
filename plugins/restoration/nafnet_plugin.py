"""NAFNet - fast-mode blur/motion restoration (medium blur_level, decided by
core/decision_engine.py), fully wired and GPU accelerated. GoPro-deblur
checkpoint (NAFNet-GoPro-width64), the upstream repo's recommended deblurring
weights.

vendor/NAFNet vendors its own `basicsr` copy under the *global* `basicsr`
name, which collides with the real pip-installed `basicsr` package that
GFPGAN/Real-ESRGAN/SwinIR already import. To avoid corrupting that shared
namespace, `_load_nafnet_class()` below temporarily swaps `sys.path`/
`sys.modules` to import vendor/NAFNet's arch module in isolation, grabs the
class object, then restores the real `basicsr` modules - this plugin's load()
is the only place that swap happens, and it happens at most once per process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from plugins.base import BasePlugin
from plugins.registry import registry

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "NAFNet"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
GDRIVE_FILE_ID = "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC"  # NAFNet-GoPro-width64.pth
WEIGHT_NAME = "NAFNet-GoPro-width64.pth"
DOWNSAMPLE_FACTOR = 16  # 4 encoder stages, each /2


def _load_nafnet_class():
    vendor_str = str(VENDOR_DIR)
    saved = {k: v for k, v in sys.modules.items() if k == "basicsr" or k.startswith("basicsr.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, vendor_str)
    try:
        import basicsr.models.archs.NAFNet_arch as nafnet_arch_module
        NAFNet = nafnet_arch_module.NAFNet
    finally:
        sys.path.remove(vendor_str)
        for k in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return NAFNet


def _download_weight(dest: Path) -> None:
    import gdown

    gdown.download(id=GDRIVE_FILE_ID, output=str(dest), quiet=False)


@registry.register
class NAFNetPlugin(BasePlugin):
    category = "restoration"
    name = "nafnet"

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        NAFNet = _load_nafnet_class()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = MODELS_DIR / WEIGHT_NAME
        if not weight_path.exists():
            _download_weight(weight_path)

        model = NAFNet(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
        state = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        state = state.get("params", state)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model.to(self.device)

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        self.ensure_loaded()
        img = image[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        img_t = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).unsqueeze(0).to(self.device)

        _, _, h, w = img_t.shape
        pad_h = (DOWNSAMPLE_FACTOR - h % DOWNSAMPLE_FACTOR) % DOWNSAMPLE_FACTOR
        pad_w = (DOWNSAMPLE_FACTOR - w % DOWNSAMPLE_FACTOR) % DOWNSAMPLE_FACTOR
        img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output = self.model(img_t)
        output = output[:, :, :h, :w]

        output = output.squeeze(0).clamp_(0, 1).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)[:, :, ::-1]  # RGB -> BGR
        return np.ascontiguousarray(output)

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
        super().unload()
