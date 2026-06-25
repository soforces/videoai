"""Restormer - high-quality blur restoration (high blur_level in balanced/
max_quality modes, decided by core/decision_engine.py), fully wired and GPU
accelerated. Uses the official Motion_Deblurring checkpoint.

Unlike NAFNet/CodeFormer, restormer_arch.py is fully self-contained (only
torch + einops, no basicsr import), so it's loaded directly via
importlib.util.spec_from_file_location under a private module name - no
sys.path/sys.modules juggling needed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from plugins.base import BasePlugin
from plugins.registry import registry

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "Restormer"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
GDRIVE_FILE_ID = "1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L"  # motion_deblurring.pth
WEIGHT_NAME = "restormer_motion_deblurring.pth"
PAD_MULTIPLE = 8

ARCH_PARAMS = dict(
    inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
    bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
)


def _load_restormer_class():
    spec = importlib.util.spec_from_file_location(
        "_vendor_restormer_arch", VENDOR_DIR / "basicsr" / "models" / "archs" / "restormer_arch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Restormer


def _download_weight(dest: Path) -> None:
    import gdown

    gdown.download(id=GDRIVE_FILE_ID, output=str(dest), quiet=False)


@registry.register
class RestormerPlugin(BasePlugin):
    category = "restoration"
    name = "restormer"

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        Restormer = _load_restormer_class()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = MODELS_DIR / WEIGHT_NAME
        if not weight_path.exists():
            _download_weight(weight_path)

        model = Restormer(**ARCH_PARAMS)
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["params"])
        model.eval()
        self.model = model.to(self.device)

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        self.ensure_loaded()
        img = image[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        img_t = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).unsqueeze(0).to(self.device)

        _, _, h, w = img_t.shape
        H = ((h + PAD_MULTIPLE) // PAD_MULTIPLE) * PAD_MULTIPLE
        W = ((w + PAD_MULTIPLE) // PAD_MULTIPLE) * PAD_MULTIPLE
        pad_h = H - h if h % PAD_MULTIPLE != 0 else 0
        pad_w = W - w if w % PAD_MULTIPLE != 0 else 0
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
