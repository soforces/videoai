"""SwinIR fallback/ensemble upscaler - real-world SR (003_realSR_BSRGAN x4),
fully wired and GPU accelerated. Used as the balanced-mode fallback and as
part of the max_quality Real-ESRGAN+SwinIR ensemble (core/decision_engine.py).

vendor/SwinIR has no `__init__.py` markers and isn't pip-installable, so
`models/network_swinir.py` (a self-contained file - only torch + timm) is
loaded directly via importlib under a private module name instead of
inserting vendor/SwinIR onto sys.path, to avoid clashing with any other
vendored repo's own top-level `models` package (NAFNet/Restormer also vendor
a `models`/`basicsr` tree).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from plugins.base import BasePlugin
from plugins.registry import registry

VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "SwinIR"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
WEIGHT_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
WEIGHT_NAME = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
WINDOW_SIZE = 8
TILE = 256
TILE_OVERLAP = 32


def _load_network_swinir_module():
    spec = importlib.util.spec_from_file_location(
        "_vendor_swinir_network", VENDOR_DIR / "models" / "network_swinir.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@registry.register
class SwinIRPlugin(BasePlugin):
    category = "upscale"
    name = "swinir"

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        from basicsr.utils.download_util import load_file_from_url

        network_swinir = _load_network_swinir_module()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = MODELS_DIR / WEIGHT_NAME
        if not weight_path.exists():
            load_file_from_url(WEIGHT_URL, model_dir=str(MODELS_DIR), file_name=WEIGHT_NAME)

        # "real_sr", non-large-model variant (SwinIR-M): matches the
        # 003_realSR_BSRGAN_DFO_*_SwinIR-M_x4_GAN.pth checkpoint layout.
        model = network_swinir.SwinIR(
            upscale=4, in_chans=3, img_size=64, window_size=WINDOW_SIZE,
            img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler="nearest+conv", resi_connection="1conv",
        )
        pretrained = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        model.load_state_dict(pretrained["params_ema"] if "params_ema" in pretrained else pretrained, strict=True)
        model.eval()
        self.model = model.to(self.device)

    def process(self, image: np.ndarray, scale: int = 4, **kwargs) -> np.ndarray:
        """`scale` is accepted for interface parity with the other upscale
        plugins but SwinIR-M's checkpoint is a fixed x4 model; output is
        resized to the requested scale if it differs from 4."""
        self.ensure_loaded()
        img = image.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_t = img_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, h_old, w_old = img_t.size()
            h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
            w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
            img_t = torch.cat([img_t, torch.flip(img_t, [2])], 2)[:, :, : h_old + h_pad, :]
            img_t = torch.cat([img_t, torch.flip(img_t, [3])], 3)[:, :, :, : w_old + w_pad]
            output = self._tiled_inference(img_t)
            output = output[..., : h_old * 4, : w_old * 4]

        output = output.squeeze(0).clamp_(0, 1).cpu().numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        if scale != 4:
            target_w = int(round(image.shape[1] * scale))
            target_h = int(round(image.shape[0] * scale))
            output = np_resize(output, target_w, target_h)
        return output

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
        super().unload()

    def _tiled_inference(self, img_t: torch.Tensor) -> torch.Tensor:
        # `out`/`weight` cover the *full* upsampled image (4x linear -> 16x
        # the pixels) - on a 4GB GPU that pair alone can be several GiB for a
        # single 4K-ish frame, even though only one small tile is ever
        # written into them at a time. Keeping them on CPU (RAM is not the
        # scarce resource here) and moving only each small `out_patch` across
        # the PCIe bus is what actually makes "tiled" mean memory-bounded.
        b, c, h, w = img_t.size()
        tile = min(TILE, h, w)
        tile = (tile // WINDOW_SIZE) * WINDOW_SIZE
        stride = tile - TILE_OVERLAP
        sf = 4

        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        out = torch.zeros(b, c, h * sf, w * sf, device="cpu")
        weight = torch.zeros_like(out)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                patch = img_t[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                out_patch = self.model(patch).to("cpu")
                out[..., h_idx * sf : (h_idx + tile) * sf, w_idx * sf : (w_idx + tile) * sf] += out_patch
                weight[..., h_idx * sf : (h_idx + tile) * sf, w_idx * sf : (w_idx + tile) * sf] += 1.0
        # Stays on CPU - process() only slices/clamps/.cpu()s it next, so
        # there's no reason to ship the full-size buffer back to the GPU
        # just to immediately copy it off again.
        return out / weight


def np_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
