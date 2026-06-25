"""rembg - fast-mode background removal, ONNXRuntime based (GPU via CUDAExecutionProvider
when available). Session is created once and reused across frames/jobs.

Note: onnxruntime-gpu's CUDAExecutionProvider needs the full CUDA 12.x toolkit
runtime libraries (e.g. cublasLt64_13.dll), not just the NVIDIA driver. If those
aren't installed, onnxruntime logs a load failure and falls back to CPU
automatically - rembg still runs correctly, just slower. Install the CUDA 12
toolkit to get GPU acceleration here.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from plugins.base import BasePlugin
from plugins.registry import registry


@registry.register
class RembgPlugin(BasePlugin):
    category = "background"
    name = "rembg"

    def __init__(self):
        super().__init__()
        self.session = None

    def load(self) -> None:
        from rembg import new_session

        providers = ort.get_available_providers()
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
        self.session = new_session("u2net", providers=[provider])

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        from rembg import remove

        self.ensure_loaded()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgba = remove(rgb, session=self.session)
        # Composite onto white background and return BGR to match the rest of the
        # pipeline's image convention, rather than leaking an RGBA array downstream.
        rgb_out = result_rgba[:, :, :3]
        alpha = result_rgba[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(rgb_out, 255)
        composited = (rgb_out.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
