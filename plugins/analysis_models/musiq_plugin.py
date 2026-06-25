"""MUSIQ (neural quality scoring) and CLIP-based anomaly detection, both
fully wired and GPU accelerated.

MUSIQ via `pyiqa` (chaofengc/IQA-PyTorch): a pure-PyTorch reimplementation of
the official koniq-trained MUSIQ checkpoint - avoids needing TensorFlow
(the official release is a TF-Hub SavedModel) alongside the existing PyTorch
stack. analysis/quality.py's heuristic scorer remains the default fast-path
quality signal; this plugin is the real model-based scorer for when a more
accurate 0-100 quality estimate is worth its extra inference cost.

CLIP-anomaly via `open_clip_torch` zero-shot classification: rather than
requiring a curated "clean frame" reference set (which doesn't exist
off-the-shelf and can't be calibrated without product-specific data), this
scores each frame by its softmax similarity to a fixed pair of text prompts
- "normal, undistorted photo" vs. "photo with severe visual artifacts/
distorted geometry" - the same zero-shot-prompt technique used by CLIP-based
anomaly detectors like WinCLIP. No training or calibration step needed.
"""
from __future__ import annotations

import numpy as np
import torch

from plugins.base import BasePlugin
from plugins.registry import registry

NORMAL_PROMPTS = [
    "a normal clear photograph, undistorted and natural",
    "a sharp, well-formed photo of a real scene",
]
ANOMALY_PROMPTS = [
    "a photo with severe visual artifacts, distorted warped geometry, glitch corruption",
    "a deformed face or body with extra or missing fingers, mangled anatomy",
]


@registry.register
class MUSIQPlugin(BasePlugin):
    category = "analysis_models"
    name = "musiq"

    def __init__(self):
        super().__init__()
        self.metric = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        import pyiqa

        self.metric = pyiqa.create_metric("musiq", device=torch.device(self.device))

    def process(self, image: np.ndarray, **kwargs) -> float:
        """Returns a 0-100 perceptual quality score (higher is better)."""
        import cv2

        self.ensure_loaded()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            score = self.metric(tensor.to(self.metric.device))
        return float(score.item())

    def unload(self) -> None:
        self.metric = None
        torch.cuda.empty_cache()
        super().unload()


@registry.register
class CLIPAnomalyPlugin(BasePlugin):
    category = "analysis_models"
    name = "clip_anomaly"

    def __init__(self):
        super().__init__()
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        model = model.to(self.device).eval()

        prompts = NORMAL_PROMPTS + ANOMALY_PROMPTS
        tokens = tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        self.model = model
        self.preprocess = preprocess
        self.text_features = text_features

    def process(self, image: np.ndarray, **kwargs) -> float:
        """Returns an anomaly probability in [0, 1] (mean softmax mass on the
        anomaly prompts vs. the normal prompts)."""
        import cv2
        from PIL import Image

        self.ensure_loaded()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ self.text_features.T).softmax(dim=-1)

        n_normal = len(NORMAL_PROMPTS)
        anomaly_mass = sims[0, n_normal:].sum().item()
        return float(anomaly_mass)

    def unload(self) -> None:
        self.model = None
        self.text_features = None
        torch.cuda.empty_cache()
        super().unload()
