"""SAM2 (object segmentation, precision background removal) - fully wired,
GPU accelerated, model loaded once and kept resident.

`process()` returns a binary mask (uint8, 0/255) for the largest foreground
object found from an automatic point-grid prompt (no user-supplied point/box
needed, since this runs unattended in a batch pipeline). Background-removal
callers composite this mask themselves (see plugins/background/rembg_plugin.py
for the same compositing pattern at the "rembg" precision tier; "sam2" is the
precision tier swapped in for max_quality mode in core/decision_engine.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from plugins.base import BasePlugin, PluginNotInstalledError
from plugins.registry import registry

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_INSTALL_HINT = "pip install sam2 (checkpoint auto-downloads on first use)."


@registry.register
class SAM2Plugin(BasePlugin):
    category = "segmentation"
    name = "sam2"

    def __init__(self):
        super().__init__()
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise PluginNotInstalledError(self.name, _INSTALL_HINT) from exc

        from basicsr.utils.download_util import load_file_from_url

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = MODELS_DIR / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            load_file_from_url(CHECKPOINT_URL, model_dir=str(MODELS_DIR), file_name=CHECKPOINT_NAME)

        sam2_model = build_sam2(MODEL_CFG, str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Returns a uint8 0/255 mask of the most prominent foreground object,
        found via a center-weighted point grid (this pipeline runs unattended,
        so there's no human in the loop to click a point/box prompt)."""
        import cv2

        self.ensure_loaded()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            self.predictor.set_image(rgb)
            # 3x3 grid of foreground point prompts biased toward the center,
            # where the subject of an unattended frame is most often located.
            xs = [int(w * f) for f in (0.3, 0.5, 0.7)]
            ys = [int(h * f) for f in (0.3, 0.5, 0.7)]
            points = np.array([[x, y] for y in ys for x in xs])
            labels = np.ones(len(points), dtype=np.int32)
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)

        best_mask = masks[int(np.argmax(scores))]
        return (best_mask * 255).astype(np.uint8)

    def unload(self) -> None:
        self.predictor = None
        torch.cuda.empty_cache()
        super().unload()
