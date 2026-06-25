"""CodeFormer - high-quality face restoration (max_quality / high face-damage
mode), fully wired and GPU accelerated. Uses the `codeformer-pip` PyPI wheel,
which vendors CodeFormer's network + facelib detector/parser under its own
`codeformer.*` namespace, so it never collides with the real `basicsr` package
that GFPGAN/Real-ESRGAN already import under the global name.

Loaded once and kept resident (network + face detector + face parser). Face
detection/alignment/paste-back is handled by `codeformer.facelib`'s
FaceRestoreHelper, the same helper the upstream sczhou/CodeFormer repo uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import normalize

from plugins.base import BasePlugin
from plugins.registry import registry

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models_cache"
WEIGHT_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
WEIGHT_NAME = "codeformer.pth"

# Fidelity weight: 0 favors quality/identity restoration, 1 favors fidelity to
# the (damaged) input. 0.5 is CodeFormer's own recommended general-purpose value.
DEFAULT_FIDELITY = 0.5


@registry.register
class CodeFormerPlugin(BasePlugin):
    category = "faces"
    name = "codeformer"

    def __init__(self):
        super().__init__()
        self.net = None
        self.face_helper = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        from codeformer.basicsr.utils.download_util import load_file_from_url
        from codeformer.basicsr.utils.registry import ARCH_REGISTRY
        from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper

        import codeformer.basicsr.archs  # noqa: F401 - registers "CodeFormer" in ARCH_REGISTRY

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weight_path = MODELS_DIR / WEIGHT_NAME
        if not weight_path.exists():
            load_file_from_url(WEIGHT_URL, model_dir=str(MODELS_DIR), file_name=WEIGHT_NAME)

        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        checkpoint = torch.load(str(weight_path), map_location=self.device, weights_only=False)["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()
        self.net = net

        # FaceRestoreHelper.__init__ constructs and loads the detector +
        # parser networks (init_detection_model/init_parsing_model) - build it
        # exactly once here and reset its per-image state via clean_all() in
        # process(), rather than rebuilding it (and reloading those nets) per frame.
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
        )

    def process(self, image: np.ndarray, fidelity: float = DEFAULT_FIDELITY, **kwargs) -> np.ndarray:
        from codeformer.basicsr.utils import img2tensor, tensor2img

        self.ensure_loaded()
        face_helper = self.face_helper
        face_helper.clean_all()
        face_helper.read_image(image)
        num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        if num_faces == 0:
            return image
        face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.net(cropped_face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            if self.device == "cuda":
                torch.cuda.empty_cache()
            face_helper.add_restored_face(restored_face.astype("uint8"))

        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=image)
        return restored_img if restored_img is not None else image

    def unload(self) -> None:
        self.net = None
        self.face_helper = None  # also frees its resident detection/parsing nets
        torch.cuda.empty_cache()
        super().unload()
