"""PaddleOCR - text extraction, gated by analysis.text_detect.has_probable_text
(decided per-frame in core/decision_engine.py). Loaded once and kept resident.

Uses CPU inference (`paddlepaddle`, not `paddlepaddle-gpu`): OCR only runs on
the small subset of frames flagged as probably containing text, so it isn't
on the <60s hot path the way upscale/face/restoration are, and staying off
the GPU avoids competing with those models for this machine's 4GB VRAM.

PP-OCRv6 (paddleocr>=3.x) + oneDNN currently crashes on this CPU
(`NotImplementedError: ConvertPirAttribute2RuntimeAttribute ... onednn`), so
oneDNN is explicitly disabled (`enable_mkldnn=False`); doc-orientation/
unwarping preprocessing is also disabled since these are extracted video
frames, not scanned documents.
"""
from __future__ import annotations

from plugins.base import BasePlugin
from plugins.registry import registry


@registry.register
class PaddleOCRPlugin(BasePlugin):
    category = "ocr"
    name = "paddleocr"

    def __init__(self):
        super().__init__()
        self.ocr = None

    def load(self) -> None:
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
        )

    def process(self, image, **kwargs) -> str:
        """Returns recognized text joined by newlines (empty string if none)."""
        self.ensure_loaded()
        results = self.ocr.predict(image)
        lines: list[str] = []
        for result in results:
            lines.extend(t for t in result.get("rec_texts", []) if t)
        return "\n".join(lines)
