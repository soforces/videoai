"""Export plugins: JPG, PNG, WebP, ZIP and contact-sheet generation.

These are pure I/O/encoding (no model weights), so they're "real" by definition -
no stubs needed here, unlike the AI plugins.
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path

from PIL import Image

from plugins.base import BasePlugin
from plugins.registry import registry


class ImageExportPlugin(BasePlugin):
    category = "export"

    def load(self) -> None:
        pass

    def process(self, image, **kwargs):
        raise NotImplementedError("Use process_file for export plugins")


@registry.register
class JpgExporter(ImageExportPlugin):
    name = "jpg"

    def process_file(self, src: Path, dst: Path, **kwargs) -> None:
        with Image.open(src) as im:
            im.convert("RGB").save(dst, format="JPEG", quality=kwargs.get("quality", 95), subsampling=0)


@registry.register
class PngExporter(ImageExportPlugin):
    name = "png"

    def process_file(self, src: Path, dst: Path, **kwargs) -> None:
        with Image.open(src) as im:
            im.save(dst, format="PNG")


@registry.register
class WebpExporter(ImageExportPlugin):
    name = "webp"

    def process_file(self, src: Path, dst: Path, **kwargs) -> None:
        with Image.open(src) as im:
            im.convert("RGB").save(dst, format="WEBP", quality=kwargs.get("quality", 95))


@registry.register
class ContactSheetExporter(ImageExportPlugin):
    name = "contact_sheet"

    def process_file(self, src: Path, dst: Path, **kwargs) -> None:
        raise NotImplementedError("contact_sheet operates on a list of frames; use build_contact_sheet")


def build_contact_sheet(frame_paths: list[Path], dst: Path, thumb_size: int = 256, columns: int = 6) -> None:
    if not frame_paths:
        raise ValueError("No frames to build a contact sheet from")
    rows = math.ceil(len(frame_paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), color=(20, 20, 20))
    for i, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_size, thumb_size))
            x = (i % columns) * thumb_size
            y = (i // columns) * thumb_size
            offset = ((thumb_size - im.width) // 2, (thumb_size - im.height) // 2)
            sheet.paste(im, (x + offset[0], y + offset[1]))
    sheet.save(dst, format="JPEG", quality=90)


def build_zip(frame_paths: list[Path], dst: Path, extra_files: list[Path] | None = None) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame_path in frame_paths:
            zf.write(frame_path, arcname=frame_path.name)
        for extra_path in extra_files or []:
            zf.write(extra_path, arcname=extra_path.name)
