"""Decode a canonical pair without model resizing or silent alpha compositing."""

import io
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from deskcheck.domain.image_frame import ImageFrame


class ImagePairLoader:
    MAX_PIXELS = 20_000_000

    def load(self, before_path, after_path, pair_id: str) -> tuple[ImageFrame, ImageFrame]:
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("pair_id must be a nonempty string")
        before = self._decode(before_path)
        after = self._decode(after_path)
        if (before.width, before.height) != (after.width, after.height):
            raise ValueError("Image dimensions differ after EXIF orientation")
        return before, after

    def _decode(self, source) -> ImageFrame:
        if not isinstance(source, (str, Path, bytes)):
            raise TypeError("Image source must be a path or encoded bytes")
        target = io.BytesIO(source) if isinstance(source, bytes) else source
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                # Ultralytics 8.4.215 patches PIL.Image.open to install a HEIF plugin on
                # any unidentified input. Use its saved native opener for C's uploads.
                opener = Image.open
                patches = sys.modules.get("ultralytics.utils.patches")
                if patches is not None and opener is getattr(patches, "image_open", None):
                    opener = patches._image_open
                with opener(target) as image:
                    if image.width * image.height > self.MAX_PIXELS:
                        raise ValueError("Image exceeds the 20-million-pixel resource limit")
                    if "A" in image.getbands() or "transparency" in image.info:
                        raise ValueError("Transparent input requires an agreed alpha policy")
                    oriented = ImageOps.exif_transpose(image)
                    try:
                        rgb = oriented.convert("RGB")
                        try:
                            pixels = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(
                                rgb.height, rgb.width, 3
                            )
                        finally:
                            rgb.close()
                    finally:
                        oriented.close()
        except (
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            raise ValueError("Cannot decode image input") from error
        return ImageFrame(
            array=pixels, width=pixels.shape[1], height=pixels.shape[0],
            source_path=None if isinstance(source, bytes) else Path(source),
        )
