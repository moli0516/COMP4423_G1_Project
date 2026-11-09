import io
import os

import numpy as np
from PIL import Image, ImageOps

from src.exceptions.preprocessing_error import PreprocessingError
from .preprocessing_config import PreprocessingConfig


class ImagePreprocessor:
    """Decode, orient, composite and resize images through one shared route."""

    def __init__(self, config=None):
        self.config = config if config is not None else PreprocessingConfig()

    def load_image(self, source):
        """Return a detached PIL image; do not resize or close caller streams."""
        try:
            if isinstance(source, Image.Image):
                source.load()
                return source.copy()
            if isinstance(source, (str, os.PathLike)):
                with Image.open(source) as image:
                    image.load()
                    return image.copy()
            if isinstance(source, (bytes, bytearray, memoryview)):
                data = bytes(source)
            elif hasattr(source, "read") and hasattr(source, "seek"):
                source.seek(0)
                data = source.read()
                if not isinstance(data, (bytes, bytearray)):
                    raise TypeError("Image streams must return bytes.")
            else:
                raise TypeError("Expected a path, encoded bytes, binary stream or PIL image.")
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                return image.copy()
        except (OSError, ValueError, TypeError, SyntaxError) as error:
            raise PreprocessingError(f"Cannot decode image: {error}") from error

    def normalise_orientation_and_colour(self, image):
        if not isinstance(image, Image.Image):
            raise PreprocessingError("Expected a PIL image for orientation/colour conversion.")
        try:
            oriented = ImageOps.exif_transpose(image)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                foreground = oriented.convert("RGBA")
                background = Image.new("RGBA", foreground.size, (self.config.pad_value,) * 3 + (255,))
                return Image.alpha_composite(background, foreground).convert("RGB")
            return oriented.convert("RGB")
        except (OSError, ValueError, TypeError) as error:
            raise PreprocessingError(f"Cannot normalise image: {error}") from error

    def transform(self, image):
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.ndim not in (2, 3):
                raise PreprocessingError("Expected a two/three-dimensional uint8 image array.")
            try:
                image = Image.fromarray(image)
            except (TypeError, ValueError) as error:
                raise PreprocessingError(str(error)) from error
        image = self.normalise_orientation_and_colour(image)
        size = (self.config.image_size,) * 2
        method = Image.Resampling.BILINEAR
        if self.config.resize_mode == "squash":
            result = image.resize(size, resample=method)
        elif self.config.resize_mode == "center_crop":
            edge = min(image.size)
            left = (image.width - edge) // 2
            top = (image.height - edge) // 2
            result = image.crop((left, top, left + edge, top + edge)).resize(size, method)
        else:
            scale = self.config.image_size / max(image.size)
            width = max(1, min(self.config.image_size, round(image.width * scale)))
            height = max(1, min(self.config.image_size, round(image.height * scale)))
            resized = image.resize((width, height), resample=method)
            result = Image.new("RGB", size, (self.config.pad_value,) * 3)
            offset = ((self.config.image_size - width) // 2, (self.config.image_size - height) // 2)
            result.paste(resized, offset)
        return np.array(result, dtype=np.uint8, copy=True, order="C")

    def load(self, source):
        return self.transform(self.load_image(source))

    def load_many(self, sources):
        return [self.load(source) for source in sources]
