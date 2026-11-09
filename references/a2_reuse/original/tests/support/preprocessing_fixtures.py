import contextlib
import dataclasses
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
import src.preprocessing as preprocessing_public
from src.preprocessing import api as preprocessing_api
from src.preprocessing.preprocessing_config import PreprocessingConfig

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def solid(size, colour):
    return Image.new('RGB', size, colour)

def left_right(size=(40, 20), left=RED, right=BLUE):
    image = Image.new('RGB', size, left)
    image.paste(right, (size[0] // 2, 0, size[0], size[1]))
    return image

def save_jpeg_with_orientation(path, image, orientation):
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, format='JPEG', quality=100, exif=exif)
