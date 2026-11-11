"""Generate clearly synthetic inputs; no ground-truth files enter inference."""

import io

import numpy as np
from PIL import Image

from deskcheck.detection.fake_color_detector import FakeColorDetector


def make_demo_pair() -> tuple[bytes, bytes]:
    before = np.full((120, 160, 3), 240, dtype=np.uint8)
    after = before.copy()
    colors = FakeColorDetector.COLORS
    before[10:25, 10:25] = colors[0]
    after[10:25, 80:95] = colors[0]
    before[40:55, 10:25] = colors[1]
    before[70:85, 10:25] = colors[2]
    after[70:85, 10:25] = colors[2]
    after[40:55, 80:95] = colors[3]
    encoded = []
    for array in (before, after):
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        encoded.append(output.getvalue())
    return tuple(encoded)
