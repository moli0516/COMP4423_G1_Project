"""Synthetic-only pixel detector. Never a trained detector or accuracy baseline."""

import numpy as np

from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection


class FakeColorDetector:
    COLORS = (
        (30, 110, 160),
        (197, 95, 50),
        (77, 82, 97),
        (127, 74, 142),
        (62, 125, 89),
        (185, 63, 80),
    )

    def predict(self, image) -> list[Detection]:
        records = []
        for class_id, color in enumerate(self.COLORS):
            y, x = np.where(np.all(image.array == color, axis=2))
            if len(x):
                records.append(
                    Detection(
                        f"fake-{class_id}",
                        class_id,
                        1.0,
                        BoundingBox(
                            float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)
                        ),
                    )
                )
        return records
