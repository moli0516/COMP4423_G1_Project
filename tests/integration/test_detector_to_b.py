"""Integration: real YoloDetector → B filter → matcher → analyzer.

Requires the E0 checkpoint and formal dataset to be present. Skips
gracefully if either is missing (so CI stays green without vision extras).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deskcheck.domain.image_frame import ImageFrame

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "runs" / "experiments" / "E0_baseline" / "best.pt"
DATASET_PHOTO = Path(
    "/home/wyc/Documents/CV Group/data/formal_photos/G01P1_before.jpg"
)


def _load_image_as_frame(path: Path) -> ImageFrame:
    from PIL import Image
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    return ImageFrame(array=arr, width=w, height=h, source_path=path)


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="E0 checkpoint not present (runs/experiments/E0_baseline/best.pt)",
)
@pytest.mark.skipif(
    not DATASET_PHOTO.exists(),
    reason="formal dataset photo not present at expected path",
)
def test_real_detector_returns_canonical_detections() -> None:
    """End-to-end smoke: real image → YoloDetector → list[Detection]."""
    from deskcheck.detection.yolo_detector import YoloConfig, YoloDetector

    cfg = YoloConfig(
        checkpoint_path=CHECKPOINT,
        confidence_threshold=0.20,
        nms_iou_threshold=0.35,
        device="0",
    )
    detector = YoloDetector(cfg)
    frame = _load_image_as_frame(DATASET_PHOTO)
    dets = detector.predict(frame)

    # 6 target objects in the formal photos: expect at least 4 detections
    assert len(dets) >= 4, f"expected ≥4 detections, got {len(dets)}"
    for d in dets:
        assert 0 <= d.class_id < 6
        assert 0.0 <= d.confidence <= 1.0
        # boxes inside original frame
        assert 0 <= d.box.x1 < d.box.x2 <= frame.width
        assert 0 <= d.box.y1 < d.box.y2 <= frame.height
