"""Owner A: production YOLO detector adapter.

Implements ObjectDetector with the canonical ImageFrame boundary from PR #8.

Contract (from B's earlier invariants):
- predict(image: ImageFrame) -> list[Detection]
- Returns boxes in ORIGINAL oriented image coordinates.
- Applies backend NMS before returning.
- Undoes any letterbox/resize exactly once.
- Works with source_path=None.
- Does NOT mutate image.array.
- Lazy-imports ultralytics (vision extra) so the package installs without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deskcheck.detection.object_detector import ObjectDetector
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection
from deskcheck.domain.image_frame import ImageFrame

APPROVED_CLASS_NAMES = ("bottle", "cup", "keyboard", "mouse", "book", "scissors")


@dataclass(frozen=True)
class YoloConfig:
    """Runtime configuration for the YOLO detector.

    All fields frozen after validation. No silent defaults for the
    checkpoint path or class map.
    """
    checkpoint_path: Path
    class_names: tuple[str, ...] = APPROVED_CLASS_NAMES
    confidence_threshold: float = 0.20
    nms_iou_threshold: float = 0.35
    image_size: int = 640
    device: str = "0"


class YoloDetector(ObjectDetector):
    """Production detector. Loads the checkpoint once at construction."""

    def __init__(self, config: YoloConfig) -> None:
        self._config = config
        self._validate_config()
        self._model = None   # lazy — loaded on first predict
        self._names: dict[int, str] | None = None

    def _validate_config(self) -> None:
        cfg = self._config
        if not cfg.checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {cfg.checkpoint_path}. "
                f"Refusing to run without a trained artifact."
            )
        if len(cfg.class_names) == 0:
            raise ValueError("class_names must be non-empty")
        if not (0.0 <= cfg.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0,1], got {cfg.confidence_threshold}"
            )
        if not (0.0 <= cfg.nms_iou_threshold <= 1.0):
            raise ValueError(
                f"nms_iou_threshold must be in [0,1], got {cfg.nms_iou_threshold}"
            )
        if cfg.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {cfg.image_size}")

    def _load_model(self):
        """Return (model, names_map). Cached after first call."""
        if self._model is not None:
            return self._model, self._names
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Run: pip install -e '.[vision]'"
            ) from exc

        self._model = YOLO(str(self._config.checkpoint_path))

        model_names = getattr(self._model, "names", None)
        if model_names is None:
            raise RuntimeError(
                "loaded model does not expose a 'names' attribute; "
                "cannot perform canonical class mapping."
            )
        self._names = dict(model_names)
        return self._model, self._names

    def _canonical_class_id(self, backend_name: str) -> int | None:
        """Map backend class name to canonical id, or None to drop."""
        try:
            return self._config.class_names.index(backend_name)
        except ValueError:
            return None

    def predict(self, image: ImageFrame) -> list[Detection]:
        """Run inference and return canonical detections in original pixels."""
        if not isinstance(image, ImageFrame):
            raise TypeError(
                f"predict() expects ImageFrame, got {type(image).__name__}"
            )

        model, names = self._load_model()

        # Do not mutate caller's array — pass a copy to ultralytics.
        # Ultralytics may resize internally but must not modify in place.
        image.array.copy()

        # Run inference. Ultralytics returns boxes in the ORIGINAL input
        # image coordinate system (it undoes its own letterbox internally).
        # Ultralytics treats ndarray input as BGR (OpenCV convention). Our
        # ImageFrame.array is canonical RGB. Explicitly reverse the channel
        # axis before predict, with ascontiguousarray so downstream C++
        # extensions get a contiguous buffer. Caller's array is not modified.
        bgr = np.ascontiguousarray(image.array[:, :, ::-1])

        results = model.predict(
            source=bgr,
            imgsz=self._config.image_size,
            conf=self._config.confidence_threshold,
            iou=self._config.nms_iou_threshold,
            device=self._config.device,
            verbose=False,
        )

        # Ultralytics contract (v8.x): predict() returns a list of Results
        # — one per input image. For single-image source, list length must
        # be exactly 1. A different type/length is a backend fault, not a
        # valid empty result.
        if not isinstance(results, (list, tuple)):
            raise TypeError(
                f"YOLO predict() returned {type(results).__name__}; "
                f"expected list of Results (one per image)."
            )
        if len(results) != 1:
            raise RuntimeError(
                f"YOLO predict() returned {len(results)} Results for one image; "
                f"expected exactly 1. Single-image inference contract violated."
            )

        result = results[0]
        # Valid empty: Results has a real Boxes object with len()==0.
        # Missing .boxes attribute or boxes=None indicates malformed output.
        if not hasattr(result, "boxes"):
            raise RuntimeError(
                "YOLO Results missing 'boxes' attribute — malformed backend output."
            )
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError(
                "YOLO Results.boxes is None — malformed output "
                "(valid empty results have Boxes with len()==0, not None)."
            )
        if len(boxes) == 0:
            return []  # valid empty detection

        detections: list[Detection] = []
        for idx, box in enumerate(boxes):
            class_id_backend = box.cls[0].item()
            if isinstance(class_id_backend, float) and class_id_backend.is_integer():
                class_id_backend = int(class_id_backend)
            if type(class_id_backend) is not int:
                raise ValueError("Detector class ID must be a finite integer")
            backend_name = names.get(class_id_backend, "")
            canonical_id = self._canonical_class_id(backend_name)
            if canonical_id is None:
                # Unsupported class — drop silently per contract, but a run
                # log could be extended here to record dropped-class counts.
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

            # Clip to image bounds (ultralytics occasionally emits +/-1 px outside)
            x1 = max(0.0, min(x1, image.width))
            y1 = max(0.0, min(y1, image.height))
            x2 = max(0.0, min(x2, image.width))
            y2 = max(0.0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box after clipping

            detections.append(
                Detection(
                    detection_id=f"yolo-{idx:04d}",
                    class_id=canonical_id,
                    confidence=confidence,
                    box=BoundingBox(x1, y1, x2, y2),
                )
            )

        # Deterministic ordering: class_id asc, confidence desc, detection_id asc
        detections.sort(key=lambda d: (d.class_id, -d.confidence, d.detection_id))
        return detections
