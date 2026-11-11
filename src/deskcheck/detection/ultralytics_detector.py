"""YOLO model adapter using an existing local checkpoint and canonical six-class mapping."""

from collections.abc import Mapping
from math import isfinite

import numpy as np

from deskcheck.application.app_config import CLASS_NAMES
from deskcheck.application.image_frame import ImageFrame
from deskcheck.application.real_settings import RealSettings
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection


def _numpy(value):
    """Convert either torch.Tensor or NumPy array into a detached NumPy copy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).copy()


class UltralyticsDetector:
    """Convert native YOLO boxes to B Detection values in oriented original pixels.

    The optional model injection is for tests only; runtime loads a local .pt file
    after RealSettings already checked its full SHA256. No weight auto-download.
    """

    def __init__(self, settings: RealSettings, model=None):
        if not isinstance(settings, RealSettings):
            raise TypeError("RealSettings required")
        self.settings = settings
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "ultralytics is not installed; install the optional vision dependencies "
                    "into this project environment before loading an existing checkpoint"
                ) from error
            model = YOLO(str(settings.checkpoint_path), task="detect")
        self.model = model
        source_names = model.names
        if isinstance(source_names, Mapping):
            items = source_names.items()
        elif isinstance(source_names, (list, tuple)):
            items = enumerate(source_names)
        else:
            raise TypeError("Model class names must be an ordered list or indexed map")
        mapping = {}
        for key, name in items:
            if type(key) is not int or not isinstance(name, str):
                raise ValueError("Model class names must map integer indices to names")
            normalized = name.strip().casefold()
            if normalized in CLASS_NAMES:
                canonical = CLASS_NAMES.index(normalized)
                if canonical in mapping.values():
                    raise ValueError(f"Ambiguous duplicate model class: {normalized}")
                mapping[key] = canonical
        missing = set(CLASS_NAMES) - {CLASS_NAMES[v] for v in mapping.values()}
        if missing:
            raise ValueError(f"Model checkpoint missing required classes: {sorted(missing)}")
        self.mapping = mapping

    def predict(self, frame: ImageFrame) -> list[Detection]:
        if not isinstance(frame, ImageFrame):
            raise TypeError("YOLO detector requires a canonical C ImageFrame")
        # Ultralytics accepts OpenCV-style BGR HWC, uint8; the loader owns RGB + EXIF.
        bgr = np.ascontiguousarray(frame.array[:, :, ::-1])
        output = self.model.predict(
            source=bgr,
            conf=self.settings.confidence_threshold,
            iou=self.settings.nms_iou_threshold,
            imgsz=self.settings.imgsz,
            device="cpu",
            verbose=False,
        )
        if not isinstance(output, (list, tuple)) or len(output) != 1:
            raise RuntimeError("Expected exactly one YOLO Results item for one image")
        result = output[0]
        if tuple(getattr(result, "orig_shape", ())) != (frame.height, frame.width):
            raise ValueError("YOLO result geometry differs from canonical oriented source")
        boxes = result.boxes
        if boxes is None:
            return []
        coords = _numpy(boxes.xyxy)
        scores = _numpy(boxes.conf)
        classes = _numpy(boxes.cls)
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError("YOLO must return xyxy coordinates of shape Nx4")
        count = coords.shape[0]
        if scores.shape != (count,) or classes.shape != (count,):
            raise ValueError("YOLO scores/classes length disagrees with boxes")
        detections = []
        for index, (xyxy, conf, class_id) in enumerate(zip(coords, scores, classes)):
            if not np.isfinite(class_id) or int(class_id) != class_id:
                raise ValueError("YOLO class_id must be finite integer")
            source_id = int(class_id)
            if source_id not in self.mapping:
                continue
            confidence = float(conf)
            if not isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("YOLO confidence must be finite within [0,1]")
            box = BoundingBox(*(float(v) for v in xyxy))
            box.validate(frame.width, frame.height)
            detections.append(
                Detection(
                    detection_id=f"yolo-{index}",
                    class_id=self.mapping[source_id],
                    confidence=confidence,
                    box=box,
                )
            )
        return detections
