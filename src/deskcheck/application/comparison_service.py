"""Owner C: fake-first orchestration through the actual upstream B modules."""

from pathlib import Path

from deskcheck.application.app_config import CLASS_NAMES, AppConfig
from deskcheck.application.image_frame import ImageFrame
from deskcheck.application.image_pair_loader import ImagePairLoader
from deskcheck.application.real_settings import RealSettings
from deskcheck.changes.change_analyzer import ChangeAnalyzer
from deskcheck.detection.detection_filter import DetectionFilter
from deskcheck.matching.object_matcher import ObjectMatcher


class ComparisonService:
    def __init__(self, *, detector, config: AppConfig | RealSettings, loader=None) -> None:
        if not isinstance(config, (AppConfig, RealSettings)) or config.mode not in ("fake", "real"):
            raise ValueError("An explicitly validated fake or real config is required")
        if not callable(getattr(detector, "predict", None)):
            raise TypeError("detector must expose predict(frame)")
        self._detector = detector
        self._config = config
        self._loader = loader if loader is not None else ImagePairLoader()
        self._filter = DetectionFilter(config.confidence_threshold)
        self._matcher = ObjectMatcher()
        self._analyzer = ChangeAnalyzer(config.movement_threshold)

    @property
    def config(self) -> AppConfig | RealSettings:
        return self._config

    def compare(self, before_path: Path, after_path: Path, pair_id: str) -> dict:
        frames = self._loader.load(before_path, after_path, pair_id)
        return self.compare_frames(*frames, pair_id=pair_id)

    def compare_frames(self, before: ImageFrame, after: ImageFrame, *, pair_id: str) -> dict:
        """Analyze the same immutable pixel snapshots used by the renderer."""
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("pair_id must be a nonempty string")
        if not all(isinstance(frame, ImageFrame) for frame in (before, after)):
            raise TypeError("compare_frames requires C-local ImageFrame values")
        if (before.width, before.height) != (after.width, after.height):
            raise ValueError("Frame dimensions disagree")
        size = (before.width, before.height)
        selected = []
        for frame in (before, after):
            raw = self._detector.predict(frame)
            filtered = self._filter.apply(raw)
            for detection in raw:
                if detection.class_id not in range(len(CLASS_NAMES)):
                    raise ValueError("Detector returned an unsupported canonical class")
                detection.box.validate(*size)
            ids = [d.detection_id for d in raw]
            if len(ids) != len(set(ids)):
                raise ValueError("Detection IDs must be unique within each image")
            selected.append(filtered)
        records = self._analyzer.analyze(self._matcher.match(*selected), size)
        metadata = {
            "mode": self.config.mode,
            "checkpoint_sha256": (
                self.config.checkpoint_sha256 if isinstance(self.config, RealSettings) else None
            ),
            "config_sha256": self.config.config_sha256,
            "rules_version": self.config.rules_version,
            "confidence_threshold": self.config.confidence_threshold,
            "movement_threshold": self.config.movement_threshold,
        }
        if isinstance(self.config, RealSettings):
            metadata.update(
                {
                    "model_origin": self.config.model_origin,
                    "nms_iou_threshold": self.config.nms_iou_threshold,
                    "imgsz": self.config.imgsz,
                    "warning": (
                        "External checkpoint inference; this output does not prove "
                        "dataset quality or academic model accuracy."
                    ),
                }
            )
        else:
            metadata["warning"] = "Synthetic software integration only; no trained model inference."
        return {
            "pair_id": pair_id,
            "image_size": list(size),
            "records": records,
            "metadata": metadata,
        }
