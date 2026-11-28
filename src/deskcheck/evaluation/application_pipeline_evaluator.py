"""Application-pipeline precision/recall/F1 from saved post-B-filter detections.

The saved ``image_detections.json`` arrays were produced after backend confidence
filtering and NMS, canonical class mapping, and B's ``DetectionFilter``, which
keeps at most one detection per class per image. Scoring them therefore measures
the *application pipeline*, not the detector: recall is non-increasing relative
to raw detector output, while precision and F1 can move either way. A true
detector metric requires pre-filter detections, which are not saved yet; see
``docs/proposals/E1_detector_prefilter_design.md``.

Owner decisions reused unchanged (2026-11-26): match at IoU 0.5 with exact class
equality, greedy one-to-one assignment by descending confidence, operating point
read from explicit configuration and recorded, pooled (micro) overall counts.
Nothing here imports the ML backend and no model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from deskcheck.application.app_config import CLASS_NAMES
from deskcheck.evaluation.evidence_labels import IMAGE_DETECTION_EVIDENCE_LABEL

MATCH_IOU_THRESHOLD = 0.5
APPLICATION_SETTING_KEYS = ("confidence_threshold", "nms_iou_threshold")
VALIDATION_SETTING_KEYS = (
    "data",
    "split",
    "imgsz",
    "batch",
    "device",
    "workers",
    "conf",
    "iou",
    "max_det",
    "rect",
    "plots",
    "save_json",
    "project",
    "name",
)

INPUT_SCOPE_LABEL = "post_B_filter_one_per_class"
# Kept as a public alias for existing imports; the value is shared with the writer.
RUNNER_EVIDENCE_LABEL = IMAGE_DETECTION_EVIDENCE_LABEL
RUNNER_EVIDENCE_SOURCE = "run_metadata.image_detection_evidence"
SCOPE_LIMITATION = (
    "Detections are limited to at most one per class per image by B's "
    "DetectionFilter. Recall is non-increasing relative to raw detector output, "
    "while precision and F1 can move either way. A true detector metric requires "
    "pre-filter detections, which are not saved yet."
)
CONFIDENCE_BOUNDARY_NOTE = (
    "The reference backend behaviour reported for this pipeline applies confidence "
    "strictly greater than the recorded threshold, while B's DetectionFilter keeps "
    "confidence greater than or equal to it; that backend comparison was not "
    "re-executed here. The evaluator does not re-apply the threshold: it accepts a "
    "saved detection at exactly the recorded threshold and rejects any detection "
    "strictly below it. When the evaluator is constructed without a minimum, no "
    "confidence validation is applied; the report path always supplies the "
    "configured threshold."
)
RAW_SCOPE_MARKERS = ("raw", "pre-filter", "prefilter", "pre_b_filter", "before_filter")
_CONFIG_REQUIRED = {"schema_version", "class_names", "application_operating_point"}
_CONFIG_OPTIONAL = {"validation_settings"}
_HASH = re.compile(r"[0-9a-f]{64}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"Nonfinite JSON value is invalid: {token}")


def _read_json_object(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{what} is unreadable or malformed: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{what} must be a JSON object")
    return value


def _as_probability(value: Any, what: str) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{what} must be a finite number within [0, 1]")
    return float(value)


def _as_box(value: Any, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{what} must have four xyxy coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(item)
        for item in value
    ):
        raise ValueError(f"{what} coordinates must be finite numbers")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError(f"{what} must have nonnegative coordinates and positive area")
    return [x1, y1, x2, y2]


def _iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


@dataclass(frozen=True)
class ApplicationPipelineConfig:
    """Explicit operating point plus optional, recorded validation settings."""

    class_names: tuple[str, ...]
    confidence_threshold: float
    nms_iou_threshold: float
    config_sha256: str
    source_path: Path
    validation_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.class_names or any(
            not isinstance(name, str) or not name.strip() for name in self.class_names
        ):
            raise ValueError("class_names must be nonempty strings")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("nms_iou_threshold", self.nms_iou_threshold),
        ):
            _as_probability(value, name)
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise ValueError("config source path must be an absolute Path")

    def application_operating_point(self) -> dict[str, Any]:
        """Recorded operating point of the saved detections; never re-applied."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "source": str(self.source_path),
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def load(cls, path: str | Path) -> ApplicationPipelineConfig:
        config_path = Path(path).resolve(strict=True)
        content = config_path.read_bytes()
        try:
            raw = json.loads(
                content,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except ValueError as exc:
            raise ValueError(f"Evaluation config is malformed: {exc}") from exc
        if not isinstance(raw, dict):
            raise TypeError("Evaluation config must be a JSON object")
        unknown = set(raw) - _CONFIG_REQUIRED - _CONFIG_OPTIONAL
        if unknown:
            raise ValueError(f"Evaluation config has unknown keys {sorted(unknown)}")
        missing = _CONFIG_REQUIRED - set(raw)
        if missing:
            raise ValueError(f"Evaluation config is missing {sorted(missing)}")
        if raw["schema_version"] != "1.0":
            raise ValueError("Evaluation config requires schema_version 1.0")
        if raw["class_names"] != list(CLASS_NAMES):
            raise ValueError("Evaluation config must use the canonical class map")

        operating_point = raw["application_operating_point"]
        if not isinstance(operating_point, dict):
            raise TypeError("application_operating_point must be a JSON object")
        if set(operating_point) != set(APPLICATION_SETTING_KEYS):
            raise ValueError(
                f"application_operating_point must contain exactly {list(APPLICATION_SETTING_KEYS)}"
            )

        validation = raw.get("validation_settings", {})
        if not isinstance(validation, dict):
            raise TypeError("validation_settings must be a JSON object")
        extra_validation = set(validation) - set(VALIDATION_SETTING_KEYS)
        if extra_validation:
            raise ValueError(f"Unknown validation settings {sorted(extra_validation)}")
        for key, value in validation.items():
            if value is None:
                raise ValueError(f"validation setting {key!r} must be an explicit value")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError(f"validation setting {key!r} must be finite")
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"validation setting {key!r} must be a JSON scalar")

        return cls(
            class_names=tuple(raw["class_names"]),
            confidence_threshold=_as_probability(
                operating_point["confidence_threshold"],
                "application_operating_point.confidence_threshold",
            ),
            nms_iou_threshold=_as_probability(
                operating_point["nms_iou_threshold"],
                "application_operating_point.nms_iou_threshold",
            ),
            config_sha256=_sha256_bytes(content),
            source_path=config_path,
            validation_settings=dict(validation),
        )


class ApplicationPipelineEvaluator:
    """Score saved post-B-filter detections with the owner-fixed matching rule.

    ``min_confidence`` defaults to ``None``: direct construction performs no
    confidence validation, so a saved detection below the operating point would
    still be scored. The report path (``evaluate_application_pipeline_predictions``
    and ``evaluate_saved_pipeline_run``) always passes the configured threshold,
    and the output records whether validation was applied.
    """

    def __init__(
        self,
        class_names: tuple[str, ...] | list[str] = CLASS_NAMES,
        iou_threshold: float = MATCH_IOU_THRESHOLD,
        min_confidence: float | None = None,
    ) -> None:
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not isfinite(iou_threshold)
            or not 0 < iou_threshold <= 1
        ):
            raise ValueError("Match IoU threshold must be finite and in (0, 1]")
        names = tuple(class_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("class_names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("class_names must be unique")
        if min_confidence is not None:
            min_confidence = _as_probability(min_confidence, "min_confidence")
        self.class_names = names
        self.iou_threshold = float(iou_threshold)
        self.min_confidence = min_confidence

    def evaluate(self, images: list[dict]) -> dict:
        """Score per-image post-B-filter detections against per-image ground truth.

        Each record is {"image_id": str, "detections": [...], "ground_truth": [...]}.
        Detections carry class_id, confidence and bbox_xyxy; ground truth carries
        class_id and bbox_xyxy. Empty detections or ground truth are valid. Missing
        images, duplicate image IDs, unknown class IDs and invalid boxes are errors.
        When ``min_confidence`` is set, any saved detection strictly below it is
        rejected; a detection at exactly that value is accepted.
        """
        normalized = self._validate_images(images)
        per_class_index = {
            index: {"tp": 0, "fp": 0, "fn": 0} for index in range(len(self.class_names))
        }
        per_image: dict[str, dict] = {}
        for image in normalized:
            outcome = self._match_image(image)
            per_image[image["image_id"]] = outcome
            for index, counts in outcome["counts_by_class"].items():
                for key in ("tp", "fp", "fn"):
                    per_class_index[index][key] += counts[key]

        per_class = {
            name: self._summarize(per_class_index[index])
            for index, name in enumerate(self.class_names)
        }
        pooled = {"tp": 0, "fp": 0, "fn": 0}
        for counts in per_class_index.values():
            for key in pooled:
                pooled[key] += counts[key]
        overall = self._summarize(pooled)
        overall["aggregation"] = "micro: pooled TP/FP/FN counts over all images and classes"
        macro_classes = [
            name
            for name, values in per_class.items()
            if values["tp"] + values["fp"] + values["fn"] > 0
        ]
        macro_f1 = (
            sum(per_class[name]["f1"] for name in macro_classes) / len(macro_classes)
            if macro_classes
            else 0.0
        )
        return {
            "protocol_version": "application-pipeline-prf-v1",
            "metric_name": "application_pipeline_precision_recall_f1",
            "metric_scope": (
                "application-pipeline precision/recall/F1 over post-B-filter detections; "
                "not a detector metric and not backend mAP"
            ),
            "input_scope": INPUT_SCOPE_LABEL,
            "scope_limitation": SCOPE_LIMITATION,
            "image_count": len(normalized),
            "class_names": list(self.class_names),
            "matching_rules": {
                "iou_threshold": self.iou_threshold,
                "class_matching": "exact class_id match",
                "assignment": "greedy one-to-one, detections in descending confidence order",
                "duplicate_detection_of_matched_ground_truth": "false positive",
                "unmatched_ground_truth": "false negative",
                "confidence_tie_break": (
                    "equal confidence ordered by ascending (x1, y1, x2, y2, detection_id); "
                    "fully identical keys keep input order"
                ),
                "iou_tie_break": (
                    "highest IoU; equal IoU ordered by ascending "
                    "(x1, y1, x2, y2, ground-truth index)"
                ),
            },
            "matching_note": (
                "These counts use a custom matching procedure on post-B-filter detections. "
                "They are not comparable to backend mAP counts and must not be reported as "
                "a detector metric."
            ),
            "confidence_boundary": {
                "applied": self.min_confidence is not None,
                "min_confidence": self.min_confidence,
                "note": CONFIDENCE_BOUNDARY_NOTE,
            },
            "zero_division": 0,
            "overall": overall,
            "per_class": per_class,
            "per_class_macro_f1": {
                "role": "secondary",
                "value": macro_f1,
                "aggregation": (
                    "unweighted mean of per-class F1 over classes with predictions or ground truth"
                ),
                "classes_included": macro_classes,
                "unavailable": not macro_classes,
            },
            "per_image": {
                image_id: {
                    "tp": outcome["tp"],
                    "fp": outcome["fp"],
                    "fn": outcome["fn"],
                    "detection_outcomes": outcome["detection_outcomes"],
                    "unmatched_ground_truth_indices": outcome["unmatched_ground_truth_indices"],
                }
                for image_id, outcome in per_image.items()
            },
        }

    def _validate_images(self, images: list[dict]) -> list[dict]:
        if not isinstance(images, list):
            raise TypeError("images must be a list")
        if not images:
            raise ValueError("At least one image is required")
        normalized = []
        seen: set[str] = set()
        for image in images:
            if not isinstance(image, dict):
                raise TypeError("Each image record must be a dictionary")
            if not {"image_id", "detections", "ground_truth"} <= image.keys():
                raise ValueError("Image record is missing required fields")
            image_id = image["image_id"]
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("image_id must be a nonempty string")
            if image_id in seen:
                raise ValueError(f"Duplicate image_id: {image_id}")
            seen.add(image_id)
            normalized.append(
                {
                    "image_id": image_id,
                    "detections": self._validate_detections(image["detections"]),
                    "ground_truth": self._validate_ground_truth(image["ground_truth"]),
                }
            )
        return sorted(normalized, key=lambda item: item["image_id"])

    def _validate_class_id(self, value: Any) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("class_id must be a nonnegative integer")
        if value >= len(self.class_names):
            raise ValueError(
                f"Unknown class_id {value}; canonical class map has {len(self.class_names)}"
            )
        return value

    def _validate_detections(self, detections: Any) -> list[dict]:
        if not isinstance(detections, list):
            raise TypeError("detections must be a list")
        normalized = []
        for detection in detections:
            if not isinstance(detection, dict):
                raise TypeError("Each detection must be a dictionary")
            if not {"class_id", "confidence", "bbox_xyxy"} <= detection.keys():
                raise ValueError("Detection is missing required fields")
            detection_id = detection.get("detection_id")
            if detection_id is not None and not isinstance(detection_id, str):
                raise TypeError("detection_id must be a string when provided")
            confidence = _as_probability(detection["confidence"], "confidence")
            if self.min_confidence is not None and confidence < self.min_confidence:
                raise ValueError(
                    f"Saved detection confidence {confidence!r} is below the recorded "
                    f"operating point {self.min_confidence!r}; this artifact was not "
                    "produced by the declared pipeline"
                )
            normalized.append(
                {
                    "class_id": self._validate_class_id(detection["class_id"]),
                    "confidence": confidence,
                    "bbox_xyxy": _as_box(detection["bbox_xyxy"], "bbox_xyxy"),
                    "detection_id": detection_id,
                }
            )
        return normalized

    def _validate_ground_truth(self, ground_truth: Any) -> list[dict]:
        if not isinstance(ground_truth, list):
            raise TypeError("ground_truth must be a list")
        normalized = []
        for item in ground_truth:
            if not isinstance(item, dict):
                raise TypeError("Each ground-truth item must be a dictionary")
            if not {"class_id", "bbox_xyxy"} <= item.keys():
                raise ValueError("Ground-truth item is missing required fields")
            normalized.append(
                {
                    "class_id": self._validate_class_id(item["class_id"]),
                    "bbox_xyxy": _as_box(item["bbox_xyxy"], "ground-truth bbox_xyxy"),
                }
            )
        return normalized

    def _match_image(self, image: dict) -> dict:
        detections = image["detections"]
        ground_truth = image["ground_truth"]
        order = sorted(
            range(len(detections)),
            key=lambda index: (
                -detections[index]["confidence"],
                *detections[index]["bbox_xyxy"],
                detections[index]["detection_id"] or "",
            ),
        )
        unmatched = set(range(len(ground_truth)))
        counts_by_class = {
            index: {"tp": 0, "fp": 0, "fn": 0} for index in range(len(self.class_names))
        }
        outcomes = [None] * len(detections)
        for index in order:
            detection = detections[index]
            best: tuple[tuple, int, float] | None = None
            for truth_index in sorted(unmatched):
                truth = ground_truth[truth_index]
                if truth["class_id"] != detection["class_id"]:
                    continue
                iou = _iou(detection["bbox_xyxy"], truth["bbox_xyxy"])
                key = (-iou, *truth["bbox_xyxy"], truth_index)
                if best is None or key < best[0]:
                    best = (key, truth_index, iou)
            if best is not None and best[2] >= self.iou_threshold:
                truth_index, iou = best[1], best[2]
                unmatched.discard(truth_index)
                counts_by_class[detection["class_id"]]["tp"] += 1
                outcomes[index] = {
                    "detection_index": index,
                    "detection_id": detection["detection_id"],
                    "class_id": detection["class_id"],
                    "confidence": detection["confidence"],
                    "outcome": "true_positive",
                    "matched_ground_truth_index": truth_index,
                    "iou": iou,
                }
            else:
                counts_by_class[detection["class_id"]]["fp"] += 1
                outcomes[index] = {
                    "detection_index": index,
                    "detection_id": detection["detection_id"],
                    "class_id": detection["class_id"],
                    "confidence": detection["confidence"],
                    "outcome": "false_positive",
                    "matched_ground_truth_index": None,
                    "iou": best[2] if best is not None else None,
                }
        for truth_index in unmatched:
            counts_by_class[ground_truth[truth_index]["class_id"]]["fn"] += 1
        tp = sum(counts["tp"] for counts in counts_by_class.values())
        fp = sum(counts["fp"] for counts in counts_by_class.values())
        fn = sum(counts["fn"] for counts in counts_by_class.values())
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "counts_by_class": counts_by_class,
            "detection_outcomes": outcomes,
            "unmatched_ground_truth_indices": sorted(unmatched),
        }

    @staticmethod
    def _summarize(counts: dict) -> dict:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        unavailable = []
        if tp + fp == 0:
            unavailable.append("precision")
        if tp + fn == 0:
            unavailable.append("recall")
        if 2 * tp + fp + fn == 0:
            unavailable.append("f1")
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "unavailable_metrics": unavailable,
        }


def evaluate_application_pipeline_predictions(
    config: ApplicationPipelineConfig,
    images: list[dict],
    *,
    map_result: dict | None = None,
) -> dict:
    """Build the pipeline report with separate operating-point and mAP sections."""
    if not isinstance(config, ApplicationPipelineConfig):
        raise TypeError("config must be a loaded ApplicationPipelineConfig")
    evaluation = ApplicationPipelineEvaluator(
        config.class_names,
        min_confidence=config.confidence_threshold,
    ).evaluate(images)
    evaluation["application_operating_point"] = {
        **config.application_operating_point(),
        "note": (
            "Saved detections are already filtered at this operating point; "
            "these values are recorded, not re-applied or tuned."
        ),
    }
    return {
        "schema_version": "1.0",
        "report_type": "application_pipeline_evaluation",
        "report_scope": (
            "application-pipeline diagnostic from saved post-B-filter detections; "
            "no model loading; not a detector metric"
        ),
        "input_scope": {
            "label": INPUT_SCOPE_LABEL,
            "limitation": SCOPE_LIMITATION,
            "true_detector_metric": ("blocked until pre-filter detections are saved and approved"),
        },
        "application_pipeline_prf": evaluation,
        "backend_validation_map": map_result
        if map_result is not None
        else {
            "state": "not_run",
            "reason": "Backend validation mAP was not supplied for this report.",
        },
    }


def load_image_ground_truth(path: str | Path) -> dict:
    """Load per-image boxes: {image_id: [{"class_id", "bbox_xyxy"}, ...]}."""
    source = Path(path)
    raw = _read_json_object(source, "Image ground-truth file")
    required = {"schema_version", "split", "dataset_release", "images"}
    if set(raw) != required:
        raise ValueError(f"Image ground-truth file must contain exactly {sorted(required)}")
    if raw["schema_version"] != "1.0":
        raise ValueError("Image ground-truth file requires schema_version 1.0")
    if raw["split"] not in ("val", "test"):
        raise ValueError("Image ground-truth file must declare split 'val' or 'test'")
    if not isinstance(raw["dataset_release"], str) or not raw["dataset_release"].strip():
        raise ValueError("Image ground-truth file requires a dataset release")
    if not isinstance(raw["images"], list) or not raw["images"]:
        raise ValueError("Image ground-truth file requires a nonempty images list")
    images: dict[str, list[dict]] = {}
    for record in raw["images"]:
        if not isinstance(record, dict) or set(record) != {"image_id", "ground_truth"}:
            raise ValueError("Each image record must contain image_id and ground_truth")
        image_id = record["image_id"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("image_id must be a nonempty string")
        if image_id in images:
            raise ValueError(f"Duplicate image_id: {image_id}")
        images[image_id] = record["ground_truth"]
    return {
        "images": images,
        "split": raw["split"],
        "dataset_release": raw["dataset_release"],
        "sha256": _sha256_bytes(source.read_bytes()),
    }


def _require_completed_run(metadata: dict) -> tuple[str, list[str]]:
    if metadata.get("schema_version") != "1.0" or metadata.get("status") != "complete":
        raise ValueError("Evaluation run is not a completed schema 1.0 result")
    if metadata.get("runner_protocol") != "event-runner-v1":
        raise ValueError("Evaluation run uses an unsupported evaluation runner protocol")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Evaluation run does not record a run_id; its identity cannot be bound")
    pairs = metadata.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Evaluation run does not record a nonempty pair inventory")
    pair_ids = []
    for pair in pairs:
        required = {"pair_id", "before_path", "after_path"}
        if not isinstance(pair, dict) or not required <= pair.keys():
            raise ValueError("Evaluation run pair inventory has an invalid record")
        pair_id = pair["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("Evaluation run pair IDs must be nonempty strings")
        pair_ids.append(pair_id)
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Evaluation run pair inventory contains duplicate IDs")
    if metadata.get("pair_count") != len(pair_ids):
        raise ValueError("Evaluation run pair_count disagrees with its pair inventory")
    return run_id, pair_ids


def _input_scope(metadata: dict, detections_present: bool) -> dict:
    evidence = metadata.get("image_detection_evidence")
    if evidence is None:
        raise ValueError(
            "Evaluation run does not record image_detection_evidence; the input scope "
            "is unknown and is not guessed"
        )
    if not isinstance(evidence, str) or not evidence.strip():
        raise TypeError("image_detection_evidence must be a nonempty string")
    if evidence == "not provided":
        if detections_present:
            raise ValueError(
                "Contradictory run: image_detection_evidence says no detections were "
                "provided, but image_detections.json exists"
            )
        raise ValueError(
            "Evaluation run recorded no per-image detections; there is nothing to score"
        )
    folded = evidence.casefold()
    if any(marker in folded for marker in RAW_SCOPE_MARKERS):
        raise ValueError(
            f"Run declares a raw/pre-filter detection scope ({evidence!r}), but no such "
            "artifact exists yet. A true detector metric needs the pre-filter artifact "
            "described in docs/proposals/E1_detector_prefilter_design.md; this evaluator "
            "scores post-B-filter detections only"
        )
    if evidence != IMAGE_DETECTION_EVIDENCE_LABEL:
        raise ValueError(
            f"Unknown image_detection_evidence label {evidence!r}; only "
            f"{IMAGE_DETECTION_EVIDENCE_LABEL!r} is accepted"
        )
    if not detections_present:
        raise ValueError(
            "Contradictory run: image_detection_evidence declares post-B-filter "
            "detections, but image_detections.json is missing"
        )
    return {
        "label": INPUT_SCOPE_LABEL,
        "source": RUNNER_EVIDENCE_SOURCE,
        "recorded_label": evidence,
        "limitation": SCOPE_LIMITATION,
    }


def load_runner_pipeline_detections(run_dir: str | Path) -> dict:
    """Reuse a completed EvaluationRunner output's post-B-filter image detections."""
    root = Path(run_dir)
    metadata = _read_json_object(root / "run_metadata.json", "Evaluation run metadata")
    run_id, pair_ids = _require_completed_run(metadata)
    detections_path = root / "image_detections.json"
    scope = _input_scope(metadata, detections_path.is_file())
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("Evaluation run metadata has no provenance object")
    checkpoint = provenance.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not _HASH.fullmatch(checkpoint):
        raise ValueError("Evaluation run does not record a lowercase SHA-256 checkpoint identity")
    raw = _read_json_object(detections_path, "Evaluation run image detections")
    if set(raw) != set(pair_ids):
        raise ValueError("Saved detections do not match the run's pair inventory")
    images: dict[str, list] = {}
    for pair_id in sorted(pair_ids):
        sides = raw[pair_id]
        if not isinstance(sides, dict) or set(sides) != {"before", "after"}:
            raise ValueError(
                f"Pair {pair_id!r} detections must contain exactly before and after lists"
            )
        for side in ("before", "after"):
            if not isinstance(sides[side], list):
                raise TypeError(f"Pair {pair_id!r} {side} detections must be a list")
            images[f"{pair_id}__{side}"] = sides[side]
    return {
        "images": images,
        "split": provenance.get("split"),
        "dataset_release": provenance.get("dataset_release"),
        "operating_point": {key: provenance.get(key) for key in APPLICATION_SETTING_KEYS},
        "run_id": run_id,
        "checkpoint_sha256": checkpoint,
        "input_scope": scope,
        "pair_ids": pair_ids,
    }


def evaluate_saved_pipeline_run(
    config: ApplicationPipelineConfig,
    run_dir: str | Path,
    ground_truth_path: str | Path,
    *,
    map_result: dict | None = None,
) -> dict:
    """Score a completed runner output against a per-image ground-truth file."""
    predictions = load_runner_pipeline_detections(run_dir)
    truth = load_image_ground_truth(ground_truth_path)
    if predictions["split"] != truth["split"]:
        raise ValueError("Prediction split differs from the ground-truth split")
    if predictions["dataset_release"] != truth["dataset_release"]:
        raise ValueError("Prediction dataset release differs from the ground-truth release")
    if set(predictions["images"]) != set(truth["images"]):
        raise ValueError("Prediction and ground-truth image inventories must match exactly")
    expected = config.application_operating_point()
    for key in APPLICATION_SETTING_KEYS:
        declared = predictions["operating_point"][key]
        if declared is None:
            raise ValueError(f"Evaluation run does not record {key}")
        # Exact JSON equality, deliberately: numeric values must match exactly, so
        # the string "0.2" does not match 0.2 and values one ulp apart do not match.
        # No tolerance or coercion is applied here.
        if declared != expected[key]:
            raise ValueError(
                f"Evaluation run {key}={declared!r} differs from the evaluation config"
            )
    images = [
        {
            "image_id": image_id,
            "detections": predictions["images"][image_id],
            "ground_truth": truth["images"][image_id],
        }
        for image_id in sorted(truth["images"])
    ]
    report = evaluate_application_pipeline_predictions(config, images, map_result=map_result)
    report["inputs"] = {
        "evaluation_run": str(Path(run_dir).resolve()),
        "run_id": predictions["run_id"],
        "checkpoint_sha256": predictions["checkpoint_sha256"],
        "ground_truth": str(Path(ground_truth_path).resolve()),
        "ground_truth_sha256": truth["sha256"],
        "split": truth["split"],
        "dataset_release": truth["dataset_release"],
        "input_scope_source": RUNNER_EVIDENCE_SOURCE,
    }
    report["binding"] = {
        "run_id": predictions["run_id"],
        "checkpoint_sha256": predictions["checkpoint_sha256"],
        "verified": [
            "run is a completed schema 1.0 event-runner-v1 result",
            "run_id is recorded in the saved run",
            "pair inventory and pair_count match the saved detections",
            "saved detections declare the accepted post-B-filter input scope",
            "recorded operating point equals the evaluation config",
            "recorded checkpoint identity is a 64-character SHA-256 string",
        ],
        "not_verified": [
            (
                "run_metadata stores no digest of image_detections.json, so the file is "
                "not provably the artifact this run produced"
            ),
            (
                "checkpoint bytes were not read, so the checkpoint identity is a "
                "recorded string only"
            ),
            (
                "the upstream filter and backend were not re-run; only saved "
                "confidences were checked against the recorded threshold"
            ),
            (
                "image bytes and their recorded before/after SHA-256 values were not "
                "re-read or compared with the saved detections"
            ),
            (
                "the run's manifest_sha256 was not compared with the ground-truth file, "
                "which carries no manifest identity"
            ),
            (
                "the run's config_sha256 was not compared with the evaluation config "
                "hash, because they identify different configuration files"
            ),
            (
                "per-pair prediction_scope was not checked; the runner does not persist "
                "it, so only the run-level evidence label is available"
            ),
        ],
        "tamper_note": (
            "Provenance fields are self-declared. A hand-edited run can satisfy local "
            "equality checks; this is not tamper proofing."
        ),
    }
    return report
