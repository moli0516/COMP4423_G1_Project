"""Synthetic fixtures only: no real images, checkpoints or backend.

The subject under test is an application-pipeline diagnostic. Several tests below
deliberately demonstrate that scoring B's one-per-class output is not a detector
metric, so go unnoticed contamination would fail the suite.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest
from test_failure_record import record_kwargs

from deskcheck.detection.detection_filter import DetectionFilter
from deskcheck.domain.bounding_box import BoundingBox
from deskcheck.domain.detection import Detection
from deskcheck.evaluation.application_pipeline_evaluator import (
    APPLICATION_SETTING_KEYS,
    INPUT_SCOPE_LABEL,
    RUNNER_EVIDENCE_LABEL,
    ApplicationPipelineConfig,
    ApplicationPipelineEvaluator,
    evaluate_application_pipeline_predictions,
    evaluate_saved_pipeline_run,
    load_image_ground_truth,
    load_runner_pipeline_detections,
)
from deskcheck.evaluation.failure_record import (
    build_failure_record,
    build_rerun_link,
    write_failure_record,
)

CLASS_NAMES = ["bottle", "cup", "keyboard", "mouse", "book", "scissors"]
CHECKPOINT_SHA = "e" * 64


def detection(class_id=0, confidence=0.9, box=(0, 0, 3, 1), detection_id=None):
    value = {"class_id": class_id, "confidence": confidence, "bbox_xyxy": list(box)}
    if detection_id is not None:
        value["detection_id"] = detection_id
    return value


def truth(class_id=0, box=(0, 0, 3, 1)):
    return {"class_id": class_id, "bbox_xyxy": list(box)}


def image(image_id="img", detections=None, ground_truth=None):
    return {
        "image_id": image_id,
        "detections": [] if detections is None else detections,
        "ground_truth": [] if ground_truth is None else ground_truth,
    }


def score(detections, ground_truth, **kwargs):
    return ApplicationPipelineEvaluator(**kwargs).evaluate(
        [image(detections=detections, ground_truth=ground_truth)]
    )


def as_evaluator_input(detection_value):
    return {
        "detection_id": detection_value.detection_id,
        "class_id": detection_value.class_id,
        "confidence": detection_value.confidence,
        "bbox_xyxy": [
            detection_value.box.x1,
            detection_value.box.y1,
            detection_value.box.x2,
            detection_value.box.y2,
        ],
    }


def write_config(tmp_path, *, confidence=0.2, nms=0.35, validation=None, class_names=None):
    payload = {
        "schema_version": "1.0",
        "class_names": CLASS_NAMES if class_names is None else class_names,
        "application_operating_point": {
            "confidence_threshold": confidence,
            "nms_iou_threshold": nms,
        },
    }
    if validation is not None:
        payload["validation_settings"] = validation
    path = tmp_path / "pipeline_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_runner_run(
    tmp_path,
    *,
    confidence=0.2,
    nms=0.35,
    split="val",
    evidence=RUNNER_EVIDENCE_LABEL,
    detections=None,
    run_id="synthetic-run-1",
    pairs=("p1",),
):
    run = tmp_path / "synthetic-run"
    run.mkdir(exist_ok=True)
    metadata = {
        "schema_version": "1.0",
        "status": "complete",
        "runner_protocol": "event-runner-v1",
        "run_id": run_id,
        "pair_count": len(pairs),
        "pairs": [
            {
                "pair_id": pair_id,
                "before_path": f"{pair_id}-before.png",
                "after_path": f"{pair_id}-after.png",
            }
            for pair_id in pairs
        ],
        "provenance": {
            "split": split,
            "dataset_release": "synthetic-release",
            "confidence_threshold": confidence,
            "nms_iou_threshold": nms,
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
    }
    if evidence is not None:
        metadata["image_detection_evidence"] = evidence
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    payload = (
        {"p1": {"before": [detection()], "after": [detection()]}}
        if detections is None
        else detections
    )
    (run / "image_detections.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


def write_ground_truth(tmp_path, *, split="val", release="synthetic-release", images=None):
    path = tmp_path / "ground_truth.json"
    payload = {
        "schema_version": "1.0",
        "split": split,
        "dataset_release": release,
        "images": images
        if images is not None
        else [
            {"image_id": "p1__before", "ground_truth": [truth()]},
            {"image_id": "p1__after", "ground_truth": [truth()]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_perfect_match_counts_true_positive():
    result = score([detection()], [truth()])
    overall = result["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 0, 0)
    assert overall["precision"] == overall["recall"] == overall["f1"] == 1
    assert overall["unavailable_metrics"] == []
    assert overall["aggregation"].startswith("micro")
    assert result["matching_rules"]["iou_threshold"] == 0.5


def test_output_is_labelled_as_application_pipeline_not_detector():
    result = score([detection()], [truth()])
    assert result["metric_name"] == "application_pipeline_precision_recall_f1"
    assert result["input_scope"] == INPUT_SCOPE_LABEL
    assert "not a detector metric" in result["metric_scope"]
    assert "one per class per image" in result["scope_limitation"]
    assert "must not be reported as a detector metric" in result["matching_note"]
    assert "detector" not in result["metric_name"]
    assert result["protocol_version"] == "application-pipeline-prf-v1"


def test_class_mismatch_counts_false_positive_and_false_negative():
    result = score([detection(class_id=1)], [truth(class_id=0)])
    assert result["overall"]["tp"] == 0
    assert result["per_class"]["cup"]["fp"] == 1
    assert result["per_class"]["bottle"]["fn"] == 1
    assert result["per_class"]["bottle"]["recall"] == 0


@pytest.mark.parametrize(
    "shift, expected_tp",
    [
        (1.0, 1),
        (1.01, 0),
        (0.99, 1),
        (1.0 - 1e-12, 1),
        (1.0 + 1e-12, 0),
    ],
)
def test_iou_boundary_is_inclusive_at_half(shift, expected_tp):
    result = score(
        [detection(box=(shift, 0, shift + 3, 1))],
        [truth(box=(0, 0, 3, 1))],
    )
    assert result["overall"]["tp"] == expected_tp
    assert result["overall"]["fp"] == 1 - expected_tp
    assert result["overall"]["fn"] == 1 - expected_tp


def test_touching_edges_have_zero_iou():
    result = score([detection(box=(3, 0, 6, 1))], [truth(box=(0, 0, 3, 1))])
    assert result["overall"]["tp"] == 0
    assert result["overall"]["fp"] == 1
    assert result["overall"]["fn"] == 1


def test_very_small_boxes_are_handled_without_error():
    tiny = (0.0, 0.0, 1e-6, 1e-6)
    identical = score([detection(box=tiny)], [truth(box=tiny)])
    assert identical["overall"]["tp"] == 1
    assert identical["matching_rules"]["iou_threshold"] == 0.5
    touching = score([detection(box=(1e-6, 0.0, 2e-6, 1e-6))], [truth(box=tiny)])
    assert touching["overall"]["fp"] == 1
    assert touching["overall"]["fn"] == 1


def test_duplicate_detection_is_false_positive():
    result = score([detection(), detection()], [truth()])
    assert (result["overall"]["tp"], result["overall"]["fp"], result["overall"]["fn"]) == (1, 1, 0)


def test_higher_confidence_duplicate_wins_and_lower_becomes_false_positive():
    result = score(
        [
            detection(confidence=0.5, detection_id="low"),
            detection(confidence=0.9, detection_id="high"),
        ],
        [truth()],
    )
    outcomes = {
        item["detection_id"]: item["outcome"]
        for item in result["per_image"]["img"]["detection_outcomes"]
    }
    assert outcomes == {"high": "true_positive", "low": "false_positive"}


def test_confidence_tie_break_uses_detection_id():
    result = score(
        [detection(detection_id="b"), detection(detection_id="a")],
        [truth()],
    )
    outcomes = {
        item["detection_id"]: item["outcome"]
        for item in result["per_image"]["img"]["detection_outcomes"]
    }
    assert outcomes == {"a": "true_positive", "b": "false_positive"}


def test_identical_detections_without_detection_id_keep_input_order():
    first, second = detection(), detection()
    result = score([first, second], [truth()])
    outcomes = [item["outcome"] for item in result["per_image"]["img"]["detection_outcomes"]]
    assert outcomes == ["true_positive", "false_positive"]
    assert result["per_image"]["img"]["detection_outcomes"][0]["detection_id"] is None


def test_iou_tie_break_uses_lowest_ground_truth_index_for_identical_boxes():
    result = score([detection()], [truth(), truth()])
    outcomes = result["per_image"]["img"]["detection_outcomes"]
    assert outcomes[0]["matched_ground_truth_index"] == 0
    assert result["per_image"]["img"]["unmatched_ground_truth_indices"] == [1]


def test_equal_iou_across_different_ground_truth_boxes_uses_bounding_box_order():
    result = score(
        [detection(box=(0, 0, 10, 10))],
        [truth(box=(0, 0, 10, 5)), truth(box=(0, 0, 5, 10))],
    )
    # Both candidate IoUs are exactly 0.5; ascending bbox (x2 5 < 10) wins over index.
    assert result["per_image"]["img"]["detection_outcomes"][0]["matched_ground_truth_index"] == 1
    assert result["per_image"]["img"]["unmatched_ground_truth_indices"] == [0]


def test_greedy_assignment_is_intentionally_suboptimal():
    # Owner rule: detections are matched in descending confidence order, so a
    # higher-confidence box with a worse IoU wins the ground truth. Intended.
    result = score(
        [
            detection(confidence=0.9, box=(0, 0, 10, 5)),
            detection(confidence=0.8, box=(0, 0, 10, 10)),
        ],
        [truth(box=(0, 0, 10, 10))],
    )
    outcomes = [item["outcome"] for item in result["per_image"]["img"]["detection_outcomes"]]
    assert outcomes == ["true_positive", "false_positive"]
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 1
    assert result["overall"]["fn"] == 0


def test_empty_predictions_are_false_negatives():
    result = score([], [truth()])
    assert result["overall"]["tp"] == 0
    assert result["overall"]["fn"] == 1
    assert result["overall"]["precision"] == 0
    assert "precision" in result["overall"]["unavailable_metrics"]


def test_empty_ground_truth_is_false_positive():
    result = score([detection()], [])
    assert result["overall"]["fp"] == 1
    assert result["overall"]["support"] == 0


def test_empty_predictions_and_ground_truth_are_allowed_and_unavailable():
    result = score([], [])
    assert result["overall"]["tp"] == result["overall"]["fp"] == result["overall"]["fn"] == 0
    assert set(result["overall"]["unavailable_metrics"]) == {"precision", "recall", "f1"}
    assert result["overall"]["precision"] == 0
    assert result["per_class_macro_f1"]["unavailable"] is True
    assert result["per_class_macro_f1"]["value"] == 0


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError, match="Unknown class_id"):
        score([detection(class_id=6)], [])
    with pytest.raises(ValueError, match="Unknown class_id"):
        score([], [truth(class_id=99)])
    with pytest.raises(ValueError):
        score([detection(class_id=-1)], [])


@pytest.mark.parametrize(
    "box",
    [(0, 0, 0, 1), (-1, 0, 2, 2), (0, 0, 2), (0, 0, float("nan"), 1)],
)
def test_invalid_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        score([detection(box=box)], [])


def test_nonfinite_confidence_is_rejected():
    with pytest.raises(ValueError):
        score([detection(confidence=float("inf"))], [])


def test_duplicate_image_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate image_id"):
        ApplicationPipelineEvaluator().evaluate([image("same"), image("same")])


def test_empty_inventory_is_rejected():
    with pytest.raises(ValueError, match="At least one image"):
        ApplicationPipelineEvaluator().evaluate([])


def test_per_class_and_pooled_counts_are_micro():
    images = [
        image(
            "a",
            [detection(class_id=0), detection(class_id=0, box=(5, 0, 8, 1))],
            [truth(class_id=0)],
        ),
        image(
            "b",
            [detection(class_id=1)],
            [truth(class_id=1, box=(0, 0, 3, 1)), truth(class_id=1, box=(9, 0, 12, 1))],
        ),
    ]
    result = ApplicationPipelineEvaluator().evaluate(images)
    assert result["per_class"]["bottle"]["tp"] == 1
    assert result["per_class"]["bottle"]["fp"] == 1
    assert result["per_class"]["cup"]["tp"] == 1
    assert result["per_class"]["cup"]["fn"] == 1
    overall = result["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (2, 1, 1)
    assert overall["precision"] == pytest.approx(2 / 3)
    assert overall["recall"] == pytest.approx(2 / 3)
    assert overall["f1"] == pytest.approx(2 / 3)
    assert result["per_class_macro_f1"]["role"] == "secondary"
    assert result["per_class_macro_f1"]["classes_included"] == ["bottle", "cup"]


def test_zero_and_unavailable_are_distinguishable():
    result = score([], [])
    empty_class = result["per_class"]["bottle"]
    assert empty_class["precision"] == 0
    assert "precision" in empty_class["unavailable_metrics"]
    detected = score([detection(class_id=0)], [])
    assert detected["per_class"]["bottle"]["precision"] == 0
    assert "precision" not in detected["per_class"]["bottle"]["unavailable_metrics"]
    assert "recall" in detected["per_class"]["bottle"]["unavailable_metrics"]


def test_post_filter_input_cannot_reproduce_raw_detector_counts():
    """Contamination guard: B's one-per-class filter changes the scored counts."""
    raw = [
        Detection("raw-0", 0, 0.9, BoundingBox(0.0, 0.0, 3.0, 1.0)),
        Detection("raw-1", 0, 0.8, BoundingBox(10.0, 0.0, 13.0, 1.0)),
    ]
    post_filter = DetectionFilter(0.2).apply(raw)
    assert len(post_filter) == 1, "B's filter must keep only one detection per class"

    ground_truth = [truth(box=(0, 0, 3, 1)), truth(box=(10, 0, 13, 1))]
    evaluator = ApplicationPipelineEvaluator()
    raw_result = evaluator.evaluate(
        [image("img", [as_evaluator_input(item) for item in raw], ground_truth)]
    )
    post_result = evaluator.evaluate(
        [image("img", [as_evaluator_input(item) for item in post_filter], ground_truth)]
    )

    assert raw_result["overall"]["tp"] == 2
    assert raw_result["overall"]["recall"] == 1.0
    assert post_result["overall"]["tp"] == 1
    assert post_result["overall"]["fn"] == 1
    assert post_result["overall"]["recall"] == 0.5
    # Raw counts are only reachable with pre-filter detections, which are not saved.
    assert raw_result["overall"] != post_result["overall"]


def test_config_requires_explicit_operating_point(tmp_path):
    path = write_config(tmp_path)
    config = ApplicationPipelineConfig.load(path)
    assert config.confidence_threshold == 0.2
    assert config.nms_iou_threshold == 0.35
    assert config.class_names == tuple(CLASS_NAMES)
    assert len(config.config_sha256) == 64
    assert config.validation_settings == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "1.0", "class_names": CLASS_NAMES},
        {
            "schema_version": "2.0",
            "class_names": CLASS_NAMES,
            "application_operating_point": {
                "confidence_threshold": 0.2,
                "nms_iou_threshold": 0.35,
            },
        },
        {
            "schema_version": "1.0",
            "class_names": CLASS_NAMES[:2],
            "application_operating_point": {
                "confidence_threshold": 0.2,
                "nms_iou_threshold": 0.35,
            },
        },
        {
            "schema_version": "1.0",
            "class_names": CLASS_NAMES,
            "application_operating_point": {"confidence_threshold": 0.2},
        },
        {
            "schema_version": "1.0",
            "class_names": CLASS_NAMES,
            "application_operating_point": {
                "confidence_threshold": 1.2,
                "nms_iou_threshold": 0.35,
            },
        },
        {
            "schema_version": "1.0",
            "class_names": CLASS_NAMES,
            "application_operating_point": {
                "confidence_threshold": 0.2,
                "nms_iou_threshold": 0.35,
            },
            "extra": 1,
        },
    ],
)
def test_invalid_configs_are_rejected(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        ApplicationPipelineConfig.load(path)


def test_report_records_operating_point_and_keeps_map_separate(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path))
    report = evaluate_application_pipeline_predictions(
        config, [image(detections=[detection()], ground_truth=[truth()])]
    )
    operating_point = report["application_pipeline_prf"]["application_operating_point"]
    for key in APPLICATION_SETTING_KEYS:
        assert operating_point[key] == getattr(config, key)
    assert operating_point["config_sha256"] == config.config_sha256
    assert "not re-applied" in operating_point["note"]
    assert report["backend_validation_map"]["state"] == "not_run"

    map_result = {"protocol_version": "backend-map-wrapper-v1", "metrics": {}}
    separate = evaluate_application_pipeline_predictions(
        config,
        [image(detections=[detection()], ground_truth=[truth()])],
        map_result=map_result,
    )
    assert separate["backend_validation_map"] == map_result
    assert "validation_settings" not in separate["application_pipeline_prf"]


def test_confidence_at_the_recorded_threshold_is_accepted(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path, confidence=0.2))
    report = evaluate_application_pipeline_predictions(
        config,
        [image(detections=[detection(confidence=0.2)], ground_truth=[truth()])],
    )
    assert report["application_pipeline_prf"]["overall"]["tp"] == 1
    boundary = report["application_pipeline_prf"]["confidence_boundary"]
    assert boundary["applied"] is True
    assert boundary["min_confidence"] == 0.2
    assert "exactly the recorded threshold" in boundary["note"]


def test_confidence_below_the_recorded_threshold_is_rejected(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path, confidence=0.2))
    with pytest.raises(ValueError, match="below the recorded operating point"):
        evaluate_application_pipeline_predictions(
            config,
            [image(detections=[detection(confidence=0.1999)], ground_truth=[truth()])],
        )


def test_saved_run_path_never_loads_a_model(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path))
    run = make_runner_run(tmp_path)
    ground_truth = write_ground_truth(tmp_path)
    report = evaluate_saved_pipeline_run(config, run, ground_truth)
    prf = report["application_pipeline_prf"]
    assert prf["image_count"] == 2
    assert prf["overall"]["tp"] == 2
    assert report["inputs"]["split"] == "val"
    assert report["inputs"]["run_id"] == "synthetic-run-1"
    assert report["inputs"]["input_scope_source"] == "run_metadata.image_detection_evidence"
    assert len(report["inputs"]["ground_truth_sha256"]) == 64
    assert report["input_scope"]["label"] == INPUT_SCOPE_LABEL
    assert report["binding"]["run_id"] == "synthetic-run-1"
    assert report["binding"]["checkpoint_sha256"] == CHECKPOINT_SHA
    assert "not tamper proofing" in report["binding"]["tamper_note"]


def test_saved_run_operating_point_must_match_config(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path, confidence=0.2))
    run = make_runner_run(tmp_path, confidence=0.5)
    with pytest.raises(ValueError, match="confidence_threshold"):
        evaluate_saved_pipeline_run(config, run, write_ground_truth(tmp_path))


def test_saved_run_split_release_and_inventory_must_match(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path))
    run = make_runner_run(tmp_path)
    with pytest.raises(ValueError, match="split"):
        evaluate_saved_pipeline_run(config, run, write_ground_truth(tmp_path, split="test"))
    with pytest.raises(ValueError, match="dataset release"):
        evaluate_saved_pipeline_run(config, run, write_ground_truth(tmp_path, release="other"))
    missing = write_ground_truth(
        tmp_path, images=[{"image_id": "p1__before", "ground_truth": [truth()]}]
    )
    with pytest.raises(ValueError, match="inventories must match"):
        evaluate_saved_pipeline_run(config, run, missing)
    extra = write_ground_truth(
        tmp_path,
        images=[
            {"image_id": "p1__before", "ground_truth": [truth()]},
            {"image_id": "p1__after", "ground_truth": [truth()]},
            {"image_id": "p2__before", "ground_truth": [truth()]},
        ],
    )
    with pytest.raises(ValueError, match="inventories must match"):
        evaluate_saved_pipeline_run(config, run, extra)


def test_saved_run_rejects_missing_side_and_unknown_pair(tmp_path):
    run = make_runner_run(tmp_path, detections={"p1": {"before": [detection()]}})
    with pytest.raises(ValueError, match="exactly before and after"):
        load_runner_pipeline_detections(run)

    run = make_runner_run(tmp_path, detections={"p2": {"before": [], "after": []}})
    with pytest.raises(ValueError, match="pair inventory"):
        load_runner_pipeline_detections(run)


def test_saved_run_rejects_duplicate_json_keys(tmp_path):
    run = make_runner_run(tmp_path)
    (run / "image_detections.json").write_text(
        '{"p1": {"before": [], "after": []}, "p1": {"before": [], "after": []}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        load_runner_pipeline_detections(run)


def test_saved_run_rejects_nonfinite_saved_box(tmp_path):
    run = make_runner_run(tmp_path)
    (run / "image_detections.json").write_text(
        '{"p1": {"before": [{"class_id": 0, "confidence": 0.9, '
        '"bbox_xyxy": [0, 0, NaN, 1]}], "after": []}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nonfinite JSON value"):
        load_runner_pipeline_detections(run)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: m.pop("run_id"), "run_id"),
        (lambda m: m.update({"status": "failed"}), "completed"),
        (lambda m: m.update({"runner_protocol": "other"}), "runner protocol"),
        (lambda m: m.pop("image_detection_evidence"), "does not record"),
        (lambda m: m.update({"image_detection_evidence": "something else"}), "Unknown"),
        (
            lambda m: m.update({"image_detection_evidence": "RAW detector output"}),
            "raw/pre-filter",
        ),
        (
            lambda m: m.update({"image_detection_evidence": "not provided"}),
            "Contradictory",
        ),
        (
            lambda m: m["provenance"].pop("checkpoint_sha256"),
            "checkpoint identity",
        ),
    ],
)
def test_saved_run_scope_and_identity_are_enforced(tmp_path, mutate, message):
    run = make_runner_run(tmp_path)
    path = run / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mutate(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=message):
        load_runner_pipeline_detections(run)


def test_scope_claiming_saved_detections_without_file_is_rejected(tmp_path):
    run = make_runner_run(tmp_path)
    (run / "image_detections.json").unlink()
    with pytest.raises(ValueError, match="Contradictory"):
        load_runner_pipeline_detections(run)


def test_run_without_recorded_detections_is_rejected(tmp_path):
    run = make_runner_run(tmp_path, evidence="not provided")
    (run / "image_detections.json").unlink()
    with pytest.raises(ValueError, match="nothing to score"):
        load_runner_pipeline_detections(run)


def test_ground_truth_file_rejects_duplicate_images(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "split": "val",
                "dataset_release": "synthetic-release",
                "images": [
                    {"image_id": "x", "ground_truth": []},
                    {"image_id": "x", "ground_truth": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate image_id"):
        load_image_ground_truth(path)


def load_cli():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_application_pipeline.py"
    spec = importlib.util.spec_from_file_location("evaluate_application_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_writes_pipeline_report_without_loading_a_model(tmp_path):
    cli = load_cli()
    config = write_config(tmp_path)
    run = make_runner_run(tmp_path)
    ground_truth = write_ground_truth(tmp_path)
    output = tmp_path / "pipeline-report"
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "--run",
                str(run),
                "--ground-truth",
                str(ground_truth),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "application_pipeline_metrics.json").read_text())
    assert report["report_type"] == "application_pipeline_evaluation"
    assert report["application_pipeline_prf"]["overall"]["tp"] == 2
    assert report["backend_validation_map"]["state"] == "not_run"
    assert report["rerun_of"] == []


def test_cli_records_failure_and_never_publishes_results(tmp_path):
    cli = load_cli()
    config = write_config(tmp_path)
    run = make_runner_run(tmp_path)
    ground_truth = write_ground_truth(tmp_path, split="test")
    output = tmp_path / "pipeline-report"
    assert (
        cli.main(
            [
                "--config",
                str(config),
                "--run",
                str(run),
                "--ground-truth",
                str(ground_truth),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    failures = list(tmp_path.glob("pipeline-report.failure-*"))
    assert len(failures) == 1
    assert sorted(p.name for p in failures[0].iterdir()) == ["failure_record.json"]
    record = json.loads((failures[0] / "failure_record.json").read_text())
    assert record["record_type"] == "evaluation_failure"
    assert "split" in record["reason"]
    assert (
        record["input_identities"]["config_sha256"]
        == ApplicationPipelineConfig.load(config).config_sha256
    )


def test_direct_evaluator_use_records_confidence_validation_not_applied():
    result = score([detection(confidence=0.05)], [truth()])
    boundary = result["confidence_boundary"]
    assert boundary["applied"] is False
    assert boundary["min_confidence"] is None
    assert "no confidence validation is applied" in boundary["note"]
    # Documented behaviour of direct use: the low-confidence box is still scored.
    assert result["overall"]["tp"] == 1


def test_operating_point_comparison_is_exact_and_not_coerced(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path, confidence=0.2))
    ground_truth = write_ground_truth(tmp_path)

    exact = make_runner_run(tmp_path, confidence=0.2)
    assert evaluate_saved_pipeline_run(config, exact, ground_truth)["inputs"]["split"] == "val"

    as_string = make_runner_run(tmp_path, confidence="0.2")
    with pytest.raises(ValueError, match="differs from the evaluation config"):
        evaluate_saved_pipeline_run(config, as_string, ground_truth)

    one_ulp = make_runner_run(tmp_path, confidence=math.nextafter(0.2, 1.0))
    with pytest.raises(ValueError, match="differs from the evaluation config"):
        evaluate_saved_pipeline_run(config, one_ulp, ground_truth)


def cli_args(tmp_path, *, output=None, rerun_of=None, rerun_reason=None, config=None, run=None):
    args = [
        "--config",
        str(config or write_config(tmp_path)),
        "--run",
        str(run or make_runner_run(tmp_path)),
        "--ground-truth",
        str(write_ground_truth(tmp_path)),
        "--output",
        str(output or (tmp_path / "pipeline-report")),
    ]
    for path in rerun_of or []:
        args.extend(["--rerun-of", str(path)])
    if rerun_reason is not None:
        args.extend(["--rerun-reason", rerun_reason])
    return args


def test_cli_refuses_existing_output_target(tmp_path):
    cli = load_cli()
    output = tmp_path / "pipeline-report"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    assert cli.main(cli_args(tmp_path, output=output)) == 2
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in output.iterdir()) == ["keep.txt"]


def test_cli_records_valid_rerun_link_and_reason(tmp_path):
    cli = load_cli()
    failure = build_failure_record(**record_kwargs())
    failure_path = write_failure_record(failure, tmp_path / "first-failure")
    output = tmp_path / "pipeline-report"

    assert (
        cli.main(
            cli_args(
                tmp_path,
                output=output,
                rerun_of=[failure_path],
                rerun_reason="fixed synthetic failure",
            )
        )
        == 0
    )
    report = json.loads((output / "application_pipeline_metrics.json").read_text())
    assert [link["failure_id"] for link in report["rerun_of"]] == ["run-synthetic-1"]
    assert report["rerun_of"][0]["reason"] == "fixed synthetic failure"
    assert (
        report["rerun_of"][0]["failure_record_sha256"]
        == build_rerun_link(failure_path, "fixed synthetic failure")["failure_record_sha256"]
    )


def test_cli_rejects_missing_or_malformed_rerun_reference_before_work(tmp_path):
    cli = load_cli()
    output = tmp_path / "pipeline-report"
    missing = tmp_path / "missing-failure.json"

    assert (
        cli.main(
            cli_args(
                tmp_path,
                output=output,
                rerun_of=[missing],
                rerun_reason="retry",
                config=tmp_path / "missing-config.json",
                run=tmp_path / "missing-run",
            )
        )
        == 2
    )
    # Rejected during argument validation: no report, no failure record.
    assert not output.exists()
    assert list(tmp_path.glob("pipeline-report.failure-*")) == []

    malformed = tmp_path / "malformed-failure.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert (
        cli.main(cli_args(tmp_path, output=output, rerun_of=[malformed], rerun_reason="retry")) == 2
    )
    assert not output.exists()
    assert list(tmp_path.glob("pipeline-report.failure-*")) == []


def test_saved_run_detections_are_hashed_in_inputs(tmp_path):
    config = ApplicationPipelineConfig.load(write_config(tmp_path))
    run = make_runner_run(tmp_path)
    ground_truth = write_ground_truth(tmp_path)
    report = evaluate_saved_pipeline_run(config, run, ground_truth)
    expected = hashlib.sha256(ground_truth.read_bytes()).hexdigest()
    assert report["inputs"]["ground_truth_sha256"] == expected
