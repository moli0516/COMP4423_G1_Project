"""Synthetic validator stubs only: no Ultralytics, torch or checkpoint access."""

import pytest

from deskcheck.evaluation.map_evaluator import evaluate_backend_map

CLASS_NAMES = ["bottle", "cup", "keyboard", "mouse", "book", "scissors"]


class StubBox:
    def __init__(self, map50_95=0.5, map50=0.7):
        self.map = map50_95
        self.map50 = map50


class StubResults:
    """Mimics the attributes the E2 script already reads from a validator result."""

    def __init__(self, *, box=None, support=None, class_rows=None):
        self.box = StubBox() if box is None else box
        self.nt_per_class = support
        self._class_rows = class_rows

    def class_result(self, index):
        if self._class_rows is None:
            raise IndexError(index)
        return self._class_rows[index]


def stub_validator(record, results):
    def run_validation(**kwargs):
        record.append(kwargs)
        return results

    return run_validation


def test_map_metrics_come_from_the_injected_validator():
    calls = []
    results = StubResults(
        box=StubBox(map50_95=0.42, map50=0.66),
        support=[1] * 6,
        class_rows=[(0.5, 0.4, 0.71, 0.43)] * 6,
    )
    report = evaluate_backend_map(
        stub_validator(calls, results),
        validation_settings={"imgsz": 640, "batch": 8},
        class_names=CLASS_NAMES,
        backend_version="8.4.215-synthetic",
    )
    assert report["metrics"] == {"mAP50_95": 0.42, "mAP50": 0.66}
    assert calls == [{"imgsz": 640, "batch": 8}]
    assert report["validation_settings"]["observed"] == {"imgsz": 640, "batch": 8}
    assert report["backend"]["version"]["state"] == "observed"
    assert report["backend"]["version"]["value"] == "8.4.215-synthetic"
    assert report["per_class_ap"]["bottle"] == {"support": 1, "AP50": 0.71, "AP50_95": 0.43}


def test_unobserved_settings_are_not_assumed():
    calls = []
    report = evaluate_backend_map(
        stub_validator(calls, StubResults()),
        validation_settings={},
        class_names=CLASS_NAMES,
    )
    assert calls == [{}]
    unobserved = report["validation_settings"]["unobserved"]
    assert "conf" in unobserved and "iou" in unobserved and "imgsz" in unobserved
    assert report["validation_settings"]["observed"] == {}
    assert report["backend"]["version"] == {"state": "unobserved"}
    assert "not assumed" in report["validation_settings"]["note"]


def test_application_operating_point_is_not_forced_into_validation_settings():
    calls = []
    evaluate_backend_map(
        stub_validator(calls, StubResults()),
        validation_settings={"imgsz": 640},
        class_names=CLASS_NAMES,
    )
    assert "conf" not in calls[0] and "iou" not in calls[0]


def test_per_class_ap_is_reported_unavailable_without_support():
    report = evaluate_backend_map(
        stub_validator([], StubResults()),
        validation_settings={},
        class_names=CLASS_NAMES,
    )
    assert report["per_class_ap"]["state"] == "unavailable"


def test_per_class_ap_is_reported_unavailable_when_rows_are_missing():
    report = evaluate_backend_map(
        stub_validator([], StubResults(support=[1] * 6)),
        validation_settings={},
        class_names=CLASS_NAMES,
    )
    assert report["per_class_ap"]["state"] == "unavailable"


@pytest.mark.parametrize(
    "box",
    [StubBox(map50_95=None), StubBox(map50=float("nan")), StubBox(map50_95=1.5)],
)
def test_invalid_backend_metrics_are_rejected(box):
    with pytest.raises(RuntimeError):
        evaluate_backend_map(
            stub_validator([], StubResults(box=box)),
            validation_settings={},
            class_names=CLASS_NAMES,
        )


def test_missing_box_metrics_are_rejected():
    class NoBox:
        pass

    with pytest.raises(RuntimeError, match="box"):
        evaluate_backend_map(
            stub_validator([], NoBox()),
            validation_settings={},
            class_names=CLASS_NAMES,
        )


def test_validator_failure_is_reported_without_metrics():
    def failing(**kwargs):
        raise OSError("synthetic validator failure")

    with pytest.raises(RuntimeError) as error:
        evaluate_backend_map(failing, validation_settings={}, class_names=CLASS_NAMES)
    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize("settings", [{"confidence_threshold": 0.2}, {"unknown": 1}])
def test_unknown_validation_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        evaluate_backend_map(
            stub_validator([], StubResults()),
            validation_settings=settings,
            class_names=CLASS_NAMES,
        )


def test_none_setting_is_rejected_rather_than_treated_as_unobserved():
    with pytest.raises(ValueError):
        evaluate_backend_map(
            stub_validator([], StubResults()),
            validation_settings={"imgsz": None},
            class_names=CLASS_NAMES,
        )


def test_scope_is_separate_from_operating_point():
    report = evaluate_backend_map(
        stub_validator([], StubResults()),
        validation_settings={},
        class_names=CLASS_NAMES,
    )
    assert "operating point" in report["metric_scope"]
    assert "not interchangeable" in report["matching_note"]
