"""Synthetic failure records only; no evaluation result is implied."""

import json

import pytest

from deskcheck.evaluation.evaluation_runner import EvaluationRunner
from deskcheck.evaluation.failure_record import (
    FAILURE_RECORD_FILENAME,
    build_failure_record,
    build_rerun_link,
    load_failure_record,
    validate_rerun_links,
    write_failure_record,
)


def record_kwargs(**overrides):
    value = {
        "run_id": "run-synthetic-1",
        "reason": "synthetic inference failure",
        "exception_type": "OSError",
        "context": {"stage": "per_pair", "pair_id": "p1", "image_id": None},
        "input_identities": {
            "manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "dataset_release": "synthetic-release",
            "split": "val",
        },
        "source_identity": {
            "source_revision": "synthetic-revision",
            "source_dirty": True,
            "source_dirty_paths": ["src/synthetic.py"],
        },
        "evaluation_config": {"confidence_threshold": 0.2, "nms_iou_threshold": 0.35},
        "command": ["python", "synthetic"],
        "occurred_at_utc": "2026-11-26T12:00:00+00:00",
    }
    value.update(overrides)
    return value


def test_failure_record_contains_required_evidence(tmp_path):
    record = build_failure_record(**record_kwargs())
    path = write_failure_record(record, tmp_path / "failure")
    assert path.name == FAILURE_RECORD_FILENAME
    assert sorted(p.name for p in path.parent.iterdir()) == [FAILURE_RECORD_FILENAME]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["record_type"] == "evaluation_failure"
    assert saved["reason"] == "synthetic inference failure"
    assert saved["exception_type"] == "OSError"
    assert saved["context"]["pair_id"] == "p1"
    assert saved["input_identities"]["manifest_sha256"] == "a" * 64
    assert saved["source_identity"]["source_dirty_paths"] == ["src/synthetic.py"]
    assert saved["evaluation_config"]["confidence_threshold"] == 0.2
    assert saved["command"] == ["python", "synthetic"]
    assert saved["rerun_of"] == []
    assert "status" not in saved
    assert load_failure_record(path) == saved


def test_failure_record_requires_free_text_reason_and_timestamp():
    with pytest.raises(ValueError):
        build_failure_record(**record_kwargs(reason="  "))
    with pytest.raises(ValueError):
        build_failure_record(**record_kwargs(occurred_at_utc="2026-11-26T12:00:00"))


def test_failure_record_rejects_unknown_hash_and_split():
    bad = record_kwargs()
    bad["input_identities"]["manifest_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        build_failure_record(**bad)
    bad = record_kwargs()
    bad["input_identities"]["split"] = "train"
    with pytest.raises(ValueError):
        build_failure_record(**bad)


def test_failure_record_never_overwrites(tmp_path):
    record = build_failure_record(**record_kwargs())
    target = tmp_path / "failure"
    write_failure_record(record, target)
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_failure_record(record, target)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in target.iterdir()) == [FAILURE_RECORD_FILENAME, "keep.txt"]


def test_malformed_record_is_rejected_on_load(tmp_path):
    record = build_failure_record(**record_kwargs())
    path = write_failure_record(record, tmp_path / "failure")
    broken = dict(record)
    broken["unexpected"] = True
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError):
        load_failure_record(path)


def test_rerun_link_binds_id_hash_and_reason(tmp_path):
    record = build_failure_record(**record_kwargs())
    path = write_failure_record(record, tmp_path / "failure")
    link = build_rerun_link(path, "fixed synthetic timeout")
    assert link["failure_id"] == "run-synthetic-1"
    assert len(link["failure_record_sha256"]) == 64
    assert link["reason"] == "fixed synthetic timeout"
    assert validate_rerun_links([link]) == [link]


def test_rerun_link_requires_reason(tmp_path):
    record = build_failure_record(**record_kwargs())
    path = write_failure_record(record, tmp_path / "failure")
    with pytest.raises(ValueError):
        build_rerun_link(path, "")
    with pytest.raises(FileNotFoundError):
        build_rerun_link(tmp_path / "missing" / FAILURE_RECORD_FILENAME, "retry")


def test_malformed_or_stale_rerun_references_are_rejected(tmp_path):
    record = build_failure_record(**record_kwargs())
    path = write_failure_record(record, tmp_path / "failure")
    link = build_rerun_link(path, "retry")

    with pytest.raises(ValueError):
        validate_rerun_links([{**link, "extra": 1}])
    with pytest.raises(FileNotFoundError):
        validate_rerun_links([{**link, "failure_record_path": str(tmp_path / "gone.json")}])
    with pytest.raises(ValueError, match="bytes changed"):
        validate_rerun_links([{**link, "failure_record_sha256": "0" * 64}])
    with pytest.raises(ValueError, match="failure_id"):
        validate_rerun_links([{**link, "failure_id": "other-run"}])
    with pytest.raises(ValueError, match="same failure twice"):
        validate_rerun_links([link, link])
    with pytest.raises(TypeError):
        validate_rerun_links("not-a-list")

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"):
        validate_rerun_links([link])


class FailingService:
    def compare(self, before_path, after_path, pair_id):
        raise OSError("synthetic backend failure")


class PassingService:
    def compare(self, before_path, after_path, pair_id):
        return {"pair_id": pair_id, "records": [EVENT]}


EVENT = {
    "class_id": 0,
    "status": "unchanged",
    "before_box": [0, 0, 2, 2],
    "after_box": [0, 0, 2, 2],
}


def pair(pair_id="p1"):
    return {
        "pair_id": pair_id,
        "before_path": f"{pair_id}-before.png",
        "after_path": f"{pair_id}-after.png",
    }


def metadata(run_id="run-synthetic-2"):
    return {
        "run_id": run_id,
        "split": "val",
        "dataset_release": "synthetic-release",
        "manifest_sha256": "d" * 64,
        "checkpoint_sha256": "e" * 64,
        "config_sha256": "f" * 64,
        "confidence_threshold": 0.2,
        "nms_iou_threshold": 0.35,
        "movement_threshold": 0.02,
        "imgsz": 640,
        "rules_version": "synthetic-rules",
        "source_revision": "synthetic-revision",
        "source_dirty": False,
        "source_dirty_paths": [],
        "command": ["python", "synthetic"],
    }


def test_runner_writes_failure_record_and_no_success_output(tmp_path):
    runner = EvaluationRunner(FailingService())
    failure_dir = tmp_path / "failure"
    with pytest.raises(RuntimeError):
        runner.run(
            [pair()],
            {"p1": [EVENT]},
            ["p1"],
            tmp_path / "result",
            metadata=metadata(),
            failure_output_dir=failure_dir,
        )
    assert not (tmp_path / "result").exists()
    record = load_failure_record(failure_dir)
    assert record["run_id"] == "run-synthetic-2"
    assert record["exception_type"] == "RuntimeError"
    assert "p1" in record["reason"]
    assert record["context"] == {"stage": "per_pair", "pair_id": "p1", "image_id": None}
    assert record["input_identities"]["manifest_sha256"] == "d" * 64
    assert record["source_identity"]["source_revision"] == "synthetic-revision"
    assert record["evaluation_config"]["event_iou_threshold"] == 0.5
    assert sorted(p.name for p in failure_dir.iterdir()) == [FAILURE_RECORD_FILENAME]


def test_runner_without_failure_directory_only_raises(tmp_path):
    with pytest.raises(RuntimeError):
        EvaluationRunner(FailingService()).run(
            [pair()], {"p1": [EVENT]}, ["p1"], tmp_path / "result"
        )
    assert not (tmp_path / "result").exists()
    assert list(tmp_path.iterdir()) == []


def test_runner_refuses_existing_output_and_records_failure(tmp_path):
    existing = tmp_path / "result"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        EvaluationRunner(PassingService()).run(
            [pair()],
            {"p1": [EVENT]},
            ["p1"],
            existing,
            metadata=metadata(),
            failure_output_dir=tmp_path / "failure",
        )
    assert marker.read_text(encoding="utf-8") == "keep"
    record = load_failure_record(tmp_path / "failure")
    assert record["context"]["stage"] == "preflight"


def test_runner_records_and_exposes_rerun_links(tmp_path):
    failure_record = build_failure_record(**record_kwargs())
    failure_path = write_failure_record(failure_record, tmp_path / "first-failure")
    link = build_rerun_link(failure_path, "fixed synthetic backend failure")

    result = EvaluationRunner(PassingService()).run(
        [pair()],
        {"p1": [EVENT]},
        ["p1"],
        tmp_path / "result",
        metadata=metadata(run_id="run-synthetic-3"),
        rerun_of=[link],
    )
    assert result["run_metadata"]["rerun_of"] == [link]
    assert result["run_metadata"]["run_id"] == "run-synthetic-3"


def test_runner_rejects_malformed_rerun_link_before_work(tmp_path):
    class CountingService:
        calls = 0

        def compare(self, before_path, after_path, pair_id):
            CountingService.calls += 1
            return {"pair_id": pair_id, "records": [EVENT]}

    with pytest.raises(ValueError):
        EvaluationRunner(CountingService()).run(
            [pair()],
            {"p1": [EVENT]},
            ["p1"],
            tmp_path / "result",
            metadata=metadata(),
            rerun_of=[{"failure_id": "x"}],
        )
    assert CountingService.calls == 0
    assert not (tmp_path / "result").exists()


def test_runner_records_run_id_and_primary_metric_in_success_output(tmp_path):
    result = EvaluationRunner(PassingService()).run(
        [pair()],
        {"p1": [EVENT]},
        ["p1"],
        tmp_path / "result",
        metadata=metadata(run_id="run-synthetic-4"),
    )
    metadata_doc = json.loads((tmp_path / "result" / "run_metadata.json").read_text())
    assert metadata_doc["run_id"] == "run-synthetic-4"
    assert metadata_doc["primary_downstream_metric"]["name"] == "change_only_macro_f1"
    assert metadata_doc["primary_downstream_metric"]["role"] == "primary"
    assert metadata_doc["primary_downstream_metric"]["value"] == pytest.approx(0.0)
    assert result["run_metadata"]["rerun_of"] == []
