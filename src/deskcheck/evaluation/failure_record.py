"""Persistent failure records and validated rerun links for evaluation runs.

A failed evaluation writes ``failure_record.json`` into a new directory; it is
never written into a completed run directory and never uses the success file
names. A later run may reference earlier failure records by identifier, and the
reference is re-validated (file present, bytes unchanged, identifier matching)
before it is recorded in the new run.

These records are local evidence only: they are not receipts, signatures or
approvals, and whether a rerun replaces a failed run is an owner decision.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FAILURE_RECORD_FILENAME = "failure_record.json"
FAILURE_RECORD_TYPE = "evaluation_failure"
SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "occurred_at_utc",
    "reason",
    "exception_type",
    "context",
    "input_identities",
    "source_identity",
    "evaluation_config",
    "command",
    "rerun_of",
}
INPUT_IDENTITY_FIELDS = (
    "manifest_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "dataset_release",
    "split",
)
SOURCE_IDENTITY_FIELDS = ("source_revision", "source_dirty", "source_dirty_paths")
RERUN_LINK_FIELDS = {
    "failure_id",
    "failure_record_path",
    "failure_record_sha256",
    "reason",
}
_HASH = re.compile(r"[0-9a-f]{64}")


def new_run_id() -> str:
    """Fresh local identifier for a run or failure record."""
    return uuid.uuid4().hex


def utc_now() -> str:
    """Current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"Nonfinite JSON value is invalid: {token}")


def _snapshot(value: Any, what: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be JSON-serialisable without nonfinite values") from exc


def _optional_hash(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{what} must be a lowercase SHA-256 digest or null")
    return value


def _optional_text(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a nonempty string or null")
    return value


def _validate_context(context: Any) -> dict[str, Any]:
    if not isinstance(context, dict) or set(context) != {"stage", "pair_id", "image_id"}:
        raise ValueError("context must contain exactly stage, pair_id and image_id")
    if not isinstance(context["stage"], str) or not context["stage"].strip():
        raise ValueError("context.stage must be a nonempty string")
    _optional_text(context["pair_id"], "context.pair_id")
    _optional_text(context["image_id"], "context.image_id")
    return dict(context)


def _validate_input_identities(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(INPUT_IDENTITY_FIELDS):
        raise ValueError(f"input_identities must contain exactly {list(INPUT_IDENTITY_FIELDS)}")
    for field in ("manifest_sha256", "checkpoint_sha256", "config_sha256"):
        _optional_hash(value[field], f"input_identities.{field}")
    _optional_text(value["dataset_release"], "input_identities.dataset_release")
    split = value["split"]
    if split is not None and split not in ("val", "test"):
        raise ValueError("input_identities.split must be 'val', 'test' or null")
    return dict(value)


def _validate_source_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(SOURCE_IDENTITY_FIELDS):
        raise ValueError(f"source_identity must contain exactly {list(SOURCE_IDENTITY_FIELDS)}")
    _optional_text(value["source_revision"], "source_identity.source_revision")
    if value["source_dirty"] is not None and type(value["source_dirty"]) is not bool:
        raise ValueError("source_identity.source_dirty must be a boolean or null")
    paths = value["source_dirty_paths"]
    if paths is not None and (
        not isinstance(paths, list) or any(not isinstance(item, str) for item in paths)
    ):
        raise ValueError("source_identity.source_dirty_paths must be a list of strings or null")
    return dict(value)


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("occurred_at_utc must be a nonempty ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("occurred_at_utc must be a valid ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("occurred_at_utc must include a timezone offset")
    return value


def build_failure_record(
    *,
    run_id: str,
    reason: str,
    exception_type: str,
    context: dict[str, Any],
    input_identities: dict[str, Any],
    source_identity: dict[str, Any],
    evaluation_config: dict[str, Any],
    command: list[str] | None,
    occurred_at_utc: str | None = None,
    rerun_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build (and fully validate) one failure record without writing it."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be nonempty free text")
    if not isinstance(exception_type, str) or not exception_type.strip():
        raise ValueError("exception_type must be a nonempty string")
    if command is not None and (
        not isinstance(command, list) or any(not isinstance(item, str) for item in command)
    ):
        raise ValueError("command must be a list of strings or null")
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": FAILURE_RECORD_TYPE,
        "run_id": run_id,
        "occurred_at_utc": _validate_timestamp(occurred_at_utc or utc_now()),
        "reason": reason,
        "exception_type": exception_type,
        "context": _validate_context(context),
        "input_identities": _validate_input_identities(input_identities),
        "source_identity": _validate_source_identity(source_identity),
        "evaluation_config": _snapshot(evaluation_config, "evaluation_config"),
        "command": None if command is None else list(command),
        "rerun_of": validate_rerun_links(list(rerun_of or [])),
    }
    _validate_record(record)
    return record


def _validate_record(record: Any) -> None:
    if not isinstance(record, dict):
        raise TypeError("Failure record must be a JSON object")
    if set(record) != REQUIRED_FIELDS:
        raise ValueError(f"Failure record must contain exactly {sorted(REQUIRED_FIELDS)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported failure record schema_version")
    if record["record_type"] != FAILURE_RECORD_TYPE:
        raise ValueError("Failure record has an unexpected record_type")
    if not isinstance(record["run_id"], str) or not record["run_id"].strip():
        raise ValueError("Failure record run_id must be a nonempty string")
    _validate_timestamp(record["occurred_at_utc"])
    for field in ("reason", "exception_type"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Failure record {field} must be nonempty")
    _validate_context(record["context"])
    _validate_input_identities(record["input_identities"])
    _validate_source_identity(record["source_identity"])
    if not isinstance(record["evaluation_config"], dict):
        raise TypeError("Failure record evaluation_config must be an object")
    _snapshot(record["evaluation_config"], "evaluation_config")
    if record["command"] is not None and (
        not isinstance(record["command"], list)
        or any(not isinstance(item, str) for item in record["command"])
    ):
        raise ValueError("Failure record command must be a list of strings or null")
    validate_rerun_links(record["rerun_of"])


def write_failure_record(record: dict[str, Any], directory: str | Path) -> Path:
    """Write a failure record into a new directory; never overwrite anything."""
    _validate_record(record)
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite failure record directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / FAILURE_RECORD_FILENAME).write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite failure record directory: {target}")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / FAILURE_RECORD_FILENAME


def load_failure_record(path: str | Path) -> dict[str, Any]:
    """Load and validate one failure record file."""
    source = Path(path)
    if source.is_dir():
        source = source / FAILURE_RECORD_FILENAME
    try:
        record = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failure record is unreadable or malformed: {exc}") from exc
    _validate_record(record)
    return record


def build_rerun_link(failure_record_path: str | Path, reason: str) -> dict[str, Any]:
    """Bind a new run to one earlier failure record, with a free-text reason."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Rerun reason must be nonempty free text")
    path = Path(failure_record_path)
    if not path.is_file():
        raise FileNotFoundError(f"Rerun link references a missing failure record: {path}")
    record = load_failure_record(path)
    return {
        "failure_id": record["run_id"],
        "failure_record_path": str(path.resolve()),
        "failure_record_sha256": _sha256_file(path),
        "reason": reason,
    }


def validate_rerun_links(links: Any) -> list[dict[str, Any]]:
    """Re-validate rerun links: present files, unchanged bytes, matching IDs."""
    if not isinstance(links, list):
        raise TypeError("rerun_of must be a list of rerun links")
    validated = []
    seen: set[str] = set()
    for link in links:
        if not isinstance(link, dict) or set(link) != RERUN_LINK_FIELDS:
            raise ValueError(f"Rerun link must contain exactly {sorted(RERUN_LINK_FIELDS)}")
        failure_id = link["failure_id"]
        if not isinstance(failure_id, str) or not failure_id.strip():
            raise ValueError("Rerun link failure_id must be a nonempty string")
        if not isinstance(link["failure_record_path"], str) or not link["failure_record_path"]:
            raise ValueError("Rerun link failure_record_path must be a nonempty string")
        if not isinstance(link["failure_record_sha256"], str) or not _HASH.fullmatch(
            link["failure_record_sha256"]
        ):
            raise ValueError("Rerun link failure_record_sha256 must be a lowercase SHA-256")
        if not isinstance(link["reason"], str) or not link["reason"].strip():
            raise ValueError("Rerun link reason must be nonempty free text")
        path = Path(link["failure_record_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Rerun link references a missing failure record: {path}")
        if _sha256_file(path) != link["failure_record_sha256"]:
            raise ValueError("Rerun link failure record bytes changed after the link was made")
        record = load_failure_record(path)
        if record["run_id"] != failure_id:
            raise ValueError("Rerun link failure_id does not match the referenced record")
        if failure_id in seen:
            raise ValueError("Rerun links must not reference the same failure twice")
        seen.add(failure_id)
        validated.append(dict(link))
    return validated
