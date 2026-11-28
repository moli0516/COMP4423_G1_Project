"""Prepare the B runner's validation manifest from A's released annotations and splits."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from evaluate_real_pairs import load_manifest as load_runner_manifest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.data.dataset_catalog import load_catalog
from deskcheck.evaluation.annotation_converter import convert_pair
from deskcheck.evaluation.event_evaluator import EventEvaluator
from deskcheck.evaluation.no_change_evaluator import NoChangeEvaluator


def _path_from_output(image_path: str, output_dir: Path) -> str:
    path = Path(image_path).resolve(strict=True)
    try:
        return Path(os.path.relpath(path, output_dir)).as_posix()
    except ValueError:  # On Windows the output and photo may be on separate drives.
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_validation(annotations: Path, splits: Path, output: Path) -> int:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    catalog, manifest = load_catalog(annotations, splits)
    if manifest.schema_version != "1.0":
        raise ValueError(f"Unsupported split schema: {manifest.schema_version!r}")
    selected = [catalog[pair_id] for pair_id in manifest.splits["val"]]
    truth = {pair.pair_id: convert_pair(pair) for pair in selected}
    eligible = [pair.pair_id for pair in selected if pair.independently_captured_no_change]
    # Use the runner's own event/eligibility contracts before publishing the manifest.
    EventEvaluator().evaluate(truth, truth)
    eligibility = NoChangeEvaluator().evaluate(truth, eligible)
    if eligibility["false_alarm_pair_count"]:
        raise ValueError("Eligible no-change ground truth contains change events")
    if any(not truth[pair_id] for pair_id in eligible):
        raise ValueError("Eligible no-change pairs require annotated unchanged objects")

    payload = {
        "schema_version": "1.0",
        "dataset_release": manifest.dataset_release,
        "split": "val",
        "source_annotations_sha256": _sha256_file(annotations),
        "source_split_manifest_sha256": _sha256_file(splits),
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "before_path": _path_from_output(pair.before_image_path, output.parent),
                "after_path": _path_from_output(pair.after_image_path, output.parent),
            }
            for pair in selected
        ],
        "ground_truth": truth,
        "eligible_no_change_pair_ids": eligible,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate in the destination directory to preserve relative image paths.
    # Linking publishes the complete file atomically and refuses a concurrent writer.
    fd, name = tempfile.mkstemp(dir=output.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        load_runner_manifest(staged)
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="A's pairs.jsonl")
    parser.add_argument("--splits", type=Path, required=True, help="A's splits.json")
    parser.add_argument("--output", type=Path, required=True, help="New validation manifest JSON")
    args = parser.parse_args()
    try:
        count = prepare_validation(args.annotations, args.splits, args.output)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"Cannot prepare validation manifest: {exc}", file=sys.stderr)
        return 1
    print(f"split=val pairs={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
