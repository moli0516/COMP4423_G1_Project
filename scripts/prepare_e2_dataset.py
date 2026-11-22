"""Stage the released formal v1 training and validation splits for A's trainer."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

FORMAL_V1_SHA256 = "0a49f3bc1308f1d5ac292cfb6274c17ef89abcc616dc21c1dd1eea0eeafa63f9"
EXPORT = "formal_photos/yolo_export/"


def prepare_dataset(archive: Path, output: Path) -> None:
    """Verify the original ZIP and copy only the required YOLO images and labels."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    root = output.resolve().as_posix()
    # A's trainer reads the root as a plain line and Ultralytics parses it as YAML.
    if any(mark in root for mark in (" #", ": ", "\r", "\n", "\t")) or root.endswith((" ", ":")):
        raise ValueError("Choose a directory representable as a plain YAML path for A's trainer")
    with archive.open("rb") as archive_file:
        archive_sha = hashlib.file_digest(archive_file, "sha256").hexdigest()
        if archive_sha != FORMAL_V1_SHA256:
            raise ValueError(
                f"Formal v1 ZIP SHA-256 differs from the verified release: {archive_sha}"
            )
        archive_file.seek(0)
        with zipfile.ZipFile(archive_file) as z:
            splits_bytes = z.read("formal_photos/splits.json")
            splits = json.loads(splits_bytes)["splits"]
            original_yaml = z.read(EXPORT + "dataset.yaml")
            lines = original_yaml.decode("utf-8").splitlines()
            local_yaml = (
                "\n".join(
                    [f"path: {root}"]
                    + [line for line in lines if not line.startswith(("path:", "test:"))]
                )
                + "\n"
            ).encode("utf-8")
            # mkdir reserves this output for this invocation, including concurrent calls.
            output.mkdir(parents=True, exist_ok=False)
            try:
                hashes = {}
                counts = {}
                for split in ("train", "val"):
                    for folder in ("images", "labels"):
                        (output / folder / split).mkdir(parents=True)
                    counts[split] = 0
                    for pair_id in splits[split]:
                        for side in ("before", "after"):
                            stem = f"{pair_id}__{side}"
                            for folder, ext in (("images", "jpg"), ("labels", "txt")):
                                name = f"{folder}/{split}/{stem}.{ext}"
                                data = z.read(EXPORT + name)
                                (output / name).write_bytes(data)
                                hashes[name] = hashlib.sha256(data).hexdigest()
                            counts[split] += 1
                (output / "dataset.yaml").write_bytes(local_yaml)
                receipt = {
                    "archive_sha256": archive_sha,
                    "source_yaml_sha256": hashlib.sha256(original_yaml).hexdigest(),
                    "split_manifest_sha256": hashlib.sha256(splits_bytes).hexdigest(),
                    "local_yaml_sha256": hashlib.sha256(local_yaml).hexdigest(),
                    "images_per_split": counts,
                    "member_sha256": hashes,
                }
                (output / "source_receipt.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            except BaseException:
                shutil.rmtree(output)
                raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="A's formal_dataset_v1.zip")
    parser.add_argument("--output", type=Path, required=True, help="New local dataset directory")
    args = parser.parse_args()
    try:
        prepare_dataset(args.archive, args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Cannot prepare E2 dataset: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared formal v1 train/val at {(args.output / 'dataset.yaml').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
