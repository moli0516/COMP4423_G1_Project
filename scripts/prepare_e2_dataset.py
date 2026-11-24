"""Stage the released formal v1 training and validation splits for A's trainer."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

FORMAL_V1_SHA256 = "0a49f3bc1308f1d5ac292cfb6274c17ef89abcc616dc21c1dd1eea0eeafa63f9"
# Digest of the sorted raw per-member hash map from the pinned v1 archive.
FORMAL_V1_MEMBERS_SHA256 = "300ea1a1a7f3ff1e102454de770fb5bd746addf4511704ce50d2650c54d4673d"
EXPORT = "formal_photos/yolo_export/"


def _check_new_directory(output: Path) -> str:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    root = output.resolve().as_posix()
    # A's trainer reads the root as a plain line and Ultralytics parses it as YAML.
    if any(mark in root for mark in (" #", ": ", "\r", "\n", "\t")) or root.endswith((" ", ":")):
        raise ValueError("Choose a directory representable as a plain YAML path for A's trainer")
    return root


def prepare_dataset(archive: Path, output: Path) -> None:
    """Verify the original ZIP and copy only the required YOLO images and labels."""
    output = output.absolute()
    root = _check_new_directory(output)
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


def _map_sha256(hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _vision_backend():
    from importlib.metadata import version

    from PIL import features

    versions = {
        "ultralytics": version("ultralytics"),
        "pillow": version("pillow"),
        "jpeg_codec": features.version_codec("jpg"),
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
    }
    if versions["ultralytics"] != "8.4.215":
        raise ValueError("Install the project vision extra with Ultralytics 8.4.215")
    from ultralytics.data.utils import check_image

    return check_image, versions


def _canonical_wh(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        if image.getexif().get(274) in (5, 6, 7, 8):
            return image.height, image.width
        return image.size


def _training_yaml(root: str) -> bytes:
    # Build a local-only trainer manifest. A relocated raw stage may retain an old absolute path.
    return (
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: 6\n"
        "names: ['bottle', 'cup', 'keyboard', 'mouse', 'book', 'scissors']\n"
    ).encode()


def _verify_members(folder: Path, hashes: dict[str, str]) -> None:
    # Image-side .npy caches can override the JPEG on later training runs.
    files = {
        p.relative_to(folder).as_posix()
        for kind in ("images", "labels")
        for split in ("train", "val")
        for p in (folder / kind / split).rglob("*")
        if p.is_file()
    }
    if files != hashes.keys():
        raise ValueError(f"Unexpected file inventory in {folder}: {sorted(files ^ hashes.keys())}")
    for name, digest in hashes.items():
        if _file_sha256(folder / name) != digest:
            raise ValueError(f"SHA-256 mismatch in {name}")


def prepare_training_copy(source: Path, output: Path) -> dict:
    """Scan copies of authentic v1 members and bind the exact trainer inputs."""
    output = output.absolute()
    root = _check_new_directory(output)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a work directory outside the raw dataset")
    raw = json.loads((source / "source_receipt.json").read_bytes())
    hashes = raw["member_sha256"]
    if raw["archive_sha256"] != FORMAL_V1_SHA256 or _map_sha256(hashes) != FORMAL_V1_MEMBERS_SHA256:
        raise ValueError("The source inventory differs from the verified formal v1 archive")
    _verify_members(source, hashes)
    check_image, versions = _vision_backend()
    yaml = _training_yaml(root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        effective, dimensions, changed = {}, {}, []
        for name, raw_digest in sorted(hashes.items()):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            if _file_sha256(target) != raw_digest:
                raise ValueError(f"Source changed while copying {name}")
            if target.suffix == ".jpg":
                original_wh = _canonical_wh(target)
                _, hw = check_image(str(target))
                if tuple(reversed(hw)) != original_wh or _canonical_wh(target) != original_wh:
                    raise ValueError(f"Backend changed canonical geometry for {name}")
                dimensions[name] = list(original_wh)
            effective[name] = _file_sha256(target)
            if effective[name] != raw_digest:
                # The trainer will call the same check again. A second scan must be stable.
                check_image(str(target))
                if _file_sha256(target) != effective[name]:
                    raise ValueError(f"Backend repeatedly changed {name}")
                changed.append(name)
        (output / "dataset.yaml").write_bytes(yaml)
        receipt = {
            "archive_sha256": raw["archive_sha256"],
            "source_member_sha256": hashes,
            "source_members_sha256": FORMAL_V1_MEMBERS_SHA256,
            "effective_member_sha256": effective,
            "effective_members_sha256": _map_sha256(effective),
            "local_yaml_sha256": hashlib.sha256(yaml).hexdigest(),
            "image_wh": dimensions,
            "changed_images": changed,
            "versions": versions,
        }
        (output / "effective_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return receipt
    except BaseException:
        shutil.rmtree(output)
        raise


def verify_training_copy(work: Path, expected: str | None = None) -> dict:
    receipt = json.loads((work / "effective_receipt.json").read_bytes())
    raw, effective = receipt["source_member_sha256"], receipt["effective_member_sha256"]
    if (
        receipt["archive_sha256"] != FORMAL_V1_SHA256
        or _map_sha256(raw) != FORMAL_V1_MEMBERS_SHA256
        or raw.keys() != effective.keys()
    ):
        raise ValueError("Training receipt differs from the formal v1 inventory")
    yaml = (work / "dataset.yaml").read_bytes()
    if yaml != _training_yaml(work.resolve().as_posix()) or (
        hashlib.sha256(yaml).hexdigest() != receipt["local_yaml_sha256"]
    ):
        raise ValueError("Training dataset.yaml differs from the expected splits, classes or path")
    digest = _map_sha256(effective)
    if digest != receipt["effective_members_sha256"] or (
        expected is not None and digest != expected
    ):
        raise ValueError("Effective input fingerprint differs from the expected value")
    _verify_members(work, effective)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--archive", type=Path, help="Stage source bytes from A's formal_dataset_v1.zip"
    )
    mode.add_argument(
        "--from-staged", type=Path, help="Make a training copy of a verified raw stage"
    )
    mode.add_argument("--verify-effective", type=Path, help="Rehash an existing training copy")
    parser.add_argument("--output", type=Path, help="New local dataset directory for preparation")
    parser.add_argument(
        "--expect-effective-sha256", help="Require a previous run's dataset fingerprint"
    )
    args = parser.parse_args()
    if (args.verify_effective is None) != (args.output is not None):
        parser.error("--output is required for preparation and unused for verification")
    if args.expect_effective_sha256 and args.verify_effective is None:
        parser.error("--expect-effective-sha256 is used with --verify-effective")
    try:
        if args.archive is not None:
            prepare_dataset(args.archive, args.output)
            print(f"Staged raw v1 train/val files at {args.output.resolve()}")
        else:
            receipt = (
                prepare_training_copy(args.from_staged, args.output)
                if args.from_staged is not None
                else verify_training_copy(args.verify_effective, args.expect_effective_sha256)
            )
            print(
                f"Effective train/val fingerprint {receipt['effective_members_sha256']}"
                f" ({len(receipt['changed_images'])} JPEGs repaired)"
            )
    except (OSError, ValueError, KeyError, ImportError, AssertionError, zipfile.BadZipFile) as exc:
        print(f"Cannot prepare E2 dataset: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
