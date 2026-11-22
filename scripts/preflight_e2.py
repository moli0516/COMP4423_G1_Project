"""Prepare E2 proposals against A's released E0 and inspect the backend call without training."""

import argparse
import copy
import hashlib
import json
import runpy
import sys
from contextlib import chdir
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deskcheck.training import trainer
from deskcheck.training.model_artifact import sha256_file
from deskcheck.training.training_config import load

E0_WEIGHTS_SHA = "82ed9f8569b705e7074f44d81006515da3e02dde3284a211c4a73a5f9d96c438"
E0_CONFIG_SHA = "bb2bb73582be5fd36c272fd586817fb9379f73596ab61bef72875ea3053725b6"
PROPOSALS = {"e2_960_b16": (960, 16), "e0_control_640_b8": (640, 8), "e2_960_b8": (960, 8)}
TRAINER_SOURCES = (
    "scripts/train.py",
    "src/deskcheck/training/trainer.py",
    "src/deskcheck/training/training_config.py",
)


class ReachedBackend(BaseException):
    """Stop at the backend boundary, outside the training CLI's exception handler."""


def probe(cli, config: Path, scratch: Path) -> tuple[str, dict]:
    call = {}

    class RecordingModel:
        def __init__(self, weights):
            call["weights"] = weights

        def train(self, **kwargs):
            call["kwargs"] = kwargs
            raise ReachedBackend

    with patch.object(trainer, "_lazy_import_ultralytics", return_value=(RecordingModel, "spy")):
        try:
            cli(
                [
                    "--config",
                    str(config),
                    "--dataset",
                    str(scratch / "dataset.yaml"),
                    "--run-dir",
                    str(scratch / config.stem),
                ]
            )
        except ReachedBackend:
            return call["weights"], call["kwargs"]
    raise RuntimeError(f"Training CLI did not reach the backend for {config.name}")


def check_effective_args(weights, kwargs, recorded, defaults, size, batch):
    if weights != recorded["model"] or (kwargs["imgsz"], kwargs["batch"]) != (size, batch):
        raise ValueError("Unexpected initialization, image size or micro-batch")
    if str(kwargs["device"]) != str(recorded["device"]):
        raise ValueError("Training device selection differs from E0")
    # Training restores the same defaults in the pinned backend. Paths vary by machine.
    excluded = {"model", "data", "project", "name", "imgsz", "batch", "device"}
    checked = (kwargs.keys() | recorded.keys()) - excluded
    for key in sorted(checked):
        actual = kwargs[key] if key in kwargs else defaults[key]
        if actual != recorded[key]:
            raise ValueError(f"E0 recorded {key}={recorded[key]!r}, candidate uses {actual!r}")


def json_bytes(data) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def preflight(release: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    checkpoint, config = release / "best.pt", release / "resolved_config.json"
    artifact = json.loads((release / "artifact.json").read_bytes())
    if sha256_file(checkpoint) != E0_WEIGHTS_SHA or artifact["checkpoint_sha256"] != E0_WEIGHTS_SHA:
        raise ValueError("Checkpoint bytes or artifact digest differ from the released E0")
    if sha256_file(config) != E0_CONFIG_SHA:
        raise ValueError("Use the standalone tuned E0 resolved_config.json")
    if version("ultralytics") != artifact["ultralytics_version"]:
        raise ValueError(f"Install Ultralytics {artifact['ultralytics_version']} to match E0")
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT

    recorded = YOLO(str(checkpoint)).ckpt["train_args"]
    base = json.loads(config.read_bytes())
    receipt = {
        "training_performed": False,
        "checkpoint_sha256": E0_WEIGHTS_SHA,
        "resolved_e0_config_sha256": E0_CONFIG_SHA,
        "cases": {},
        "e0_runtime": {
            key: artifact[key] for key in ("python_version", "torch_version", "ultralytics_version")
        },
        "preflight_runtime": {key: version(key) for key in ("torch", "ultralytics")},
        "trainer_source_sha256_lf": {
            name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in TRAINER_SOURCES
        },
        "e0_checkpoint_training_args": {
            key: recorded[key]
            for key in (
                "nbs",
                "amp",
                "deterministic",
                "close_mosaic",
                "workers",
                "warmup_epochs",
                "cos_lr",
                "seed",
                "imgsz",
                "batch",
                "device",
                "model",
            )
        },
    }
    files = {}
    with TemporaryDirectory() as temporary, chdir(ROOT):
        scratch = Path(temporary).resolve()
        for folder in ("images/train", "images/val", "labels/train", "labels/val"):
            (scratch / folder).mkdir(parents=True)
        (scratch / "dataset.yaml").write_text(f"path: {scratch.as_posix()}\n", encoding="utf-8")
        cli = runpy.run_path(str(ROOT / "scripts/train.py"))["main"]
        _, inherited = probe(cli, ROOT / "configs/experiments/resolution.json", scratch)
        receipt["upstream_inherited_size_and_batch"] = [inherited["imgsz"], inherited["batch"]]
        for name, (size, batch) in {"e0_reference": (640, 16), **PROPOSALS}.items():
            path = config
            if name in PROPOSALS:
                candidate = copy.deepcopy(base)
                candidate.update(experiment_id="E0-C" if size == 640 else "E2", owner="C")
                candidate["preprocessing"]["image_size"] = size
                candidate["training"]["batch_size"] = batch
                candidate["outputs"]["run_dir"] = f"runs/experiments/{name}"
                candidate["notes"] = "Proposed pending selection of the comparison and GPU."
                path = scratch / f"{name}.json"
                files[path.name] = json_bytes(candidate)
                path.write_bytes(files[path.name])
            if load(path).inference != base["inference"]:
                raise ValueError("Resolved inference thresholds differ from the released E0")
            weights, kwargs = probe(cli, path, scratch)
            check_effective_args(weights, kwargs, recorded, DEFAULT_CFG_DICT, size, batch)
            receipt["cases"][name] = {
                "imgsz": size,
                "batch": batch,
                "config_sha256": sha256_file(path),
                "initial_weights_name": weights,
                "backend_options_match_e0": True,
            }
    # Publish only after all probes pass. Write the receipt last to identify a complete run.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "receipt.json").write_bytes(json_bytes(receipt))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="A's extracted E0 folder")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for proposals and receipt"
    )
    args = parser.parse_args()
    try:
        preflight(args.release.resolve(), args.output.absolute())
    except (OSError, ImportError, KeyError, ValueError, RuntimeError) as exc:
        print(f"E2 preflight failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified E0 and three proposals; training_performed=false; output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
