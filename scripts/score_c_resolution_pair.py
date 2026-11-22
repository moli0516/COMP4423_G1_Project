"""Compare the validation-selected 640 and 960 checkpoints from a matched C experiment."""

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from time import perf_counter

from deskcheck.detection.yolo_detector import APPROVED_CLASS_NAMES
from deskcheck.training.model_artifact import sha256_file

# Hash of the sorted relative-path -> SHA-256 mapping for the released val images and labels.
VAL_INVENTORY = "f2916ea819798de9b7246956d63c79351da76ff272c234c8fbefcf560df510f6"
VAL_PAIRS = ("G01P1", "G01P2", "G03P1", "G03P2")


def check_recipes(control, candidate, control_batch, candidate_batch):
    """Check the actual settings embedded in best.pt, including any backend batch reduction."""
    if (control["imgsz"], candidate["imgsz"]) != (640, 960):
        raise ValueError("Expected checkpoint imgsz 640 then 960")
    if len({control_batch, candidate_batch, control["batch"], candidate["batch"]}) != 1:
        raise ValueError("Requested and effective batch sizes differ")
    if control["epochs"] != 100 or candidate["epochs"] != 100:
        raise ValueError("Expected the agreed 100-epoch cap in both arms")
    vary = {"imgsz", "project", "name", "save_dir"}
    for key in sorted((control.keys() | candidate.keys()) - vary):
        if key not in control or key not in candidate or control[key] != candidate[key]:
            raise ValueError(f"Training argument differs across arms: {key}")


def read_receipt(run, explicit_path):
    if explicit_path is None:
        local = run / "receipt.json"
        explicit_path = local if local.is_file() else run.parent / "receipt.json"
    return json.loads(explicit_path.read_bytes())


def check_provenance(receipts, artifacts):
    """Bind both actual training receipts to their checkpoints and compare the physical inputs."""
    binds = {
        "best_sha": "checkpoint_sha256",
        "data_yaml_sha": "training_manifest_sha256",
        "batch": "batch_size",
        "size": "image_size",
        "epochs_requested": "epochs_requested",
        "python": "python_version",
        "torch": "torch_version",
        "ultralytics": "ultralytics_version",
    }
    shared = (
        "init_sha",
        "effective_train_val_sha256",
        "source_trainer_sha",
        "data_yaml_sha",
        "device",
        "gpu_uuid",
        "cuda",
        "actual_amp",
        "actual_workers",
        "torch",
        "ultralytics",
        "python",
    )
    required = set(binds) | set(shared) | {"exit_code", "torch_steps"}
    for size, receipt, artifact in zip((640, 960), receipts, artifacts, strict=True):
        missing = required.difference(receipt)
        if missing:
            raise ValueError(f"Missing {size} training receipt fields: {sorted(missing)}")
        if type(receipt["exit_code"]) is not int or receipt["exit_code"] != 0:
            raise ValueError(f"Unsuccessful training receipt for {size}: exit_code")
        for key in (
            "init_sha",
            "effective_train_val_sha256",
            "source_trainer_sha",
            "data_yaml_sha",
            "best_sha",
        ):
            if not isinstance(receipt[key], str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[key]):
                raise ValueError(f"Invalid {size} training receipt SHA-256: {key}")
        for key, attribute in binds.items():
            if receipt[key] != artifact[attribute]:
                raise ValueError(f"Training receipt differs from {size} artifact: {key}")
        if receipt["size"] != size or receipt["epochs_requested"] != 100:
            raise ValueError(f"Unexpected training size or epochs_requested for {size}")
        if type(receipt["actual_workers"]) is not int or receipt["actual_workers"] < 0:
            raise ValueError(f"Invalid actual_workers for {size}")
        if type(receipt["actual_amp"]) is not bool:
            raise TypeError(f"Invalid actual_amp for {size}")
        cpu = str(artifact["device"]).lower() == "cpu"
        if cpu:
            if receipt["device"] != "cpu" or receipt["gpu_uuid"] is not None:
                raise ValueError(f"Conflicting CPU device or gpu_uuid in {size} training receipt")
            if not isinstance(receipt.get("machine_id"), str) or not receipt["machine_id"].strip():
                raise ValueError(f"Missing CPU machine_id for {size}")
        else:
            for key in ("device", "gpu_uuid", "cuda"):
                if not isinstance(receipt[key], str) or not receipt[key].strip():
                    raise ValueError(f"Missing GPU {key} for {size}")
        steps = receipt["torch_steps"]
        uninstrumented = (
            cpu and steps is None and receipt.get("step_count_method") == "not instrumented"
        )
        if not uninstrumented and (type(steps) is not int or steps < 1):
            raise ValueError(f"Invalid torch_steps for {size}")
    fields = (*shared, "machine_id") if receipts[0]["device"] == "cpu" else shared
    for field in fields:
        if receipts[0].get(field) != receipts[1].get(field):
            raise ValueError(f"Training receipts differ across arms: {field}")
    return {
        **{field: receipts[0][field] for field in fields},
        "torch_steps": [r["torch_steps"] for r in receipts],
        "step_count_method": [r.get("step_count_method", "observed") for r in receipts],
    }


def validation_files(root):
    files = {}
    for folder, extension in (("images", "jpg"), ("labels", "txt")):
        for pair in VAL_PAIRS:
            for side in ("before", "after"):
                name = f"{folder}/val/{pair}__{side}.{extension}"
                files[name] = sha256_file(root / name)
        expected = {root / name for name in files if name.startswith(folder + "/")}
        if set((root / folder / "val").iterdir()) != expected:
            raise ValueError(f"Unexpected validation inventory in {folder}")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    if digest.hexdigest() != VAL_INVENTORY:
        raise ValueError("Validation files differ from formal v1")
    return [root / name for name in files if name.startswith("images/")]


def read_arm(run):
    from ultralytics import YOLO

    artifact = json.loads((run / "artifact.json").read_bytes())
    path = run / "best.pt"
    if sha256_file(path) != artifact["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint digest differs from the trainer artifact in {run}")
    with (run / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != artifact["epochs_completed"]:
        raise ValueError(f"Incomplete training history in {run}")
    model = YOLO(str(path))
    if tuple(model.names[i] for i in range(len(model.names))) != APPROVED_CLASS_NAMES:
        raise ValueError(f"Unexpected checkpoint classes in {run}")
    return model, artifact, float(rows[-1]["time"])


def area_support(root, images):
    """Count labeled boxes in COCO area bins using original image pixels."""
    from PIL import Image

    counts = {"small": 0, "medium": 0, "large": 0}
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
        label = root / "labels" / "val" / (path.stem + ".txt")
        for line in label.read_text(encoding="utf-8").splitlines():
            _, _, _, w, h = line.split()
            area = width * float(w) * height * float(h)
            category = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            counts[category] += 1
    return counts


def latency(models, images, device):
    import torch
    from PIL import Image

    samples = [[], []]
    gpu = device != "cpu"

    def synchronize():
        if gpu:
            torch.cuda.synchronize(int(device))

    for path in images:
        with Image.open(path) as source:
            image = source.convert("RGB")
        for repetition in range(4):
            # Discard the first pass on each image, then alternate model order.
            for i in (0, 1) if repetition % 2 == 0 else (1, 0):
                model, size = models[i], (640, 960)[i]
                synchronize()
                start = perf_counter()
                model.predict(
                    source=image, device=device, imgsz=size, conf=0.20, iou=0.35, verbose=False
                )
                synchronize()
                if repetition:
                    samples[i].append((perf_counter() - start) * 1000)
    return samples


def evaluate(
    control, candidate, root, output, device, control_receipt=None, candidate_receipt=None
):
    import torch
    import ultralytics

    inputs = validation_files(root)
    arms = [read_arm(run) for run in (control, candidate)]
    models = [arm[0] for arm in arms]
    check_recipes(*(m.ckpt["train_args"] for m in models), *(a[1]["batch_size"] for a in arms))
    for key in ("torch_version", "ultralytics_version", "training_manifest_sha256", "split_seed"):
        if arms[0][1][key] != arms[1][1][key]:
            raise ValueError(f"Training provenance differs across arms: {key}")

    provenance = check_provenance(
        [read_receipt(control, control_receipt), read_receipt(candidate, candidate_receipt)],
        [arm[1] for arm in arms],
    )
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "val.yaml"
    spec = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "names": list(APPROVED_CLASS_NAMES),
    }
    dataset.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    scores = []
    for size, (model, artifact, seconds) in zip((640, 960), arms, strict=True):
        metrics = model.val(
            data=str(dataset),
            imgsz=size,
            device=device,
            batch=min(8, artifact["batch_size"]),
            workers=0,
            split="val",
            rect=True,
            conf=0.001,
            iou=0.70,
            max_det=300,
            plots=False,
            save_json=False,
            project=str(output),
            name=f"val_{size}",
        )
        if list(metrics.nt_per_class) != [8, 5, 8, 8, 8, 7]:
            raise ValueError("Validator support differs from the 44 formal validation labels")
        scores.append(
            {
                "size": size,
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "epochs_completed": artifact["epochs_completed"],
                "selected_epoch": artifact["selected_epoch"],
                "training_csv_seconds": seconds,
                "training_batch": artifact["batch_size"],
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
                "per_class": {
                    name: {
                        "support": int(metrics.nt_per_class[i]),
                        "AP50": float(metrics.class_result(i)[2]),
                        "AP50_95": float(metrics.class_result(i)[3]),
                    }
                    for i, name in enumerate(APPROVED_CLASS_NAMES)
                },
            }
        )
    for score, samples in zip(scores, latency(models, inputs, device), strict=True):
        score["latency_median_ms"] = statistics.median(samples)
        score["latency_samples_ms"] = samples
    report = {
        "split": "val",
        "validation_inventory_sha256": VAL_INVENTORY,
        "photos": len(inputs),
        "capture_groups": ["G01", "G03"],
        "box_area_counts_original_pixels": area_support(root, inputs),
        "validation_batch": min(8, arms[0][1]["batch_size"]),
        "training_provenance": provenance,
        "evaluation_device": device,
        "evaluation_torch": torch.__version__,
        "evaluation_ultralytics": ultralytics.__version__,
        "latency_protocol": "Decoded PIL RGB input, batch 1, application thresholds. One warmup "
        "per image and model, three timed calls per image, alternating order.",
        "control": scores[0],
        "candidate": scores[1],
        "delta_960_minus_640": {
            key: scores[1][key] - scores[0][key]
            for key in ("mAP50", "mAP50_95", "latency_median_ms")
        },
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "dataset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for side in ("control", "candidate"):
        parser.add_argument(
            f"--{side}-receipt", type=Path, help=f"Override {side} training receipt.json location"
        )
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index")
    args = parser.parse_args()
    if args.device != "cpu" and not args.device.isdecimal():
        parser.error("--device must be cpu or a CUDA device index")
    report = evaluate(
        args.control,
        args.candidate,
        args.dataset_root.resolve(),
        args.output.resolve(),
        args.device,
        args.control_receipt,
        args.candidate_receipt,
    )
    print(json.dumps(report["delta_960_minus_640"]))


if __name__ == "__main__":
    main()
