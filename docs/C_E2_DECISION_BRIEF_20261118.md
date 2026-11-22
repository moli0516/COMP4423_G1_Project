# Part C E2 resolution experiment

C chose a fresh 640 control and 960 candidate with matched settings on the same machine. This page records the configuration preflight, raw data preparation and earlier engineering smoke. [Measured results and scope](C_E2_RESOLUTION_PROTOCOL.md)

## Reproduce the configuration check

Use A’s extracted E0 release directory containing `best.pt`, `artifact.json` and the tuned `resolved_config.json`. Install the project's vision extra in the environment. Run from the checkout using a new output directory.

```bash
python scripts/preflight_e2.py --release path/to/E0_baseline --output runs/e2_preflight
```

The command verifies the checkpoint and configuration digests, reads the training arguments stored in the released weights, and exercises A’s real CLI, parser and trainer with an in-memory backend spy. It emits three proposed configurations and writes `receipt.json` after all checks pass. It reads no photos and starts no optimizer. If interrupted while publishing, use a fresh output directory.

A's CLI at `45f0e0a` resolves its E2 template through `baseline_reference`. The earlier inheritance issue was fixed. The three standalone proposals use A's released E0 settings, including the tuned inference confidence of 0.20 and NMS IoU of 0.35. [A's trainer CLI](https://github.com/moli0516/COMP4423_G1_Project/blob/45f0e0a/scripts/train.py)

## Set up the formal dataset on the training machine

The released YOLO YAML points to A's training machine. In addition, Ultralytics may rewrite JPEGs in its first dataset scan. Use a byte-faithful source directory and a separate working directory for training. An existing source stage with its original `source_receipt.json` can be reused if all images and labels still match the release.

```bash
python scripts/prepare_e2_dataset.py --archive path/to/formal_dataset_v1.zip --output runs/formal_v1_raw
python scripts/prepare_e2_dataset.py --from-staged runs/formal_v1_raw --output runs/formal_v1_work
python scripts/prepare_e2_dataset.py --verify-effective runs/formal_v1_work
```

The first command checks the frozen archive and extracts 28 training and eight validation JPEGs with their 36 label files. The second checks all 72 source hashes against the formal v1 inventory. It runs the pinned Ultralytics JPEG check on individual working copies, verifies the canonical image dimensions, and records the source and effective member hashes, runtime and JPEG codec in `effective_receipt.json`. Keep both receipts. The archive's original YAML digest is `e02cc0bc03b20cd1cb649cefd2bc3bb86e76c8a436ca5f5946481eeb1d95fe92`. Each machine receives its own local YAML. The held-out test split stays in the archive. [Formal v1 archive and split digests](REAL_BATCH_EVALUATION.md#formal-v1-results)

Two independent preparations on Linux using Ultralytics 8.4.215, Pillow 12.3.0 and libjpeg-turbo 3.1.4.1 produced the same effective fingerprint, `2b7412542f8e447e57fe9ee0da4f9f39e0a68d12e9ce1bd0b6de91fe4400e6b2`. All 28 training JPEGs were re-encoded. All eight validation JPEGs and 36 labels retained their original bytes. All 36 images retained their 4096 × 3072 canonical dimensions.

Record the printed effective fingerprint before training. Verify it before and after each arm. A changed file, removed label or additional image-side cache fails the check. If a previous trainer has modified the raw source, extract a fresh raw stage from the original ZIP.

```bash
python scripts/prepare_e2_dataset.py --verify-effective runs/formal_v1_work --expect-effective-sha256 <recorded-fingerprint>
```

For each training run, put the same initial `yolov8n.pt` in the working directory, record its SHA-256, and pass the working YAML to the shared CLI. For example, the batch-16 command is

```bash
python scripts/train.py --config runs/e2_preflight/e2_960_b16.json --dataset runs/formal_v1_work/dataset.yaml --run-dir runs/e2_960_b16
```

Use a fresh run directory per arm. On a different GPU, run the 640 control there with the same initialization, effective data, batch and environment as the 960 arm. Check the backend log for an out-of-memory retry or a change in effective batch before interpreting the comparison.

## Training pipeline verification

C ran the shared training CLI at `b965d5b` on the 28 training images and eight validation images for one epoch at 960 pixels, batch 1, on CPU. A PyTorch post-step hook observed one SGD update. The training CSV reported 29.0053 seconds. Both the backend's validation-selected checkpoint and the copy next to the run metadata had SHA-256 `1632c34fbbbef57bae11605ac5999345d148274d0c0ed3ddccfbe827eecbfe3d`. Reloading it on a validation image produced nine finite detections. This run used a fresh `yolov8n.pt` with SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. A's historical initialization digest is still unconfirmed.

The one-epoch run verified the data, optimizer, checkpoint and relative output directory. The matched experiment uses separately trained checkpoints at both sizes.

## Earlier configuration options

| Configuration | Image size | Micro-batch | Use |
| --- | --- | --- | --- |
| Released E0 | 640 | 16 | Existing reference on A's training machine |
| `e2_960_b16.json` | 960 | 16 | Compare against released E0 when the same environment, original initialization, dataset and memory capacity can be reproduced |
| `e0_control_640_b8.json` | 640 | 8 | Matched control if the 960 run needs batch eight |
| `e2_960_b8.json` | 960 | 8 | Pair with the 640 batch-eight control on the same machine |

On a different machine, train both sizes with its own common batch size. The comparison used by C does not depend on the identity of A’s historical GPU. The local hardware measurements and actual batches are recorded with each pair.

## Inputs and evidence

The released E0 checkpoint confirms YOLOv8n initialized from `yolov8n.pt`, SGD, learning rate 0.01, weight decay 0.0005, seed 42, 100 requested epochs, patience 20 and nominal batch 64. AMP and deterministic mode were enabled. The artifact reports 84 completed epochs and epoch 64 selected. The training environment was Python 3.12.3, Ultralytics 8.4.215 and Torch 2.11.0+cu128. The logged device number is 0. Use the original pretrained initialization and record its SHA-256 and physical GPU for an exact historical comparison.

Keep the same release, capture-group split, augmentation, optimizer and validation checkpoint selection across matched runs. A's `training_manifest_sha256` covers the training dataset YAML. Check image and label identities from the release separately. Record effective batch/optimizer steps, validation AP and per-class support, peak memory, wall time and inference latency. Use validation for selecting the model and settings. [E2 protocol](C_E2_RESOLUTION_PROTOCOL.md)

The 18 November preliminary measurements are preserved in [the original preflight notes](C_E2_PREFLIGHT_20261118.md). The current preflight tests configuration reproducibility. The experiment protocol links to the measured accuracy and performance.
