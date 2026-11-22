# Part C E2 resolution experiment

Updated 22 November 2026. C has verified the training settings and prepared the candidate configurations. The comparison design and GPU budget remain open.

## Reproduce the configuration check

Use A’s extracted E0 release directory containing `best.pt`, `artifact.json` and the tuned `resolved_config.json`. Install the project's vision extra in the environment. Run from the checkout using a new output directory.

```bash
python scripts/preflight_e2.py --release path/to/E0_baseline --output runs/e2_preflight
```

The command verifies the checkpoint and configuration digests, reads the training arguments stored in the released weights, and exercises A’s real CLI, parser and trainer with an in-memory backend spy. It emits three proposed configurations and writes `receipt.json` after all checks pass. It reads no photos and starts no optimizer. If interrupted while publishing, use a fresh output directory.

A's CLI at `45f0e0a` resolves its E2 template through `baseline_reference`. The earlier inheritance issue was fixed. The three standalone proposals use A's released E0 settings, including the tuned inference confidence of 0.20 and NMS IoU of 0.35. [A's trainer CLI](https://github.com/moli0516/COMP4423_G1_Project/blob/45f0e0a/scripts/train.py)

## Set up the formal dataset on the training machine

A's archived YOLO dataset YAML points to the absolute path on the original training machine. From the project checkout, run the following once on each machine, choosing a fresh output directory.

```bash
python scripts/prepare_e2_dataset.py --archive path/to/formal_dataset_v1.zip --output runs/formal_v1
```

The command verifies the v1 archive fingerprint and copies the released YOLO export bytes for the 28 training and eight validation images together with their labels. It creates a local `dataset.yaml` for A's trainer and a `source_receipt.json` with all 72 member hashes, the split manifest digest and both YAML digests. The held-out test images and labels stay in the archive. The verifier measured the archived YAML before substituting the Windows path. Its digest matches A's artifact at `e02cc0bc03b20cd1cb649cefd2bc3bb86e76c8a436ca5f5946481eeb1d95fe92`. [Formal v1 archive and split digests](REAL_BATCH_EVALUATION.md#formal-v1-results)

For each selected training run, put the agreed initial `yolov8n.pt` in the working directory, record its SHA-256, and pass the local YAML to the shared CLI. For example, the batch-16 command on an allocated GPU is

```bash
python scripts/train.py --config runs/e2_preflight/e2_960_b16.json --dataset runs/formal_v1/dataset.yaml --run-dir runs/e2_960_b16
```

Use a separate fresh run directory per arm. On a different GPU, run the 640 control there with the same initialization, batch and environment as the 960 arm. Check the backend log for an out-of-memory retry or a change in effective batch before interpreting the comparison.

## Training pipeline verification

C ran the shared training CLI at `b965d5b` on the 28 training images and eight validation images for one epoch at 960 pixels, batch 1, on CPU. A PyTorch post-step hook observed one SGD update. The training CSV reported 29.0053 seconds. Both the backend's validation-selected checkpoint and the copy next to the run metadata had SHA-256 `1632c34fbbbef57bae11605ac5999345d148274d0c0ed3ddccfbe827eecbfe3d`. Reloading it on a validation image produced nine finite detections. This run used a fresh `yolov8n.pt` with SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. A's historical initialization digest is still unconfirmed.

The 640/960 comparison still requires matched runs on the selected GPU. The CPU epoch verifies that the data, optimizer, checkpoint and relative output directory work together.

## Choose the comparison

| Configuration | Image size | Micro-batch | Use |
| --- | --- | --- | --- |
| Released E0 | 640 | 16 | Existing reference on A's training machine |
| `e2_960_b16.json` | 960 | 16 | Compare against released E0 when the same environment, original initialization, dataset and memory capacity can be reproduced |
| `e0_control_640_b8.json` | 640 | 8 | Matched control if the 960 run needs batch eight |
| `e2_960_b8.json` | 960 | 8 | Pair with the 640 batch-eight control on the same machine |

Ask the lead and A which comparison and compute budget they approve. If only one new 960/batch-eight run is feasible, an E0-to-E2 comparison measures the joint change in resolution and batch size. For a resolution-only claim on a different machine or runtime, run the matched control there as well. Any GPU memory or latency claims need measurements from the chosen hardware.

## Inputs and evidence

The released E0 checkpoint confirms YOLOv8n initialized from `yolov8n.pt`, SGD, learning rate 0.01, weight decay 0.0005, seed 42, 100 requested epochs, patience 20 and nominal batch 64. AMP and deterministic mode were enabled. The artifact reports 84 completed epochs and epoch 64 selected. The training environment was Python 3.12.3, Ultralytics 8.4.215 and Torch 2.11.0+cu128. The logged device number is 0. Use the original pretrained initialization and record its SHA-256 and physical GPU for an exact historical comparison.

Keep the same release, capture-group split, augmentation, optimizer and validation checkpoint selection across matched runs. A's `training_manifest_sha256` covers the training dataset YAML. Check image and label identities from the release separately. Record effective batch/optimizer steps, validation AP and per-class support, peak memory, wall time and inference latency. Use validation for selecting the model and settings. [E2 protocol](C_E2_RESOLUTION_PROTOCOL.md)

The 18 November preliminary measurements are preserved in [the original preflight notes](C_E2_PREFLIGHT_20261118.md). The current preflight tests configuration reproducibility. E2 accuracy and performance await the experiment.
