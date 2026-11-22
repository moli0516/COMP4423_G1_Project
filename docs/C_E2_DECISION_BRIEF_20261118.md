# Part C E2 resolution experiment

Updated 22 November 2026. C has verified the training settings and prepared the candidate configurations. The comparison design and GPU budget remain open.

## Reproduce the configuration check

Use A’s extracted E0 release directory containing `best.pt`, `artifact.json` and the tuned `resolved_config.json`. Install the project's vision extra in the environment. Run from the checkout using a new output directory.

```bash
python scripts/preflight_e2.py --release path/to/E0_baseline --output runs/e2_preflight
```

The command verifies the checkpoint and configuration digests, reads the training arguments stored in the released weights, and exercises A’s real CLI, parser and trainer with an in-memory backend spy. It emits three proposed configurations and writes `receipt.json` after all checks pass. It reads no photos and starts no optimizer. If interrupted while publishing, use a fresh output directory.

A's CLI at `45f0e0a` resolves its E2 template through `baseline_reference`. The earlier inheritance issue was fixed. The three standalone proposals use A's released E0 settings, including the tuned inference confidence of 0.20 and NMS IoU of 0.35. [A's trainer CLI](https://github.com/moli0516/COMP4423_G1_Project/blob/45f0e0a/scripts/train.py)

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
