# C’s 640 and 960 resolution experiment

C trained a fresh 640-pixel control and a 960-pixel candidate on the same machine from the same YOLOv8n initialization. A's released E0 remains the production checkpoint. A owns the detector and common trainer, B owns the event rules and evaluation, and C ran the comparison through those existing interfaces.

## Matched RTX 2060 SUPER result

C completed both training runs on the same 8 GiB card at batch 2. The run used Python 3.11.18, Torch 2.9.1+cu126, Ultralytics 8.4.215, mixed precision and zero data-loader workers. The starting weights and all 72 effective files matched the CPU study. The desktop training environment used Pillow 13.0.0. The application declares Pillow below 13. The production comparison service uses CPU inference by default.

| Desktop measurement | 640 | 960 |
| --- | ---: | ---: |
| Completed epochs (selected epoch) | 46 (26) | 28 (8) |
| Observed SGD updates | 18 | 10 |
| Native detector mAP50 | 0.98140 | 0.99500 |
| Native detector mAP50–95 | 0.84079 | 0.88475 |
| Correct localized application events (CPU inference) | 21 / 23 | 23 / 23 |
| Event macro-F1 | 0.86184 | 1.00000 |
| Change-only macro-F1 | 0.83333 | 1.00000 |
| Total elapsed training time | 408.70 s | 355.17 s |
| Peak CUDA tensor memory allocated | 406 MiB | 655 MiB |
| Peak memory reserved by PyTorch | 426 MiB | 746 MiB |
| Median batch-one CUDA prediction | 126.46 ms | 126.54 ms |

The desktop-trained 640 checkpoint reported an unchanged book as removed on both G01P1 and G01P2. The 960 checkpoint classified both correctly. Both scored zero false alarms on the sole eligible independent control. The event evaluation used the regular CPU application and the four original validation pairs.

The 960 checkpoint gained 0.04396 mAP50–95. Its largest class gains were book and scissors. Bottle and keyboard scored lower. GPU latency samples spanned about 109–148 ms at 640 and 113–174 ms at 960 on the occupied desktop. A median difference under 0.1 ms is too small to establish a speed difference here. The 960 run stopped earlier, so its shorter total wall time includes 18 fewer epochs. Memory values are PyTorch process peaks. Other desktop applications also occupied the GPU.

| Class | Validation boxes | 640 AP50–95 | 960 AP50–95 |
| --- | ---: | ---: | ---: |
| Bottle | 8 | 0.8759 | 0.8548 |
| Cup | 5 | 0.8155 | 0.8550 |
| Keyboard | 8 | 0.8955 | 0.8790 |
| Mouse | 8 | 0.8449 | 0.8524 |
| Book | 8 | 0.8283 | 0.9950 |
| Scissors | 7 | 0.7846 | 0.8723 |

Selected desktop checkpoint digests are `8b4e3c46962a8dc79feb0d7637fc2af6fc6cdc23f1a275ac96a9bdbf4003a5e5` (640) and `81d6e3523092179aca8ac67b3933fe5ad480dbbcc70933d7bf3759fa849e916d` (960). The independent native comparison report has SHA-256 `c9ffbadfbc2c5866a1e0e037524e74b9734737e706f9fa103ec8c16fed892e7c`.

### Supported application runtime on the desktop

C also ran the production E0 CLI on the original 4096 × 3072 G06P1 pair using Python 3.12.16, Torch 2.11.0+cu128, Ultralytics 8.4.215 and Pillow 12.3.0. All project core and vision constraints and the dependency check passed. The stock CLI used CPU and wrote JSON, CSV and both PNGs. A second comparison used the same frames and settings with the detector explicitly on CUDA. Both reported a moved cup and five unchanged objects. The maximum coordinate difference was 0.00049 original-image pixels. This train/dev pair validates desktop execution. The accuracy results use the separate validation split.

[Machine-readable experiment results](evidence/c_e2_validation.json)

## Matched CPU result

| Formal v1 validation | 640 | 960 |
| --- | ---: | ---: |
| Effective training micro-batch | 2 | 2 |
| Completed epochs (selected epoch) | 33 (13) | 28 (8) |
| Detector mAP50 | 0.99500 | 0.99500 |
| Detector mAP50–95 | 0.82783 | 0.86614 |
| Application event macro-F1 | 0.82018 | 1.00000 |
| Change-only macro-F1 | 0.77778 | 1.00000 |
| Correct localized events | 21 / 23 | 23 / 23 |
| Alarms on eligible independent controls | 0 / 1 | 0 / 1 |
| Training elapsed, including final validation and output | 248.8 s | 328.7 s |
| Median batch-one prediction from decoded RGB | 78.3 ms | 107.3 ms |

At 640 the application reported an added book in G01P1 and a moved book in G01P2. Both books were annotated unchanged. At 960 both classifications agreed with the annotations. On these eight images, detector mAP50–95 rose by 0.03831 while median CPU prediction took 29.0 ms longer. The original-pixel bounding boxes of all 44 annotated objects exceed the COCO large-area threshold. This split supplies no direct evidence of an improvement on small objects. A single training seed, two validation capture groups and one eligible control limit generalization.

| Class | Validation boxes | 640 AP50–95 | 960 AP50–95 |
| --- | ---: | ---: | ---: |
| Bottle | 8 | 0.8101 | 0.8326 |
| Cup | 5 | 0.7195 | 0.8075 |
| Keyboard | 8 | 0.8162 | 0.8270 |
| Mouse | 8 | 0.8464 | 0.8462 |
| Book | 8 | 0.9558 | 0.9799 |
| Scissors | 7 | 0.8190 | 0.9036 |

## Training and evaluation

Both CPU arms ran the shared trainer on an AMD EPYC 9V74 using Python 3.12.14, Torch 2.11.0+cpu and Ultralytics 8.4.215. They used the E0 augmentation settings, SGD, learning rate 0.01, weight decay 0.0005, seed 42, early-stopping patience 20 and a 100-epoch cap. Each arm started independently from the same pretrained weights. The historical released E0 trained at batch 16 and is documented [separately](REAL_BATCH_EVALUATION.md#formal-v1-results).

The frozen export contains 28 training and eight validation JPEGs with 36 label files. C verified all source and effective file hashes. The backend re-encoded 28 training JPEGs on the working copy. The eight validation JPEGs and all labels retained their released bytes. [Dataset preparation and verification commands](C_E2_DECISION_BRIEF_20261118.md#set-up-the-formal-dataset-on-the-training-machine)

| CPU study input or output | SHA-256 |
| --- | --- |
| Starting YOLOv8n weights, both sizes | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Effective training and validation file inventory | `2b7412542f8e447e57fe9ee0da4f9f39e0a68d12e9ce1bd0b6de91fe4400e6b2` |
| Selected 640 checkpoint | `36170c7d9d02999ea86734b00967e33e960410a4b87bf2474ffe7d5beffcbd88` |
| Selected 960 checkpoint | `4e7d8f57fe60498586c14ad36e51f8e7067c0b7581237bbe859f9552998da9a2` |

C evaluated each selected checkpoint at its training resolution with the native validator on the same eight images. AP used confidence 0.001 and NMS IoU 0.70. Application events used the eight original photographs with confidence 0.20, NMS IoU 0.35 and B’s movement threshold of 0.02152951369355096. Each timing series contains 24 predictions with model order alternating, after one warm-up per image and model. Decoding, display and serialization are outside the timed interval. CPU elapsed times include execution on a shared host.

Run the scorer with both output directories, their receipts and the prepared dataset. It checks effective batches and checkpoint arguments, initialization and dataset hashes, trainer identity, runtime and hardware provenance before writing a fresh comparison. The private evidence package contains the selected checkpoints and receipts. Keep the held-out test split reserved for the team’s final freeze.

```bash
python scripts/score_c_resolution_pair.py --control runs/640 --candidate runs/960 --dataset-root runs/formal_v1_work --device cpu --output runs/validation_comparison
```
