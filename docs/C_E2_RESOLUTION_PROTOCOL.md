# Part C resolution experiment protocol

A released the E0 checkpoint, formal v1 dataset and shared trainer at `45f0e0a`. C reproduced E0 application results on the full validation split and exercised the three candidate training configurations through A's trainer. E2 training and GPU measurements are pending. [Configuration receipts and comparison choices](C_E2_DECISION_BRIEF_20261118.md)

## Question and comparison

Does moving from E0 **640 × 640** to E2 **960 × 960** model input
resolution improve small-object detection, and at what cost in inference
time and training memory? E0 used 640. The proposed E2 run uses 960.

| Component | E0 reference | E2 proposal | Control rule |
|---|---|---|---|
| Input size | 640 | 960 | Treatment variable |
| Dataset release/splits | A's v1 | Identical hash & groups | No cross-split leakage |
| Model/init | A's YOLOv8n + source weights | Same verified initialization | A pins model/version |
| Data order / train seed | A's 42 | Same seed (or paired independent seeds) | Compare same conditions |
| Training optimizer/lr/epochs/augmentations | A's E0 config | Same effective values | No E2-specific tuning from test |
| Batch size | 16 in released E0 | 16 if feasible | If 960 requires batch 8, compare it with a 640/batch-8 control on the same hardware |
| Validation | `val` | `val` | Select/checkpoint only from validation |
| Test | Held out | Held out | Final evaluation only after full freeze |
| Reported outputs | Detector mAP, per-class results | Same | A owns scorer and threshold protocol |
| Cost metrics | Latency, memory, training wall-time | Same | Same hardware, warmup and runs |

## Training and evaluation

1. Use A's released dataset and trainer. Record the physical GPU and SHA-256 of the original `yolov8n.pt` initialization. The trained E0 output is kept for inference.
2. Agree on the GPU allocation and a matched batch size. If batch size changes along with resolution, report both changes.

3. Run E0/E2 using the **same trainer** on the **same data and hardware**,
   each writing a fresh run directory with immutable settings/usage receipts.
   Record seeds, checkpoint hash, epoch count, data/split hash, actual settings,
   training time, wall-clock and peak memory. Never overwrite E0.
4. On validation, record mAP@0.5, mAP@[0.5:0.95], per-class AP (especially
   small objects), paired failure examples, and inference latency (identical
   warmed-up hardware / repetition protocol). If tuning is allowed, do it
   only on validation; disclose comparisons and budget.
5. Freeze the settings and chosen checkpoint **before test**. Run the full
   held-out test once for reporting. B handles event/no-change metrics and
   C's app must use the matching model/thresholds, not alternate defaults.
6. Include honest nulls for missing timing/memory metrics; no synthetic
   `FakeColorDetector` output, stub metric or missing checkpoint is evidence
   of the E2 hypothesis.

B's movement threshold is frozen at 0.02152951369355096. The completed four-pair E0 event evaluation is in [the formal validation results](REAL_BATCH_EVALUATION.md#formal-v1-results).

## Report evidence to preserve after execution

E0/E2 resolved configs + SHA-256, dataset/split manifest SHA, scripts/commit,
trainer dependencies, seed, model checkpoints and hashes, per-class detector AP
and val/test split, time/memory unit/measurement method, paired examples,
negative or inconclusive results, and a personally written interpretation.
