# Member C — E2 resolution experiment: gated protocol (not run)

**Evidence:** A's unmerged PR #8, `configs/experiments/resolution.json`
at commit `29f65797c19379a678dc6ce1b5fc39d1b010ffd3`
declares `experiment_id=E2`, `owner=C` and a proposed
input-resolution comparison against A's E0 baseline.
**Status: NOT EXECUTED.** The file explicitly says STUB, so no
measurements or selected checkpoint are claimed.

12 November recheck of the same A commit: `scripts/train.py` still exposes
only `--config` and exits 2 with `Not implemented: validate/export training
data and implement ModelTrainer first.` No runnable trainer interface exists
for a C execution wrapper yet. Do not invent `--data`/`--run-dir` flags from
the older blueprint or add a competing trainer. Next executable E2 step belongs
to A: deliver the runnable trainer and frozen E0 inputs; C then binds a preflight
and fresh-run invocation to that actual interface, before any approved training.

## Question and comparison

Does moving from E0 **640 × 640** to E2 **960 × 960** model input
resolution improve small-object detection, and at what cost in inference
time and training memory? These numbers are **A's provisional settings**,
not a frozen academic choice.

| Component | E0 proposed | E2 proposed | Control rule |
|---|---|---|---|
| Input size | 640 | 960 | Treatment variable |
| Dataset release/splits | A's v1 | Identical hash & groups | No cross-split leakage |
| Model/init | A's YOLOv8n + source weights | Same verified initialization | A pins model/version |
| Data order / train seed | A's 42 | Same seed (or paired independent seeds) | Compare same conditions |
| Training optimizer/lr/epochs/augmentations | A's E0 config | Same effective values | No E2-specific tuning from test |
| Batch size | 16 | 8 | **Confound:** A's stub changes this too; agree equivalent effective batch via accumulation or explicitly qualify the comparison |
| Validation | `val` | `val` | Select/checkpoint only from validation |
| Test | Held out | Held out | Final evaluation only after full freeze |
| Reported outputs | Detector mAP, per-class results | Same | A owns scorer and threshold protocol |
| Cost metrics | Latency, memory, training wall-time | Same | Same hardware, warmup and runs |

## Protocol after A makes a runnable training baseline

1. A supplies real frozen dataset release ID, SHA-256 for split manifest,
   dataset YAML, model/init, dependency version and working trainer.
   C does **not** implement a competing trainer.
2. A/B/C agree any necessary E2 batch-size compensation and how to compare
   effective gradient batch. Predeclare any unavoidable difference; do not
   present such a result as an isolated resolution effect.
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

## Known blockers, not discretionary work by C

- A PR #8 is still open; no approved merged trainer/checkpoint/dataset release.
- B has not frozen the movement threshold for event metrics.
- The proposed E2 `training.batch_size=8` differs from E0's 16;
  requires A review because batch effects can confound resolution.
- **A/B approval is needed before running this experiment or changing shared
  training settings.** This document is a reproducibility plan, not an
  authorization to run costly training.

## Report evidence to preserve after execution

E0/E2 resolved configs + SHA-256, dataset/split manifest SHA, scripts/commit,
trainer dependencies, seed, model checkpoints and hashes, per-class detector AP
and val/test split, time/memory unit/measurement method, paired examples,
negative or inconclusive results, and a personally written interpretation.
