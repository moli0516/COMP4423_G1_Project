# Part C E2 — decision brief for A and the group leader

Date: 18 Nov 2026 (HKT)
Status: PROPOSED SETTINGS / DECISIONS OPEN / NO TRAINING AUTHORIZED

This is a short decision brief responding to the group leader's request. It reuses the existing C E2 protocol and 14-check preflight rather than replacing them. Exact A source: PR #8, commit 1a4974b6ea153dbf7f83571f89a15b36c700caa1 (unmerged at inspection). C evidence: PR #9, commit c40ee61538b0ed4fcf9f798b7c0196b506f123e3.

## A. Current E0 vs proposed E2 values (not frozen)

| Field | A's E0 baseline | A's E2 proposal |
| --- | --- | --- |
| Experiment owner | A | C |
| Model and initialization | YOLOv8n; yolov8n.pt | Same as E0 |
| Training resolution | 640 x 640 | 960 x 960 |
| Micro-batch | 16 | 8 |
| Epochs; patience; seed | 100; 20; 42 | Inherit E0 |
| Optimizer; lr0; weight_decay | SGD; 0.01; 0.0005 | Inherit E0 |
| Dataset; split seed; classes | Proposed v1; 42; six target classes | Same frozen release when available |
| Color; letterbox | RGB; true | Same |
| Augmentations | hsv_h=.015, hsv_s=.7, hsv_v=.4, fliplr=.5, mosaic=1, scale=.5 | Inherit |
| Checkpoint selection | Validation best mAP50-95 | Same |
| Confidence; NMS IoU | 0.25; 0.45 | Same |
| Run directory | E0_baseline | E2_resolution |

E0 describes its hyperparameters as placeholders. E2 is explicitly STUB. These are candidates, not final commands or accepted results.

## B. Main decision — define the comparison before running

The proposal simultaneously changes two knobs: 640 to 960 pixels and batch 16 to 8. The input pixel area per image is 2.25x, but real memory, latency and training cost are not measured. A raw E0-vs-E2 AP delta would be a resolution-plus-batch comparison, not an isolated resolution effect.

Recommended decision IF the scientific goal is the effect of resolution:
- Preserve A's existing E0 reference (640 / batch 16) and its history.
- Subject to A's GPU/memory and experiment budget approval, compare a separate matched-batch E0-C control (640 / batch 8) versus E2 (960 / batch 8), on identical frozen data, initialization, optimizer, seed, augmentation, trainer, checkpoint protocol and hardware.
- Do NOT silently relabel the original E0 as batch 8. A matched control is a separate run and needs explicit time/compute approval.
- If the E2 batch 8 cannot fit, approve a lower identical micro-batch for both runs, or use the operational trade-off framing below.

Lower-compute fallback: retain 640 / batch 16 versus 960 / batch 8. Label the result an operational resolution-plus-batch trade-off and explicitly disclose that batch-size effects are not identifiable separately. Do not make resolution-only causal claims.

Gradient-accumulation alternative: the current A trainer passes batch but does not explicitly pass or log Ultralytics nominal batch (nbs) / accumulation. Ultralytics documents nbs-dependent accumulation and weight-decay scaling; A must confirm the behavior in the *pinned installed version* and save actual optimizer-step and effective-batch receipts. Equal effective batch does not automatically eliminate differences from unequal micro-batches. External version-dependent reference: https://docs.ultralytics.com/modes/train

## C. Confirmed E2 config compatibility issue

A's shared trainer has now been implemented in PR #8 and runs load_config(config_path) without a baseline_path. The raw E2 resolution.json has inherit-valued sections, so A's config loader raises ValueError: config block 'model' must be a JSON object when given that file directly.

C already executed 14/14 preflight checks against the exact A source, showing that:
- E0 loads standalone; raw E2 fails without baseline.
- Passing E0 as baseline resolves E2.
- A fully resolved standalone E2 candidate then loads.
- The effective changes are image_size 640 to 960; batch_size 16 to 8; one training note; output folder. Other declared fields inherit E0.

Decision requested from A: either make the shared trainer safely resolve the referenced E0 baseline, or approve a versioned self-contained E2 JSON generated from the *final frozen* E0. Never execute the current raw E2 STUB. C's offline candidate is NOT approved (SHA256 6f392a128e088ad79c4f0370a4f1e073e8fc00a31a2d880829b609db394aae10).

## D. Decisions for A and leader

| ID | Decision needed | Owner | Current state |
| --- | --- | --- | --- |
| D1 | Which claim: resolution-controlled comparison (preferred), or operational 640/16 versus 960/8 trade-off? | Leader + A + C | OPEN |
| D2 | Approve matched micro-batch and, if necessary, the additional E0-C run; fix maximum hardware/time/seed budget | Leader + A | OPEN |
| D3 | Pin training software, machine/GPU, actual batch/accumulation, AMP, deterministic controls, optimizer/schedule, augmentation, seed(s) | A | OPEN |
| D4 | Finalize E0 and resolve E2 inheritance using A's approved parser/interface; publish full config hashes | A with C preflight | OPEN |
| D5 | Provide frozen dataset.yaml, split/group manifest hashes, class map and initial weights hash; no capture-group leakage or test tuning | A + B | NOT SUPPLIED TO C |
| D6 | Freeze validation AP protocol (mAP50/mAP50-95/per-class and small-object support), test-set holdout rules, inference resize/size for E0/E2, confidence/NMS | A + B + C | OPEN |
| D7 | Define measured costs: warmed-up repeated inference latency (e.g. median and p95), hardware, GPU peak memory, train wall time, epoch count, optimizer steps and any OOM/retries | A + C | OPEN |
| D8 | Validate checkpoint metadata: A trainer currently records selected_epoch=epochs_completed after locating best.pt or last.pt; this may misstate which validation epoch actually yielded best.pt. Also record which weights were actually selected | A | REVIEW NEEDED |
| D9 | Explicit approval to run E2 only after D1-D8; no model training or extra spending beforehand | Leader + A + C | HOLD |

Minimal agreed experiment record: comparison label, source/trainer commit, frozen E0/E0-C/E2 configs and hashes, pretrained weights hash, dataset/split manifest hash, device + seed(s), micro/effective batch and optimizer steps, checkpoint + actual selected best epoch, validation AP and per-class support, latency/memory/time method, failure logs and training budget. Evaluate held-out test once after the protocol is frozen, not for tuning.

## E. Source and execution boundary

- A baseline: https://github.com/moli0516/COMP4423_G1_Project/blob/1a4974b6ea153dbf7f83571f89a15b36c700caa1/configs/experiments/baseline.json
- A E2: https://github.com/moli0516/COMP4423_G1_Project/blob/1a4974b6ea153dbf7f83571f89a15b36c700caa1/configs/experiments/resolution.json
- A trainer: https://github.com/moli0516/COMP4423_G1_Project/blob/1a4974b6ea153dbf7f83571f89a15b36c700caa1/src/deskcheck/training/trainer.py
- Existing C detail: docs/C_E2_RESOLUTION_PROTOCOL.md and docs/C_E2_PREFLIGHT_20261118.md
- C raw config-preflight receipt: Tamslaptop, %USERPROFILE%\Documents\COMP4423_C_E2_PREFLIGHT_20261118_02\

No A/B code was edited. No dataset/trained checkpoint was created, and no E2 training was started. PR #9 remains Draft. This is a decision document, not a request to merge.
