# Part C E2 resolution experiment — executable preflight against A PR #8 (18 Nov 2026)

**Current status: CONFIG PREFLIGHT PASS / TRAINING NOT AUTHORIZED OR RUN.**
This addendum supersedes the old trainer-placeholder observation in the earlier C E2 protocol. Do not edit A's branch or claim model results.

## New A-side implementation (source inspected)

Source snapshot: A PR #8 commit 1a4974b6ea153dbf7f83571f89a15b36c700caa1 (still OPEN and unmerged when inspected).
- scripts/train.py now has a shared trainer wrapper with --config, --run-dir and optional --dataset.
- src/deskcheck/training/trainer.py now contains a genuine Ultralytics train() code path, but this is **not** proof of an executed training run or a checked model.
- The trainer calls load_config(config_path) **without** providing baseline_path.
- The E0 baseline remains a placeholder for hyperparameters and the E2 resolution.json remains a STUB containing "inherit" markers and batch size 8 (vs E0 16).
- Current standalone config validation rejects the raw E2 file: ValueError: config block 'model' must be a JSON object.

Source:
https://github.com/moli0516/COMP4423_G1_Project/blob/1a4974b6ea153dbf7f83571f89a15b36c700caa1/src/deskcheck/training/trainer.py
https://github.com/moli0516/COMP4423_G1_Project/blob/1a4974b6ea153dbf7f83571f89a15b36c700caa1/src/deskcheck/training/training_config.py
https://github.com/moli0516/COMP4423_G1_Project/blob/1a4974b6ea153dbf7f83571f89a15b36c700caa1/configs/experiments/resolution.json

## C-owned preflight actually executed — zero model runs

Read-only snapshots of A's actual training_config.py, E0/E2 JSON, trainer.py and CLI were tested with Python 3.12.10 on Tamslaptop. **14 / 14 check PASS**:
1. E0 loads and validates standalone.
2. Raw E2 without baseline fails with the predicted config validation error.
3. With baseline_path=E0 supplied, E2 fully resolves.
4. A generated complete E2 candidate passes A's standalone config loader without baseline_path.
5. Differences E0->E2: resolution 640->960; batch size 16->8; an added training note; and expected run-directory name. Other effective model/data/optimizer/augmentation settings are inherited, provisionally.
6. Jerry-C currently does not contain A's frozen dataset.yaml or merged A training implementation; there is no accepted E0 checkpoint or full experiment output.

**Candidate is not approved and must NOT be used for training yet.**
Candidate: E2_RESOLVED_CANDIDATE_NOT_APPROVED.json
SHA-256: 6f392a128e088ad79c4f0370a4f1e073e8fc00a31a2d880829b609db394aae10

Read-only A snapshots, actual preflight script and JSON receipt saved outside Git:
%USERPROFILE%\Documents\COMP4423_C_E2_PREFLIGHT_20261118_02\
Handoff ZIP: %USERPROFILE%\Documents\COMP4423_C_RELEASE_COMPAT_20261118\C_E2_PREFLIGHT_A_PR8_1a4974_20261118.zip
ZIP SHA-256: c5f64af9125fdc3c7fb474491b75c7b04764fe34eceac611a53d5ef6722888c2; ZIP CRC PASS.

## Remaining dependencies and scientific controls (external)

- A/B: finalize/review A's real trainer and shared frame/detector, provide exact merged commit, tested Ultralytics/PyTorch versions, frozen dataset release/split SHA, and verified E0 configuration/checkpoint/results.
- A/B/C: settle the E2 batch-size change from 16 to 8; either agree an equivalent effective batch/accumulation comparison or disclose the confound. Image resolution alone cannot explain an uncontrolled result.
- C AFTER shared approval: resolve approved E2 config using the actual merged interface, use A's single trainer on the same data, seeds and hardware; collect validation AP, per-class AP, cost/latency/memory with provenance and failure examples; do not tune on the held-out test. No competing trainer should be created.
- Group leader: confirm experiment control approvals and independently review setup/release tools; neither approval is implied by this preflight.
- Student: write their own individual report interpretation and Statement of Contribution. Do not present AI-generated text as personally written.

The E2 candidate and preflight receipts are separate from A's branch, the Jerry-C production runtime and the final course ZIP. This operation did not download models, run training, call paid inference/training APIs or change the team dataset.
