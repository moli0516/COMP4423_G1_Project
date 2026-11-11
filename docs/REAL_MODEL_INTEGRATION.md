# Jerry-C real-model inference entry point

This is **new executable Part C integration** for an externally supplied local
Ultralytics YOLO detect-model checkpoint. It does **not** include a real model
or a new training result; A has not yet supplied one. No network/API/model
download is performed by Jerry-C's inference command.

## Two distinct modes

- **Synthetic:** existing `Start Jerry-C.cmd`, built-in fake color test and
  `scripts/predict_pair.py`. Never report these as real-photo inference.
- **Real (requires A's checkpoint):** `scripts/predict_real_pair.py` and the
  "Run local YOLO checkpoint" choice in the SAME Streamlit app.
  Both use the existing B detection filter, matcher, analyzer and
  `ResultRenderer` / `ResultSerializer`. They require explicit real config
  and a local `.pt` file with matching SHA-256.

## When A's file arrives

1. Keep A's weight outside git, for example
   `C:\Users\qwqwh\Documents\COMP4423_MODEL_PRIVATE\best.pt`.
   Do not pretend the proposed A PR's `configs/experiments/*.json`
   are trained models. Confirm the actual checkpoint is for object detection.
2. In a Python shell, calculate the complete weight hash:

   ```python
   import hashlib
   from pathlib import Path
   p = Path(r"C:\absolute\path\to\best.pt")
   h = hashlib.sha256()
   with p.open("rb") as stream:
       for chunk in iter(lambda: stream.read(1024 * 1024), b""):
           h.update(chunk)
   print(h.hexdigest())
   ```

3. Create an **untracked local config outside git** (e.g.
   `C:\Users\qwqwh\Documents\COMP4423_MODEL_PRIVATE\real.json`)
   using this schema, with actual verified values:

   ```json
   {
     "schema_version": "1.0",
     "mode": "real",
     "checkpoint_path": "best.pt",
     "checkpoint_sha256": "REPLACE_WITH_ACTUAL_64_HEX_SHA256",
     "model_origin": "REPLACE_WITH_A_SOURCE_AND_TRAINING_RUN_ID",
     "confidence_threshold": 0.25,
     "movement_threshold": 0.02,
     "nms_iou_threshold": 0.5,
     "imgsz": 640,
     "rules_version": "REPLACE_WITH_APPROVED_RULES_VERSION",
     "class_names": ["bottle", "cup", "keyboard", "mouse", "book", "scissors"]
   }
   ```

   **Numbers in this illustration are not approved/frozen team settings**;
   replace them from A/B's actual validated run. The local real config is
   rejected if missing, malformed, nonfinite, wrongly hashed or inconsistent.

4. Only once the local checkpoint exists, install the optional Vision
   dependencies into the **separate Jerry-C environment** if not already
   installed: `python -m pip install -e ".[vision]"`. This may install a
   large CPU/torch stack; do not run an installation automatically just to
   test the UI. GPU use is not required; the adapter explicitly selects CPU.
   The model is created with its absolute, already verified local path,
   not with an auto-download model identifier.
5. In the `Jerry-C` repository from PowerShell, run:

   ```powershell
   .\.venv\Scripts\python.exe -B scripts/predict_real_pair.py --config "C:\absolute\path\to\real.json" --before "C:\photos\before.jpg" --after "C:\photos\after.jpg" --pair-id real-p1 --output "runs\real-p1-new"
   ```

   Output directory must be new. Four files:
   `result.json`, `result.csv`, `annotated_before.png`,
   `annotated_after.png`. The result metadata includes actual checkpoint
   SHA-256 and resolved config SHA-256. A failed real model load/inference
   returns a nonzero exit code **without fake fallback**.
6. To use the existing UI, set `DESKCHECK_REAL_CONFIG` to the absolute
   local config path before launching `Start Jerry-C.cmd`, choose
   "Run local YOLO checkpoint", upload photos and compare. Without the
   config, the real button is disabled and no fake inference is published.

## Settings integrity and A/B ownership

`RealSettings` is now **source-bound**: an instance carries its immutable
config bytes and original resolved source path, and `__post_init__` validates
all effective fields against that source and checkpoint SHA-256.
`dataclasses.replace(settings, movement_threshold=...)` or direct construction
with a mismatched `config_sha256` is rejected; a legitimate change requires
reloading a new source and produces a new hash. This protects the metadata
used in UI, CLI and B runner. It is a safeguard against accidental
in-process changes, not a substitute for A's real-checkpoint validation.

Detector ownership is **not frozen by Jerry-C**. The current
`UltralyticsDetector` is C's candidate only; A owns the final model loader,
preprocess/NMS/box restoration and may adopt or replace the candidate.
C keeps image loading, UI/CLI orchestration, exports, and an explicit frame
adapter seam; B owns filter/matcher/change/evaluation. See
[A/B/C detector handoff](A_B_C_DETECTOR_HANDOFF.md).

## Exact contract and limitations

- A model supplies native `Results.boxes.xyxy/conf/cls` and its `names`
  mapping; the adapter converts those to B's six canonical `Detection`
  classes. It supports six-class custom models as well as models whose
  native class list contains extra classes, but it **requires all six**
  names, rejects duplicate class mappings, and drops unsupported labels.
- The loader applies EXIF orientation and RGB conversion once. The adapter
  supplies a *copy* as OpenCV-style BGR to Ultralytics, requests CPU
  inference and expects model boxes in the original oriented pixel system.
  Ultralytics handles its internal letterbox/rescale and backend NMS.
- B's existing filter keeps at most one prediction/class. Matching and
  status meanings are **unchanged**. A/B must still approve the final
  confidence, movement threshold, NMS setting, class policy and alpha policy.
- The only current real-mode tests **inject a controlled model API stub**,
  assert class mapping/B integration and explicit no-fallback failures.
  This is genuine application wiring, **not proof of a successful
  Ultralytics inference, academic checkpoint, held-out real-photo quality,
  E2 resolution experiment or course marking**. Those remain blocked by
  A's yet-to-be-delivered weights and frozen dataset.

Primary API reference: https://docs.ultralytics.com/modes/predict/
and https://docs.ultralytics.com/reference/engine/results/
