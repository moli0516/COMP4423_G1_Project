> **11 November update:** the Jerry-C UI, full-test and actual-browser status is documented in [jerry_c_handoff.md](jerry_c_handoff.md). The 10 November notes below are historical; their pytest/Ruff blockers have been resolved in the new isolated environment.

# C application quickstart

This is a locally implemented C increment on `feat/tam-c-fake-first-20261110`, based on main `146c8cd`. It does not modify the original dirty C checkout, main, A's open PR, B's algorithms, or the shared Path-based ObjectDetector.

## Current capability

The application decodes an oriented RGB pair without resizing, passes pixel-only synthetic detections through the actual upstream B filter/matcher/analyzer, renders copied images, and exports strict JSON/CSV. It also runs through the actual EvaluationRunner. All four statuses, missing-side boxes, empty scenes and explicit failures are covered.

**Only fake mode exists in this increment.** Exact-color shapes are a software integration fixture, not a trained detector. Empty detections do not establish a real-world no-change scene. Configuration thresholds are synthetic examples, not approved scientific settings.

The C-local ImageFrame has immutable bytes-backed pixels and exposes no path or pair identity to the detector. It is deliberately not a claim that A's proposed shared ImageFrame/adapter interface is merged. Transparent images are rejected until alpha-compositing policy is agreed.

## Dependencies and running

Use an already-installed Python with NumPy and Pillow. The UI additionally requires Streamlit. No packages were installed during this work. On this laptop the previous standalone C application's existing Python 3.12 venv supplies those dependencies.

From this worktree, set `PYTHONPATH` to its `src` directory. Run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -B -m unittest discover -s tests/c_fake_first -v
python -B scripts/predict_pair.py --config configs/fake_app.json --before before.png --after after.png --pair-id demo --output runs/demo-001
python -B -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false --server.maxUploadSize=32
```

Replace `python` with the existing appropriate venv executable when required. The UI's built-in synthetic demo needs no external input files. The example CLI inputs must use the fake detector's six exact RGB colors; arbitrary photographs are not supported by this fake detector.

CLI output is a new directory containing exactly:
- result.json
- result.csv
- annotated_before.png
- annotated_after.png

Existing targets are rejected rather than overwritten. All four files are staged before publication. Do not concurrently target the same directory. The CLI analyzes and renders the exact same decoded pixel snapshots. Exit 0 means successful fake output, 2 means an existing output target or argument error, and 1 means input/comparison/write failure.

## Verified on 10 November 2026

- C tests: 16/16 passed, including headless Streamlit interaction and actual CLI subprocess.
- Regression tests reproduce and guard immutable pixels, source-bound config provenance and single-decode publication.
- Real B EvaluationRunner accepts the C service and publishes its three evaluation JSON files using independently specified synthetic truth.
- Actual CLI smoke: exit 0, four records and three changes, four saved outputs.
- Live Streamlit server: loopback health `ok`, HTTP 200; the test server was stopped afterwards.
- Git diff whitespace check passed. Upstream B and ObjectDetector have no diffs.
- Full upstream pytest and Ruff: NOT RUN because these packages are absent in both relevant interpreters. This is not a full-suite/CI/clean-install pass.
- No browser upload acceptance, stress test, real photo inference, model training or real accuracy evaluation is claimed.

Local evidence: `runs/c_actual_20261110T143223Z/execution.json` and its bundle. Run outputs are ignored by Git.

## What is needed for a genuine model run

A must supply the actual DeskCheck dataset release and annotations, frozen capture-group split, usable image/label YOLO export, implemented trainer, trained checkpoint with class/preprocessing identity, and detector adapter. The current main and latest A PR do not contain those deliverables. Movement semantics, inference settings and alpha policy must be agreed; do not invent them or reuse unrelated A2 classification data.

A trained detector can then be integrated through an explicitly agreed in-memory adapter, with canonical-coordinate and real-data tests. This fake increment is not the completion of that real-model goal. No remote push or merge has been performed.
