# DeskCheck

## Part C ? Jerry-C integration candidate

See [Jerry-C handoff](docs/jerry_c_handoff.md) for the tested synthetic UI/CLI and [real model integration](docs/REAL_MODEL_INTEGRATION.md) for the local-checkpoint YOLO backend and CLI, and [real batch evaluation](docs/REAL_BATCH_EVALUATION.md) for B's existing EvaluationRunner. Install this checkout with `python -m pip install -r requirements.txt`. A real-model inference run is NOT verified until A provides the actual .pt checkpoint and a frozen configuration.


COMP4423 Group 1: compare two desk photographs and report object changes.

## Status
On the unmerged `Jerry-C` branch, Part C now includes a working Streamlit UI, image-pair loading, B-rule comparison, JSON/CSV/PNG export, synthetic demo, and a separately gated YOLO inference adapter with local-checkpoint CLI. The YOLO integration was verified with an injected model-API stub, **not an actual trained checkpoint**. The real DeskCheck dataset, training implementation, checkpoint and academic test metrics are not bundled. The training entrypoint remains a placeholder owned by A. Do not use synthetic detections as real model evidence.

## Setup
Python 3.11+. From repository root:
```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
```
Install training or UI dependencies later with `pip install -e ".[vision,ui]"`. Versions are not a validated lockfile: record resolved versions before experiments.

## Ownership
- A: data capture/audit/split/export, detector and training.
- B: filtering, matching, change rules and evaluation.
- C: configuration, comparison service, visualization, local UI and release.
All members help collect/review data and write their own reports.

## Design
One class per snake_case.py file; PascalCase class names. Domain types are shared contracts. Match same-class objects before checking movement. Train/fine-tune a deep detector; pretrained inference alone is insufficient. See docs/implementation_plan.md.

## Data and rules
Six proposed classes: bottle, cup, keyboard, mouse, book, scissors. Proposed statuses: added, removed, moved, unchanged. Movement threshold is deliberately unset and must be approved before test annotation. Split related capture groups near 60/20/20; never train on validation/test images. No inference from filenames or labels.

## Git
This initializes main because the repository is empty. Subsequent work uses feature branches and reviewed PRs. Do not commit raw images, weights, credentials or run outputs. Branch protection is not configured by this commit.
