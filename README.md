# DeskCheck

## Part C — Jerry-C integration candidate

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
The `dev` extra includes Streamlit for UI tests. No vision dependency or model is
needed for the synthetic demo. Do not install/download a model to satisfy a setup
check. For the tested Windows/Python 3.12 constraints and release procedure, see
[C source release](docs/C_SOURCE_RELEASE.md).

## Run and diagnose

On Windows, from the repository root:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_c_setup.py --profile ui --port 8503
.\.venv\Scripts\python.exe -B scripts/run_c_demo.py --output runs/demo-NEW-NAME
```

Then double-click `Start Jerry-C.cmd` and open `http://127.0.0.1:8503`.
The launcher checks setup and port availability first. Keep its window running;
Ctrl+C stops the local server. A connection error after stopping it requires a
new launch. If the port is occupied, inspect the existing UI/service; the checker
does not terminate it. JSON diagnostics are available with `--json`.

Images and the table share pair-local `object-N` correspondence labels; PNGs and
the table show selected detection confidence. JSON retains original oriented
pixel boxes, IDs and confidence. CSV keeps the B event fields and flattened boxes.
The serializer rejects contradictory retained-detection/event evidence before
rendering or export. Legacy event-only results remain supported, with unknown
confidence displayed as empty rather than invented.

`--real-config <path>` on the checker validates config/checkpoint hashes only.
It never loads a model, certifies its runtime, or substitutes synthetic output.
Post-B-filter diagnostic arrays are **not detector AP input**; A defines the
detector-level evaluation boundary. The additive result extension still needs
A/B review. B's batch `predictions.json` contains event records only.

Build and verify a C source archive from a clean, committed Jerry-C checkout:

```powershell
.\.venv\Scripts\python.exe -B scripts/package_c_release.py --output runs/C_Source_NEW.zip
.\.venv\Scripts\python.exe -B scripts/package_c_release.py --verify runs/C_Source_NEW.zip
```

This archive excludes data, weights, environments and individual reports. It is
marked `C_SOURCE_ONLY`, **not a complete Canvas submission**. See
[contribution evidence](docs/C_CONTRIBUTION_EVIDENCE.md) for factual code/evidence
pointers; each student writes and verifies their own report.

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
This candidate is on `Jerry-C`, in Draft PR #9. Main and A/B branches are not
modified. A PR #8 remains a separate integration gate. Do not commit raw images,
weights, credentials or run outputs; do not merge without team review.
