# DeskCheck

DeskCheck compares two photographs of the same desk and reports which target objects were added, removed or moved.

## Run the E0 application

Place A's released `best.pt` in `artifacts/`. The UI, single-pair CLI and batch runner use A's `YoloDetector` through `ComparisonService.for_real`. The E0 application settings and checkpoint digest are in `configs/app.json`. Follow the [real-model instructions](docs/REAL_MODEL_INTEGRATION.md) for setup, commands and real-photo verification, or the [batch instructions](docs/REAL_BATCH_EVALUATION.md) for evaluation manifests.

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

- A owns data capture, audit, split and export, the detector, the shared trainer and the E0 baseline.
- B owns filtering, matching, change rules and evaluation.
- C owns application configuration, the comparison service, visualization, UI, CLI, release packaging and the 640/960 resolution experiment using A's trainer. [C's experiment protocol](docs/C_E2_RESOLUTION_PROTOCOL.md)

All members help collect/review data and write their own reports.

## Design
One class per snake_case.py file; PascalCase class names. Domain types are shared contracts. Match same-class objects before checking movement. Train/fine-tune a deep detector; pretrained inference alone is insufficient. See docs/implementation_plan.md.

## Data and rules
The class order is bottle, cup, keyboard, mouse, book, scissors. B reports movement when the center displacement divided by the oriented image diagonal exceeds 0.02152951369355096. Preserve capture-group splits for training and evaluation.

## Git

PR #9 merged the original C application. The follow-up C integration uses A's detector from PR #8 at `45f0e0a`. Keep the weights, photo releases and per-run outputs outside Git.
