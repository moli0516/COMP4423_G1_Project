# DeskCheck

## Part C ? Jerry-C integration candidate

See [Jerry-C handoff and running instructions](docs/jerry_c_handoff.md) for the current synthetic UI, CLI, tests and exact real-model limitations. Install this checkout with `python -m pip install -r requirements.txt`. The UI, CLI and B evaluation runner use the same comparison service; no trained inference is claimed.


COMP4423 Group 1: compare two desk photographs and report object changes.

## Status
Project skeleton only. No trained model, dataset, detector implementation or working UI is included yet. Placeholder entry points fail explicitly.

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
