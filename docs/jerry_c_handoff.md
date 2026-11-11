# Jerry-C: Part C handoff (11 November 2026)

## Scope and current boundary
This branch integrates the actual upstream B filtering, matching, change analysis and evaluation runner with C image loading, CLI, UI, rendering and export. It is **synthetic-mode software**, not a trained object detector. Real photographs can be inspected manually; the UI does not present fake detections as real-photo AI results.

The original standalone demo, original dirty checkout and 10 November worktree are preserved. Work is isolated on `Jerry-C`, based on main `146c8cd6f40d160dd3583d64c42ee1e0b450f310` plus the existing C increment. No A/B algorithm, shared domain class, shared ObjectDetector, main branch or trained artifact has been changed.

## Start on Windows (from this repository root)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Do not recreate an existing environment unnecessarily. Requirements now install this checkout, not the old remote skeleton commit. `requirements-windows-py312.lock.txt` records the tested Windows Python 3.12 dependency versions, excluding the optional browser QA tools; use it as constraints only on a compatible Python 3.12 Windows environment. The default requirements allow platform-appropriate versions, including Python 3.11 CI.

Double-click **Start Jerry-C.cmd** to launch the UI on `127.0.0.1:8503`. It does not terminate a conflicting service. Stop with Ctrl+C. The UI is not a permanently hosted website.

```powershell
.\.venv\Scripts\python.exe -B scripts/run_c_demo.py --output runs/jerry-c-demo-001
```

Choose a fresh output name each time. The demo creates `before.png`, `after.png` and `bundle/` containing `result.json`, `result.csv`, `annotated_before.png`, and `annotated_after.png`. Successful synthetic output contains four records and three changes. JSON and CSV bytes match the browser downloads on Windows; the newline regression has a test. Do not concurrently publish to the same destination.

## UI and data behavior
- Built-in synthetic demo, synthetic image uploads, and separate no-AI photograph inspection.
- Oriented original previews, annotated/original/wipe views, object and change counts, readable result table, change-only display filter and object evidence.
- Four downloads always contain the full result, regardless of display filtering.
- Input content, pair ID, source or config changes invalidate old results and downloads. A bad replacement cannot leave old downloads active.
- Canonical RGB uint8 pixels, original oriented dimensions, no application-side model resize. Transparent images are rejected until alpha policy is agreed. Encoded uploads are limited to 32 MiB and decoded images to 20 million pixels.
- Six canonical classes: bottle, cup, keyboard, mouse, book, scissors. One object per class is the current scope. B owns thresholds, matching and status semantics. No detections is not proof that a real scene is unchanged.
- The UI and CLI analyze and render the same decoded pixel snapshot. Model failure is not replaced by empty records.

## A-frame integration seam (not a trained model)
`deskcheck.application.frame_adapter.FrameDetectorAdapter(detector, frame_factory)` explicitly converts C's immutable frame to a supplied A/shared-frame factory using copied canonical pixels and `source_path=None`. It rejects resizing/pixel mutation and propagates detector failure. It never imports an unmerged branch automatically or changes the shared Path-based detector interface in main.

In addition to repository tests with a representative frame type, a local check loaded A's exact proposed ImageFrame from `feat/a-catchup-contracts` commit `29f65797c19379a678dc6ce1b5fc39d1b010ffd3`, Git blob `9ac1f67d2d07faa6bc1c1df8a34463a4d599d75e`. The bridge produced exactly the same four records as direct C fake detection through actual B. This proves frame conversion for that source revision, not A's future trained adapter, model preprocessing, or final interface approval.

## Verification obtained on the laptop
- New project-only Python 3.12 environment installed successfully; old environments unchanged.
- Complete repository suite: 257 tests passed (including upstream B and C UI/upload tests), zero skips in that run.
- Ruff passed after scoped C fixes. No lint exclusions were added to hide production failures.
- Actual headless Edge: HTTP health, actual file uploads, corrupt replacement/recovery, cleared stale downloads, wipe keyboard endpoints, no-AI photo mode, and four browser downloads byte-equal to CLI. Zero page exceptions and zero request failures. Mobile viewport/content were both 390 px.
- Test browser and test server were stopped after verification. Browser acceptance is distinct from Streamlit AppTest.
- The existing upstream GitHub workflow is kept byte-identical: Ubuntu/Python 3.11, `pip install -e .[dev]`, Ruff and pytest. The dev extra now includes Streamlit so UI/upload tests are exercised rather than skipped. Windows/Python 3.12 was validated locally. A proposed two-platform workflow is retained as documentation in `ci_matrix_proposal.yml.txt`, not as an active workflow: the current OAuth credential lacks workflow-write scope. A maintainer must separately approve/apply that optional workflow update. Consult actual PR checks for remote CI status.
- No clean installation on another person's computer, real-photo accuracy, real model training, stress benchmark, independent teammate approval or merge is claimed.

## Remaining team handoff
A still needs to provide the real dataset release/split, trainer, checkpoint identity, canonical class map, preprocessing/box-restoration settings and working detector. A/B/C must agree the shared interface and final alpha/threshold/NMS ownership. C real mode must validate checkpoint/config provenance before being enabled; it intentionally remains rejected here. B remains responsible for evaluation semantics; UI/CLI/runner agreement is covered on synthetic inputs.

Before merging, another teammate should review `Jerry-C`, read the CI outcome, and run the commands above. Do not merge A's unreviewed interface branch merely to make this demo work. The existing 14 November milestone can use this synthetic demo, clearly labelled; it is not completion of the real-model requirement.

## Contributions and evidence
Existing 10 November C code was carried forward; this increment adds the review UI, explicit frame bridge, upload/recovery and six-class tests, one-command demo, reproducible checkout requirements, CI and Windows byte-serialization regression fix. Tests and browser scripts were AI-assisted and executed; the individual course report must reflect the student's own work and verification.

Local raw receipts and screenshots are retained outside Git in the separate `COMP4423_Jerry_C_EVIDENCE_20261111` directory. The earlier failed browser check is preserved alongside the corrected passing run. They are not fabricated real-model metrics.

## Publication permission boundary
The first push was rejected solely because it modified `.github/workflows/ci.yml` without the required workflow scope. No credentials or scopes were changed. The publishable branch restores the exact upstream workflow and adds the required UI test dependency through ordinary project packaging. The workflow proposal and the original local commit are preserved; this does not claim the optional Windows CI job is active.
