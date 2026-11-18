# C release compatibility and macOS handoff — 18 Nov 2026

Scope: focused C release-tool update requested by group leader. No modification to ComparisonService, B algorithms, A dataset splitter, YOLO inference or E2 training. The group lead reports Mac execution and code review completed; keep PR #9 Draft until final documentation review, required checks against main, and explicit merge approval.

## Python 3.13 reserved-path compatibility
The old source release verifier used PureWindowsPath.is_reserved(), deprecated on Python 3.13. The group leader observed 507 warnings on macOS Python 3.13.7. Direct replacement with os.path.isreserved() is not portable to macOS/Linux and does not support Python 3.11/3.12.

The portable _reserved_windows_component recognizes Windows DOS device basenames (CON, PRN, AUX, NUL, CONIN$, CONOUT$, COM1–9, LPT1–9, superscript digit variants, and names with extensions such as NUL.txt) on ALL host operating systems. Existing checks for traversal, reserved characters, trailing spaces/period, symlinks, case-colliding entries, missing/extra files and tampered contents remain.

## New Mac launcher
Start Jerry-C.sh is a macOS/Linux Bash script:
- Finds the project root even when called from another working directory.
- Requires this project's .venv/bin/python (no fallback to global Python).
- Validates DESKCHECK_PORT (default 8503), probes dependencies/config/port using check_c_setup.py, and only then launches Streamlit on localhost.
- Never installs dependencies, downloads a model, disables checks or terminates unrelated services.
- Is included as a required entry in deterministic C source ZIP with Unix executable permissions.

## Verified native Mac execution (group-lead report, 18 Nov 2026)

At C source commit `273d1007`, the lead ran a **fresh checkout and new venv** on macOS 26.1 / Python 3.13.7. Installation and pip check, targeted release tests with deprecation warnings treated as errors, full pytest, Ruff, setup checker, synthetic CLI (4 records / 3 changes), and real `bash "Start Jerry-C.sh"` startup passed; localhost root and health both returned HTTP 200. The earlier manual-photo UI inspection was not repeated. The lead also reports completing code review and a local A+B+C integration at `24738f5`; its **546 passed** and **34 coverage checks** belong to the earlier tested integration tree, not a new full-suite rerun of the final documentation/permission-only revision. This is not trained-model or E2 acceptance, nor an automatic merge authorization. See the lead's [PR #9 Mac receipt](https://github.com/moli0516/COMP4423_G1_Project/pull/9#issuecomment-6980402696).

## Fresh macOS installation and review

Use a NEW directory with a verified source SHA. Create a NEW environment instead of reusing another project's .venv. Run:

    python3 --version
    python3 -m venv .venv
    ./.venv/bin/python -m pip install -r requirements.txt
    ./.venv/bin/python -B scripts/check_c_setup.py --profile ui --json
    ./.venv/bin/python -B scripts/run_c_demo.py --output runs/fresh_demo_unique
    bash "Start Jerry-C.sh"

Open http://127.0.0.1:8503, select Synthetic demo and click Compare. Verify 4 records, 3 changes, and all four downloads (JSON, CSV, annotated before/after PNGs). For another free port: DESKCHECK_PORT=8504 bash "Start Jerry-C.sh". Stop only the test server you launched using Ctrl+C.

Record source SHA, OS, Python/pip versions, exact installation commands and exit codes, actual UI startup/browser observations, hashes and failure logs. Reusing a prior .venv proves source relocation only, not a fresh dependency installation.

## Targeted reviewer commands

    ./.venv/bin/python -m pytest -q tests/c_release/test_source_release.py tests/c_release/test_release_windows_portable.py tests/c_release/test_mac_launch.py
    ./.venv/bin/python -m ruff check .

## Evidence boundaries
- Offline Linux sandbox Python 3.13.5: 62 relevant pytest cases passed, including 507 successive release verifications with DeprecationWarning treated as an error. NOT macOS.
- Windows 11 / Python 3.12.10 clean-venv verification was completed at `273d1007`: full pytest 381 passed, 10 POSIX skips; real Windows launcher, synthetic CLI and actual Edge UI/export parity passed. See `docs/C_STANDALONE_ACCEPTANCE_20261118.md` for source ZIP and raw receipt paths.
- No real YOLO inference, held-out evaluation, or E2 run performed; A/B gates still apply.
- The reported A splitter 9 failures/25 passes are owned by A. This release change does not touch their code.
- The lead now reports code review completed; final documentation-delta review and required checks against `main` remain before any explicit merge approval. Passing tests do not by themselves authorize merging.
