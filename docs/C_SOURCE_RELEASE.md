# C source release and local setup

The archive contains the DeskCheck application, A's E0 detector, B's pipeline
and their source tests. The synthetic demo runs with the UI dependencies. For
real photos, install the vision extra and place A's released `best.pt` under
`artifacts/` in the extracted directory. Set `DESKCHECK_REAL_CONFIG` to the
absolute path of `configs/app.json` before starting the UI. The application
checks the checkpoint hash. [Real-model setup](REAL_MODEL_INTEGRATION.md)

On macOS/Linux, create a fresh `.venv` and run `bash "Start Jerry-C.sh"`
from the extracted directory. The launcher checks the environment and starts
the local UI. [macOS setup and launcher checks](C_RELEASE_MAC_REVIEW_20261118.md)

## Install a fresh copy

Use a new directory for a checkout or verified archive. Preserve an existing
working `.venv` rather than recreating it. From the new source root on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-windows-py312.lock.txt
.\.venv\Scripts\python.exe -B scripts/check_c_setup.py --profile ui --port 8503
```

The constraints record the existing tested Windows/Python 3.12 environment;
availability on a new machine/index is not guaranteed. For Python 3.11/Linux,
use platform-appropriate resolution (`python -m pip install -e '.[dev,ui]'`),
then run the same checker and tests. Choose the synthetic demo or select the
local E0 checkpoint in the UI for real inference.

The checker uses stdlib before dependency probes, verifies core/UI version ranges
from `pyproject.toml` (using `packaging`, supplied by the existing dev/UI set), runs
`pip check`, validates explicit fake config, and optionally tests port binding.
Exit 0 means selected setup checks passed; 1 means a diagnosed problem; invalid
CLI arguments exit 2. `--json` emits a structured report. Port availability is
only a momentary observation, not a reservation. An already running UI can own it.

`--real-config path/to/real.json` reuses `RealSettings.load` for config/checkpoint
integrity. Even `HASH_VALIDATED` always says `real_backend=UNVERIFIED` and
`real_inference=NOT_RUN`. A dummy file with the right hash can pass integrity;
this does not establish that it is a usable checkpoint. Failure never falls back.

## Launch, smoke and recover

Double-click `Start Jerry-C.cmd`; it diagnoses prerequisites/port 8503 before
starting Streamlit at `http://127.0.0.1:8503`. Keep its console open. Stop with
Ctrl+C. If connection fails, rerun the checker: a stopped server needs a launch;
an occupied port needs inspection of the existing service. For a different port,
run the checker with that port and launch Streamlit manually with matching
`--server.port`; do not stop an unknown service or change firewall settings.

```powershell
.\.venv\Scripts\python.exe -B scripts/run_c_demo.py --output runs/demo-NEW-NAME
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check --no-cache .
```

Inspect `bundle/result.json`, `result.csv` and both annotation PNGs: four synthetic
objects, three changes, original 160x120 geometry, same pair-local object IDs,
selected confidence, and post-B-filter diagnostic arrays. UI downloads remain
complete when the displayed table is filtered. PNG labels use compact IDs/score
with nearby free-space placement; class/status and full detector IDs are in the
table/JSON/CSV. Colors are added green, removed red, moved amber, unchanged blue.
Extremely small/crowded images can still clip/overlap labels; source boxes remain
unchanged. No new manual browser acceptance or real-photo result is implied.

## Build and verify without overwriting

Build from a clean `Jerry-C` or `feat/c-e0-detector-integration` worktree.

```powershell
.\.venv\Scripts\python.exe -B scripts/package_c_release.py --output runs/C_Source_NEW.zip
.\.venv\Scripts\python.exe -B scripts/package_c_release.py --verify runs/C_Source_NEW.zip
```

Builds use committed Git bytes. The archive includes source, scripts, tests,
documentation, the synthetic and E0 app configs, and the public experiment
configs. The team distributes weights and photos separately. The archive
has deterministic ordering/timestamps and per-file SHA256
plus full source commit. Repeat builds under the same compressor produce identical
bytes. This is integrity checking, not cryptographic proof of authorship or safety.
Dirty worktrees, symlinks, incomplete inventories and existing output names fail.
Verification works without Git and rejects missing/extra/tampered/unsafe entries,
Windows reserved names, case aliases and file/directory collisions.

Extract a verified archive to a fresh directory; create/install a new environment
there using the first section, then run setup and demo. A check using the existing
environment with extracted source proves source relocation, not clean dependency
installation on another computer. Keep that distinction in evidence reports.

Rollback: keep the prior checkout/commit and all generated outputs. Switch to a
separately prepared prior version only after stopping your own UI; no reset, clean,
deletion or rewriting other branches is required. A failed output is never reused;
diagnose it and choose a fresh name.

## Course submission

The course brief (pp. 4 and 6) calls for a local app using the submitted model
and settings, a group code ZIP and individual reports. This C source ZIP
contains the code component. The group submission also needs the released
weights, data and each student’s report. C has run E0 on the complete formal
validation split. The held-out test remains available for the team's final
evaluation. [Formal validation evidence](REAL_BATCH_EVALUATION.md#formal-v1-results)
