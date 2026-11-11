# P1: annotation-to-runner verification

This patch adds tests only; it does not modify production code or approve PR #8.

## Inspected dependency baseline

PR #8 API head when prepared: `29f65797c19379a678dc6ce1b5fc39d1b010ffd3`.
Inspected Git blob IDs:
- evaluation_runner.py: `5a2607be32f049f33e4fa68e86d95c2155502fc5`
- annotation_converter.py: `5098d2a2c057dbb6971b8f207516d2560b9f74dc`
- event_evaluator.py: `da038d95ca354e63214c1cb23f2952d1bf2a8db2`
- no_change_evaluator.py: `81f85a608d8c820355b68240598229c2d7954844`

Apply to an isolated branch/worktree containing A's catalog, validator and converter.
Do not cherry-pick A into main merely to run this test. Record the actual A and B
commit SHAs used for local verification; these blob IDs are not a runtime provenance claim.

## Scope

Synthetic JSONL + full train/val/test manifest -> catalog -> successful semantic
audit with explicit test-only tau -> existing convert_dataset -> real runner and
evaluators -> serialized outputs. Only validation pairs reach the fake service.
Predictions are independently specified, not derived from ground truth conversion.
All six classes, all four statuses, two eligible validation controls and an explicit
empty scene are covered. A 24-pixel displacement on an 800-pixel diagonal gives
0.03, strictly above the test-only tau 0.02. This is not a project tau decision.

Six tests cover perfect predictions, a wrong status and 1/2 false-alarm rate,
service failure without publication, invalid annotation eligibility, wrong runtime
eligibility, and rejection of an empty scene as an eligible annotated control.
Empty scenes remain valid event inventory entries, but current runner policy
requires nonempty unchanged annotations for eligible no-change controls.

No images are decoded and no model is called. This does not verify P2, exporter,
splitter search, trainer, detector AP, or UI. The fake service intentionally accepts
paths without opening them, matching the runner's injection boundary.

## Verification

The generated Python was syntax-checked during preparation. Pytest and Ruff were
NOT run against the repository in this environment. Do not describe this patch as
passing until the commands below succeed in your clean integration environment.

From the repository root, apply the attached patch, then run:

```bash
git apply --check <patch-file>
git apply <patch-file>
python -m pytest tests/integration/test_annotation_runner_handoff.py -ra --tb=short
ruff check .
python -m pytest -ra --tb=short
git diff --check
```

`<patch-file>` is a placeholder for the downloaded patch. Before applying, inspect
`git status` and confirm the two added paths do not already contain local work.
Use the project's clean CI-compatible environment and declared dev dependencies.
Do not weaken audit assertions or modify the runner merely to make these tests pass.
