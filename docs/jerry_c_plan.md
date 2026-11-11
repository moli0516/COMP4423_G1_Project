# Jerry-C implementation plan ? 2026-11-11

## Approved scope and design
User requested completing all presently feasible Part C work on a personal branch named exactly Jerry-C. Preserve existing worktrees; no main writes or merge. Only publish Jerry-C after local verification. The real detector, training and actual-photo AI accuracy remain outside the available evidence.

Reuse the 10 November C service with real B algorithms. Bring the older demo's useful interaction patterns (previews, human-readable table, change filtering, per-object detail, wipe, recoverable uploads, downloads) to this worktree rather than copying its incompatible B implementation.

Do not change shared A/B domain interfaces or silently adopt an unmerged branch. Add a C-local explicit frame adapter seam with supplied factory, validating dimensions and snapshot ownership. Keep fake configuration visibly fake and real mode rejected until actual A deliverables exist.

## Global constraints
- Branch: Jerry-C. Original source snapshot: main 146c8cd + saved C working changes.
- Only this worktree is writable. Preserve source SHA256 receipt outside repository.
- Python >=3.11; no model download, no paid API, no training or invented real metrics.
- UI/CLI/runner reuse ComparisonService and the same resolved settings.
- Outputs must use fresh directories; no overwrite of inputs or outputs.

## Review focus
1. Bad/alpha/mismatched/oversized uploads must recover and remove old downloads.
2. Filtering the displayed rows must not filter exported results.
3. Image snapshot must match inference and rendering; external frame conversion must preserve pixels.
4. Different pair IDs, configuration or inputs must invalidate previous UI results.
5. Full pytest must include upstream tests and UI; clean install must not pull an old editable Git revision.

## Task 1 ? environment and regression baseline
- [ ] Install dev/ui dependencies into a new project-only .venv; do not alter old environment.
- [ ] Run full pytest, Ruff and pip check; record baseline defects without hiding upstream failures.

## Task 2 ? C boundary and publication hardening
Files: application/frame_adapter.py, application/result_renderer.py, tests/c_release/test_boundary.py.
Interfaces: FrameDetectorAdapter(detector, frame_factory).predict(CImageFrame) -> list[Detection].
- [ ] Add tests first: preserve canonical arrays, no path reopen, no input mutation, reject invalid conversion; six classes and four states; reversed inputs; finite evidence.
- [ ] Observe RED then implement minimal C-only bridge and any demonstrated correctness fixes.
- [ ] Re-run tests and full regression.

## Task 3 ? unified user interface
Files: application/presentation.py, app/streamlit_app.py, tests/c_release/test_ui_release.py.
Interfaces: comparison_rows(result, changes_only=False) returns display dictionaries without modifying result; comparison_wipe(before, after, percent) returns a fresh PIL image.
- [ ] Add tests for summary/table/filter/details/downloads, same-snapshot rendering, input invalidation and error recovery; observe RED.
- [ ] Implement preview + annotated/original/wipe views, four downloads and explicit fake limitations. Retain old UI test behavior where possible.
- [ ] Test with AppTest and an actual local browser if an available browser automation runtime supports it.

## Task 4 ? reproducibility and delivery
Files: requirements.txt, .github/workflows/ci.yml, README.md, docs/jerry_c_handoff.md, scripts/run_c_demo.py.
- [ ] Use current checkout installation, not old e3a6417 editable remote requirement; CI includes dev and ui.
- [ ] Fresh installation, full tests/lint/pip check, CLI four outputs, live server health and browser acceptance.
- [ ] Verify originals unchanged, review scoped diff, commit only C-owned changes, push Jerry-C only and open a draft PR (never merge). Record actual CI result or pending state.

## Decisions
The user already approved execution; no repeated design confirmation. New isolated worktree uses a snapshot of uncommitted C work so no original working changes are lost. A shared ImageFrame merge is not claimed by a C-local adapter test. Self-review is not an independent reviewer.
