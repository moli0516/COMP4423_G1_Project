# Jerry-C — Part C completion and handoff

**Prepared:** 2026-11-11 (HKT)

**Personal branch:** `Jerry-C` (never `main`)

**PR:** https://github.com/moli0516/COMP4423_G1_Project/pull/9 (Draft; review and merge are separate)

**Owner scope:** Part C application and UI; A/B retain ownership of their respective pipeline and evaluation semantics.

## Independent C deliverables

| Capability | Exact implementation / proof | Boundary |
|---|---|---|
| Read before/after images in original orientation | `image_pair_loader.py`; explicit EXIF/RGB, matching dimensions, 20M pixel cap | Alpha rejected pending team policy; no inference preprocessing resize |
| Compare detections and classify changes | `comparison_service.py` calls unchanged B filter, matcher, analyzer | Synthetic-only detector; no trained weights |
| Demonstrate all four status types | `demo_pair.py`, `test_release_core.py` and independent `test_demo_oracle.py` | The fixture is not real-world accuracy evidence |
| Export a complete comparison | `result_serializer.py`, `result_renderer.py`, `predict_pair.py` | Fresh output directory; input images preserved |
| Display and inspect C results | `app/streamlit_app.py`, `presentation.py` | Synthetic results labelled; real-photo inspection does **not** run AI |
| Recover from UI/input errors | `test_release_ui.py` and real Edge acceptance receipts | Replacing uploads invalidates old results/downloads |
| Prepare A-frame conversion seam | `frame_adapter.py`, `test_release_core.py` | Not an implementation/approval of A's real YOLO detector |
| Refuse false demo success | `demo_verifier.py`, `test_demo_truthfulness.py`, `test_demo_oracle.py` | Checks independent expected fixture boxes/distances and actual annotation pixels |
| Same result in each entrance | `test_three_entrypoint_parity.py` checks Streamlit, CLI, ComparisonService, actual B EvaluationRunner on same pair | Uses synthetic fixture and separately authored manual event truth |
| Reproducible setup | `requirements.txt`, `requirements-windows-py312.lock.txt`, `Start Jerry-C.cmd`, `scripts/run_c_demo.py` | Windows verified locally; Linux/Python 3.11 CI separate |
| GitHub handoff | Draft PR #9 on own branch, not merged | Team review required before main |

## Running your own copy (Windows PowerShell)

From the local `COMP4423_Jerry_C_20261111` worktree, keep the existing
`.venv` if it works. To verify without changing code:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B scripts/run_c_demo.py --output runs/jerry-check-NEW-NAME
```

Use an unused output directory. Start `Start Jerry-C.cmd` to open the local
Streamlit UI at `127.0.0.1:8503`. Stop the UI with Ctrl+C. The fake demo
is deliberately **not** a test of real photographs or a trained model.

## External prerequisites: not C-complete yet

1. **A → C:** actual annotated DeskCheck data release, frozen group split and class map, YOLO training implementation and reproducible checkpoint, model-specific preprocessing and un-letterboxed original-image boxes.
2. **A/B/C interface approval:** final `ImageFrame`, `ObjectDetector.predict` input, alpha policy and confidence/NMS responsibility. A's PR #8 proposes an ImageFrame change; this branch does not merge it or assume it has been approved.
3. **B → C:** frozen movement threshold and exact result/evaluation definitions; real validation/test fixtures. C's current `configs/fake_app.json` uses **0.05 only for synthetic demos**. A's PR describes a proposed pilot **0.02**. Do not copy either value silently into a real test.
4. **Real C inference acceptance:** with A's deliverables installed, perform new tests on real before/after photos, ensure UI and CLI show the same boxes/changes and the B evaluator gets equivalent records; verify weights/config provenance and error cases.
5. **Experimental E2 (resolution)** if confirmed as C's assigned experiment: A's common trainer and frozen split are required. The proposed `configs/experiments/resolution.json` in A PR #8 is only a stub, not a completed C training run.

## School work and personal evidence

The **11/14** local demo milestone and **12/11 23:59 HKT** course deadline come from the existing local `PLAN.md`; reconfirm against official course notices if the timetable changes. For the personal report, cite personally examined code, test commands, screenshots, PR review and actual limitations. Do not present synthetic results as real precision/recall, a trained model, or independently written human work.

## Before marking the group deliverable complete

- [ ] A and B have reviewed PR #9 and agreed the cross-module interfaces
- [ ] Model weights and real dataset release are available with hashes and configuration
- [ ] Real photos are tested in both UI and CLI; matched B records and event evaluation agree
- [ ] Selected settings are frozen before the final test set; no leakage
- [ ] A different group member performs a clean installation/demo using the actual submission package
- [ ] PR is merged only after code review; this document does not authorize merging
