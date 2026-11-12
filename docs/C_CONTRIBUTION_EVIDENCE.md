# C factual contribution/evidence index

This is an index of code and verification, not an individually authored academic
report or a claim that the student personally wrote every line. Implementation and
tests in this chat are AI-assisted. Each student must inspect/run/explain the work,
record what they actually did, and write their own Statement of Contribution and
answers to the supplied report-template questions.

| C work | Inspectable source / test | Evidence boundary |
|---|---|---|
| Source-bound real settings | `real_settings.py`; commit `365cf13`; `test_real_settings_provenance.py` | Artifact/config consistency; not real inference |
| Real batch wiring to actual B runner | `scripts/evaluate_real_pairs.py`; commit `e317d5e` | STUB_ONLY; B metrics unchanged |
| Retained selected detections | commit `29af104`; `test_retained_detections_contract.py` | Post-B diagnostics; not detector AP |
| Oriented frame seam compatibility | commit `e896ace`; `test_frame_boundary_compatibility.py` | EXIF/pixel/geometry and representative A shape; no merged A adapter |
| Export/visual evidence consistency | `result_serializer.py`, `result_renderer.py`, `presentation.py`; `test_result_evidence_consistency.py` | Synthetic ID/box inventory checks and presentation; B classifications untouched |
| Setup/launch diagnosis | `scripts/check_c_setup.py`, `Start Jerry-C.cmd`; `test_setup_check.py` | Read-only imports/pip/config/port/hash checks; never loads model |
| Reproducible C source packaging | `scripts/package_c_release.py`; `test_source_release.py` | Allowlisted committed source/integrity/relocation; incomplete final submission |

For exact current file commits use `git log -1 -- <path>` and the source ZIP's
`RELEASE_MANIFEST.json`. Local full test/CLI/live-health outputs go in a fresh
ignored `runs/` verification directory; the final CODEX_RETURN_PACK supplies its
exact path, final SHA and CI links. Do not cite a test count as a scientific result.

Observed GenAI limitations in this increment: existing transport checks validated
arrays separately but missed contradictions with event records; long PNG labels
overlapped in the tiny demo despite passing text-capture tests. Negative export
tests plus actual generated-image inspection exposed these defects; serializer
cross-checks and compact label placement corrected them. A read-only reviewer
also identified the stale oracle fixture and cwd-dependent repair command.

Not established: independently captured/annotated data, actual model training or
inference, detector AP, accepted movement threshold, A/B review/merge, a clean
install on a teammate's computer, or a newly performed manual browser walkthrough.
