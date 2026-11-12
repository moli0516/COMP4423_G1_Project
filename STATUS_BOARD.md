# Jerry-C sprint status

Goal: a locally runnable C application with consistent visual/export evidence and
reproducible setup/release. Reader: C user and A/B reviewers. No substitute detector,
trainer, invented annotations, or claim of real inference.

Updated: 2026-11-12 19:22 HKT. Application/test server running: no (owned server stopped).
Baseline: `e896ace8ad5d14ae7c78a5698518b24f8c5beaaa`; worktree initially clean.

Usable result: consistent JSON/CSV evidence + ID/confidence annotations/table,
read-only setup/port/artifact checker in launcher, committed-source ZIP builder/verifier.
Local verification: 355 pytest passed; Ruff and pip check passed; synthetic demo
4 records/3 changes; HTTP root/health 200 and negative setup paths passed.
Read-only independent review: 2 rounds; identified issues corrected, no open blockers.
Final commit/publication/archive/CI evidence is recorded after execution in
`runs/c-sprint-verification-20261112-01/CODEX_RETURN_PACK.md`.

Next external action / owner / gate: A supplies its reviewed adapter/runtime/checkpoint
handoff with matching config and SHA256 before C real-photo acceptance. No model
download, training or real inference is running. Need user action: none for this increment.

Acceptance / short execution plan:
1. Serializer, presentation and renderer: reject inconsistent retained evidence;
   show correspondence IDs and confidence. Verify focused regressions and demo.
2. `scripts/check_c_setup.py` and Windows launcher: diagnose dependencies,
   configuration/artifact errors and occupied port without inference/install.
3. C source release builder: fresh output, explicit source inventory/integrity,
   no private data/checkpoints/venv; extracted-source demo verification.
4. Full pytest, Ruff, pip check, live HTTP health, read-only independent review;
   publish only Jerry-C and update Draft PR #9 with actual CI.

| External blocker | Owner | Default action | Review deadline |
|---|---|---|---|
| PR #8 shared frame/converter unmerged | A/B | Keep temporary C frame seam | Before production contract integration |
| Final adapter/runtime/checkpoint/data absent | A | Keep real inference UNVERIFIED | Before real-photo acceptance |
| Movement threshold not approved | B | No scientific freeze or held-out evaluation | Before real acceptance |
| E2 trainer is placeholder; 16 vs 8 batch confound | A/C | No training; reuse A trainer when runnable | Before any E0/E2 run |

These gates never become approval merely through elapsed time. Synthetic readiness
and a C source archive cannot satisfy the final course submission on their own.
