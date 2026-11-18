# Member C — standalone acceptance and remaining external gates (18 Nov 2026, HKT)

**Status (updated 18 Nov 2026):** C-owned synthetic application, Windows clean-install/release, and portable shell tests have evidence. The lead has now reported native Mac execution PASS and code review completed. Final documentation-delta review, required checks against `main`, and explicit merge approval remain open; PR #9 stays Draft. **Real-model accuracy and E2 training remain unverified/not run.** This is a reviewer index, not the course submission.

## Exact source and team boundaries

- Verified C implementation source: Jerry-C commit f20869668791a0d4e7493b87c590a0cbdd35db66.
- GitHub Linux CI: https://github.com/moli0516/COMP4423_G1_Project/actions/runs/44242055390 — Python 3.11.18, **391 passed**, Ruff PASS at f208696.
- Windows full suite: **381 passed, 10 POSIX-only skips**, Ruff PASS, pip check PASS in fresh Python 3.12.10 venv extracted from C-source commit 273d100.
- Commit f208696 only changed Start Jerry-C.sh Git mode 100644 to 100755. Source-file bytes are 103/103 identical between the 273d100 and f208696 source ZIPs; mode-only claim is independently verified, not inferred from CI.
- The leader reviewed RealSettings provenance and retained-detection consistency at 9372416. Do not reopen those decisions without a new failure.
- A still owns training/detector/preprocessing/letterbox/NMS/class map, B still owns filter/matcher/status/evaluation. No A/B code was changed in this increment.

## Genuine clean Windows dependency and launch check

Machine: Tamslaptop, Windows 11 Home, Python 3.12.10. A new source directory and a brand-new .venv were created; this was **not** source relocation with an existing interpreter.

- Source archive C_SOURCE_273d100.zip SHA-256: 22dcc3ba5dece7e15272020b585e684a76bae8d6f18a67fbdd7a24757b0d6dbd.
- pip install -r requirements.txt -c requirements-windows-py312.lock.txt exited 0; pip check and check_c_setup --profile ui returned PASS.
- run_c_demo produced 4 synthetic records and 3 changes; the separate UI/CLI/B runner parity test passed.
- Real Streamlit process started in the NEW venv, root and /_stcore/health returned HTTP 200; the test-owned server alone was stopped.
- Actual Microsoft Edge headless clicked Compare in the new app. JSON, CSV and both annotated PNG downloads were byte-for-byte identical to fresh CLI outputs; zero browser page exceptions.
- Actual Windows Start Jerry-C.cmd, invoked from the NEW venv/source archive, returned root and health HTTP 200 and stopped only its own process tree.
- A separate read-only checker rehashed 33 evidence claims against the saved source ZIPs, raw logs and browser downloads: 33 PASS.

Raw evidence (local to Tamslaptop): %USERPROFILE%\Documents\COMP4423_C_RELEASE_COMPAT_20261118\
- WINDOWS_FRESH_EVIDENCE_273d100_20261118.zip (SHA-256 d447c300101c104368e04d77e75bd7a09b87628a9aacf43175c8f58848aa820d)
- fresh_full_pytest.txt / fresh_full_ruff.txt / fresh_pip_install.txt / fresh_setup.json
- fresh_streamlit_receipt.json / fresh_windows_cmd_launcher_receipt.json
- fresh_edge_20261118/acceptance.json and ui_after_compare.png
- VERIFY_WINDOWS_PROOF_READ_ONLY.ps1 (separate 33-check verifier)

## Independent C-owned portable/release validation (no extra model tokens)

1. Git Bash on Windows: 10/10 isolated launcher cases PASS (success, setup failure does not launch, invalid ports, missing venv and shell syntax). Receipt: git_bash_launcher_20261118/POSIX_GIT_BASH_STUB_VERIFICATION.json. Uses a **stub** interpreter; this is **not macOS** or genuine Linux Streamlit execution.
2. C source-release CLI and archive safety: 18/18 checks PASS. Archive verified **at f208696**: C_SOURCE_f208696.zip, SHA-256 3e9e1ae748a5146216378961d4b2b04c022e3236fcbacb3c30b5023321b6eda9, 103 files, all allowed and hash-verified, no datasets/photos/checkpoints/environments. Existing-output path refused; tampered ZIP rejected. Receipt: release_negative_20261118/RELEASE_SECURITY_ACCEPTANCE.json.
3. Stronger adversarial tests with internally CONSISTENT forged manifests: 5/5 PASS. A legal new source file was accepted as the positive control; DOS-reserved name, case-colliding paths, symlink and path traversal were rejected. Receipt: release_adversarial_v2_20261118/ADVERSARIAL_RECEIPT.json. Unlike an inventory-mismatch-only test, the forbidden cases included valid manifest inventories and matching hashes, so the checks specifically exercise safety constraints.
4. Earlier isolated Linux Python 3.13.5 compatibility tests: 62 relevant tests passed and 507 repeated source-archive verifications with DeprecationWarning escalated to error passed. This is not evidence of **macOS Python 3.13.7** execution.

These supplementary standalone tests and logs are separate from the committed repository test suite. All original source files and previous artifacts were preserved.

## What is NOT done by C, and who owns the next gate

- **Completed by group lead:** native macOS 26.1 / Python 3.13.7 fresh checkout/new-venv checks at `273d1007`, real `Start Jerry-C.sh` startup, and code review. The 546-passed A+B+C integration result belongs to the earlier tested integration tree, not a rerun for final documentation changes. See docs/C_RELEASE_MAC_REVIEW_20261118.md and https://github.com/moli0516/COMP4423_G1_Project/pull/9#issuecomment-6980402696. **Still open:** lead's final doc-delta check, required checks against `main`, and explicit merge authorization.
- GitHub workflow permission: the prepared optional macOS CI workflow could not be pushed because this account's OAuth token lacks the workflow scope. It remains **local-only** under backup ref local-only-macos-ci-permission-blocked-20261118; do not bypass the permission or alter main.
- A: final merged shared ImageFrame and trainer (code now exists in unmerged PR #8), production detector, accepted trained checkpoint, source/config hashes and frozen split. The raw E2 config still needs A-approved inheritance resolution.
- B: accepted event/evaluation semantics, final batch controls and held-out event metrics. A splitter's reported 9 failed / 25 passed train-coverage regressions are not a C defect.
- C **after** A/B handoff: actual YOLO UI/CLI/runner parity and E2 resolution comparison using A's trainer under agreed effective-batch controls. Both remain NOT RUN.
- Student: personally authored individual COMP4423 report, GenAI limitations and Statement of Contribution; source evidence and an editable working draft exist, but an AI working draft is not the finished individual submission.

## C ownership acceptance boundary

The independent C release/app tasks that do not require the group leader, A or B have executable evidence and no newly identified failing C-owned test. Preserve PR #9 as **DRAFT**. Stop adding features only to increase the test count; proceed on a **specific** new reviewer defect or the agreed model handoff. Do not merge, push to main, update A/B branches, replace teammate artifacts or run E2 on synthetic checkpoints.
