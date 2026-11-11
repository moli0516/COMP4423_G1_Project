# C fake-first integration plan

Base: upstream main 146c8cd6f40d160dd3583d64c42ee1e0b450f310.
Branch: feat/tam-c-fake-first-20261110. Only this new worktree is a writer.

Scope: integrate image loading, actual B filtering/matching/change analysis, strict serialization, visualization and a synthetic-only CLI/UI. No model, dataset training, PR merge, push or modification of the original dirty checkout.

1. Add failing C tests for all four statuses, empty/one-sided scenes, EXIF, RGB, missing/corrupt/mismatched inputs, finite serialization, no input mutation and EvaluationRunner.
2. Implement C-local in-memory ImageFrame/loader and explicit fake configuration. Until A/B freeze alpha policy, reject transparent input rather than silently compositing it. Preserve canonical oriented dimensions without resizing.
3. Inject a pixel-only fake detector into ComparisonService, using real upstream B modules unchanged. Keep the current upstream Path-based ObjectDetector unchanged; this increment does not claim the proposed shared ImageFrame adapter is merged or compatible.
4. Add strict JSON/CSV, copied-image rendering, fresh-directory CLI publication and a clearly synthetic-only Streamlit UI.
5. Run red/green C tests, CLI/demo and UI checks. Run upstream pytest/Ruff only if available; otherwise report NOT RUN with exact dependency blocker. Preserve original checkout/untracked file hashes.

Training remains blocked on a real dataset, usable YOLO export, trainer, A adapter and checkpoint. Synthetic passes are software integration evidence only.
