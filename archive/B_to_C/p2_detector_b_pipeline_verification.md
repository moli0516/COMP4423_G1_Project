# P2: detector-boundary to B pipeline integration

## Scope

Tests use ImageFrame(source_path=None), a scripted ObjectDetector subclass, and
production DetectionFilter, ObjectMatcher and ChangeAnalyzer. No production code
is added or modified. The helper composes those APIs only inside the test module.

The inspected remote application/comparison_service.py is a NotImplementedError
stub (blob 840258e519b8da3972f4ff6e1562fa2530ff10bd). This patch does not claim
that C's service, loader, runner bridge or UI is integrated. No images are decoded,
no checkpoint is loaded and no detector AP/model accuracy is measured.

## Inspected API blobs

- detection_filter.py: 8826d4bf451666ea444d5b2653213e7bdbef97c0
- object_matcher.py: e821a122c255cd4ca58ac1793c9e510aa9dad875
- change_analyzer.py: 7b2e9cd4ed378ec40fea9ff97e6ccdd5622dc792
- detection.py: 716b1d0a28e647bb9554eefa960625b73b115edb

The inspected filter currently implements confidence >= threshold, highest
confidence per class, then ascending geometry and detection ID. Tests document
that implementation; they do not constitute team approval of the policy.
TEST_TAU=0.1 and confidence=0.5 are synthetic test settings, not release settings.
An 80x60 frame has diagonal 100, allowing exact below/equal/above test displacements.

## Coverage

11 collected cases (one test is parameterized over three values):
- all four statuses, absence of absent-both rows, sorted outputs;
- normalized movement below, equal to and above tau;
- confidence below/equal threshold and resulting side presence;
- duplicate selection by confidence, geometry and identifier;
- input-order invariance;
- far same-class matching without a spatial association gate;
- identical geometry with different classes remains removed + added;
- explicit empty outputs;
- out-of-frame rejection.

Every successful helper run verifies two frame-based predict calls, unchanged
input detection lists, unchanged filtered lists and byte-identical input arrays.
This checks these pipeline executions, not that every possible detector adapter
is incapable of mutating pixels. The fake returns scripted detections and is not
a test of semantic image understanding.

## Apply and verify

Use the existing local test/a-catchup-contracts branch, with P1 retained as its own
commit. Inspect git status before applying; do not overwrite existing local work.
Replace <patch-file> with the downloaded patch location.

```bash
git apply --check <patch-file>
git apply <patch-file>
python -m pytest tests/integration/test_detector_b_pipeline.py -ra --tb=short
ruff check .
python -m pytest -ra --tb=short
git diff --check
```

Generated Python syntax was checked. Repository pytest and Ruff were NOT executed
during patch preparation. Record actual branch/commit, Python version, commands
and outcomes after local execution. Do not label this patch passing before that.

After review and successful checks, stage only these two added files and commit
P2 separately. No push or GitHub changes were performed during preparation.
