# Implementation plan

## Skeleton conventions
One class per file. Domain contracts are shared. JSON configuration is used initially to avoid a mandatory YAML dependency; this is an explicit simplification of the earlier design proposal. Detector API initially accepts a Path; the adapter must decode consistently and must never use filename text as a feature. An ImageFrame contract can replace this through an agreed versioned interface change. Current matcher tuples and event dictionaries are provisional until ObjectMatch/ChangeRecord are introduced.

## A: data and detection
Implement DatasetAuditor, GroupSplitter, YoloDatasetExporter, YoloDetector and ModelTrainer. Record image/pair/group IDs, boxes, physical correspondences and events. Split related groups near 60/20/20 with no leakage; cover classes/statuses. Train/fine-tune all six classes and compare a documented baseline. Save logs, resolved settings, dependencies and checkpoint.

## B: matching and evaluation
Implement DetectionFilter, ObjectMatcher and ChangeAnalyzer. Retain at most one prediction/class after documented filtering. Match same class regardless of displacement. Compute center displacement divided by image diagonal. moved iff d > approved tau; equality is unchanged. added/removed use side absence; absent both creates no record. Approve/freeze tau before test annotation. Add all four statuses, exact boundary, far match, empty scenes and duplicate cases to tests. Implement detector AP, per-status PR/F1, macro-F1, change-only results and independent no-change pair false alarms.

## C: integration and UI
Implement AppConfig, ImagePairLoader, typed records, ComparisonService, ResultRenderer and ResultSerializer. UI, CLI and evaluation share settings and service. Reject dimension mismatch, corrupt images, missing checkpoint and incompatible class mapping. Add predict_pair.py, real application integration and clean-install checks.

## Dataset contract
Original-image xyxy coordinates; one instance/class/image; same physical object if a class occurs in both images. Each pair uses fixed camera and equal dimensions. Annotate all target objects and events; include genuine separate no-change captures and lighting variation. Inference sees no labels or collection notes. No fixed dataset size is prescribed; justify diversity/support.

## Experiment policy
Proposed model/settings are not measured optima. Select model/confidence/NMS on validation, freeze before final full test evaluation; do not revise rules using test results. Event-localization scoring protocol must be documented separately from inference matching. No failed sample is silently excluded.

## Git policy
After this initial commit, use feature branches and PRs with a non-author review and passing CI. Branch protection requires manual repository administration. Keep raw data and weights outside ordinary Git. Save contribution/AI-verification records; individual reports must be independently authored.
