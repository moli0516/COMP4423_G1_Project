# Member A: start here

Proposed paths only. Reconcile with current main; do not rename working code.

```text
DeskCheck/
|-- src/deskcheck/
|   |-- data/
|   |   |-- dataset_inventory.py
|   |   |-- dataset_catalog.py
|   |   |-- dataset_validator.py
|   |   |-- dataset_splitter.py
|   |   `-- yolo_exporter.py
|   |-- training/
|   |   |-- training_config.py
|   |   `-- trainer.py
|   `-- detection/
|       `-- yolo_detector.py
|-- scripts/
|   |-- update_images_csv.py
|   |-- validate_dataset.py
|   |-- split_dataset.py
|   |-- export_yolo_dataset.py
|   `-- train.py
|-- configs/experiments/
|   |-- baseline.json
|   |-- augmentation.json
|   `-- resolution.json
`-- tests/
    |-- unit/
    |   |-- test_dataset_validator.py
    |   |-- test_dataset_splitter.py
    |   |-- test_training_config.py
    |   `-- test_yolo_detector.py
    `-- integration/
        `-- test_detector_to_b.py
```

## Source to destination

- update_images_csv.py -> dataset_inventory.py + thin CLI wrapper.
- split_dataset.py -> dataset_splitter.py + thin CLI wrapper.
- validate_dataset.py -> dataset_validator.py + thin CLI wrapper.
- dataset_catalog.py -> DeskCheck catalog.
- polyu_lr_config.py -> training_config.py (replace classifier settings).
- Detector exporter/trainer/adapter are new work, not supplied by this patch.

## First steps

1. Agree image/pair/capture-group/annotation manifests and class map.
2. Adapt inventory/validator on tiny disposable fixtures; protect stable IDs,
   annotations and pair references when updating records.
3. Remove binary labels, geographic verification and single-label-group rules.
4. Assign each capture group once. Related images and shared-image pairs must
   not cross splits. Target 60/20/20 without breaking groups to hit ratios.
5. Check all object classes in each split; status coverage in validation/test;
   independently captured no-change counts. Reject impossible coverage.
6. Freeze movement definitions before test annotation; do not use test model
   results to redraw splits or select rules.
7. Freeze release/manifests; build common trainer and detector output adapter.
8. Fine-tune a baseline with actual parameter updates. Save resolved config,
   logs, environment, checkpoint identity and validation-selection policy.

You own pipeline infrastructure, not everyone else's experiments.
