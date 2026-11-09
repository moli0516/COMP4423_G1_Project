# Member C: start here

Proposed paths only. Framework choice is flexible; REST is not required.

```text
DeskCheck/
|-- src/deskcheck/
|   |-- application/
|   |   |-- image_pair_loader.py
|   |   |-- comparison_service.py
|   |   |-- result_renderer.py
|   |   `-- result_serializer.py
|   `-- ui/
|       `-- app.py
|-- examples/
|   `-- fake_comparison_demo.py
|-- tests/
|   |-- unit/
|   |   |-- test_image_pair_loader.py
|   |   |-- test_result_renderer.py
|   |   `-- test_result_serializer.py
|   `-- integration/
|       |-- test_comparison_service.py
|       `-- test_service_to_runner.py
`-- docs/
    `-- application_quickstart.md
```

## Source to destination

Extract image_preprocessor.py decode/orientation/RGB capabilities into the pair
loader. Inspect preprocessing_config.py and preprocessing tests before adapting.
Do not blindly call the old load(): it also resizes. Classification crop/resize
logic does not supply a detector box-restoration contract.

## First steps

1. Read B's actual API/types and runnable fixtures.
2. Agree canonical orientation and original-image coordinate conventions with A/B.
3. Implement pair loading/dimension checks without duplicating detector resize.
4. Fake detections -> real B filter/matcher/analyzer -> renderer/serializer.
5. Cover four statuses, missing-side boxes, empty scenes and explicit failures.
6. Implement compare(before_path, after_path, pair_id) for the runner.
7. Replace fake detector with A's adapter; rerun integration tests.

Do not recreate B's matching/status logic. UI and evaluation use the same
comparison implementation and settings. A trained checkpoint is not required
to start fake-first integration.
