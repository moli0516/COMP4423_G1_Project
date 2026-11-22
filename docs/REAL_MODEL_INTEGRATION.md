# Run A's E0 detector in DeskCheck

The 22 November C integration uses A's detector and shared image frame from commit `45f0e0a`. C's loader applies EXIF orientation and decodes RGB once. The three real entry points construct `ComparisonService.for_real(settings)` and pass the shared frame straight to `YoloDetector.predict`. B's filter, matcher, change rules and evaluator then consume the detections.

## Set up the released model

Use the team's [E0 checkpoint release](https://drive.google.com/drive/folders/1g5Ri_W7Q4R7QpWqL0dbFkxqp9dSpbily) and put `best.pt` in the checkout under `artifacts/`. Install the UI and vision extras in the project environment. For CPU-only installations, use the PyTorch CPU wheel appropriate to the platform. The verification environment used Python 3.12.14, Ultralytics 8.4.215, PyTorch 2.11.0+cpu and torchvision 0.26.0+cpu.

```bash
python -m pip install -e ".[ui,vision]"
python scripts/check_c_setup.py --profile ui --real-config configs/app.json
python scripts/predict_real_pair.py --config configs/app.json --before before.jpg --after after.jpg --pair-id desk-pair --output runs/desk-pair
```

Select a new output directory for each run. For the UI, set the environment variable to the same configuration, start Streamlit, choose the local YOLO checkpoint, then upload the two photographs. In PowerShell from the project root:

```powershell
$env:DESKCHECK_REAL_CONFIG = (Resolve-Path configs/app.json).Path
python -m streamlit run app/streamlit_app.py
```

In Bash:

```bash
DESKCHECK_REAL_CONFIG="$PWD/configs/app.json" python -m streamlit run app/streamlit_app.py
```

## Settings and ownership

| Setting | Effective value |
| --- | --- |
| A checkpoint SHA-256 | `82ed9f8569b705e7074f44d81006515da3e02dde3284a211c4a73a5f9d96c438` |
| C application config SHA-256 | `b6eec16b8d5289665035ce40d4bf961a8145b81fd30c1037b1fb4fec9d69798f` |
| CPU inference | confidence 0.20, backend NMS IoU 0.35, input size 640 |
| B movement rule | center displacement / oriented image diagonal strictly greater than 0.02152951369355096 |
| Canonical IDs in order | 0 bottle, 1 cup, 2 keyboard, 3 mouse, 4 book, 5 scissors |

A's standalone `resolved_config.json`, tuning invocation and top-level `artifact.json` agree on 0.20/0.35. The nested training snapshot in `artifact.json` lists the earlier 0.25/0.45 settings. The movement threshold comes from [B's freeze recorded in section 3c](https://github.com/moli0516/COMP4423_G1_Project/blob/45f0e0a/docs/a_contracts_freeze.md). `RealSettings` checks the checkpoint bytes and binds effective parameters to the exact configuration bytes on every construction.

A owns preprocessing, model loading, backend NMS and the output class/box mapping. The C integration branch proposes an A-side correction at the NumPy input boundary. Ultralytics takes NumPy images in BGR order. Converting the shared RGB snapshot there preserves the UI pixels and matches the backend's PIL input. The adapter checks source geometry and requires one mapping for each target class. [Ultralytics inference sources](https://docs.ultralytics.com/modes/predict/#inference-sources)

## Real checkpoint checks

On the two 2048 × 1536 P1 pilot originals, the unmodified A adapter and an independent PIL reference produced different predictions on both images. On the before image the counts were 9 and 10 respectively. With the channel correction the raw BGR input matched PIL on both photographs. After B's filter, all 12 selected boxes, classes and confidence scores matched the independent reference exactly in oriented pixels.

The app reported six objects and one change on P1. The bottle's normalized displacement was 0.04369357. Chromium uploads and the standalone CLI produced byte-identical JSON, CSV and both annotated PNGs. P1 was used for inference transport only. The separate external coffee identity check also produced four byte-identical exports, but predicted a mouse and a book as well as the visible cup. Entry-point agreement measures implementation consistency. Formal validation and E2 training have separate evidence requirements.

The two external photo pairs also ran through B's regular batch CLI. Across the three pairs, all 12 browser downloads matched the CLI bytes. Both external pairs' six batch event records matched the corresponding UI/CLI records, with identical config and checkpoint hashes. One external pair was the same coffee photograph supplied twice. The other was an unchanged ALOI mug photographed twice at a fixed camera position with different lighting. On that pair the model reported a removed book. The external run used only the independently captured ALOI pair for the no-change metric.

The combined C branch includes A's exact handoff as a Git ancestor. A's PR #8 remains subject to its own review. Formal v1 validation needs its manifests, validation photos and at least one annotated independently captured control. The 356,090,127-byte archive exceeded the available 268,435,456-byte transfer limit during this verification. The E2 960-pixel training comparison still requires the team's batch-size decision and training data.
