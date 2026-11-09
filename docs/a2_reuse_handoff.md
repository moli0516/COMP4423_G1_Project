# A2 reuse handoff

## Labels

| Label | Meaning |
|---|---|
| REUSE | Extract compatible code, adjust integration dependencies, test |
| ADAPT | Preserve general logic; replace A2-specific assumptions |
| REFERENCE | Borrow design; no wholesale migration expected |
| PENDING REVIEW | Inspect dependencies/tests before choosing |

Staging under `references/` does not mean everything is REFERENCE-only.

## Selected files (paths relative to the source repository)

| Source | Owner | Classification / scope |
|---|---|---|
| scripts/update_images_csv.py | A | ADAPT: inventory and metadata retention |
| scripts/split_dataset.py | A | ADAPT: whole-group split, seed, dry run, report |
| scripts/validate_dataset.py | A | ADAPT: metadata/image/duplicate validation |
| src/data/dataset_catalog.py | A | ADAPT: manifest loading |
| src/experiments/polyu_lr_config.py | A | ADAPT: JSON/config validation structure |
| src/preprocessing/image_preprocessor.py | C | REUSE decode; ADAPT orientation/RGB; REFERENCE resize |
| src/preprocessing/preprocessing_config.py | C | PENDING REVIEW |
| src/utils/file_io.py | Shared | REUSE hash; ADAPT CSV/path/atomic-write integration |
| src/utils/serialization.py | Shared | REUSE NumPy conversion; add schema/finite checks separately |
| src/data/constants.py, src/exceptions/ | A | PENDING REVIEW dependencies |
| tests/splitting/, tests/preprocessing/ | A/C | PENDING REVIEW tests and fixtures |

No models, datasets, old split assignments, results, lockfile replacement or
`.bak` files should be migrated. App/web code is optional later work, not part
of this source snapshot.

## Contracts: agree together before integration

- A to B: per-image detections; shared class map; confidence; canonical-original
  image xyxy boxes. Supply image dimensions through the actual B interface.
- B to C: use implemented record fields/types, four statuses, missing-side box
  semantics and empty/error behavior. Do not invent a second event schema.
- C to runner: `compare(before_path, after_path, pair_id)` returns
  `{"pair_id": "...", "records": [...]}`.
- Raise explicit service failures. Never disguise failure as empty records.
- Same-class matching is not rejected merely for large displacement.
- Confirm movement threshold, confidence settings, NMS/duplicate ownership and
  orientation. These are not frozen by this patch.

B supplies exact imports/signatures, fixtures for four statuses plus empty/error
cases, and a runnable fake integration example. The prior integration guide must
be reconciled with main before treating it as authoritative.

## Acceptance / non-goals

This is a handoff/bootstrap patch, not integrated functionality. Existing B,
production CLI, training behavior and evaluation behavior must remain unchanged.
Keep existing working paths rather than renaming modules to match a diagram.
Inspect lint/test configuration before fetching: exclude the reference snapshot
from collection if necessary, without weakening production checks. Gitignore
alone does not exclude files from pytest or Ruff.

A and C adapt code through focused PRs and add DeskCheck-specific tests. Preserve
B runner's fresh-directory, validation and staged-publication semantics: a
single-file atomic writer is not a multi-file transaction.

## Experiment plan (proposed; team confirmation required)

One shared trainer, one frozen dataset split, separate configs. A baseline,
B augmentation, C resolution. Each member executes/checks/analyzes their run.
Validation selects settings; test is reserved for frozen final evaluation.
