# A2 reuse handoff

This directory stages selected original A2 sources for adaptation to DeskCheck.
It is not a production package. Do not run the old dataset scripts against the
live DeskCheck dataset. Preserve fetched originals; adapt them in separate PRs.

## Populate the source snapshot

From the destination repository root:

```sh
python tools/import_a2_reference.py
```

The importer uses the pinned source revision recorded in its source, never a
moving branch. It downloads selected Python sources from GitHub, discovers
relative/`src` import dependencies and test support imports, records SHA-256
hashes, and publishes a fresh `original/` directory only after success.
Existing `original/` snapshots are refused, never overwritten.

Original source code is NOT embedded in this patch; the importer fetches it.
Network access is required for that step. Inspect the generated manifest before
merging. The pin was observed in the earlier review, not freshly validated here.

See ../../docs/a2_reuse_handoff.md and the separate A/C start guides.
