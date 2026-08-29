# Push-reader upstream

This crate was imported exactly from the `vortex-morsel` tree at
`ae8b9800409a60d1ceebb2b8181a144581a0cc45` on `codex/morsel-push-optimized`.
The exact import is commit `e592bf4269add47dfb3994d105e55652cae30503`. It is deliberately
separate from `vortex-morsel`, which contains the pull reader.

The currently integrated upstream tree is
`ae8b9800409a60d1ceebb2b8181a144581a0cc45:vortex-morsel`.

Use the checked helper to verify the import boundary and its integration overlay:

```bash
vortex-morsel-push/scripts/refresh-upstream.sh
```

Dry-run a refresh before applying it:

```bash
vortex-morsel-push/scripts/refresh-upstream.sh --check <new-push-commit>
vortex-morsel-push/scripts/refresh-upstream.sh --apply <new-push-commit>
```

The helper refuses a dirty push subtree. After applying, resolve only integration-overlay
conflicts, run the crate checks, and update `CURRENT_UPSTREAM_REF` in the helper and the currently
integrated upstream tree recorded above. Keep the exact-import provenance unchanged.

The allowed integration surface is `Cargo.toml`, `README.md`, `UPSTREAM.md`, the two evaluation
binaries, `src/build.rs`, `src/driver.rs`, `src/executor.rs`, `src/lib.rs`, and
`src/nodes/conjunct.rs`, plus this documentation and its refresh helper. Anything else is
unexpected upstream drift and must be reviewed before expanding the allowlist.

The shared array, buffer, and mask prerequisites were imported separately. They intentionally
retain the pull branch's newer sparse `intersect_by_rank` optimization instead of replacing it
with the older implementation at the push source commit.
