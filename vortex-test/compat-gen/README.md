# vortex-compat

Backward-compatibility testing for the Vortex file format. Ensures the
current reader can always decode `.vortex` files written by any older
released version.

See [DESIGN.md](DESIGN.md) for full architecture and design rationale.

## Quick Start

```bash
# Generate fixtures locally
cargo run -p vortex-compat --release -- generate --output /tmp/fixtures

# Check fixtures against current reader
cargo run -p vortex-compat --release -- check --dir /tmp/fixtures --mode exact

# Publish to S3 (requires AWS credentials)
python scripts/compat.py publish

# Check all published versions
python scripts/compat.py check

# Publish from a historical tag via git worktree
python scripts/compat.py publish --git-ref v0.62.0

# Local-only workflow (no S3)
python scripts/compat.py publish --store /tmp/compat-store
python scripts/compat.py check --store /tmp/compat-store
```

## Rust Binary: `vortex-compat`

A thin binary with two commands. It has **no** knowledge of versions, S3,
manifests, or orchestration.

**Output protocol:** progress/diagnostics to stderr, structured JSON to
stdout (`check` only).

### `generate --output <DIR> [--exclude <CSV>]`

Three phases:

1. **Setup** — run each fixture's `setup()` concurrently via
   `tokio::spawn_blocking`. Used by TPC-H and ClickBench fixtures to
   download external data.
2. **Build** — construct arrays in parallel threads via `std::thread::scope`.
   All must succeed before any files are written.
3. **Write** — serialize each fixture's arrays as a `.vortex` file, then
   write `fixtures.json` listing all generated files.

Output:
```
<DIR>/
├── fixtures.json
├── primitives.vortex
├── strings.vortex
└── ...
```

`fixtures.json` format:
```json
{
  "fixtures": [
    {"name": "primitives.vortex", "description": "..."},
    {"name": "strings.vortex", "description": "..."}
  ]
}
```

### `check --dir <DIR> --mode <MODE> [--exclude <CSV>]`

For each `.vortex` file in the directory:

1. Run `setup()` + `build()` to reconstruct expected arrays from current code
2. Read the file bytes and decode via `adapter::read_file()`
3. Combine chunks into `ChunkedArray` and compare with `assert_arrays_eq!`

JSON result to stdout:
```json
{
  "passed": ["primitives.vortex"],
  "failed": [{"name": "foo.vortex", "error": "mismatch at row 42"}],
  "skipped": ["old_fixture.vortex"]
}
```

Check modes:

| Mode | Extra files in dir | Missing fixtures |
|------|--------------------|------------------|
| `superset` (default) | Error | Skipped |
| `exact` | Error | Error |
| `subset` | Skipped | Error |

Use `superset` when checking old versions (they may be missing newly-added
fixtures). Use `exact` for the current version.

## Python Orchestrator: `compat.py`

### `publish [--git-ref <REF>] [--store <SPEC>] [--dry-run] [--exclude <CSV>]`

1. Detect version from nearest git tag at HEAD (or `<REF>`)
2. Generate fixtures (from current tree, or from a worktree at `<REF>`)
3. Fetch previous version's manifest, merge `since` values, enforce additive-only
4. Upload `.vortex` files + `manifest.json` to `v{version}/arrays/`
5. Update `versions.json`

### `check [--versions <CSV>] [--store <SPEC>] [--exclude <CSV>]`

1. Read `versions.json` from store
2. For each version, download `arrays/manifest.json` + all `.vortex` files
3. Run `vortex-compat check --dir <tmpdir> --mode superset`
4. Aggregate results, exit 1 if any failures

### `generate --output <DIR> [--git-ref <REF>] [--exclude <CSV>]`

Generate fixtures locally without publishing. Writes `.vortex` files and a
`manifest.json`.

### `list [--store <SPEC>] [--version <VER>]`

Without `--version`: print all version numbers.
With `--version`: print that version's `manifest.json`.

### `validate-manifest [--store <SPEC>]`

Walk all versions in order and verify no fixtures were removed between
consecutive versions (additive-only property).

### Store abstraction

| Spec | Type | Auth |
|------|------|------|
| `s3://vortex-compat-fixtures` (default) | S3 | Public reads (HTTPS), AWS creds for writes |
| `/tmp/compat` | Local directory | None |

### Git worktree workflow

`--git-ref` automates publishing from historical releases:

```bash
python compat.py publish --git-ref v0.62.0
```

This creates a worktree at the tag, builds the binary against that code,
generates fixtures, then cleans up.

### Environment variables

| Variable | Description |
|----------|-------------|
| `VORTEX_COMPAT_BIN` | Path to pre-built `vortex-compat` binary. Skips `cargo run`. |
