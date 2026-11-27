# vortex-compat

Backward-compatibility testing for the Vortex file format. Covers both
directions: ensuring the current reader can decode `.vortex` files written by
older releases, and ensuring the current writer targeting any supported frozen
`core` edition can be read by that edition's oldest release.

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

# Generate fixtures targeting an older edition
cargo run -p vortex-compat --release -- generate --output /tmp/fixtures-core2025.05.0 --edition core2025.05.0

# Check all frozen core editions with each edition's oldest vortex-data reader
uv run scripts/compat.py --profile ci check-editions

# Publish current tree using a historical tag for version detection
python scripts/compat.py publish --git-ref v0.62.0

# Local-only workflow (no S3)
python scripts/compat.py publish --store /tmp/compat-store
python scripts/compat.py check --store /tmp/compat-store
```

## Rust Binary: `vortex-compat`

A thin binary with four commands (`generate`, `editions`, `dump-expected`,
`check`). It has **no** knowledge of S3, published manifests, or orchestration.

**Output protocol:** progress/diagnostics to stderr, structured JSON to
stdout (`check` and `editions`).

### `generate --output <DIR> [--edition <EDITION>] [--exclude <CSV>]`

When `--edition` is passed, `vortex-compat` enables that edition on the write
session and passes the session into both flat and dataset writers. Fixtures
whose required encoding is not permitted by the target edition are skipped and
recorded under `"skipped"` in `fixtures.json`; any other error fails generation.

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

### `editions`

Prints a JSON array of all frozen `core` editions and their `min_library_version`.

### `check --dir <DIR> --mode <MODE> [--exclude <CSV>]`

For each known fixture:

1. Write a fresh file into a temporary directory with `write_expected()`.
   This builds the expected arrays from the current code. For a synthetic
   fixture, it writes the result of `build()`.
2. Validate the stored file's full layout tree via `adapter::read_layout_tree()`.
3. Read both the stored file and the fresh file with `adapter::read_file()`
   and compare them with `assert_arrays_eq!`, catching errors and panics per
   fixture so every fixture runs to completion.

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

Pass `--profile <PROFILE>` before the subcommand to override the Cargo build
profile (default: `release`, e.g. `compat.py --profile ci check`).

### `publish [--git-ref <REF>] [--store <SPEC>] [--dry-run]`

1. Detect version from nearest git tag at HEAD (or `<REF>`)
2. Generate fixtures from the current working tree
3. Verify against the previous version's manifest (additive-only)
4. Upload `.vortex` files + `manifest.json` to `v{version}/arrays/`
5. Update `versions.json`

### `check [--versions <CSV>] [--mode <MODE>] [--store <SPEC>]`

1. Read `versions.json` from store
2. For each version, download `arrays/manifest.json` + all `.vortex` files
3. Run `vortex-compat check --dir <tmpdir> --mode superset`
4. Aggregate results across all versions and fixtures, exit 1 if any failures

### `check-editions [--editions <CSV>] [--exclude <CSV>] [--accepted-failures <PATH>] [--python <VER>] [--json-output <PATH>]`

1. Queries `vortex-compat editions` for frozen `core` editions and their `min_library_version`
2. Generates fixtures at each selected edition with `vortex-compat generate --edition <EDITION>`
3. Dumps expected Arrow IPC tables directly from HEAD via `vortex-compat dump-expected`, then verifies each edition using its oldest `vortex-data` release wheel (applying any documented `reader_overrides` in `accepted-edition-failures.json`)
4. Fails on any failure not listed in `accepted-edition-failures.json` and on any listed entry that now passes (shrink-only)

### `generate --output <DIR> [--edition <EDITION>] [--git-ref <REF>]`

Generate fixtures locally from the current working tree without publishing. Writes `.vortex` files and a
`manifest.json` tagged with the version detected from HEAD (or `<REF>`).

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

### Version detection and git worktree workflow

`--git-ref` overrides only the git ref used for `git describe --tags --abbrev=0`
version detection; `compat.py` always builds `vortex-compat` from the current
working tree (or uses `VORTEX_COMPAT_BIN`):

```bash
python compat.py publish --git-ref v0.62.0
```

To generate and publish fixtures using the code at a historical release itself,
create a temporary git worktree at that ref and run `compat.py` from inside it:

```bash
git worktree add /tmp/vortex-compat-v0.62.0 v0.62.0
(cd /tmp/vortex-compat-v0.62.0 && uv run vortex-test/compat-gen/scripts/compat.py publish)
git worktree remove /tmp/vortex-compat-v0.62.0
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `VORTEX_COMPAT_BIN` | Path to pre-built `vortex-compat` binary. Skips `cargo build`. |
