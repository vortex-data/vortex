# vortex-compat

Compatibility testing for the Vortex file format:

- **Backward compatibility:** the current reader can always decode `.vortex`
  files written by any older released version.
- **Forward compatibility:** a file the current writer produces under a frozen
  edition is readable by every release from that edition's
  `min_library_version` on. See the `generate-forward` and `forward` commands below.

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

# Read files the current writer produces, per frozen edition, with every
# released reader that edition covers (no S3, no credentials)
uv run scripts/compat.py forward

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

For each known fixture:

1. Write a fresh file into a temporary directory with `write_expected()`.
   This builds the expected arrays from the current code. For a synthetic
   fixture, it writes the result of `build()`.
2. Read the stored file and the fresh file with `adapter::read_file()`.
3. Compare the two arrays with `assert_arrays_eq!`.

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

### `generate-forward --output <DIR> [--exclude <CSV>]`

Writes every fixture once per frozen `core` edition. While an edition's files
are written, it is the only enabled `core` edition and the writer uses its
default strategy, so the files hold what a real writer emits: compressed
encodings, layouts, zone maps, and file statistics. Each fixture's `build()`
is canonicalized first, so the writer chooses every encoding itself.

Output:
```
<DIR>/
├── forward.json
├── expected/primitives.arrow      # expected values, Arrow IPC
├── core2026.08.0/primitives.vortex
└── ...
```

`forward.json` lists the fixtures, and for each edition its
`min_library_version`, the fixtures it wrote, and the fixtures it could not
write, with the writer's error. An edition may be unable to represent a fixture,
for example a dtype that joined a later edition. The command fails if a fixture
that one edition wrote is unwritable in a later edition, or if the newest frozen
edition cannot write a fixture. Every written file is read back with the current
reader and compared with `build()`.

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

### `forward [--mode <MODE>|--versions <CSV>] [--editions <CSV>] [--exclude <CSV>] [--dir <DIR>] [--expected-failures <TOML>] [--results <JSON>]`

1. Run `vortex-compat generate-forward` into a temp dir (or use `--dir`)
2. List the final releases of the `vortex-data` Python package on PyPI
3. For each release, read the files of every edition whose
   `min_library_version` it has reached. Each release runs
   `scripts/forward_reader.py` in an isolated `uv` environment with that
   release's published wheel, so no old code is compiled.
4. Report results per edition, and exit 1 if any file fails to read or differs
   from its expected values, unless the failure is documented as expected.
   An edition the current writer cannot write any fixture with also fails: its
   forward guarantee cannot be tested.

The reader uses `vortex.open(path).to_arrow()`, which every release since
0.36.0 provides. It compares the result with the expected Arrow IPC file by
value, so an older reader may export an equivalent Arrow type, such as `string`
for `string_view`. NaN equals NaN, and `0.0` differs from `-0.0`. Temporal
columns are compared through their integer storage.

Known failures are documented in `scripts/forward_expected_failures.toml`, each
with the editions, fixtures, release range, and error text it covers, and the
reason. A failure an entry documents passes the check; any other failure fails
it. A covered file that reads correctly also fails the check, so an entry must
be narrowed or removed as soon as its failure is fixed. To see every failure,
and to refresh the entries, run:

```bash
uv run scripts/compat.py forward --expected-failures /dev/null --results failures.json
```

A file with several incompatibilities can report a different one from run to
run, depending on which column the reader decodes first. Such a file is listed
under every entry whose error it shows.

`--mode all` (the default) reads with every release from each edition's
`min_library_version` up to the latest release. `--mode last` reads only with
each edition's `min_library_version` and the latest release. `--jobs` sets how
many releases read concurrently, and `--python` sets the reader environments'
Python version (default 3.12).

Some old wheels declare their dependencies loosely. Every reader environment
therefore adds `typing_extensions` and pins `substrait<0.24`.

The reader reports each file as it reads it. If an old release aborts the
process, for example with a Rust panic, the file it was reading fails with the
panic message and the reader restarts without it, so one crash costs one file.

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
