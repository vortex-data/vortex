# Vortex Backward-Compatibility Testing

## The Problem

Vortex is a columnar file format. Users write `.vortex` files with one
version of the library and expect to read them with any future version. If a
code change silently breaks the ability to decode old files, we ship data
loss.

## The Solution

We maintain a library of `.vortex` fixture files, one set per released
version, stored in S3. A test reads every old fixture with the current reader
and compares the decoded values against a known-good oracle. If any fixture
from any version decodes to the wrong values, the test fails.

## How Fixtures Work

A fixture is a small `.vortex` file with known contents. The expected
contents are defined by a Rust function — `build()` — that deterministically
constructs the arrays. The same `build()` code is used at both ends:

- **At publish time:** `build()` produces arrays -> the writer serializes them
  to a `.vortex` file -> the file is uploaded to S3.
- **At check time:** `build()` produces the same arrays -> the reader decodes
  the old file -> the two are compared value-by-value.

If the reader is correct, the values match. If a code change breaks
decoding, they don't.

### The oracle question

"Isn't comparing against `build()` circular? What if `build()` itself
changes?"

`build()` is the specification — it defines what the fixture *should*
contain. The contract is that `build()` for a given fixture name is immutable
once defined. It must never change its output. If someone modifies it, the
check fails loudly against every old version.

### Fixture evolution

Because `build()` is immutable, you cannot add a column to an existing
fixture or change its schema. To test a new type, encoding, or structural
pattern, create a **new fixture file** with a new name.

## Fixture Trait

Every fixture implements this trait in Rust:

```rust
pub trait Fixture: Send + Sync {
    /// Filename, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// Optional async setup (download external data, etc).
    /// Called before build(). Default is a no-op.
    fn setup(&self, _tmp_dir: &Path) -> VortexResult<()> { Ok(()) }

    /// Build the expected arrays. Must be deterministic.
    /// Returns a Vec to support chunked fixtures (multiple chunks).
    fn build(&self, tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>>;
}
```

All fixtures are registered in `all_fixtures()` in `src/fixtures/mod.rs`.

### Current fixture suite

| Fixture | File | Description |
|---------|------|-------------|
| Primitives | `primitives.vortex` | u8-u64, i32, i64, f32, f64 at min/mid/max |
| Strings | `strings.vortex` | Empty, ASCII, multibyte (Japanese), emoji |
| Booleans | `booleans.vortex` | Mixed true/false |
| Nullable | `nullable.vortex` | Nullable int and string columns with interspersed nulls |
| Nested Struct | `struct_nested.vortex` | Two-level struct-in-struct |
| Chunked | `chunked.vortex` | 3 chunks of 1000 rows, deterministic values |
| TPC-H Lineitem | `tpch_lineitem.vortex` | TPC-H lineitem at SF 0.01 |
| TPC-H Orders | `tpch_orders.vortex` | TPC-H orders at SF 0.01 |
| ClickBench Hits | `clickbench_hits_1k.vortex` | First 1000 rows of ClickBench hits |

### Adding a new fixture

1. Create a struct implementing `Fixture` in `src/fixtures/`.
2. Register it in `all_fixtures()` in `src/fixtures/mod.rs`.
3. Run `cargo run -p vortex-compat --release -- generate --output /tmp/test` to verify.
4. Publish — the orchestrator merges the manifest so the new fixture gets
   `since` set to the current version.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Python orchestrator (compat.py)                 │
│  Version-agnostic: S3, manifests, worktrees      │
│                                                  │
│  Commands: publish, check, generate, list,       │
│            validate-manifest                     │
└──────────────────┬───────────────────────────────┘
                   │ invokes
┌──────────────────▼───────────────────────────────┐
│  Rust binary (vortex-compat)                     │
│  Version-specific: builds arrays, writes/reads   │
│  .vortex files, compares arrays                  │
│                                                  │
│  Commands: generate, check                       │
└──────────────────────────────────────────────────┘
```

**Why the split?** Only the Rust binary links against Vortex. When Vortex
APIs change across versions, only this small binary needs to recompile. The
Python orchestrator handles S3, manifests, git worktrees, and additive-only
enforcement without touching Vortex code.

### Rust binary: `vortex-compat`

A thin binary with two commands. It has **no** knowledge of versions, S3,
manifests, or orchestration.

**Output protocol:** progress/diagnostics to stderr, structured JSON to
stdout (`check` only).

#### `generate --output <DIR> [--exclude <CSV>]`

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

#### `check --dir <DIR> --mode <MODE> [--exclude <CSV>]`

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
| `subset` (default) | Skipped | Error |
| `exact` | Error | Error |
| `superset` | Error | Skipped |

Use `subset` when checking old versions (they may have extra fixtures not in
current code). Use `exact` for the current version.

#### Adapter module (`src/adapter.rs`)

The only code that calls Vortex file I/O APIs directly:

```rust
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()>;
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>>;
```

Uses the Epoch C API (`session.write_options()` /
`session.open_options().open_buffer()`), valid from v0.58.0 through HEAD.

### Python orchestrator: `compat.py`

#### `publish [--git-ref <REF>] [--store <SPEC>] [--dry-run] [--exclude <CSV>]`

1. Detect version from nearest git tag at HEAD (or `<REF>`)
2. Generate fixtures (from current tree, or from a worktree at `<REF>`)
3. Fetch previous version's manifest, merge `since` values, enforce additive-only
4. Upload `.vortex` files + `manifest.json` to `v{version}/arrays/`
5. Update `versions.json`

#### `check [--versions <CSV>] [--store <SPEC>] [--exclude <CSV>]`

1. Read `versions.json` from store
2. For each version, download `arrays/manifest.json` + all `.vortex` files
3. Run `vortex-compat check --dir <tmpdir> --mode subset`
4. Aggregate results, exit 1 if any failures

#### `generate --output <DIR> [--git-ref <REF>] [--exclude <CSV>]`

Generate fixtures locally without publishing. Writes `.vortex` files and a
`manifest.json`.

#### `list [--store <SPEC>] [--version <VER>]`

Without `--version`: print all version numbers.
With `--version`: print that version's `manifest.json`.

#### `validate-manifest [--store <SPEC>]`

Walk all versions in order and verify no fixtures were removed between
consecutive versions (additive-only property).

#### Store abstraction

| Spec | Type | Auth |
|------|------|------|
| `s3://vortex-compat-fixtures` (default) | S3 | Public reads (HTTPS), AWS creds for writes |
| `/tmp/compat` | Local directory | None |

#### Git worktree workflow

`--git-ref` automates publishing from historical releases:

```bash
python compat.py publish --git-ref v0.62.0
```

This creates a worktree at the tag, builds the binary against that code,
generates fixtures, then cleans up.

#### Environment variables

| Variable | Description |
|----------|-------------|
| `VORTEX_COMPAT_BIN` | Path to pre-built `vortex-compat` binary. Skips `cargo run`. |

## Store Format

### `versions.json`

A JSON array of version strings:

```json
["0.62.0", "0.63.0"]
```

### Directory layout

```
store/
├── versions.json
├── v0.62.0/
│   └── arrays/
│       ├── manifest.json
│       ├── primitives.vortex
│       └── ...
└── v0.63.0/
    └── arrays/
        ├── manifest.json
        └── ...
```

Each version gets a directory named `v{version}` with an `arrays/`
subdirectory for fixture files. Re-publishing overwrites the existing
directory.

### Per-version manifest

```json
{
  "version": "0.63.0",
  "generated_at": "2026-03-14T08:00:00Z",
  "fixtures": [
    {
      "name": "primitives.vortex",
      "description": "All primitive types with boundary values",
      "since": "0.62.0"
    }
  ]
}
```

The `since` field tracks which version introduced each fixture. It is carried
forward automatically by manifest merging — when publishing version N, the
orchestrator reads version N-1's manifest and preserves existing `since`
values.

## CI Workflows

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

Triggered via **manual dispatch** with optional `git_ref` input.

1. Checkout with full history (`fetch-depth: 0`)
2. Assume `GitHubBenchmarkRole` IAM role
3. Run `compat.py publish [--git-ref <input>]`

### Weekly validation (`.github/workflows/compat-test-weekly.yml`)

Runs **every Monday at 06:00 UTC** and on **manual dispatch**.

1. Checkout `main` at HEAD
2. Run `compat.py check`
3. Exit 1 if any fixture from any version fails

## Module Layout

```
vortex-test/compat-gen/
├── Cargo.toml
├── DESIGN.md              # this file
├── README.md              # quickstart
├── scripts/
│   └── compat.py          # Python orchestrator
└── src/
    ├── main.rs            # clap CLI: generate + check
    ├── lib.rs             # pub mod exports
    ├── generate.rs        # fixture generation (setup -> build -> write)
    ├── check.rs           # fixture validation (read -> compare -> JSON)
    ├── adapter.rs         # Vortex file I/O (Epoch C)
    └── fixtures/
        ├── mod.rs         # Fixture trait + all_fixtures()
        ├── synthetic.rs   # Primitives, Strings, Booleans, Nullable, Struct, Chunked
        ├── tpch.rs        # TPC-H lineitem and orders
        └── clickbench.rs  # ClickBench hits (first 1000 rows)
```

## Known Limitations

**Does catch:** schema deserialization breaks, encoding algorithm changes
that corrupt values, null bitmap changes, UTF-8/multibyte handling, chunk
boundary regressions, struct field ordering, encoding removal.

**Does not catch:** predicate pushdown (no filters applied), column
projection (reads all columns), missing DType coverage (no decimal, temporal,
list, binary, extension), degenerate cases (empty files, all-null, float
specials), partial reads / mmap.

**Dependency drift:** If a dependency like `tpchgen` silently changes output
across versions, old fixtures will fail validation even though the reader is
fine. If you see unexpected failures across *all* old versions for a specific
fixture, check whether its `build()` deps changed first.
