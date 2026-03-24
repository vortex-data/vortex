<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Vortex Backward-Compatibility Testing

## The Problem

Vortex is a columnar file format. Users write `.vortex` files with one
version of the library and expect to read them in any future version. 
We need back-compat!

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
pub trait Fixture {
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

See [README.md](README.md) for CLI usage.

#### Adapter module (`src/adapter.rs`)

The only code that calls Vortex file I/O APIs directly:

```rust
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()>;
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>>;
```

Uses the Epoch C API (`session.write_options()` /
`session.open_options().open_buffer()`), valid from v0.58.0 through HEAD.

### Python orchestrator: `compat.py`

See [README.md](README.md) for CLI usage.

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

Used to upload fixtures for each new release. Triggered via **manual
dispatch** with an optional `git_ref` input (defaults to HEAD).

Two-phase workflow:

1. **dry-run** — auto-detects the version from the nearest git tag, builds
   fixtures, and prints what would be uploaded.
2. **upload** — requires manual approval via the `compat-upload` GitHub
   environment, then performs the actual upload to S3.

### Compat validation (`.github/workflows/compat-validation.yml`)

Validates the current code against previously published fixtures. Runs weekly
(Monday 6am UTC) and via manual dispatch with a `mode` input:

| Mode | Behavior |
|------|----------|
| `last` (default) | Compare against the most recent published release only |
| `all` | Compare against all previously published versions (pre-release check) |

## Known Limitations

**Does catch:** schema deserialization breaks, encoding algorithm changes
that corrupt values, null bitmap changes, UTF-8/multibyte handling, chunk
boundary regressions, struct field ordering, encoding removal.

**Does not catch:** predicate pushdown (no filters applied), column
projection (reads all columns), missing DType coverage (no decimal, temporal,
list, binary, extension), degenerate cases (empty files, all-null, float
specials), partial reads / mmap.
