# Vortex File Backward Compatibility Testing — Implementation Plan

RFC: https://github.com/vortex-data/rfcs/pull/23

## Overview

A standalone crate (`vortex-test/compat-gen/`) that generates deterministic `.vortex` fixture files
and validates them across versions. Not a workspace member — uses path deps to workspace crates.

Two binaries:
- **`compat-gen`**: Build fixture arrays → write `.vortex` files + `manifest.json`
- **`compat-test`**: Fetch fixtures from S3 (plain HTTPS) → read → compare via `assert_arrays_eq!`

## API Epochs

The Vortex file write/read API has 3 distinct epochs. The adapter layer (`adapter.rs`) is the only
file that changes when cherry-picking to old release branches.

| Epoch | Versions       | Write API                                               | Read (in-memory)                                     | Scan output              | Session |
|-------|----------------|---------------------------------------------------------|------------------------------------------------------|--------------------------|---------|
| **A** | 0.36.0         | `VortexWriteOptions::default().write(sink, stream) → W` | `VortexOpenOptions::in_memory().open(buf).await?`    | `into_array_stream()` async | None    |
| **B** | 0.45.0–0.52.0  | `VortexWriteOptions::default().write(sink, stream) → W` | `VortexOpenOptions::in_memory().open(buf)?` (sync)   | `into_array_iter()` sync | Exists, not wired |
| **C** | 0.58.0–HEAD    | `session.write_options().write(sink, stream) → WriteSummary` | `session.open_options().open_buffer(buf)?` (sync) | `into_array_stream()` async | Central |

### Key Breaking Changes

- **A→B**: In-memory `open()` changed from async to sync; scan switched to sync `into_array_iter()`
- **B→C**:
  - `VortexWriteOptions` lost `Default`, now constructed from `VortexSession`
  - `write()` return type: `W` (sink) → `WriteSummary`
  - `VortexOpenOptions` lost the `FileType` generic parameter
  - `in_memory().open()` → `open_options().open_buffer()`
  - Scan: `into_array_iter()` → `into_array_stream()` (async restored)

## Array Construction API Stability

Array construction is stable across ALL versions — fixture builders need NO adaptation:

| API | Status |
|-----|--------|
| `StructArray::try_new(field_names, fields, len, validity)` | Stable 0.36.0–HEAD |
| `PrimitiveArray::new(buffer![...], validity)` | Stable 0.36.0–HEAD |
| `buffer![1, 2, 3].into_array()` | Stable 0.36.0–HEAD |
| `VarBinArray::from(vec!["a", "b"])` | Stable 0.36.0–HEAD |
| `BoolArray::from_iter([true, false])` | Stable 0.36.0–HEAD |
| `ArrayRef::from_arrow(record_batch, false)` | Stable 0.36.0–HEAD |
| `ChunkedArray::try_new(chunks, dtype)` | Stable 0.36.0–HEAD |

## Crate Layout

```
vortex-test/compat-gen/
  Cargo.toml                # standalone, path deps to workspace
  PLAN.md
  src/
    main.rs                 # compat-gen CLI: --version, --output
    test_main.rs            # compat-test CLI: --fixtures-url
    adapter.rs              # write_file() + read_file() — ONLY branch-specific file
    manifest.rs             # Manifest serde struct
    validate.rs             # fetch from HTTPS + assert_arrays_eq! loop
    fixtures/
      mod.rs                # Fixture trait + all_fixtures() registry
      synthetic.rs          # 6 synthetic fixtures
      tpch.rs               # 2 TPC-H fixtures (lineitem, orders)
      clickbench.rs         # ClickBench hits 1k fixture
      encodings.rs          # per-encoding fixture stubs (todo!())
```

## Fixture Suite

### Trait

```rust
pub trait Fixture: Send + Sync {
    fn name(&self) -> &str;
    fn build(&self) -> Vec<ArrayRef>;
}
```

Returns `Vec<ArrayRef>` to support chunked fixtures. Single-array fixtures return a one-element vec.

### Synthetic Fixtures (implemented)

| File | Schema | Purpose |
|------|--------|---------|
| `primitives.vortex` | `Struct{u8, u16, u32, u64, i32, i64, f32, f64}` | Primitive round-trip |
| `strings.vortex` | `Struct{Utf8}` | String encoding |
| `booleans.vortex` | `Struct{Bool}` | Bool round-trip |
| `nullable.vortex` | `Struct{Nullable<i32>, Nullable<Utf8>}` | Null handling |
| `struct_nested.vortex` | `Struct{Struct{i32, Utf8}, f64}` | Nested types |
| `chunked.vortex` | Chunked `Struct{u32}` (3 x 1000 rows) | Multi-chunk files |

### Realistic Fixtures (implemented)

| File | Source | Rows | Purpose |
|------|--------|------|---------|
| `tpch_lineitem.vortex` | TPC-H SF 0.01 via `tpchgen-arrow` | ~60K | Numeric + string schema |
| `tpch_orders.vortex` | TPC-H SF 0.01 via `tpchgen-arrow` | ~15K | Date + decimal types |
| `clickbench_hits_1k.vortex` | First 1000 rows of ClickBench `hits` parquet (pinned URL) | 1000 | Wide table (105 cols) |

### Per-Encoding Fixture Stubs (todo)

One fixture per stable encoding to exercise encoding-specific read paths. These are stubbed with
`todo!()` until the "stable encodings" RFC defines what's frozen.

| File | Encoding | Stub? |
|------|----------|-------|
| `enc_dict.vortex` | DictArray | `todo!()` |
| `enc_runend.vortex` | RunEndArray | `todo!()` |
| `enc_constant.vortex` | ConstantArray | `todo!()` |
| `enc_sparse.vortex` | SparseArray | `todo!()` |
| `enc_alp.vortex` | ALPArray | `todo!()` |
| `enc_bitpacked.vortex` | BitPackedArray | `todo!()` |
| `enc_fsst.vortex` | FSSTArray | `todo!()` |

## Adapter Layer

Only `adapter.rs` changes per epoch. Contains two functions:

```rust
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()>;
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>>;
```

### Epoch A — v0.36.0

```rust
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let file = tokio::fs::File::create(path).await?;
        let _sink = VortexWriteOptions::default().write(file, stream).await?;
        Ok(())
    })
}

pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let file = VortexOpenOptions::in_memory().open(bytes).await?;  // async
        let arr = file.scan()?.into_array_stream()?.read_all().await?;
        Ok(vec![arr])
    })
}
```

### Epoch B — v0.45.0–v0.52.0

```rust
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    // identical to Epoch A — same VortexWriteOptions::default() API
    // ...
}

pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    let file = VortexOpenOptions::in_memory().open(bytes)?;  // sync now
    let arr = file.scan()?.into_array_iter()?.read_all()?;   // sync
    Ok(vec![arr])
}
```

### Epoch C — v0.58.0+/HEAD

```rust
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    let session = VortexSession::default();
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let mut file = tokio::fs::File::create(path).await?;
        let _summary = session.write_options().write(&mut file, stream).await?;
        Ok(())
    })
}

pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    let session = VortexSession::default();
    let file = session.open_options().open_buffer(bytes)?;
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let arr = file.scan()?.into_array_stream()?.read_all().await?;
        Ok(vec![arr])
    })
}
```

## Validation Strategy

Comparison uses `assert_arrays_eq!` with `ChunkedArray` wrapping:

```rust
fn validate(actual: Vec<ArrayRef>, expected: Vec<ArrayRef>) -> VortexResult<()> {
    let actual_dtype = actual[0].dtype().clone();
    let expected_dtype = expected[0].dtype().clone();
    let actual_chunked = ChunkedArray::try_new(actual, actual_dtype)?;
    let expected_chunked = ChunkedArray::try_new(expected, expected_dtype)?;
    assert_arrays_eq!(actual_chunked, expected_chunked);
    Ok(())
}
```

The writer may re-chunk across versions, but `assert_arrays_eq!` compares element-by-element
so chunk boundaries don't matter.

## Fixture Fetching

Fixtures are stored in a public S3 bucket accessible via plain HTTPS. `compat-test` uses
`reqwest` (blocking) to fetch — no AWS SDK needed.

```
https://vortex-compat-fixtures.s3.amazonaws.com/v{VERSION}/manifest.json
https://vortex-compat-fixtures.s3.amazonaws.com/v{VERSION}/{fixture}.vortex
```

Version discovery: `compat-test` takes a `--versions` flag listing which versions to test,
or discovers them from a top-level `versions.json` in the bucket.

## CI Workflows

### `compat-gen-upload.yml` — on tag push or manual dispatch

```yaml
on:
  push:
    tags: ["[0-9]+.[0-9]+.[0-9]+"]
  workflow_dispatch:
    inputs:
      tag: { description: "Git tag", required: true }

jobs:
  upload-fixtures:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
        with: { ref: "${{ inputs.tag || github.ref_name }}" }
      - uses: dtolnay/rust-toolchain@stable
      - run: |
          VERSION=${{ inputs.tag || github.ref_name }}
          cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
            --bin compat-gen -- --version "$VERSION" --output /tmp/fixtures/
      - run: |
          VERSION=${{ inputs.tag || github.ref_name }}
          aws s3 cp /tmp/fixtures/ s3://vortex-compat-fixtures/v${VERSION}/ --recursive
```

### `compat-test-weekly.yml` — weekly + manual

```yaml
on:
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch: {}

jobs:
  compat-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: |
          cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
            --bin compat-test -- \
            --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com
```

## Code Size Summary

| Component | ~Lines | Shared across branches? |
|-----------|--------|------------------------|
| `Cargo.toml` | 35 | Yes |
| `src/main.rs` (compat-gen CLI) | 56 | Yes |
| `src/test_main.rs` (compat-test CLI) | 40 | Yes |
| `src/adapter.rs` (write + read) | 55 | **No — 3 epoch variants** |
| `src/manifest.rs` | 10 | Yes |
| `src/validate.rs` (fetch + compare) | 60 | Yes |
| `src/fixtures/mod.rs` (trait + registry) | 40 | Yes |
| `src/fixtures/synthetic.rs` (6 fixtures) | 170 | Yes |
| `src/fixtures/tpch.rs` (2 fixtures) | 45 | Yes |
| `src/fixtures/clickbench.rs` (1 fixture) | 50 | Yes |
| `src/fixtures/encodings.rs` (stubs) | 60 | Yes |
| CI workflows (2 YAML files) | 80 | Yes |
| **Total** | **~700** | **~645 shared (92%), ~55 branch-specific (8%)** |

## What Changes Per Version When Cherry-Picking

| Component | Changes? |
|-----------|----------|
| Fixture trait + registry | No |
| All fixture builders | No |
| `adapter.rs` | **Yes — ~55 lines, 3 variants** |
| `main.rs`, `test_main.rs`, `manifest.rs`, `validate.rs` | No |
| `Cargo.toml` | No (path deps resolve to local version) |
| CI workflows | No |

## Usage

```bash
# Generate fixtures for the current version
cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
  --bin compat-gen -- --version 0.62.0 --output /tmp/fixtures/

# Validate fixtures from S3 against the current reader
cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
  --bin compat-test -- \
  --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com

# Validate from a local directory (for development)
cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
  --bin compat-test -- \
  --fixtures-dir /tmp/fixtures/
```

## Implementation Order

1. ~~Fixture trait + synthetic builders~~ ✅
2. ~~TPC-H fixtures~~ ✅
3. ~~compat-gen binary (main.rs)~~ ✅
4. ~~Adapter write path (3 epochs)~~ ✅
5. ClickBench fixture
6. Per-encoding fixture stubs
7. Adapter read path (3 epochs)
8. compat-test binary (test_main.rs + validate.rs)
9. CI workflows
