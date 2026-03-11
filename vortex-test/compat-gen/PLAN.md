# Vortex File Backward Compatibility Testing — Implementation Plan

RFC: https://github.com/vortex-data/rfcs/pull/23

## Overview

A standalone crate (`vortex-test/compat-gen/`) that generates deterministic `.vortex` fixture files
for backward compatibility testing. Not a workspace member — uses path deps to workspace crates.

## API Epochs

The Vortex file write/read API has 3 distinct epochs. The adapter layer (`adapter.rs`) is the only
file that changes when cherry-picking to old release branches.

| Epoch | Versions       | Write API                                               | Read (in-memory)                                     | Session |
|-------|----------------|---------------------------------------------------------|------------------------------------------------------|---------|
| **A** | 0.36.0         | `VortexWriteOptions::default().write(sink, stream) → W` | `VortexOpenOptions::in_memory().open(buf).await?`    | None    |
| **B** | 0.45.0–0.52.0  | `VortexWriteOptions::default().write(sink, stream) → W` | `VortexOpenOptions::in_memory().open(buf)?` (sync)   | Exists, not wired |
| **C** | 0.58.0–HEAD    | `session.write_options().write(sink, stream) → WriteSummary` | `session.open_options().open_buffer(buf)?` (sync) | Central |

### Key Breaking Changes

- **A→B**: In-memory `open()` changed from async to sync
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

## Fixture Suite

### Trait

```rust
pub trait Fixture: Send + Sync {
    fn name(&self) -> &str;
    fn build(&self) -> Vec<ArrayRef>;
}
```

Returns `Vec<ArrayRef>` to support chunked fixtures naturally.

### Synthetic Fixtures

| File | Schema | Purpose |
|------|--------|---------|
| `primitives.vortex` | `Struct{u8, u16, u32, u64, i32, i64, f32, f64}` | Primitive round-trip |
| `strings.vortex` | `Struct{Utf8}` | String encoding |
| `booleans.vortex` | `Struct{Bool}` | Bool round-trip |
| `nullable.vortex` | `Struct{Nullable<i32>, Nullable<Utf8>}` | Null handling |
| `struct_nested.vortex` | `Struct{Struct{i32, Utf8}, f64}` | Nested types |
| `chunked.vortex` | Chunked `Struct{u32}` (3 x 1000 rows) | Multi-chunk files |

### Realistic Fixtures

| File | Source | Rows |
|------|--------|------|
| `tpch_lineitem.vortex` | TPC-H SF 0.01 | ~60K |
| `tpch_orders.vortex` | TPC-H SF 0.01 | ~15K |

## Adapter Layer

Only `adapter.rs` changes per epoch (~15 lines). See `src/adapter.rs` for the current (Epoch C)
implementation. The git history shows all 3 epoch variants.

## What Changes Per Version When Cherry-Picking

| Component | Changes? |
|-----------|----------|
| Fixture trait + registry | No |
| Fixture builders (synthetic) | No |
| Fixture builders (TPC-H) | No |
| `adapter.rs` | Yes — ~15 lines, 3 variants |
| `main.rs`, `manifest.rs` | No |
| `Cargo.toml` | No (path deps resolve to local version) |

## Usage

```bash
# Generate fixtures for the current version
cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
  --bin compat-gen -- --version 0.62.0 --output /tmp/fixtures/

# Outputs:
#   /tmp/fixtures/manifest.json
#   /tmp/fixtures/primitives.vortex
#   /tmp/fixtures/strings.vortex
#   ...
```
