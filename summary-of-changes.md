# Decompose TurboQuant: Handoff Guide

## Branch

`claude/decompose-turboquant-encoding-35hBD` — single commit rebased on latest `develop`.

## Background & Motivation

TurboQuant is a lossy vector quantization encoding
([arXiv:2504.19874](https://arxiv.org/abs/2504.19874),
[RFC 0033](https://vortex-data.github.io/rfcs/rfc/0033.html)) for high-dimensional vector data. It
was previously implemented as a monolithic `TurboQuantArray` VTable that bundled three concerns:

1. **Scalar quantization** — mapping coordinates to codebook centroids (codes + centroids)
2. **SORF rotation** — structured random rotation to spread energy across coordinates
   (rotation_signs)
3. **L2 normalization** — already extracted to `L2Denorm` in PR #7349

The goal of this change is to decompose TurboQuant so each layer can be independently swapped. For
example, SORF could be replaced with a learned rotation (motivated by
[Ashwin's tweet](https://x.com/ashwingop/status/2041117804615897162)). Standard array types
(`DictArray`, `FixedSizeListArray`) replace custom encodings, and compute pushdowns (slice, take,
filter) come for free from existing `ScalarFnArray` rules.

## What Changed

### Before

```
ScalarFnArray(L2Denorm, [TurboQuantArray{codes, centroids, rotation_signs}, norms])
```

`TurboQuantArray` was a custom VTable with its own metadata, serialization, compute pushdowns
(slice, take, scalar_at), and decompression logic.

### After

```
ScalarFnArray(L2Denorm, [
    ScalarFnArray(SorfTransform, [
        FixedSizeListArray(
            DictArray(codes=Primitive<u8>, values=Primitive<f32>),
            padded_dim
        )
    ]),
    norms
])
```

Each layer is a standard Vortex array type with existing compute rules:

- **DictArray** — `take(values, codes)` dequantizes; slice pushes into codes, keeps values shared
- **FixedSizeListArray** — slice pushes into inner elements
- **ScalarFnArray (SorfTransform)** — slice/take/filter push through to children automatically via
  `ScalarFnSliceReduceRule`
- **ScalarFnArray (L2Denorm)** — same pushdown rules; re-applies norms at execution

Decompression is automatic: `array.execute::<ExtensionArray>(ctx)` walks the tree.

### Architecture: Modularity

Each layer is independently swappable:

| Layer         | Current impl                         | Could be swapped with                              |
| ------------- | ------------------------------------ | -------------------------------------------------- |
| Normalization | `L2Denorm` ScalarFn                  | Skip (pre-normalized data), different norm         |
| Rotation      | `SorfTransform` ScalarFn             | `LearnedTransform`, `OrthogonalTransform`, etc.    |
| Quantization  | `DictArray(codes, centroids)` in FSL | Product quantization, different codebook structure |

The encode path reflects this modularity:

- `build_quantized_fsl()` — pure Dict construction, knows nothing about rotation
- `SorfTransform::try_new_array()` — pure rotation metadata wrapping, knows nothing about
  quantization
- `TurboQuantScheme::compress()` — composes normalize → rotate+quantize → wrap in L2Denorm

### Key design decisions

**Rotation parameters via seed, not as a child array.** ScalarFnArray requires all children to have
the same length. Rotation signs have length=`num_rounds` (typically 3), not `N` (number of vectors).
We store the PRNG seed in `SorfOptions` and regenerate the `RotationMatrix` deterministically at
decode time via `RotationMatrix::try_new(seed, dimension, num_rounds)`.

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SorfOptions {
    pub seed: u64,
    pub num_rounds: u8,
    pub dimension: u32,       // original dim (before padding)
    pub element_ptype: PType, // target output element type
}
```

**DictArray for codes + centroids.** Flat `Primitive<u8>` codes (N×padded_dim indices) +
`Primitive<f32>` centroids (2^bit_width values). This replaces three custom child slots with a
standard encoding.

## Files Changed

### New files

| File                                                         | Purpose                                                                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `vortex-tensor/src/scalar_fns/sorf_transform.rs`             | `SorfTransform` ScalarFnVTable — inverse SORF rotation at decode time                                   |
| `vortex-tensor/src/encodings/turboquant/tests/mod.rs`        | Shared test helpers (make_fsl, normalize_and_encode, tree navigation helpers)                           |
| `vortex-tensor/src/encodings/turboquant/tests/roundtrip.rs`  | Roundtrip, MSE quality, edge case, f16/f64 input tests                                                  |
| `vortex-tensor/src/encodings/turboquant/tests/compute.rs`    | Slice, take, scalar_at, L2 norm readthrough tests                                                       |
| `vortex-tensor/src/encodings/turboquant/tests/structural.rs` | Centroids match, seed determinism, dtype verification, quantized accuracy, SorfTransform isolation test |
| `vortex-tensor/src/encodings/turboquant/tests/nullable.rs`   | Nullable vector roundtrip, validity propagation, slicing                                                |

### Modified files

| File                                                        | Changes                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vortex-tensor/src/scalar_fns/mod.rs`                       | Added `pub mod sorf_transform;`                                                                                                                                                                                                                                                                |
| `vortex-tensor/src/lib.rs`                                  | Register `SorfTransform` scalar fn; removed `TurboQuant` array registration and `ArraySessionExt` import                                                                                                                                                                                       |
| `vortex-tensor/src/encodings/turboquant/mod.rs`             | Complete rewrite: removed VTable re-exports, added module-level constants (`MIN_DIMENSION`, `MAX_BIT_WIDTH`, `MAX_CENTROIDS`), `validate_vector_dtype()` standalone function, updated module docs for new tree structure                                                                       |
| `vortex-tensor/src/encodings/turboquant/scheme/compress.rs` | Rewrote to produce `SorfTransform(FSL(Dict))` instead of `TurboQuantArray`. Split into `build_quantized_fsl()` (Dict construction) and `SorfTransform::try_new_array()` (rotation wrapping). Removed `build_turboquant`, `bitpack_rotation_signs`, `rotation` field from `QuantizationResult`. |
| `vortex-tensor/src/encodings/turboquant/scheme/mod.rs`      | Updated `TurboQuantScheme` to use `validate_vector_dtype()`. Updated `estimate_compression_ratio` to remove rotation sign overhead. Removed `decompress` module.                                                                                                                               |
| `vortex-tensor/src/encodings/turboquant/centroids.rs`       | Moved from `array/centroids.rs`. Replaced `TurboQuant::MAX_BIT_WIDTH` → `MAX_BIT_WIDTH`, `TurboQuant::MIN_DIMENSION` → `MIN_DIMENSION`. Fixed broken `compute_boundaries` doc link → `compute_centroid_boundaries`.                                                                            |
| `vortex-tensor/src/encodings/turboquant/rotation.rs`        | Moved from `array/rotation.rs`. Added `#[cfg(test)]` to `export_inverse_signs_u8` and `from_u8_slice` (only used in rotation's own tests now). Removed `num_rounds()` accessor (unused).                                                                                                       |
| `vortex-file/src/strategy.rs`                               | Removed `TurboQuant` from `ALLOWED_ENCODINGS` (no longer a custom array encoding).                                                                                                                                                                                                             |

### Removed files (~2,500 lines)

| File                                                          | Was                                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `vortex-tensor/src/encodings/turboquant/vtable.rs`            | `TurboQuant` VTable impl, `TurboQuantArray` type alias, serde, validation                                     |
| `vortex-tensor/src/encodings/turboquant/metadata.rs`          | Protobuf metadata for bit_width/num_rounds                                                                    |
| `vortex-tensor/src/encodings/turboquant/array/data.rs`        | `TurboQuantData`, `TurboQuantArrayExt` trait                                                                  |
| `vortex-tensor/src/encodings/turboquant/array/slots.rs`       | Slot enum (Codes, Centroids, RotationSigns)                                                                   |
| `vortex-tensor/src/encodings/turboquant/array/mod.rs`         | Module declaration                                                                                            |
| `vortex-tensor/src/encodings/turboquant/compute/mod.rs`       | `float_from_f32` helper (moved to sorf_transform.rs)                                                          |
| `vortex-tensor/src/encodings/turboquant/compute/ops.rs`       | Execute parent kernels                                                                                        |
| `vortex-tensor/src/encodings/turboquant/compute/rules.rs`     | Slice/take reduce rules                                                                                       |
| `vortex-tensor/src/encodings/turboquant/compute/slice.rs`     | Custom slice impl                                                                                             |
| `vortex-tensor/src/encodings/turboquant/compute/take.rs`      | Custom take impl                                                                                              |
| `vortex-tensor/src/encodings/turboquant/scheme/decompress.rs` | `execute_decompress` — manual dequantize+inverse-rotate (now handled by Dict execute + SorfTransform execute) |
| `vortex-tensor/src/encodings/turboquant/tests.rs`             | Old monolithic test file (split into tests/ directory)                                                        |

## Known Issues / Remaining Work

1. **Serde roundtrip tests removed.** The old `TurboQuantArray` had serde support
   (serialize/deserialize via protobuf metadata). `ScalarFnArray` doesn't support serde yet. When it
   does, serde tests should be re-added.

2. **SorfTransform validity consistency.** `return_dtype()` preserves child nullability, `execute()`
   always produces `NonNullable` output. In practice this isn't a bug because TurboQuant children
   are always non-nullable (nullability is tracked by L2Denorm's norms child). But if SorfTransform
   is ever used with nullable children, `execute()` should propagate validity.

3. **Compression ratio estimates** no longer include rotation sign overhead (since signs are no
   longer stored — rotation is deterministic from seed). The test bounds were updated accordingly.

4. **The `RotationMatrix::export_inverse_signs_u8` and `from_u8_slice` methods** are now
   `#[cfg(test)]` only. If a future encoding needs to serialize rotation signs explicitly, these
   would need to be made public again.

## How to Verify

```bash
cargo build -p vortex-tensor
cargo test -p vortex-tensor                           # 183 tests pass
cargo clippy --all-targets --all-features -p vortex-tensor  # clean
cargo +nightly fmt --all                              # formatted
cargo build --benches -p vortex --features unstable_encodings  # benchmarks compile
RUSTDOCFLAGS="-D warnings" cargo doc -p vortex-tensor --no-deps  # no doc warnings
```

## Public API Changes

**Removed exports:**

- `TurboQuant`, `TurboQuantArray`, `TurboQuantData`, `TurboQuantArrayExt`

**Added exports:**

- `SorfTransform`, `SorfOptions` (from `vortex_tensor::scalar_fns::sorf_transform`)

**Moved:**

- `MIN_DIMENSION`, `MAX_BIT_WIDTH`, `MAX_CENTROIDS` — from `TurboQuant::` associated constants to
  module-level constants in `vortex_tensor::encodings::turboquant`
- `validate_vector_dtype()` — standalone function replacing `TurboQuant::validate_dtype()`

**Preserved (unchanged API):**

- `TurboQuantConfig`, `TurboQuantScheme`, `turboquant_encode`, `turboquant_encode_unchecked`
