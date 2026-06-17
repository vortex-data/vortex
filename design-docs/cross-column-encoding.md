# Design: Cross-Column ("Virtual") Compression for Vortex

Status: Draft / RFC (design only — no implementation yet)
Author: investigation prompted by joe.isaacs@live.co.uk
Date: 2026-06-17

## 1. Motivation

Single-column encodings (bit-packing, FoR, ALP, dictionary, RLE) have largely
plateaued. The next axis of compression is *cross-column correlation*: many
columns are approximately a function of one or more sibling columns, so the
column can be replaced by a compact model plus a small per-row residual.

Two pieces of prior art motivate this:

- **Corra: Correlation-Aware Column Compression** (arXiv:2403.17229). Introduces
  *peer* encoding (one column = another column + a bounded per-row diff) and
  *subaltern* encoding (a one-to-many dictionary mapping, e.g. `city -> zip`).
- The **`virtual` / `virtual-parquet`** system (same group), which generalizes
  this to *learned* functions: per-column linear regression (`sparse-lr`) and
  piecewise / segmented linear regression (`k-regression`), with exact residuals.

This document records what those algorithms actually find and produce, then
proposes how to express the same idea as a Vortex array encoding.

## 2. Reproduced evidence (NYC taxi, `taxi_2019_04`, 7.43M rows)

`virtual-parquet==0.2.2` was run exactly as the upstream `demo-taxi.ipynb`:

```python
virtual.to_format('taxi.parquet', 'taxi_virtual.parquet',
                  model_types=['sparse-lr', 'k-regression', 'custom'])
```

It drilled **11 candidate functions**, kept the **3** that shrink the file:

| target column          | predictors                                                   | model kept            | per-column on-disk |
|------------------------|--------------------------------------------------------------|-----------------------|--------------------|
| `dropoff_at` (ts)      | `pickup_at`                                                  | `pickup_at + Δs` (exact, mse 0) | 29.3 → 12.3 MB (**−58%**) |
| `total_amount` (f32)   | `fare_amount`, `tip_amount`, `tolls_amount`, `congestion_surcharge` | 16-segment piecewise linear (mse 0.093) | 12.1 → 8.5 MB (**−29%**) |
| `congestion_surcharge` | `improvement_surcharge`                                     | 2-segment linear      | 0.99 → 0.96 MB (−3%) |

Whole-file: 122 MB → 112 MB. Round-trip verified bit-exact for `dropoff_at` and
`total_amount`.

### 2.1 What a "function" is

- **Peer offset** (timestamp/integer): `target = ref_col + Δ`, where `Δ` is a
  per-row integer. When the relationship is exact the residual *is* the encoded
  representation (mse 0).
- **k-regression**: a per-row *switch* `s ∈ [0, k)` selects one of `k`
  coefficient sets `(intercept_s, coeffs_s[])`; the prediction is
  `intercept_s + Σ_j coeffs_s[j] * predictor_j`. The switch lets a single column
  be modeled by a mixture of locally-linear regions (e.g. different tariff
  regimes for taxi fares).

### 2.2 How `virtual` physically stores it (the important part)

In the output Parquet, the original columns are **removed** and replaced by small
derived columns; the model parameters live in Parquet key/value metadata:

```
dropoff_at            -> dropoff_at_offset            : int32  (exact Δ seconds)
total_amount          -> total_amount_switch (int16)  + total_amount_offset (f32 residual)
congestion_surcharge  -> congestion_surcharge_switch  + congestion_surcharge_offset
```

Reconstruction per row `i`:

```
target[i] = round( intercept_switch[i] + Σ_j coeffs_switch[i][j] * predictor_j[i],
                   scale )
            + offset[i]          # exact residual = actual - rounded_prediction
```

The win is that `offset` (the residual) and `switch` have **tiny magnitude and
cardinality** compared to the target, so existing single-column encodings crush
them. Measured here:

| derived column                | dtype | distinct | range                |
|-------------------------------|-------|----------|----------------------|
| `dropoff_at_offset`           | i32   | 13,048   | [−3,279, 271,088]    |
| `total_amount_switch`         | i16   | 16       | [0, 15]              |
| `total_amount_offset`         | f32   | 1,878    | [−4.65, 16.94]       |
| `congestion_surcharge_switch` | i16   | 2        | [0, 1]               |
| `congestion_surcharge_offset` | f32   | 8        | [−0.75, 2.70]        |

**Key observation:** this is structurally identical to Vortex's existing
`reference + exceptions` encodings (FoR, ALP) — except the *reference is computed
from other columns* instead of from a constant scalar.

## 3. Vortex background (current state)

### 3.1 The `VTable` contract

Every encoding implements `vortex_array::vtable::VTable`
(`vortex-array/src/array/vtable/mod.rs`). Key obligations, as seen in FoR
(`encodings/fastlanes/src/for/vtable/mod.rs`) and ALP
(`encodings/alp/src/alp/array.rs`):

- `type TypedArrayData` — the in-memory metadata struct (e.g. `FoRData { reference: Scalar }`,
  `ALPData { exponents, patches_data }`). Must impl `ArrayHash` + `ArrayEq` + `Display`.
- `id()` → a `CachedId` like `CachedId::new("fastlanes.for")`.
- `validate(data, dtype, len, slots)` — invariant checks over child slots.
- `serialize` / `deserialize` — metadata to/from bytes (prost `Message`), plus
  reattaching child arrays from `ArrayChildren`.
- `execute(array, ctx)` — the decode/canonicalize entry point; returns an
  `ExecutionResult`.
- Optional `reduce_parent` / `execute_parent` — pushdown rules/kernels.
- `ValidityVTable` — typically `ValidityVTableFromChild` (validity follows a
  designated child).

Child arrays are stored as **slots** (`ArraySlots`, a `SmallVec<[Option<ArrayRef>; 4]>`).
The `#[array_slots(Name)]` macro generates named slot indices and accessor exts
(see `ALPSlots`). Slot count is variable, so an encoding can hold N children.

### 3.2 Exceptions: `Patches`

`vortex_array::patches::Patches { array_len, offset, indices, values, chunk_offsets }`
is the shared "sparse exceptions" primitive used by ALP and Sparse. It is exactly
the right tool for storing a residual list where *most* rows reconstruct cleanly
and only some need an explicit value. `PatchesData` / `PatchesMetadata` handle the
slot plumbing and serialization.

### 3.3 Closest existing analogs

- **FoR** — `reference: Scalar` in metadata + one `encoded` child. `canonicalize =
  encoded + reference`. This is a *single-column* peer offset.
- **ALP** — model params (`exponents`) in metadata + `encoded` child + optional
  `Patches`. This is *model + exceptions* for one column.
- **datetime-parts** (`encodings/datetime-parts/src/`) — splits ONE timestamp into
  `days` / `seconds` / `subseconds` children and reconstructs from all three. This
  is the best template for **multi-child reconstruction logic**, even though its
  children derive from a single source column.

### 3.4 `StructArray` and the core gap

`StructArray` (`vortex-array/src/arrays/struct_/array.rs`) stores
`[validity?, field_0 .. field_N]` as slots. Fields are **independent** — there is
no mechanism today for one field to be reconstructed from another. An encoding's
`execute()` only sees its **own** slots, never its sibling struct fields. This is
the central design constraint for cross-column compression.

## 4. The core design decision: where does the predictor come from?

Reconstruction needs the predictor columns at decode time. There are two ways to
make them reachable, which define the two variants below.

### Variant A — Self-contained `LinearModelArray`

The target array carries *its own references* to the predictor arrays as extra
slots. It is a normal encoding that happens to have predictor children. Decoding
is purely local: it never needs to know it lives inside a struct.

```
LinearModelArray  (logical dtype = the target column's dtype)
  metadata (prost):
    output_scale: i32                 # decimal places for round(), or "none"
    segments: repeated Segment        # k of them; k == 1 for plain sparse-lr
      intercept: f64
      coeffs: repeated { predictor_slot: u32, coeff: f64 }
    residual: optional PatchesMetadata # exact residual as exceptions, or...
  slots:
    [ predictor_0, .., predictor_{m-1},   # the columns the model reads
      switch?,                            # u8/u16 segment selector (omit if k==1)
      residual_encoded ]                  # dense residual child (or Patches slots)
```

`execute()` algorithm:

```
preds   = [ canonicalize(slot) for predictor slots ]      # m primitive arrays
switch  = canonicalize(switch slot) or all-zeros
acc     = per-row prediction:
            for row i: seg = segments[switch[i]]
                       acc[i] = seg.intercept + Σ_j seg.coeff_j * preds[j][i]
acc     = round(acc, output_scale)        # match the column's stored scale
out     = acc + residual                  # dense add, or apply Patches
```

Pros: fits the existing `VTable` model with **zero changes** to `StructArray`;
small, testable in isolation; mirrors ALP almost exactly (model in metadata +
encoded + residual exceptions).

Cons: the predictor columns are also stored as their own struct fields, so on
disk they are **duplicated**. In memory predictors are cheap (`Arc`-shared
`ArrayRef`), but the file layout writes each array's subtree independently. For
high-cardinality targets the residual savings dwarf the duplicate; for low-card
targets it can be a net loss — so the compressor must cost it (see §6).

### Variant B — Struct-aware reconstruction (`VirtualStructArray`)

Move ownership of the models up to the struct so predictors are stored exactly
once and shared by reference.

```
VirtualStructArray  (logical dtype = Struct{..all original fields..})
  metadata:
    base_fields:    the physically-stored columns (predictors + independents),
                    as a normal StructArray
    virtual_fields: repeated {
      name, dtype, model (segments as in Variant A),
      predictor_field_indices: [u32],   # point into base_fields
      switch_field_index, residual_field_index
    }
  canonicalize():
    materialize base_fields, then for each virtual field compute
    prediction(base_fields[predictors]) + residual, and assemble the full struct
    in the original field order.
```

Pros: no predictor duplication; matches the paper's table-level framing; one place
owns the dependency graph (and can express chains, e.g. B depends on A which
depends on C). Cons: new machinery above the per-array `VTable`; projection
pushdown must learn that selecting a virtual field also requires reading its
predictor fields; more invasive to the scan/layout path.

## 5. Recommendation & phasing

Build **Variant A**, narrowest case first, and only graduate to **Variant B** once
the value and the round-trip are proven.

1. **Phase 1 — exact peer offset.** A `LinearModelArray` (or a dedicated
   `PeerOffsetArray`) restricted to `k == 1`, single integer/timestamp predictor,
   unit coeff, exact residual. This is literally "FoR where the reference is a
   column." Covers `dropoff_at = pickup_at + Δ` — the single biggest win (−58%),
   fully lossless, minimal surface. Validates: slot wiring, predictor
   canonicalization, serialization round-trip, validity propagation.
2. **Phase 2 — k-regression + float residual.** Add the segment table, the
   `switch` child, multi-predictor linear combinations, `round(_, scale)`, and the
   float residual via `Patches`. Covers `total_amount`. Validates the lossy-model +
   exact-residual path.
3. **Phase 3 — predictor sharing (Variant B).** Promote model ownership to a
   struct-level encoding so predictors are stored once, and teach the compressor /
   projection path about field dependencies.

## 6. Compressor integration

Discovery is offline cost-based selection, mirroring `virtual`'s greedy driller
and Corra's diff-encoding graph. It plugs into the existing
`CompressorPlugin::compress_chunk` (`vortex-layout/src/layouts/compressed.rs`):

1. On a struct chunk, sample rows and, for each candidate target column, fit
   `sparse-lr` and a few `k`-regressions against correlated sibling columns
   (correlation prefilter to bound the search).
2. Estimate encoded size of `(switch, residual)` vs. the column's best
   single-column encoding (Variant A must also charge for predictor duplication;
   Variant B does not).
3. Greedily select the dependency set that minimizes total size while keeping the
   dependency graph acyclic (no column used as both a virtualized target and a
   predictor of its own predictor).
4. Emit `LinearModelArray` for chosen targets; leave the rest to normal
   single-column compression.

Model fitting (least squares, segment clustering) is the only genuinely new
numeric code; everything downstream reuses `Patches` + existing integer/float
encodings on the residual and switch children.

## 7. Open questions / risks

- **Floating-point determinism.** Reconstruction must be bit-exact across
  platforms. The residual makes the *result* exact, but the *prediction* must be
  computed identically everywhere (fixed evaluation order, no fused-multiply-add
  surprises) so that `prediction + residual` lands on the original. Pin the
  arithmetic; test on the residual, not on an approximate compare.
- **`round(_, scale)`.** The taxi columns are decimals-as-f32 with a known scale.
  We must store and reapply that scale exactly; otherwise residuals balloon.
- **Predictor duplication (Variant A).** Acceptable for high-card targets; the
  compressor must reject cases where it loses. Quantify before committing.
- **Pushdown / projection.** A scan that selects only a virtual column must also
  fetch its predictors. Trivial for Variant A (predictors are children of the
  target). Requires layout awareness for Variant B.
- **Validity.** Null in any predictor (or in the target) must propagate. Reuse
  `ValidityVTableFromChild` semantics; decide whether a null predictor forces a
  residual-stored null or an exception.
- **Scope vs. Corra subaltern.** This design covers peer + (k-)regression. The
  one-to-many dictionary case (`city -> zip`) is a separate encoding shape
  (dictionary + offset-into-group) and is intentionally out of scope here.

## 8. Test plan (when implemented)

- Round-trip `assert_arrays_eq!` for: exact peer offset; k-regression with float
  residual; with row-level validity; with nulls in predictors; empty and
  single-row arrays; sliced arrays (mirror ALP's `rstest` size cases incl.
  1023/1024/1025 boundaries).
- Serialize → deserialize → execute equality (metadata + slot reattachment).
- A regression fixture built from the taxi sample asserting the measured savings
  do not regress.
- Compressor selection unit tests: chooses virtualization only when it wins;
  never produces a cyclic dependency.

## Appendix: artifacts

Reproduction (`virtual-parquet==0.2.2`, Python 3.12) and the captured
`layout/driller/estimates` JSON were generated against
`https://blobs.duckdb.org/data/taxi_2019_04.parquet`. The chosen-function JSON
(`greedy.chosen`) is the authoritative description of the three models above.
