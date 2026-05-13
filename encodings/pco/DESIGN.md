<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Layered Pco in Vortex — Design

## Goal

Today `vortex-pco` wraps the entire pco pipeline (recast → mode → delta →
bin partition → tANS → page → chunk) inside a single opaque `PcoArray`. That
array compresses well, but it hides the per-stage decorrelations from Vortex.
You can't share its mode detection with non-pco data, you can't reuse its
entropy coder, and you can't pick a point on the compression / random-access
curve other than the one pco's authors chose.

This document specifies a decomposition into ~12 small Vortex arrays, each
corresponding to one stage of the pco algorithm, plus a "layered compressor"
that selects and composes them. The arrays are first-class: they have
decompression kernels, validity, slicing, and (where the layer allows it)
element-level random access.

We build **two compression profiles** out of the same parts so we can
empirically compare them:

- **Fast-RA**: every layer supports O(1) `scalar_at`. Achieved by dropping
  tANS and multi-order Consecutive deltas. Expected to be a ratio loss vs
  full pco but a large latency win on point queries.
- **High-ratio**: matches pco byte-for-byte. Element-level random access
  degrades to *page-level*: a point query decodes the containing page (256
  values per batch × pages of 2k–256k values).

## Non-goals

- Replacing the existing `PcoArray` in the short term. The monolith stays as
  a baseline. The layered stack is an alternative.
- A new file format. All new arrays serialize through the existing Vortex
  array framework (`ArrayParts`, prost metadata, child slots).
- Compression for types pco doesn't support (e.g. variable-length strings).

## Layer inventory

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Input: Primitive<T>, T ∈ {u8,u16,u32,u64,i8,i16,i32,i64,f16,f32,f64}    │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────▼─────────────┐
                │ 1. OrderedLatentArray<T>  │   (recast to u8/u16/u32/u64)
                └─────────────┬─────────────┘
                              │  Primitive<L>, L ∈ {u8,u16,u32,u64}
                ┌─────────────▼─────────────┐
                │ 2. Mode layer (one of):   │
                │    ClassicArray           │   identity
                │    IntMultArray           │   p,s + base:u64
                │    FloatMultArray         │   p,s + base:f64
                │    FloatQuantArray        │   p,s + k:u8
                │    PcoDictArray           │   indices + dict
                └─────────────┬─────────────┘
                              │  PerLatentVar { delta?, primary, secondary? }
                ┌─────────────▼─────────────┐
                │ 3. Delta layer (one of):  │
                │    (passthrough = NoOp)   │
                │    ConsecutiveDeltaArray  │   order, state moments
                │    LookbackDeltaArray     │   window
                │    Conv1DeltaArray        │   bias, weights, qshift
                └─────────────┬─────────────┘
                              │  PerLatentVar of latent streams (entropy-rich, still L)
                ┌─────────────▼─────────────┐
                │ 4. BinPartitionArray<L>   │   bin_idx + offset, bins:Vec<Bin<L>>
                └─────────────┬─────────────┘
                              │  bin_idx: Primitive<u16>  +  offset: var-width bits
                ┌─────────────▼─────────────┐ ┌─────────────────────────────┐
                │ 5. AnsArray (high-ratio)  │ │ 5'. VarWidthBitPackedArray  │
                │    or pass-through (RA)   │ │     offsets keyed by bin    │
                └─────────────┬─────────────┘ └─────────────┬───────────────┘
                              └──────────────┬──────────────┘
                                             ▼
                                       byte buffers
```

Layers 1–4 are "structural": they reshape data without using an entropy
code. Layer 5 is the only one that turns lower entropy into fewer bytes.
This split is what makes the two profiles cheap: Fast-RA keeps 1–4 and
swaps layer 5 for a fixed-width bin_idx bitpack; High-ratio keeps tANS.

## Per-layer specs

Each section gives the array's shape (children/buffers), the encode and
decode kernel signatures, and the random-access cost. All arrays implement
`VTable` with the standard `OperationsVTable` + `ValidityVTable`.

### 1. `OrderedLatentArray<T>`

A typed bijective recast to an unsigned latent. Lives at the bottom of every
stack.

| Aspect | Value |
|---|---|
| `T → L` | `i8/i16/i32/i64 → u8/u16/u32/u64`; `f16/f32/f64 → u16/u32/u64`; unsigned → identity |
| Children | `validity` (slot 0) |
| Buffers | one `Buffer<L>` |
| Metadata | none (T inferred from dtype, L derived from T) |
| Encode | `for x in input { out.push(to_latent_ordered(x)) }` |
| Decode | `for l in buf { out.push(from_latent_ordered(l)) }` |
| Random access | **O(1) per element** |
| Slicing | O(1), slices the underlying buffer |
| Reuse | useful outside pco wherever order-preserving unsigned casts help |

The transforms (from pco source):

```rust
// signed
fn to_latent_ordered(x: i_N) -> u_N { x.wrapping_sub(i_N::MIN) as u_N }

// float
fn to_latent_ordered(x: f_N) -> u_N {
    let bits = x.to_bits();
    if bits & SIGN != 0 { !bits } else { bits ^ SIGN }
}
```

### 2. Mode layer

All mode arrays reconstruct the latent. Their output is `Primitive<L>`; the
final `OrderedLatentArray<T>` above sits on top of *that*.

The natural child layout for every mode-and-delta layer is the pco
`PerLatentVar<T>`: `delta?`, `primary`, `secondary?`. We add a Vortex helper
trait so each multi-stream layer declares its slot names consistently.

#### 2a. `ClassicArray` — pure passthrough

Probably not a distinct array; we'd just skip the mode layer entirely. Listed
for completeness so the Mode enum stays total.

#### 2b. `IntMultArray`

| Aspect | Value |
|---|---|
| Children | `validity`, `primary: Primitive<L>`, `secondary: Primitive<L>` |
| Buffers | none |
| Metadata | `base: u64` |
| Encode | choose `base` heuristically (pco's stats sampler); split each `n` into `(n / base, n % base)` |
| Decode kernel | `out[i] = base.wrapping_mul(primary[i]) + secondary[i]` (wrapping in `L`) |
| Random access | **O(1) per element** (provided children do) |
| Compression contribution | high on data with a true scale (timestamps in coarser units, currency in millicents, …) |

#### 2c. `FloatMultArray`

| Aspect | Value |
|---|---|
| Children | `validity`, `primary: Primitive<L>`, `secondary: Primitive<L>` |
| Metadata | `base: f64` (or `f32` for f32 inputs — TBD, see open questions) |
| Decode kernel | `out[i] = to_latent(base * (primary[i] as f64) + ulp_from_latent(secondary[i]))` |
| Random access | **O(1) per element** |
| Compression contribution | high on float data with finite decimal precision |

#### 2d. `FloatQuantArray`

| Aspect | Value |
|---|---|
| Metadata | `k: u8` (bits of quantization) |
| Decode kernel | `out[i] = from_bits((primary[i] << k).wrapping_add(secondary[i]))` |
| Random access | **O(1) per element** |
| Compression contribution | medium-high on bandwidth-limited floats (ML weights, lossy-acquired sensor data) |

#### 2e. `PcoDictArray`

Distinct from Vortex's general `DictArray` because pco's dictionary lives
in chunk metadata (small, dense) and the index width is sized to the
dictionary, not the input cardinality. We may end up just folding into
Vortex's dict encoding — see open question 3.

| Aspect | Value |
|---|---|
| Children | `validity`, `primary: Primitive<L>` (indices) |
| Buffers | one `Buffer<L>` for the dictionary values |
| Decode kernel | `out[i] = dict[primary[i] as usize]` |
| Random access | **O(1) per element** |
| Compression contribution | very high on low-cardinality columns |

### 3. Delta layer

All delta arrays operate on the *primary* latent stream of the layer above.
Some produce an additional `delta` latent stream (lookback), and all carry
a small per-page initial state that pco serializes into the page header.

#### 3a. `ConsecutiveDeltaArray`

| Aspect | Value |
|---|---|
| Children | `validity`, `primary` |
| Metadata | `order: u8` (1..=7), `initial_states: PerPage<[L; order]>` |
| Encode | apply nth-order difference (`diff_n`) to primary; record `order` initial values per page |
| Decode kernel | prefix-sum reconstructing primary; SIMD-friendly via paired-summation |
| Random access | **page** (must replay from page boundary); element-level only when `order == 0` |
| Compression contribution | high on monotonic / smooth time series |

The Fast-RA profile caps `order` at 0 (i.e. no Consecutive delta) or 1 with
per-element checkpoints (TBD). High-ratio uses pco's default `Auto`.

#### 3b. `LookbackDeltaArray`

| Aspect | Value |
|---|---|
| Children | `validity`, `primary`, `lookbacks: Primitive<L>` (the extra `delta` latent) |
| Metadata | window size |
| Decode | sequential reconstruction within page using lookback indices |
| Random access | **page** |
| Compression contribution | high on data with repeating periods |

#### 3c. `Conv1DeltaArray`

| Aspect | Value |
|---|---|
| Children | `validity`, `primary` |
| Metadata | `bias`, `weights: Vec<i64>`, `qshift: u8` |
| Decode | autoregressive: `out[i] = primary[i] + (bias + Σ w_j * out[i-j]) >> qshift` |
| Random access | **page** |
| Compression contribution | high on smoothly varying signals |

### 4. `BinPartitionArray<L>`

This is where pco's "the ranges that the latent falls into" lives.

| Aspect | Value |
|---|---|
| Children | `validity`, `bin_idx: Primitive<u16>`, `offset: VarWidthBitPackedArray` |
| Metadata | `bins: Vec<Bin<L>>` where `Bin { weight: u32, lower: L, offset_bits: u8 }` |
| Encode | partition the latent range into bins by quantile, compute weights for the entropy coder |
| Decode kernel | `out[i] = bins[bin_idx[i]].lower + offset[i]` (offset is variable-width based on `bin_idx[i]`) |
| Random access | **O(1) per element** (assuming children do) |
| Compression contribution | medium — exploits per-bin offset widths to bitpack tightly |

`bin_idx` deliberately leaves the small-alphabet stream uncompressed at
this layer. Whether the entropy code is tANS (high-ratio) or a fixed-width
bitpack (fast-RA) is decided by the layer 5 choice.

### 5. Entropy code (high-ratio) — `AnsArray`

| Aspect | Value |
|---|---|
| Children | `validity` |
| Buffers | the compressed tANS state stream |
| Metadata | `ans_size_log: u8`, per-bin `weight: u32`, 4 initial states per page |
| Encode | build the tANS table from bin weights; emit symbols (in reverse, as tANS requires) |
| Decode kernel | 4-way interleaved tANS decode (matches pco's SIMD layout) producing `bin_idx[i]` |
| Random access | **batch** (256 elements). Within a batch, decode is sequential |
| Compression contribution | this is the entropy code — typically 1.5–2× on top of structural layers |

### 5'. Entropy code (fast-RA) — `VarWidthBitPackedArray` over `bin_idx`

If we don't use tANS, `bin_idx` is just a small-integer column. We can:

- bitpack it at `ceil(log2(bins.len()))` bits (already an existing capability
  in `vortex-array`); or
- promote per-bin offsets and indices into one combined bit-packed stream
  (slight ratio gain, mild code complexity).

| Aspect | Value |
|---|---|
| Random access | **O(1) per element** |
| Compression contribution | strictly worse than tANS, equal to "Shannon × log-quantization loss" |

We will measure the gap.

### `VarWidthBitPackedArray` (shared, used at layer 4 for offsets)

Distinct from existing `BitPackedArray` because the width is **per-bin**,
not global. Concretely: given `bin_idx[i]`, the i-th element occupies
`bins[bin_idx[i]].offset_bits` bits in a packed buffer with a precomputed
prefix-sum index for `O(1)` random access.

| Aspect | Value |
|---|---|
| Children | `validity`, `bin_idx` (shared with the bin partition layer) |
| Buffers | packed bit buffer, prefix-sum offsets (or, in practice, page-level prefix-sum + per-element within-page scan) |
| Random access | **O(1) per element** with prefix-sums; **page** without |
| Reuse | independently useful for any "variable-width per category" encoding |

## The two profiles

```
Fast-RA profile:                       High-ratio profile:

  OrderedLatentArray<T>                  OrderedLatentArray<T>
        │                                      │
   Mode (any 2a–2e)                       Mode (any 2a–2e)
        │                                      │
   (no Delta, or                          Delta (any 3a–3c)
    Consecutive order=0/1)                     │
        │                                 BinPartitionArray
   BinPartitionArray                            │
        │                                 AnsArray over bin_idx
   BitPackedArray over bin_idx                  │
        │                                 VarWidthBitPackedArray of offsets
   VarWidthBitPackedArray of offsets
```

The structural layers (1–4) are **identical** between profiles. Only the
choice of entropy layer and delta order differs. This is what makes
side-by-side measurement cheap.

## Layered compressor

A new `LayeredPcoCompressor` chooses the stack per column. The selection
algorithm mirrors pco's `mode_spec=Auto`:

1. Sample the column.
2. Run mode detection (try Classic / IntMult base candidates / FloatMult /
   FloatQuant / Dict). Pick the one with the lowest predicted post-mode
   entropy.
3. Run delta detection on the primary latent. Same idea: lowest entropy.
4. Build bin partition on the post-delta latent.
5. Profile = Fast-RA → emit BitPacked bin_idx; profile = High-ratio → emit
   tANS.

The compressor's `compress(array, profile) -> ArrayRef` returns a stack of
nested Vortex arrays. No new file format: the result is just a tree of
existing Vortex arrays that Vortex already knows how to read.

## Random-access strategy

Each `scalar_at(i, ctx)` walks the stack top-down:

```
OrderedLatent: scalar_at(i) → from_latent_ordered(inner.scalar_at(i))
IntMult:       scalar_at(i) → base*p.scalar_at(i) + s.scalar_at(i)
FloatMult:     scalar_at(i) → f64-recover from p,s
FloatQuant:    scalar_at(i) → bits-recover from p,s
PcoDict:       scalar_at(i) → dict[p.scalar_at(i)]
Consecutive:   scalar_at(i) → decode page containing i, then index
Lookback:      scalar_at(i) → decode page containing i, then index
Conv1:         scalar_at(i) → decode page containing i, then index
BinPartition:  scalar_at(i) → bins[idx.scalar_at(i)].lower + offset.scalar_at(i)
Ans:           scalar_at(i) → decode batch containing i, then index
BitPacked:     scalar_at(i) → O(1)
VarWidthBP:    scalar_at(i) → O(1) with prefix-sums
```

The recursion stops at the first "page-granular" or "batch-granular" layer.
In Fast-RA, every step is element-level, so `scalar_at` is O(1) modulo log
factors from prefix-sum lookups.

## Measurement plan

We will produce one Criterion bench suite and one summary table. The
dimensions are:

- **dataset** ∈ {ClickBench, TPC-H lineitem/orders, NYC taxi, synthetic}
- **column type** ∈ {i32, i64, f32, f64, low-card dict-like}
- **profile** ∈ {Fast-RA, High-ratio, monolithic PcoArray (baseline)}
- **operation** ∈ {full decompress, scalar_at × 1k random indices, filter scan}

### Metrics per (dataset, column, profile, operation)

| Metric | Why |
|---|---|
| compressed bytes | the headline ratio |
| compressed bytes by layer | tells us *which* layer earned each fold |
| full-decompress time | bulk read latency |
| random-access p50 / p99 | point-query latency |
| filter-scan throughput | analytical-query latency |
| encode time | not the priority, but informative |

The "compressed bytes by layer" comes for free in the layered stack: each
nested array reports its byte size. For the monolithic `PcoArray` we'll
treat its total as the ratio reference.

### Synthetic micro-benchmarks (per layer)

For each layer we generate a small dataset that *should* favor it, plus a
control that shouldn't. This validates each layer in isolation before we
trust the combined numbers.

| Layer | Favorable | Control |
|---|---|---|
| OrderedLatent | any (it's free) | — |
| IntMult | `[k*base + r for k]` random `r∈[0,base)` | uniform random |
| FloatMult | `[k * 0.01 for k in 0..N]` | uniform random `f64` |
| FloatQuant | floats with mantissa zeroed below k | uniform random `f64` |
| PcoDict | low-cardinality strings of ints | uniform random |
| Consecutive | monotone timestamps | uniform random |
| Lookback | periodic signal | uniform random |
| Conv1 | smoothly varying signal | uniform random |
| BinPartition + tANS | skewed-distribution ints | uniform random |

### Datasets (real)

- **ClickBench** — wide mix; we'll focus on integer and float columns.
  Already wired into `vortex-bench`.
- **TPC-H** — `lineitem.l_orderkey` (monotone i64, big delta win),
  `lineitem.l_extendedprice` (decimal-like, IntMult win),
  `orders.o_orderdate` (date i32). Already wired.
- **NYC taxi fares** — `fare_amount`, `trip_distance` (FloatMult/FloatQuant
  signal). Add a small subset (~1M rows) under `vortex-bench/data` if not
  present.

## Phasing

| Phase | Deliverable | Acceptance |
|---|---|---|
| **P0** | This design doc, reviewed | landed on `claude/analyze-pco-schemes-91DAI` |
| **P1** | `OrderedLatentArray` + `ClassicArray` (no-op pass) + tests | one round-trip test per `T` |
| **P2** | The four mode arrays (`IntMult`, `FloatMult`, `FloatQuant`, `PcoDict`) | per-mode round-trip on synthetic favorable data |
| **P3** | `ConsecutiveDeltaArray`; defer `Lookback` and `Conv1` until benches show they're worth the complexity | round-trip on synthetic favorable data |
| **P4** | `BinPartitionArray` + `VarWidthBitPackedArray` | round-trip on uniform & skewed data |
| **P5** | `AnsArray` (tANS) | round-trip parity with pco's tANS output on a fixture |
| **P6** | `LayeredPcoCompressor` + the two profiles | per-column profile selection works end-to-end |
| **P7** | Bench suite, summary table | populated table covering all (dataset, profile, op) cells |

P1 through P3 are low-risk; the values can be cross-checked against the
existing monolithic `PcoArray`'s decompressed output. P5 (tANS) is the
single risky phase — it warrants its own design pass before implementation.

## Open questions

1. **Crate boundaries.** `OrderedLatentArray`, `VarWidthBitPackedArray`, and
   `AnsArray` are reusable beyond pco. Do they live in `vortex-array`,
   their own crates, or in `encodings/pco` until we have a second user?
2. **`PerLatentVar` shape.** Should we add a Vortex helper for "up to three
   named child slots (delta, primary, secondary)" or just open-code the
   slots per array?
3. **PcoDictArray vs general DictArray.** Pco's dict mode is mechanically
   the same as Vortex's dict encoding. The only differences are: (a) pco
   sizes the index width to `ceil(log2(|dict|))` rather than a `PType`, and
   (b) pco carries the dict in chunk metadata. Is reusing `DictArray` worth
   the indirection?
4. **f32 FloatMult base.** Pco always uses `f64` for the base. Keep parity,
   or use `f32` for f32 inputs to save 4 bytes of chunk metadata?
5. **u8/i8 support.** Pco gates on `enable_8_bit`. Worth wiring in this
   pass, or defer?
6. **Where do page boundaries go in the layered world?** Today they live
   inside `PcoArray`. Options: (a) keep page-as-array (each page a leaf
   array, chunk is a `ChunkedArray`); (b) keep page slicing inside the
   per-layer arrays. Option (a) plays nicely with `vortex-layout` and
   makes page-level random access free.
7. **Fast-RA Consecutive.** Is `order=1` Consecutive worth supporting in
   Fast-RA via per-element checkpoints (e.g. one anchor per 64 elements)?
   Cheap to add; would let us keep some delta signal at O(1) per element.
