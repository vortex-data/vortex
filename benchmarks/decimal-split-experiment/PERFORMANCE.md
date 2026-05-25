<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Decimal limb-split layout: performance summary

This crate is an experiment comparing a **limb-split (struct-of-arrays)** layout
for wide decimals (`i128`/`i256`) against **Arrow-rs interleaved decimals**, for
both compression and compute, under an identical AVX-512 build.

A decimal value is stored not as one contiguous 16-/32-byte integer (Arrow's
array-of-structs) but as separate 64-bit *limb* streams: `lo`/`hi` for `i128`,
four limbs for `i256` (struct-of-arrays). This enables lane-parallel SIMD,
per-limb bit-packing, and skipping limbs that are constant.

## How to reproduce

Everything is gated on `target-cpu=native` so Arrow and the split kernels are
compiled with the **same** instruction set (a CPU-parity table is printed at the
top of the analyze run and must show the features as both compiled-in and
detected).

```bash
# Full narrative report (compression, every op, roofline, micro-opt studies):
RUSTFLAGS="-C target-cpu=native" \
  cargo run --release -p decimal-split-experiment --bin decimal_split_analyze

# Rigorous, reproducible per-op benchmarks (best split vs Arrow), pinned:
RUSTFLAGS="-C target-cpu=native" taskset -c 1 \
  cargo bench -p decimal-split-experiment --bench decimal_ops

# Add-only sweep across magnitudes / scalar baselines:
RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p decimal-split-experiment --bench decimal_arith

# Correctness (every kernel == Arrow / native reference):
cargo test -p decimal-split-experiment
```

Each `decimal_ops` benchmark has an `*_arrow` and an `*_split` sibling; divide
the two `Mitem/s` figures to get the speedup.

## Test machine

4-core x86-64, **L1d 48 KiB, L2 2 MiB, L3 260 MiB**, AVX-512F/BW/DQ/VL/CD. It is
a shared/contended host: absolute numbers vary ±10-20% run to run, so the tables
below are representative and the **ratios** are the stable signal. Use pinned
best-of-N (`taskset -c 1`, the analyze binary's micro-opt sections, or the divan
benches) for the cleanest figures.

The analyze binary (best-of-7 custom timer) and the `decimal_ops` divan bench
(fastest-of-100 samples) are different harnesses, so a given ratio can land a
bit differently between them (e.g. `lt` ~1.2-1.6×, `sum` ~1.4-2.1×); both agree
the split is faster on every op. The doc tables quote the analyze run; reproduce
the direction with `cargo bench`.

## Compute: best split kernel vs Arrow

Identical AVX-512, 65 Ki values (~2 MiB, L2-resident — the regime a chunked
engine runs in), pinned best-of-7. M items/s.

| operation | Arrow | best split | speedup | best variant |
|---|---:|---:|---:|---|
| i128 add | 82 | 649 | **7.9×** | single-vector SIMD |
| i128 lt | 773 | 1231 | **1.6×** | unrolled-by-4 SIMD |
| i128 sum (hi varies) | 2069 | 4314 | **2.1×** | widening SIMD, exact i256 |
| i128 sum (hi const 0) | 2063 | 6328 | **3.1×** | lo-only 4-acc, skips hi |
| i128 min | 1898 | 2934 | **1.6×** | lane-parallel SIMD |
| i128 mul | 279 | 566 | **2.0×** | `vpmullq`+`mulhi` SIMD |
| i128 div\* | 67 | 228 | **3.4×** | scalar (\*different semantics) |
| i256 add | 29 | 297 | **10.4×** | 4-limb SIMD |
| i256 lt | 290 | 510 | **1.8×** | 4-limb lexicographic SIMD |

Notes:
- **add/sub**: lane-parallel `vpaddq` + `vpcmpltuq` carry. Huge multiple because
  Arrow's per-element i128 add is dominated by dispatch/validity/alloc overhead.
- **sum (hi varies)**: accumulates into an exact **i256** (lane accumulators +
  carry counters). Faster than Arrow *and* overflow-correct — Arrow's `sum`
  wraps i128 silently.
- **sum (hi const 0)**: the high stream is skipped entirely (8 B/value vs 16),
  the structural win Arrow physically cannot match.
- **mul**: `vpmullq` for low/cross products + a 32-bit `vpmuludq` `mulhi`. The
  win is mostly Arrow overhead; SIMD width adds little (compute is cheap, work
  is bandwidth-shaped at these sizes).
- **div**: no SIMD divide exists; the split gives no leverage. `*` ours
  truncates, Arrow rescales+rounds (more work) — throughput-only, not identical.

## Compute: cache roofline (why `lt` looks modest)

`lt` over two i128 columns reads 32 B/value. Arrow's scalar i128 compare is
**compute-bound** (~900 M/s, flat across cache levels); the split's 8-wide mask
compare is fast enough to be **bandwidth-bound**, so it tracks cache bandwidth
and they converge at the L3 wall.

| working set | Arrow | split full | split const-hi |
|---|---:|---:|---:|
| 32 KiB (L1) | ~900 | ~4000 (**4.5×**) | ~6000 (7×) |
| 256 KiB (L2) | ~990 | ~2300 (**2.3×**) | ~4000 (4×) |
| 2 MiB (L2 edge) | ~900 | ~1700 (**1.8×**) | ~5900 (6.6×) |
| 32 MiB (L3) | ~830 | ~770 (**~1.0×**) | ~1700 (2×) |
| 256 MiB (DRAM) | ~310 | ~520 (1.7×) | ~1600 (2.6×) |

Takeaway: the SIMD compute win is large when the column is cache-resident; at
the L3/DRAM streaming wall both saturate the same bus, and the only lever left
is moving fewer bytes (const-hi skip).

## Compression

zstd level 3, plus FFoR (frame-of-reference) bit-width per limb (what FastLanes
achieves). The **bit-pack ratio** is the headline: bit-packing the split limbs
is something a 128/256-bit value cannot do directly (no FastLanes lane that
wide), so it is unique to the split.

| column | ffor bits (lo,hi) | bit-pack ratio vs raw i128 |
|---|---|---:|
| synthetic small (fits i64) | (30, 0) | **4.3×** |
| TPC-H l_quantity | (13, 0) | **9.8×** |
| TPC-H l_extendedprice | (24, 0) | **5.3×** |
| TPC-H l_discount / l_tax | (4, 0) | **32×** |
| synthetic full-range i128 | (64, 61) | 1.0× |

Every real money column has a **0-bit high limb** — that is the same property
that powers the const-hi compute fast paths. General-purpose zstd alone barely
benefits (~1.0-1.1×) because it already eats the zero high bytes; the split's
edge is specifically under **bit-packing**.

## Micro-optimization study (try-to-beat-the-asm)

Pinned best-of-7. The lever everywhere is hiding cache latency with more
independent work; it only pays in the **L2-resident** regime.

| kernel | technique | result |
|---|---|---|
| `lt` | unroll-by-4 | **1.1-1.2×** at L2 (more loads in flight) → shipped |
| `sum` | 4 accumulators | **1.07-1.12×** at L2 (breaks loop-carried chain) → shipped |
| `add` | unroll-by-4 | **0.7-0.85×** — *regression* (two zmm outputs → register spill) → not shipped |

Conclusion: micro-asm tuning caps out at ~10-20% in one cache regime. The
order-of-magnitude levers are algorithmic — skip constant limbs, bit-pack.

## Storage-regime caveat

The compute tables assume decimals are **stored split**. If data is stored
interleaved (Arrow's layout) and must be transposed to limbs just to run one
kernel, the transpose (~450 M/s AoS→SoA) erases the win for fast ops (sum/lt
drop to 0.2-0.9×). Conversely, when stored split, *Arrow* pays the gather. So
the split is a win when the data is born/kept split (i.e. it is the storage
encoding), not as a transient conversion.

## Where Arrow wins

- **Interop**: Arrow is the zero-copy lingua franca; the split must convert at
  the boundary.
- **Random access / `scalar_at`**: one contiguous read vs 2-4 limb gathers.
- **Narrow types (i32/i64)**: no benefit — Arrow already SIMDs and FastLanes
  bit-packs them.
- **mul/div**: the split gives no algorithmic leverage.
- **Maturity**: Arrow's kernels are complete; the split needs a kernel per op or
  a canonicalize fallback.

## Recommendation

Do **not** switch the canonical decimal representation. Add the split as a
**compressor-selected encoding** for wide (`i128`/`i256`) decimals, with per-limb
stats and stat-driven kernels (compare, sum, min/max, add) plus a
canonicalize-to-Arrow fallback. This captures the upside (compression 5-32×,
compute 1.6-10×, exact sum) automatically per column, and degrades to today's
behavior where it does not fit (narrow types, random access, interop, mul/div).
The genuinely new value over the existing `decimal-byte-parts` encoding is the
**compute kernels + constant-limb skipping**, which pay in the cache-resident /
constant-high-limb regime — most valuable when decimal columns are wide *and*
small-valued (money columns).
