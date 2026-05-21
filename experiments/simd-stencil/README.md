<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Copy-and-patch SIMD stencils for columnar decode (prototype)

A prototype that decodes three *stacked* Vortex encodings

- `delta(bitpacking)` — `u32`
- `alp(delta(ffor(bitpacking)))` — `f64`
- `rle(alp(delta(ffor(bitpacking))))` — `f64` run values + delta-bitpacked run ends

using several composition strategies that share the **same** pre-compiled SIMD
kernels ("stencils"). The only variable is *how the stencils are composed*, so
the benchmark isolates the composition strategy rather than the kernels.

## The strategies

| Strategy | What it is | Where |
|---|---|---|
| `materialized` | Decode each encoding layer into a full-column heap buffer before the next layer reads it. Models Vortex's array-by-array `execute`, which canonicalises a `PrimitiveArray` per layer. | `src/strategies/materialized.rs` |
| `fused` | Tiled L1-resident pipeline: per 1024-element tile, run every layer through register/L1 scratch, never touching DRAM for intermediates. Copy-and-patch with stencils kept as ordinary function pointers; runtime constants (bit-width, FoR reference, ALP scale) passed as arguments. | `src/strategies/fused.rs` |
| `patched` | Copy-and-patch with a *single* runtime-emitted leaf: the ALP-scale stage is machine code copied into an executable page with the scale patched into a `movabs` immediate; integer stages reuse the `fastlanes` per-width stencils, selected by bit-width. | `src/patched.rs` |
| `stitched` | **Body-stitching copy-and-patch**: several op bodies are concatenated into *one* executable AVX-512 loop, with constants patched in and the loop back-edge relocated at build time. No per-op calls, no inter-op materialization. | `src/stitched.rs` |
| `aot` | Ahead-of-time upper bound: a fully-inlined, const-generic pipeline monomorphised for the exact `(stack, bit-width)`, dispatched by a `match` over every width. Every combination compiled offline. | `src/strategies/aot.rs` |

The integer kernels delegate to the `fastlanes` crate, whose per-bit-width
unpack routines are themselves pre-compiled SIMD stencils — exactly the
"fixed set of kernels over varied data" shape the columnar copy-and-patch idea
targets. The ALP-scale (`i64 -> f64 * scale`) and RLE-expand kernels are written
here (`src/kernels.rs`). ALP exception patching is intentionally omitted.

## Two faces of "patch in the constant"

The prototype demonstrates that columnar decode has **two** kinds of constant,
and copy-and-patch handles each differently:

1. **Enumerable constants — patch by *selection*.** The bit-width has 1..=64
   possible values, so the fully-optimized kernel for every width is compiled
   ahead of time (this is what `fastlanes` ships). "JIT" just indexes the right
   stencil. No code generation, no immediate rewriting.
2. **Data-dependent constants — patch by *immediate*.** The ALP scale (and, in a
   fuller system, the FoR reference) is an arbitrary 64-bit value that cannot be
   enumerated. Here copy-and-patch earns its keep: the constant is baked into the
   instruction stream as an immediate (`src/patched.rs`), so the inner loop folds
   it instead of carrying it in a register or reloading it from memory.

## Why was Vortex faster than the prototype at first?

An earlier version of this prototype reported that Vortex's production `execute`
beat the stencil pipeline. Dumping the encoding trees (test `inspect_vortex_trees`)
explained why — **Vortex's compressor stored uncompressed primitive children**:

```text
=== stack A: Vortex Delta of gen_u32 ===
fastlanes.delta  len=65536
  .deltas  vortex.primitive  nbytes=262144   <- raw u32, NOT bitpacked
=== stack B: Vortex ALP of gen_f64 ===
vortex.alp  len=65536
  .encoded vortex.primitive  nbytes=524288    <- raw i64, NOT delta/for/bitpacked
```

So "Vortex" never decoded the 4-layer stack at all: its `execute` was essentially
*just the last kernel over uncompressed memory* (undelta+untranspose for A; an
i64→f64 scale for B), 1–2 passes versus the prototype's 3–4. It was faster
because it did **less work**, on a shallower encoding. (A second bug compounded
it: the prototype's stack-B FoR used an *unsigned* min, so wrapping-signed deltas
bit-packed to width 64 — zero compression. Fixed to a signed reference; width is
now ~17.)

The fair question is: decode the **same** stack both ways. Two same-stack Vortex
baselines (`src/vortex_baseline.rs`) build genuine `delta(bitpacking)` (stack A)
and `delta(ffor(bitpacking))` (stack-B integer core) as real `DeltaArray` /
`FoRArray` / `BitPackedArray` trees and decode them through Vortex's per-layer
`execute`.

## Results

Hardware: Intel Xeon (Skylake-class), AVX-512F/DQ/BW/VL/CD + BMI2, 4 shared vCPUs
(a noisy cloud box — treat ratios, not absolutes, as the signal).
`RUSTFLAGS="-C target-cpu=native"`, ~1M elements/column. Median items/s.

| Stack | materialized | **fused** | **aot** | patched | vortex (same stack) | vortex (shallow) |
|---|---|---|---|---|---|---|
| A `delta(bitpack)` u32 | 739 M | **1.16 G** | **1.22 G** | — | 1.01 G | 1.44 G¹ |
| B core `delta(ffor(bitpack))`→i64 | — | **617 M** | **613 M** | — | 454 M | — |
| B full `alp(delta(ffor(bitpack)))` f64 | 204 M | **534 M** | **541 M** | 433 M | — | 887 M¹ |
| C `rle(alp(delta(ffor(bitpack))))` f64 | 352 M | **403 M** | **408 M** | ~367 M | — | — |

¹ *shallow* = Vortex's own (uncompressed-child) encoding, i.e. **less work** — shown for context, not a same-stack comparison.

### What the numbers say

1. **The prototype beats Vortex on the same stack.** `fused`/`aot` decode genuine
   `delta(bitpacking)` 1.1–1.2× faster than Vortex's `execute` of the identical
   array (A: 1.16–1.22 G vs 1.01 G), and the `delta(ffor(bitpacking))` integer
   core 1.3–1.4× faster (B core: 617 M vs 454 M). The win is structural: Vortex
   materialises a `PrimitiveArray` between every layer; the fused pipeline keeps
   intermediates in L1. (Vortex's *shallow* encoding is still fastest where it
   applies — but only because it decodes far less, and a planner could choose that
   shallow encoding too.)

2. **Tiled fusion beats per-layer materialization with identical kernels** —
   2.0–2.6× on the deep f64 stack B (534 M vs 204 M), 1.15× on the
   memory-bound RLE stack C.

3. **`fused` matches `aot`.** Runtime-width fused is within noise of the
   const-generic-per-width AOT build (A: 1.16 vs 1.22 G; B: 534 vs 541 M; C: tie).
   The combinatorial AOT compile buys almost nothing — selecting pre-built
   stencils at runtime already reaches AOT-class throughput.

4. **A single-op `patched` leaf still trails `fused`** (B: 433 vs 534 M): one
   indirect call per tile plus a materialised `digits` buffer between untranspose
   and scale costs more than baking the scale saves. This is the motivation for
   body-stitching, below.

### Body-stitching matches AOT (`--bench stitch`)

The fix for (4): stitch op bodies into one loop. The `stitch` bench runs a 6-op
affine tail (`x = x*a + b` chained — a stand-in for FoR-add → ALP-scale → …) four
ways:

| | items/s | vs stitched |
|---|---|---|
| `aot_const` (ops baked as constants, LLVM-vectorized) | 851 M | 1.08× |
| **`stitched`** (bodies concatenated, constants hoisted into a patched pool) | **786 M** | 1.0× |
| `per_op_materialized` (one pass per op) | 307 M | 0.39× |
| `aot_dynamic` (ops in a runtime slice — can't vectorize) | 183 M | 0.23× |

**Body-stitching reaches 92% of AOT** while beating per-op materialization 2.6×
and a naive plan interpreter 4.3×. The build assembles one AVX-512 loop at run
time: copy prologue + N op bodies + epilogue, patch the constant pool, and
relocate the loop's back-edge `rel32` (the branch distance depends on how many
bodies were stitched). Getting constants *hoisted out of the loop* (into a patched
pool addressed via `r8`) rather than re-broadcast per iteration is what closes the
gap to AOT — re-broadcasting per iteration left it at ~74%.

### Build ("compile") latency

| Operation | Median |
|---|---|
| `build_patched_stencil` (mmap + copy + patch + mprotect) | 6.2 µs |
| `build_stitched_6op` (mmap + 8 fragments + 2 relocations + pool + mprotect) | 4.6 µs |
| `build_and_run_one_tile` | 11 µs |

The copy + patches are sub-microsecond; the few µs are the `mmap`/`mprotect`
syscalls. Amortised over a multi-millisecond column decode this is negligible, and
pooling executable pages would push per-stencil build into the sub-µs "memcpy"
regime — orders of magnitude below the seconds an LLVM recompile needs for the
same code quality.

## How this becomes a framework

The prototype hard-codes three pipelines, but the structure generalises to a
*decode planner* over a small, fixed stencil library. The design:

### 1. Stencil library (compiled once, offline)

A stencil is a tile transform with a fixed ABI:

```text
fn(input_tile, output_tile, &Constants)         // tile = 1024 elements
```

The library is the cross product of `{operation} x {phys-type} x {enumerable
const}` — e.g. `bitunpack[u32][w=1..32]`, `unffor[u64][w=1..64]`,
`undelta[u32|u64]`, `untranspose[...]`, `alp_scale`, `rle_expand`. All are built
with `-O3`/AVX-512 by the normal Rust/LLVM toolchain. This is bounded: a few
hundred stencils, not a combinatorial explosion, because only *enumerable*
constants are monomorphised.

### 2. Plan = a list of (stencil, constants)

A stacked encoding lowers to an ordered list of stencil selections plus the
data-dependent constants pulled from the array's metadata (per-tile width, FoR
reference, ALP exponent/scale, run layout). Building the plan is a `Vec` push
per layer — the sub-microsecond "compile". This is the `fused` strategy made
data-driven instead of hand-written.

### 3. Two execution backends behind one plan

- **Interpreted/fused backend** (`fused` here): walk the plan per tile, calling
  each selected stencil through a function pointer with constants as arguments.
  Robust, portable, already within a small factor of AOT.
- **Stitched backend** (true copy-and-patch): concatenate the selected stencil
  *bodies* into one executable loop and patch in constants + relocations. The
  `stitched` prototype shows this reaches 92% of AOT and beats the per-op
  `patched`/materialized form 2.6×, because the call boundaries and inter-op
  buffers disappear and constants are hoisted out of the loop. This is the backend
  for plans whose tight inner loops are dominated by data-dependent constants.

The planner picks the backend per column: cheap plans run interpreted (`fused`
already matches AOT); constant-heavy tight loops get stitched. Both backends share
the identical stencil library, so correctness is established once.

### 4. Where it plugs into Vortex

Vortex already decodes array-by-array via `execute`, materialising a
`PrimitiveArray` per layer (the `materialized` strategy here). A stencil planner
would slot in at the scan boundary: instead of `child.execute()?` per layer, the
`LayoutReader` lowers a recognised cascade (e.g. ALP over FoR over bitpacking)
to a tiled plan and produces canonical output in one pass. Unrecognised cascades
fall back to today's path, so it is incremental.

### Open questions / limits of the prototype

- **Stitching is demonstrated only for elementwise ops** (the affine tail). The
  hand-authored fragments are AVX-512 written by hand, so the heavy permutation
  kernels (bit-unpack, undelta, untranspose) are still *selected* pre-built
  stencils, not stitched. Stitching arbitrary bodies needs relocation-aware
  extraction of compiled stencils (a build-time `object`-crate pass) — the
  natural next step to fold the whole tail (incl. untranspose) into one loop and
  remove the `digits` round-trip that holds `patched` back on the full stack B.
- The stitched engine caps at `MAX_OPS = 6` (zmm register budget) and patches a
  back-edge `rel32` plus a constant pool by hand; a general stitcher would handle
  arbitrary register allocation and relocation types.
- **ALP exceptions / nullability / non-tile-aligned tails** are out of scope.
- The same-stack Vortex baselines cover the integer cascades (stack A fully;
  stack B's integer core). Wrapping the deep stack back in Vortex's ALP needs its
  exact exponents+patches, so stack B's ALP scale is compared as the equal
  constant overhead it is.

## Running

```bash
cargo test  -p simd-stencil
RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench stacks
RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench stitch
RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench dispatch
```
