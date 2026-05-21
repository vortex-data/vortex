<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Copy-and-patch SIMD stencils for columnar decode (prototype)

A prototype that decodes three *stacked* Vortex encodings

- `delta(bitpacking)` — `u32`
- `alp(delta(ffor(bitpacking)))` — `f64`
- `rle(alp(delta(ffor(bitpacking))))` — `f64` run values + delta-bitpacked run ends

using four composition strategies that all share the **same** pre-compiled SIMD
kernels ("stencils"). The only variable is *how the stencils are composed*, so
the benchmark isolates the composition strategy rather than the kernels.

## The four strategies

| Strategy | What it is | Where |
|---|---|---|
| `materialized` | Decode each encoding layer into a full-column heap buffer before the next layer reads it. Models Vortex's array-by-array `execute`, which canonicalises a `PrimitiveArray` per layer. | `src/strategies/materialized.rs` |
| `fused` | Tiled L1-resident pipeline: per 1024-element tile, run every layer through register/L1 scratch, never touching DRAM for intermediates. Copy-and-patch with stencils kept as ordinary function pointers; runtime constants (bit-width, FoR reference, ALP scale) passed as arguments. | `src/strategies/fused.rs` |
| `patched` | **Literal copy-and-patch**: the ALP-scale leaf is emitted as machine code at run time by copying a pre-compiled template into an executable page and patching the scale into a `movabs` immediate. Integer stages reuse the `fastlanes` per-width stencils, *selected* by bit-width. | `src/patched.rs` |
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

## Results

Hardware: Intel Xeon (Skylake-class), AVX-512F/DQ/BW/VL/CD + BMI2, 4 cores.
`RUSTFLAGS="-C target-cpu=native"`, `~1M` elements per column. Median decode
throughput (higher is better); `xN` is speedup over `materialized`.

| Stack | materialized | fused | aot | patched | vortex (real) |
|---|---|---|---|---|---|
| `delta(bitpack)` u32 | 812 M/s (1.0x) | 1.27 G/s (1.57x) | 1.36 G/s (1.68x) | — | **1.45 G/s** (1.79x) |
| `alp(delta(ffor(bitpack)))` f64 | 259 M/s (1.0x) | 638 M/s (2.46x) | 640 M/s (2.47x) | 518 M/s (2.00x) | **1.11 G/s** (4.29x) |
| `rle(alp(delta(ffor(bitpack))))` f64 | 406 M/s (1.0x) | 456 M/s (1.12x) | 441 M/s (1.08x) | 428 M/s (1.05x) | — |

### What the numbers say

1. **Tiled fusion beats per-layer materialization, same kernels.** With identical
   stencils, the `fused` L1-tiled pipeline is 1.6x (A) to 2.5x (B) faster than
   `materialized`, which writes a full-column buffer per layer. This is the core
   result: the win comes from never spilling intermediates to DRAM, not from
   better kernels. On stack C the RLE expand stage is memory-bound and dominates,
   so the gap shrinks to 1.1x.

2. **Cheap-to-build `fused` already matches AOT.** Const-generic monomorphisation
   per bit-width (`aot`) is within noise of the runtime-width `fused` path
   (A: 1.36 vs 1.27 G/s; B: 640 vs 638 M/s; C: tie). The `fastlanes` kernels are
   `#[inline(never)]` either way and per-tile width dispatch is cheap, so the
   combinatorial AOT build buys almost nothing here. A copy-and-patch planner that
   *selects* pre-built stencils gets AOT-class throughput without AOT-class build
   cost.

3. **A single-op patched leaf is slower than fused — copy-and-patch needs
   body-stitching.** The `patched` path (B: 518 M/s) is *slower* than `fused`
   (638 M/s) despite baking the ALP scale as an immediate. The cost is the
   per-tile call boundary into the runtime-emitted stencil plus the lost
   cross-stage inlining (untranspose can no longer fuse with the scale multiply).
   The constant-folding win is real but smaller than what one indirect call per
   1024 elements costs. **Conclusion: patching constants into a single op does not
   pay; the technique only wins if op *bodies* are stitched into one loop so the
   call boundaries disappear.** That is the body-stitching variant listed under
   future work, and this negative result is the strongest argument for building
   it.

4. **vs real Vortex.** Vortex's production `execute` is fastest on A and B — but
   it decodes its *own* encoding of the data, which its compressor made shallower
   (≈2 layers: ALP/Delta over bitpacking) than the explicit 4-layer stacks here.
   So the prototype does not beat Vortex's decode of *these* columns; it beats the
   same-kernel `materialized` model of per-layer decode. The apples-to-apples
   4-layer comparison Vortex's public API cannot construct is precisely
   `materialized` vs `fused`, where fusion wins.

### Build ("compile") latency

| Operation | Median |
|---|---|
| `build_patched_stencil` (mmap + copy + patch + mprotect) | 4.7 µs |
| `build_and_run_one_tile` (build + decode 1024 elems) | 8.0 µs |

The copy + 8-byte patch is sub-microsecond; the ~4.7 µs is dominated by the
`mmap`/`mprotect` syscalls that allocate and re-protect the page. Amortised over
a multi-millisecond column decode this is negligible, and a real framework would
pool executable pages to drive per-stencil build into the sub-µs "memcpy" regime.
Either way it is many orders of magnitude below the seconds an LLVM recompile
would need to reach the same code quality.

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
- **Patched backend** (true copy-and-patch): stitch the selected stencil
  *bodies* into one executable buffer and patch immediates in. The result (3)
  above shows why this must stitch bodies, not call per-op stencils: a patched
  *single* op adds a call boundary per tile and loses cross-stage inlining, so it
  ran slower than `fused`. The payoff only materialises once all the op bodies
  live in one loop with no internal calls — then constants fold and there is no
  dispatch, approaching AOT quality with `~memcpy`-cost "compilation". The
  `patched` strategy in this prototype is the partial form (patched leaf +
  selected integer stages); the full backend is future work.

The planner would pick the backend per column: cheap plans run interpreted;
since `fused` already matches AOT here, body-stitched patching is only worth it
for plans dominated by data-dependent constants in tight inner loops. Both
backends share the identical stencil library, so correctness is established once.

### 4. Where it plugs into Vortex

Vortex already decodes array-by-array via `execute`, materialising a
`PrimitiveArray` per layer (the `materialized` strategy here). A stencil planner
would slot in at the scan boundary: instead of `child.execute()?` per layer, the
`LayoutReader` lowers a recognised cascade (e.g. ALP over FoR over bitpacking)
to a tiled plan and produces canonical output in one pass. Unrecognised cascades
fall back to today's path, so it is incremental.

### Open questions / limits of the prototype

- **Stitching real stencil bodies** (true Xu/Kjolstad copy-and-patch, fusing
  multiple op bodies into one loop) is *not* done here: the `patched` backend
  patches a single-op leaf and selects the rest. Fusing arbitrary bodies needs
  relocation-aware extraction (a build-time `object`-crate pass), which is the
  natural next step.
- **FoR reference** is passed as an argument, not patched as an immediate; the
  `unchecked_unfor_pack` stencil already accepts it at runtime.
- **ALP exceptions / nullability / non-tile-aligned tails** are out of scope.
- The Vortex baselines decode Vortex's own Delta / ALP encodings of the same
  data (the public API doesn't expose a hand-built 4-deep cascade), so they are
  an end-to-end anchor; the same-kernel `materialized` strategy is the
  apples-to-apples model of Vortex's per-layer decode.

## Running

```bash
cargo test  -p simd-stencil
RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench stacks
RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench dispatch
```
