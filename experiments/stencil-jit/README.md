# stencil-jit (research prototype)

A minimal copy-and-patch JIT for fused SIMD `eq` / `neq` on 32 packed `u8`
lanes. Demonstrates the mechanism the dynamic-bitpacking-kernels discussion
named "clone machine code and adjust from one op to another."

**Not** wired into Vortex's compute layer. **Not** production-ready. Lives
outside the Cargo workspace via the top-level `exclude` so it stays out of
`cargo build --workspace`, clippy, and `public-api.lock`.

## Scope

| Dimension      | This prototype                       | Real version would need                              |
|----------------|---------------------------------------|------------------------------------------------------|
| Target         | x86-64 Linux only                     | + Windows/macOS, NEON for aarch64, AVX-512 path      |
| Vector ISA     | AVX2 (32 x u8 lanes)                  | SSE2 / AVX2 / AVX-512 / NEON / SVE selection         |
| Bit width      | `bw = 8` (no real unpack)             | One stencil per `bw` in 1..=32, or a single unpack-by-shift kernel parameterized at compile time |
| Ops            | `eq, neq, lt, le, gt, ge` (signed)    | + unsigned variants, plus arithmetic and fused predicates |
| Patch sites    | One 8-byte slot                       | Multiple slots with relocation table (stencil graph) |
| Constant       | `u8` broadcast                        | Type-generic broadcast, or in-register predicate     |

The stencil is 39 bytes; the splice slot is 8 of them.

## The six ops

All six fit in the 8-byte splice slot:

```
eq : vpcmpeqb ymm0,ymm0,ymm1            ; 4-byte nop pad
neq: vpcmpeqb ymm0,ymm0,ymm1            ; vpxor ymm0,ymm0,ymm2
gt : vpcmpgtb ymm0,ymm0,ymm1            ; 4-byte nop pad  (signed)
lt : vpcmpgtb ymm0,ymm1,ymm0            ; 4-byte nop pad  (signed; operands swapped)
ge : vpcmpgtb ymm0,ymm1,ymm0            ; vpxor ymm0,ymm0,ymm2     (!lt)
le : vpcmpgtb ymm0,ymm0,ymm1            ; vpxor ymm0,ymm0,ymm2     (!gt)
```

`ymm2` is materialised once in the prologue as all-ones via `vpcmpeqb ymm2,ymm2,ymm2`,
so the inversion is one 4-byte `vpxor` rather than a load.

Run `cargo run --example dump` to print the exact 39-byte buffer for each
op side-by-side. Bytes 21..29 are the only ones that change between the six.

## Code-size impact

Companion crate `experiments/aot-size` measures the AOT cost of every
`unpack_cmp` monomorphization in `fastlanes-rs`. Headline numbers:

| Variant                                  | Size       |
|------------------------------------------|------------|
| AOT, eq only, u32, all widths            | 47 KB      |
| AOT, 6 ops, u32, all widths              | 316 KB     |
| AOT, 6 ops, u8/u16/u32/u64, all widths   | 1.9 MB     |
| `vx` (vortex-tui) `.text` for context    | 70 MB      |
| `vx` fastlanes kernels today             | 525 KB     |

A copy-and-patch JIT that keeps **one stencil per (T, W)** and splices in
the op collapses the 6-way op fan-out to ~316 KB across all types (≈6×
reduction, ≈83% saving on the 1.9 MB AOT figure). See `aot-size/README.md`.

## How it works

1. **AOT**: `src/stencil.rs` emits the stencil into `.rodata` via
   `core::arch::global_asm!`, with four `.globl .hidden` labels marking
   `stencil_start`, `patch_start`, `patch_end`, `stencil_end`. The body
   between `patch_start` and `patch_end` is 8 NOPs — the eq pattern.
2. **JIT**: `Kernel::compile(op)` mmaps a page `PROT_READ | PROT_WRITE`,
   `memcpy`s the stencil bytes in, overwrites the 8-byte slot with either
   `EQ_PATCH` (8 NOPs) or `NEQ_PATCH` (the two-instruction sequence below),
   then `mprotect`s the page to `PROT_READ | PROT_EXEC`.
3. **Call**: `Kernel::call(packed, constant, out)` invokes the entry point
   as `extern "sysv64" fn(*const u8, u64, *mut u32)`.

W^X is preserved end-to-end: the page is never simultaneously writable and
executable. `mprotect` serializes; x86-64 has coherent icaches; no explicit
flush is needed.

## The patch

```
default (eq): 90 90 90 90 90 90 90 90
                                       (8 x nop — identity, eq mask passes through)

patched (neq): c5 f5 76 c9             vpcmpeqb ymm1, ymm1, ymm1   ; ymm1 := all-1s
               c5 fd ef c1             vpxor    ymm0, ymm0, ymm1   ; invert eq mask
```

Both forms are exactly 8 bytes, so the surrounding `vpmovmskb / mov / vzeroupper / ret`
stays at the same address. The neq form clobbers `ymm1`, but it's dead after
the broadcast.

## Running

```
cd experiments/stencil-jit
cargo test           # 5 tests, exhaustive over constants 0..=255
cargo run --example dump   # prints stencil bytes for eq and neq side-by-side
```

## What this is not

`fastlanes-rs`'s `BitPackingCompare::unpack_cmp<W, B, V, F>` already gives
you the AOT version of this with a closure for the op, and `rustc`
monomorphizes a fully-specialized kernel per `(W, op)` pair. For everything
known at compile time, that is strictly better than a JIT — same generated
code, no runtime cost, no `unsafe`, no platform-specific runtime.

A real copy-and-patch JIT pays for itself only when the op cannot be known
ahead of time: user-supplied predicates from a query planner, dictionary
code mappings discovered mid-scan, or fused operator chains whose Cartesian
product is too large to ship statically. For Vortex today the higher-ROI
work is wiring `unpack_cmp` into `BitPackedArray::compare`; this prototype
is a sketch of where the door leads if that turns out not to be enough.

## SPDX

```
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
```
