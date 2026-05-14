# aot-size (research)

Measures the AOT code size and compile time of `fastlanes::BitPacking::unpack`
and `BitPackingCompare::unpack_cmp` monomorphizations, to put the
copy-and-patch JIT savings on a real numeric footing.

Standalone crate (workspace-excluded). `lto = false`, `codegen-units = 1`,
`strip = "none"` so symbol sizes line up with the actual emitted code.

## Binaries

| Binary                  | What it monomorphizes                                              |
|-------------------------|---------------------------------------------------------------------|
| `unpack_only`           | `BitPacking::unpack` for u8/u16/u32/u64 (reachable through one dispatch) |
| `cmp_one_op`            | `unpack_cmp` for u32, single op (`eq`)                              |
| `cmp_all_ops_u32`       | `unpack_cmp` for u32 × 6 ops                                        |
| `cmp_all_ops_all_types` | `unpack_cmp` for u8/u16/u32/u64 × 6 ops — the full fastlanes matrix |

## Run

```
./measure.sh
```

Times a clean release build per binary, reads `.text` from the ELF header,
and sums the size of every `fastlanes::bitpacking*` symbol it finds.

## Results (Rust 1.91, x86-64, codegen-units=1, no LTO)

```
unpack_only             build= 13.75s  .text=   469952 B  fastlanes kernels:   125 syms /   194673 B
cmp_one_op              build= 14.62s  .text=   313088 B  fastlanes kernels:    32 syms /    46795 B
cmp_all_ops_u32         build= 18.72s  .text=   588048 B  fastlanes kernels:   190 syms /   315742 B
cmp_all_ops_all_types   build= 38.37s  .text=  2189552 B  fastlanes kernels:   712 syms /  1895816 B
```

Per-type breakdown of `cmp_all_ops_all_types`:

| Type | Syms | Bytes     | Avg/kernel |
|------|------|-----------|------------|
| u8   |  46  |    19,021 |     413 B  |
| u16  |  94  |    70,778 |     752 B  |
| u32  | 190  |   315,742 |   1,661 B  |
| u64  | 382  | 1,490,275 |   3,901 B  |

(`46 = 7 widths × 6 ops + dispatchers/etc`, and similarly upward.)

## Context: a real Vortex binary

| Binary                | .text       | fastlanes kernels                  | kernels / .text |
|-----------------------|-------------|-------------------------------------|-----------------|
| `vx` (vortex-tui)     | 70,125,002 B (≈ 70 MB) | 525,027 B (188 unpack + 121 pack + 12 dispatch, **0 `unpack_cmp`**) | 0.75 %           |
| `aot-size cmp_all_ops_all_types` | 2,189,552 B | 1,895,816 B                  | 86.6 %           |

Note `vx` has **zero `unpack_cmp` monomorphizations** — confirming that
Vortex doesn't wire `BitPackingCompare` into its compute layer today. Any
"all 6 compare ops" figures we cite here are *hypothetical incremental cost*
if that wiring landed.

## What the JIT saves

The interesting axis is **op fan-out at fixed `(T, W)`**. A copy-and-patch
stencil keeps one body per `(T, W)` and splices in the op:

```
              AOT (6 ops, all widths)  Stencil-JIT (1 stencil per width, 8-byte slot)
              -----------------------  ----------------------------------------------
u8 only           19,021 B                  ~3,170 B    (~6× reduction)
u16 only          70,778 B                 ~11,800 B    (~6× reduction)
u32 only         315,742 B                 ~52,600 B    (~6× reduction)
u64 only       1,490,275 B                ~248,400 B    (~6× reduction)
all 4 types    1,895,816 B                ~316,000 B    (~6× reduction)
```

So **collapsing the 6-way op fan-out via copy-and-patch saves ~1.58 MB of
code (≈83%) across the full cartesian.** Adding more axes — signed vs
unsigned ordering, masked-output variants, fused predicate chains — would
multiply on top of that.

In `vx`-sized terms: **adding all 6 ops AOT would grow `.text` from 70 MB
to 72 MB (+2.7 %); JIT'd, it would grow to ~70.3 MB (+0.45 %).**

It also collapses compile time: `cmp_one_op` → `cmp_all_ops_all_types`
adds 24s of `rustc` work. JIT'd, those 24s become a few microseconds of
patching per kernel that's actually used.

## What the JIT does NOT save here

- The **unpack body** dominates kernel size (the largest u64 kernel is 8,231
  bytes; the compare itself is one to two SIMD instructions out of hundreds).
  A real stencil-JIT for bitpacking would still need one stencil per `(T, W)`
  — the savings come entirely from collapsing op fan-out, not unpack shapes.
- A genuinely **W-parameterised** unpack would need runtime shift/mask
  values, which costs cycles. Worth it only if the (T, W) cartesian is also
  large enough to matter.

## SPDX

```
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
```
