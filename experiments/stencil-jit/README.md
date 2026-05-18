# stencil-jit (research prototype)

A minimal copy-and-patch JIT for fused SIMD compare on 32 packed `u8` lanes,
extended with a second splice slot for an optional **FFoR-add** fragment.
Demonstrates the mechanism the dynamic-bitpacking-kernels discussion named
"clone machine code and adjust from one op to another," and the
*chain* extension that lets multiple fragments compose at JIT time.

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
| Patch sites    | Two 8-byte slots (FFoR + compare)     | More slots + relocation tables (stencil graph), or Cranelift-at-build-time |
| Constant       | `u8` broadcast                        | Type-generic broadcast, or in-register predicate     |

The stencil is 59 bytes total; the two splice slots are 8 bytes each.

## Chains: FFoR + compare

The stencil has two patch slots, applied in order on `ymm0`:

```
SLOT 1 (8 bytes)   FFoR-add        vpaddb ymm0,ymm0,ymm3   ; nop4
                   off             8 x nop

SLOT 2 (8 bytes)   compare op      one of {eq, neq, gt, lt, ge, le}
```

The prologue always broadcasts both the compare constant (into `ymm1`) and
the FFoR reference (into `ymm3`). With SLOT 1 NOP'd, `ymm3` is set up but
unread — a couple of wasted cycles, gone behind one stencil that covers
both "with FFoR" and "without."

`ChainConfig { ffor: true | false, op: CmpOp }` gives 2 × 6 = 12
distinct kernels from the same 59-byte AOT body. `cargo test` exhaustively
verifies every (op, ffor) pair across all 256 constants and 6 reference
values.

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
| `compress-bench` `.text` for context     | 35 MB      |
| `compress-bench` fastlanes kernels today | 524 KB     |

A copy-and-patch JIT that keeps **one stencil per (T, W)** and splices in
the op collapses the 6-way op fan-out to ~316 KB across all types (≈6×
reduction, ≈83% saving on the 1.9 MB AOT figure). See `aot-size/README.md`.

## How it works

1. **AOT**: `src/stencil.rs` emits the stencil into `.rodata` via
   `core::arch::global_asm!`, with six `.globl .hidden` labels marking
   `stencil_start`, `ffor_start`, `ffor_end`, `op_start`, `op_end`,
   `stencil_end`. Both slots default to 8 NOPs.
2. **JIT**: `Kernel::compile(config)` mmaps a page `PROT_READ | PROT_WRITE`,
   `memcpy`s the stencil bytes in, overwrites SLOT 1 with FFoR-add or NOPs,
   overwrites SLOT 2 with the chosen compare op, then `mprotect`s the page
   to `PROT_READ | PROT_EXEC`.
3. **Call**: `Kernel::call(packed, constant, out, ffor_ref)` invokes the
   entry as `extern "sysv64" fn(*const u8, u64, *mut u32, u64)`.

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
cargo test                        # 4 tests, exhaustive over (op, ffor, constant, ref)
cargo run --example dump          # stencil bytes for several (ffor, op) configs
cargo run --release --example bench   # microbenchmark vs AOT alternatives
```

## Benchmark: FFoR-add + compare(==) on 32 packed u8 lanes

Three runs at 50 M iterations each (median ± ~0.05 ns of variance), AVX2,
warm cache:

| Variant                              | ns/call | GB/s   |
|--------------------------------------|---------|--------|
| stencil-JIT (FFoR + eq)              |  3.22   |  9.93  |
| AOT closure-based (autovectorized)   |  4.41   |  7.25  |
| AOT AVX2 intrinsics (`#[target_feature]`) | 1.33 | 24.0   |
| Scalar baseline                      |  4.41   |  7.25  |

JIT compile cost: **~5 µs** to mmap, memcpy, patch, mprotect.

What the numbers say:

* The JIT's body is **hand-tuned AVX2** (literally the same instructions
  the intrinsics version compiles to), so per-instruction cost is identical
  to the intrinsics version. The 1.9 ns gap is the **indirect-call tax**:
  the JIT kernel lives behind a function pointer and can't be inlined into
  the timing loop, while the intrinsics version is inlined.
* Against the **closure-based AOT** version (the realistic "no JIT" choice),
  the JIT is actually ~25 % faster — closures bring their own overhead and
  rustc/LLVM don't always nail the autovec.
* In a real query path where each kernel processes thousands or millions of
  blocks, the indirect-call tax is amortized down to ~irrelevant.
* The **5 µs compile cost** is amortized after about 1.5 k blocks (50 KB
  of data). For Vortex's 1024-element blocks that's essentially nothing.

So the real value of the JIT here is **code size and compile-time**, not
per-call speed: ~6× smaller binary for the op fan-out (see code-size
section above), and zero rustc time spent monomorphizing kernels nobody
calls.

## Delta — what it would take

`fastlanes-rs`'s `Delta::undelta_pack<LANES, W, B>` (delta.rs) works on the
transposed FastLanes layout: 1024 values laid out as `LANES = 32 or 64`
lanes of `1024 / LANES` values each, packed contiguously per lane. The
trick is that delta-undo runs *independently per lane*, with each lane's
"prev" carried in scalar code:

```
for lane in 0..LANES {
    let mut prev = base[lane];
    for value_idx in lane_indices {
        let next = unpacked[value_idx].wrapping_add(prev);
        output[value_idx] = next;
        prev = next;
    }
}
```

That looks scalar — but because each iteration of the outer loop is
independent, **the SIMD shape is "do all 32 (or 64) lanes' i-th step in
parallel via one register-wide add."** Concretely for u8:

```
ymm_prev := vmovdqu base                ; 32 lane-prevs in one register
loop i = 0..32:
    ymm_cur  := unpack-one-step-into ymm_cur     ; fragment per bw
    ymm_prev := vpaddb ymm_prev, ymm_cur         ; one add per step
    vmovdqu  output + i*stride, ymm_prev         ; or feed into compare
```

So a stencil-JIT fragment for "undelta one step" is one `vpaddb`/`vpaddd`
(~4 bytes). The chain `unpack(bw=N) → undelta → compare` would unroll 32
of these steps; each step concatenates one unpack-step fragment, one add
fragment, and (for the last step) a compare fragment that produces the
final mask. The total kernel grows roughly linearly with `step_count *
fragment_size` plus the constant prologue/epilogue.

Two practical wrinkles for a real implementation:

1. **Block boundary carry.** Each 1024-block needs its `base` (the per-lane
   final values from the previous block) fed in via a register or memory
   address. The header would `vmovdqu` it into `ymm_prev`.
2. **Output policy.** For a fused `delta + compare` chain you usually don't
   want to materialize the undeltad values — just compare each step's
   `ymm_prev` against the broadcast constant and OR the mask bit into a
   running predicate register. That folds the compare into the inner loop
   instead of running once at the end.

I'd put delta at "small extra work" relative to what's here — maybe a
day's coding to add the lane-base prologue, the per-step add fragment,
and the unrolled-loop generator. The real engineering cost is per-`bw`
unpack fragments, not delta itself.

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
