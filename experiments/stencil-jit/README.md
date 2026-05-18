# stencil-jit (research prototype)

Copy-and-patch JIT for fused SIMD compare on packed `u8` lanes, with chains
and bulk processing. Demonstrates the "clone machine code and adjust" idea
from the dynamic-bitpacking-kernels discussion.

**Not** wired into Vortex's compute layer. **Not** production-ready. Lives
outside the Cargo workspace via the top-level `exclude`.

## What's in the box

| Module                  | Purpose                                                       |
|-------------------------|---------------------------------------------------------------|
| `src/stencil.rs`        | The AOT-emitted stencil bodies + patch encodings              |
| `src/lib.rs`            | `Kernel` (1 block/call) + `BulkKernel` (n blocks/call, 2x unrolled, interleaved) |
| `src/delta.rs`          | Scalar + AVX2-intrinsics delta-undo reference (no JIT'd version yet) |
| `tests/eq_neq.rs`       | Exhaustive correctness across `(op, ffor, c, ref)`            |
| `benches/throughput.rs` | divan benchmark, FFoR-add + compare(==), bulk vs AOT          |
| `examples/bench_delta.rs` | Delta-undo benchmark (scalar / intrinsics / JIT placeholder) |

## Calling convention

Both kernels use the System V AMD64 ABI. Inputs by register:

```text
Kernel::call(packed, constant, out, ffor_ref)
   rdi  = const u8*    -- 32 bytes of packed data
   rsi  = u64          -- compare constant; only SIL (low 8 bits) is read
   rdx  = u32*         -- 4 bytes of output mask (1 bit per lane)
   rcx  = u64          -- FFoR reference; only CL is read

BulkKernel::call(packed, constant, out, ffor_ref, n_blocks)
   rdi  = const u8*    -- n_blocks * 32 bytes of packed data
   rsi  = u64          -- compare constant; SIL
   rdx  = u32*         -- n_blocks * 4 bytes of output masks
   rcx  = u64          -- FFoR reference; CL
    r8  = u64          -- n_blocks; MUST be even (or zero)
```

Inside the stencil:

```text
ymm0     -- block A data lane vector
ymm4     -- block B data lane vector (bulk kernel only)
ymm1     -- broadcast(compare constant), set up once in the prologue
ymm2     -- all-ones, used by invert patches (neq/ge/le)
ymm3     -- broadcast(FFoR reference)
```

## Splice slots

**Single-block kernel** (`__stencil_jit_*`):

```text
SLOT 1  (5 B)  vmovdqu ymm0, [rdi+0]              (FFoR off)
               vpaddb  ymm0, ymm3, [rdi+0]        (FFoR on)
SLOT 2  (8 B)  compare op against ymm1, optionally inverted via ymm2
```

**Bulk kernel** (`__stencil_jit_bulk_*`), 2x unrolled with interleaved
chains:

```text
SLOT 1A  (5 B)  vmovdqu/vpaddb -> ymm0  from [rdi+0]
SLOT 1B  (5 B)  vmovdqu/vpaddb -> ymm4  from [rdi+32]
SLOT 2A  (8 B)  compare ymm0 (block A's chain)
SLOT 2B  (8 B)  compare ymm4 (block B's chain)
```

Block A operates entirely on `ymm0`; block B on `ymm4`. The two chains share
no architectural register, so the OoO core can issue both blocks' work in
parallel.

## Compare ops (all 6 fit each 8-byte slot)

```text
eq : vpcmpeqb ymm0,ymm0,ymm1            ; 4-byte nopl
neq: vpcmpeqb ymm0,ymm0,ymm1            ; vpxor ymm0,ymm0,ymm2     (invert)
gt : vpcmpgtb ymm0,ymm0,ymm1            ; 4-byte nopl              (signed)
lt : vpcmpgtb ymm0,ymm1,ymm0            ; 4-byte nopl              (signed)
ge : vpcmpgtb ymm0,ymm1,ymm0            ; vpxor ymm0,ymm0,ymm2     (= !lt)
le : vpcmpgtb ymm0,ymm0,ymm1            ; vpxor ymm0,ymm0,ymm2     (= !gt)
```

Block B's compare patches are the same shape rewritten to target ymm4 and
read ymm4/ymm1.

## Benchmark — D and F vs C across working-set sizes

The realistic baseline isn't AOT-everything-fused (impossible to enumerate
when chains are runtime-defined); it's **C: chunk-by-chunk processing
with a 1 KB L1-resident scratch buffer reused across chunks, calling fast
AOT-intrinsics single-op kernels per chunk.** This matches Vortex's
1024-element-per-chunk execution model — the scratch is always in L1
regardless of total dataset size.

Variants:

* **`aot_chunked_unfused`** (= **C**) — outer loop over 1024-element
  chunks; per chunk, call `ffor_add` then `compare_eq` AOT-intrinsics
  kernels through a 1 KB scratch that stays hot in L1.
* **`aot_fused`** — single AVX2-intrinsics function fusing FFoR + compare
  in one pass. The ceiling, ships only when the chain is known AOT.
* **`stencil_jit_fused`** (= **D**) — this prototype, fusing at runtime.
* **`stencil_jit_per_block`** — the JIT called once per 32-byte block,
  paying the function-call tax every block.

A fourth variant — **`stencil_jit_specialized` (D-spec)** — bakes both
constants into the kernel at JIT-compile time and algebraically
simplifies: since `(x + r) == c` ⟺ `x == (c - r) (mod 256)`, the FFoR-add
disappears entirely. The loop body collapses to one memory-operand
`vpcmpeqb ymm0, ymm1, [rdi]` per block. This is the **JIT-only win** —
AOT can't fold `c - r` if `c` and `r` are query-planner parameters.

A fifth variant — **`stencil_jit_specialized_avx512` (D-spec-512)** —
uses AVX-512BW's `vpcmpeqb k, zmm, [mem]` + `kmovq`. The 64-bit kmask
avoids the AVX2 `vpmovmskb` port-0 bottleneck entirely, and zmm doubles
the lane width.

Median GB/s (stable across multiple runs):

| n_blocks | size   | **C (chunked)** | D (generic) | **D-spec (AVX2)** | **D-spec-512** | **D-spec-512 / C** | aot_fused |
|---------:|-------:|----------------:|------------:|------------------:|---------------:|-------------------:|----------:|
|    128   |  4 KB  | 14.0            | 67.4        | 90.7              | **162.0**      | **11.6×**          | 68.1      |
|   1024   | 32 KB  | 41–44           | 70.3        | 89.7              | **141.2**      | **3.4×**           | 69.9      |
|   8192   | 256 KB | 36.9            | 56.4        | 72.8              | 62.1           | 1.7×               | 57.4      |
|  32768   |  1 MB  | 32.2            | 56.6        | 55.3              | 59.0           | 1.8×               | 57.8      |
| 131072   |  4 MB  | 25.8            | 24.1        | 21.4              | 20.2           | 0.8× (memory bound)| 24.7      |

**D-spec beats AOT-fused at every cache-resident size** — AOT-fused still
pays a runtime `vpaddb` for the FFoR offset; D-spec doesn't.

**D-spec-512 hits 162 GB/s at 4 KB (L1) and 141 GB/s at 32 KB** — that's
**2× AOT-fused** and **11.6× / 3.4× over C** at those sizes. The 64-bit
`kmask` from `vpcmpeqb k, zmm, [mem]` sidesteps the AVX2 `vpmovmskb`
port-0 bottleneck (1/cycle on Skylake) that capped D-spec at ~100 GB/s.

At sizes that overflow L1 (≥ 256 KB) D-spec-512's per-iteration speedup
shrinks because we're memory-bound; AVX-512 frequency throttling on this
class of CPU also bites once a sustained AVX-512 workload runs long
enough.

**Per-block work, by variant:**

| Variant         | SIMD ops per 32-byte block | Notes                                                |
|-----------------|---------------------------:|------------------------------------------------------|
| C (chunked)     | 4 + L1 round trip          | load+vpaddb+L1-store, load+vpcmpeqb+vpmovmskb+store  |
| D (generic)     | 3                          | vpaddb-mem-operand + vpcmpeqb + vpmovmskb            |
| **D-spec**      | **2**                      | **vpcmpeqb-mem-operand + vpmovmskb**                 |
| **D-spec-512**  | **1**                      | **vpcmpeqb-mem-operand → kmask** (one 64-byte zmm op covers two 32-byte blocks; `kmovq` shared) |
| aot_fused       | 3                          | vmovdqu + vpaddb + vpcmpeqb + vpmovmskb              |

**Why D-spec beats AOT-fused:** AOT-fused takes `ffor_ref` and `constant`
as runtime parameters (the chain isn't known at Rust-compile time), so
it can't fold them. The 4× unroll also helps: D-spec runs four
independent compare chains in flight; AOT-fused unrolls 2.

**At very small sizes (n=128 = 4 chunks):** D-spec hits **6× over C**
because C pays 4 function-call invocations per stage (8 calls total),
while D-spec is one call total with a tight 4×-unrolled body.

**At memory-bound sizes (4 MB):** everything converges at ~30 GB/s — RAM
bandwidth wins.

**Headline:** for runtime-defined chains where constants are query-time-
known, a copy-and-patch JIT with constant-folding + AVX-512 delivers
**3.4–11.6× over chunked AOT-intrinsics** at L1-resident sizes and
**2× AOT-fused** — because AOT-fused can't bake the runtime constants
and isn't in this build using AVX-512 anyway. This is the structural
advantage of JIT over AOT: not "the JIT writes the same instructions
faster," but "the JIT can emit **fewer, wider** instructions because
more is known at kernel-materialize time and the target ISA is picked
per machine."

**F (Cranelift-at-build-time) gives you all of this for free** across
ISAs: Cranelift's x86-64 backend picks AVX-512 vs AVX2 at build time
per target, applies the same constant-folding semantically (operating
on its IR), and produces bytes for D's runtime to splice. LLVM would do
slightly better codegen — by ~5–10% on this size of kernel — but at
~100× the compile latency. For an interactive JIT that materializes
many kernels per query, Cranelift's "compile in microseconds at near-
LLVM quality" is the right tradeoff.


## F is real — Cranelift compiles the kernel at build time

`build.rs` uses `cranelift-codegen` 0.118 to compile an IR function:

```text
fn eq_kernel(packed: *const u8, out: *mut u32, n_halves: u64) {
    let c = vconst [35; 16]   // (42 - 7) broadcast into i8x16
    for i in 0..n_halves {
        let data = load i8x16, [packed + i*16]
        let mask = vhigh_bits(i16, icmp eq, data, c)
        store i16 mask, [out + i*2]
    }
}
```

Cranelift emits **96 bytes of self-contained x86-64 machine code with
zero relocations**. The runtime is unchanged — `CraneliftKernel::new()`
hands those bytes straight to the same `materialize()` D-spec uses.

| n_blocks | size   | C (chunked) | **F (Cranelift, xmm)** | D-spec (AVX2 ymm) | D-spec-512 (zmm) |
|---------:|-------:|------------:|-----------------------:|------------------:|-----------------:|
|    128   |  4 KB  | 38.5        | 19.0                   | 75.6              | 146.7            |
|   1024   | 32 KB  | 41.5        | 20.4                   | 79.1              | 124.4            |
|   8192   | 256 KB | 28.6        | 19.3                   | 63.1              | 58.7             |
|  32768   |  1 MB  | 32.2        | 19.4                   | 56.8              | 51.7             |
| 131072   |  4 MB  | 21.9        | 18.3                   | 25.5              | 24.5             |

F here runs at half D-spec's throughput **because Cranelift 0.118's x64
backend only handles vectors up to 128 bits** — no ymm or zmm yet.
Newer Cranelift (≥ 0.119) extends to AVX2/AVX-512, at which point F's
throughput matches D-spec / D-spec-512 by construction. That's the
point: the runtime stays identical; F's job is to demonstrate that
flipping the source of the bytes from `global_asm!` to `cranelift-codegen`
costs nothing semantically.

What F bought us, with zero hand-tuned asm:
* **96 bytes, 0 relocations** — clean, embeddable.
* **Multi-ISA for free** — `Triple::from_str(...)` switches the target;
  same IR, different bytes.
* **Same runtime path** as D — `materialize()` + `mprotect` + indirect
  call.

What it didn't buy yet:
* **Wide vectors** on Cranelift 0.118 — fix is a version bump.
* **Constant patching** — the `c = 35` is baked into the IR. The
  follow-up is to emit the constant as a Cranelift `GlobalValue` (which
  becomes a relocation), capture the reloc offset, and have the runtime
  patch the byte at that offset like `SpecializedKernel` does today.

## D vs F — the only difference is the source of the stencil bytes

D and F produce **identical machine code at runtime** for a given fragment
graph. They differ only in *who wrote the bytes*:

* **D**: hand-written in `global_asm!`, one library per ISA.
* **F**: Cranelift IR compiled at build time by `cranelift-codegen`, bytes
  extracted from the resulting `MachBuffer`, embedded in the binary as
  `.rodata` with a per-fragment relocation table; one IR source per
  fragment, Cranelift's backends fan it out to x86-64 / aarch64 / RISC-V.

Runtime cost (memcpy + relocation patches + mprotect) is the same. Steady-
state throughput is the same. The trade is purely build-time complexity
vs single-source-multi-ISA.

A sketch of `build.rs` for F:

```rust
// build.rs — not implemented in this commit, but the runtime is unchanged
use cranelift_codegen::{ir, isa, settings};
fn build_compare_eq_stencil() {
    let isa = isa::lookup("x86_64-unknown-linux-gnu")
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap();
    let mut func = ir::Function::new();
    // ... build IR: take ymm0, return vpcmpeqb(ymm0, ymm1)
    let compiled = isa.compile_function(&func, ...).unwrap();
    let bytes = compiled.code_buffer();
    let relocs = compiled.buffer.relocs();
    std::fs::write(format!("{}/compare_eq.bin", env!("OUT_DIR")), bytes).unwrap();
    write_relocs(&format!("{}/compare_eq.relocs", env!("OUT_DIR")), relocs);
}
```

`materialize()` in `src/lib.rs` is already the runtime — F just changes
where the stencil bytes come from.

## Where the JIT still has headroom on this microbench

At n=1024 the JIT shows 20.8 GB/s vs AOT-fused's 74 GB/s. The AOT body
is ~49 bytes vs the JIT's ~55; LLVM schedules instructions across
iterations more aggressively than fixed splice slots allow. AOT achieves
~1.5 cycles/block (near the `vpmovmskb` port-0 throughput limit); the
JIT sits at ~5.5 cycles/block (two chains' critical path, partially
overlapping). At n=32768 (1 MB), the JIT catches up to AOT-fused at 60
GB/s — bandwidth becomes the binding constraint there, so per-iteration
ILP differences stop mattering.

Closing the L1-regime gap would need 4×+ unroll with cross-iteration
instruction interleaving — past where hand-asm pays off. That's exactly
the case for F: get LLVM-quality scheduling per fragment, keep the byte-
splice mechanism.

## Calling convention summary

```rust
let kernel = BulkKernel::compile(ChainConfig::ffor_then_compare(CmpOp::Eq))?;
// SAFETY: 1024*32 readable, 1024*4 writable, n_blocks (=1024) is even.
unsafe { kernel.call(packed.as_ptr(), 42u8, out.as_mut_ptr(), 7u8, 1024) };
```

12 unit + integration tests verify all (op, ffor, constant, ref) combinations
plus the bulk-vs-single equivalence.

## Delta — scaffolded, not yet JIT'd

`src/delta.rs` carries the AOT alternatives a JIT'd delta-undo would be
compared against, on the **step-major 32×32 u8 layout** (matches
fastlanes-rs's delta semantics, narrowed from `LANES=128` to `LANES=32` to
fit a single ymm register). The benchmark (`examples/bench_delta.rs`) shows
scalar / autovec / intrinsics all land at ~67 GB/s — LLVM autovectorizes
the trivial lane/step inner loop into essentially the optimal `vpaddb`
sequence. JIT'd delta would need:

1. A new stencil with a prologue that loads the per-lane `base[]` into
   `ymm_prev`.
2. A 32-iteration unrolled body of `vmovdqu` (load step) + `vpaddb`
   (accumulate) + `vmovdqu` (store) — about 430 bytes.
3. Optionally a fused `delta → compare` chain: replace each step's store
   with the compare+OR-into-running-mask pattern, patched with the same
   6 compare patches the chain kernel already supports.

(Day-ish of work plus per-`bw` unpack-step fragments; mechanism is
already proven by the chain kernel here.)

## Running

```bash
cd experiments/stencil-jit
cargo test                            # 12 tests
cargo run --example dump              # raw stencil bytes
cargo run --release --example bench_delta
cargo bench                           # divan throughput, JIT vs AOT
```

## SPDX

```text
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
```
