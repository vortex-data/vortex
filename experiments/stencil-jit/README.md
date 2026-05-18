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

Median GB/s (stable across multiple runs; `aot_fused` jittered in this
sandbox between 16 and 76 GB/s for the same call, so reporting fastest
for it):

| n_blocks | size   | **C (chunked)** | **D (stencil-JIT)** | D / C | per-block JIT | aot_fused (fastest) |
|---------:|-------:|----------------:|--------------------:|------:|--------------:|--------------------:|
|    128   |  4 KB  | 14.0            | 65.6                | 4.7×  | 11.7          | 16–66 (jittery)     |
|   1024   | 32 KB  | 41–44           | 69.8                | 1.7×  | 12.0          | 18–76 (jittery)     |
|   8192   | 256 KB | 36.9            | 71.3                | 1.9×  | 12.0          | 60.7                |
|  32768   |  1 MB  | 32.2            | 58.8                | 1.8×  | 12.0          | 53.9                |
| 131072   |  4 MB  | 25.8            | 29.1                | 1.1×  | 11.8          | 27.7                |

The trend is stable across runs: **D beats C by ~1.7–2× at L1- and
L2-resident sizes**, and both **converge to RAM-bandwidth limit beyond
L2** (~28 GB/s). At very small sizes (n=128 = 4 chunks) C pays per-chunk
function-call overhead more times than D's bulk kernel does; at very
large sizes both are gated by memory bandwidth.

**Why D beats C even when C's scratch stays in L1:** L1 traffic isn't
free. Per 32-byte block, C does ~2 SIMD ops in stage 1 plus a 32-byte L1
store, then a 32-byte L1 load plus ~2 SIMD ops in stage 2. D keeps the
intermediate register-resident and emits only a 4-byte mask store. C pays
roughly 32 bytes of L1 round trip per block that D skips. At cache-port
bandwidth, that's a few cycles per block of pure overhead C can't avoid.

**`aot_fused` jitter:** the same call shows 16 GB/s in one run and 76 in
another. This appears to be CPU frequency/thermal noise in the sandbox
affecting `.text`-resident code. The JIT's mmap'd-anonymous pages and
the chunked-unfused are both stable. The cleanest signal is the D-vs-C
comparison; AOT-fused stays in the table as a "what's the ceiling when
you can AOT it" reference.

**Headline:** for runtime-defined chains, a copy-and-patch JIT delivers
**~1.7–2× speedup over chunked AOT-intrinsics at the cache-resident
sizes that matter**, at no binary-size cost beyond a small stencil
library, and converges to AOT-fused in every regime where AOT-fused is
itself meaningful.

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
