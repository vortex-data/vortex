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
when chains are runtime-defined); it's **C**, an AOT library of single-op
kernels chained at runtime via a scratch buffer. The benchmark sweeps the
input size so we can see when the unfused pipeline's intermediate-buffer
traffic starts hurting:

* **`aot_fused`** — single AVX2-intrinsics function doing FFoR + compare
  in one pass. The ceiling, ships only when the chain is known AOT.
* **`aot_unfused_pipeline`** (= **C**) — two AOT kernels (`ffor_add` then
  `compare_eq`) chained via a `Vec<u8>` scratch buffer.
* **`stencil_jit_fused`** (= **D**) — this prototype, fusing at runtime.
* **`stencil_jit_per_block`** — the JIT called once per 32-byte block.

Median GB/s, parameterized by working-set size:

| n_blocks | size   | aot_fused | C (unfused) | **D (stencil-JIT)** | per-block JIT |
|---------:|-------:|----------:|------------:|--------------------:|--------------:|
|   128    | 4 KB   | 16.6      | **33.0**    | 19.3                | 11.7          |
|  1024    | 32 KB  | **74.0**  | 26.0        | 20.8                | 12.0          |
|  8192    | 256 KB | 66.1      | 26.1        | 20.6 (best 71.6)    | 12.0          |
| 32768    | 1 MB   | 55.3      | **11.7**    | **59.8**            | 12.0          |
|131072    | 4 MB   | 29.0      |  9.1        | 29.1                | 11.8          |

What this says, by regime:

**Small (≤ L1, 4–32 KB):** Per-call overhead dominates. **C wins** —
its scratch buffer stays in L1 and the two single-op kernels inline
tightly. The JIT's indirect call still costs ns/block at this scale.

**Medium (L2-resident, 256 KB – 1 MB):** This is the regime that matters
for analytical query workloads on bitpacked columns. **D matches AOT-fused**
(60 GB/s at 1 MB), and **both beat C by ~5×** (D 60 GB/s vs C 12 GB/s).
The scratch buffer no longer fits in L1, so C pays a full write-then-read
round trip against the next cache level. Fusion finally earns its keep.

**Large (memory-bound, 4 MB):** Everything converges to ~30 GB/s — RAM
bandwidth is the ceiling. Fusion doesn't help, doesn't hurt.

The headline: **for runtime-defined chains, a copy-and-patch JIT delivers
AOT-fused-quality performance in the cache-pressure regime where fusion
matters, at no binary-size cost beyond a small stencil library.**

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
