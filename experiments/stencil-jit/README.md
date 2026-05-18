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

## Benchmark — bulk-mode throughput

`cargo bench` runs a divan harness that compiles each kernel once (JIT
compile cost ignored — it's ~5 µs of mmap + memcpy + mprotect, amortized
over millions of calls) then measures per-call throughput. Each iteration
processes 1024 32-byte blocks = 32 KB, hot in L1.

| Variant                                | ns/call | GB/s   |
|----------------------------------------|---------|--------|
| AOT AVX2 intrinsics (#[inline(never)]) |  438    | 74.7   |
| **stencil-jit bulk (2x unroll)**       | **1574**| **20.8** |
| stencil-jit per-block loop             | 2764    | 11.9   |
| AOT closure-based (autovec)            | 9934    |  3.3   |
| Scalar baseline                        | 9943    |  3.3   |

What changed since the first run:

1. **Single-block in a loop → bulk kernel**: amortized function-call
   overhead and the constant-broadcast prologue.  +2.7× over the per-block
   loop.
2. **Fused load + FFoR-add**: replaced `vmovdqu ymm0,[rdi]` + `vpaddb
   ymm0,ymm0,ymm3` with a single `vpaddb ymm0,ymm3,[rdi]` memory-operand
   instruction. One fewer µop per block.
3. **Multi-byte `nopl` padding**: replaced 4× single-byte `0x90` per
   compare slot with one 4-byte `nopl 0x0(%rax)`. Frees decode bandwidth.
4. **2x unroll with independent registers**: block A flows through `ymm0`
   end-to-end, block B through `ymm4`. Two chains, no WAW hazards.
5. **Loop entry 32-byte aligned**: stabilizes throughput (variance shrank
   from 14 µs spread to <0.02 µs).

Where the remaining 3.6× gap to AOT comes from:

The AOT body is ~49 bytes vs our 53–55 bytes, and LLVM schedules
instructions across iterations more aggressively than we can with fixed
splice slots. Specifically, AOT achieves ~1.5 cycles/block (close to the
vpmovmskb port-0 throughput limit), while ours sits at ~5.5 cycles/block
(roughly the critical-path-per-block depth, with two chains overlapping).
Closing the gap would need 4x+ unroll with full instruction interleaving
across blocks — which means encoding distinct patches per slot and
emitting more elaborate machine code. Tractable, but past the point where
hand-asm makes sense; the natural next step is Cranelift-at-build-time
codegen for the stencil.

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
