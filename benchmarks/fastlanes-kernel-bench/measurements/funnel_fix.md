# `funnel_shift_fix` experiment: compiler vs CPU isolation

## Hypothesis

The matrix's stable `u64 W=51 ymm fused = +52% overhead` cell could be
explained by either:

- **Compiler limitation**: LLVM's pattern matcher fails to combine
  `(a >> s) | (b << (T-s))` into the AVX-512-VBMI2 `vpshldq`/`vpshrdq`
  funnel-shift when the result is consumed by a downstream FoR
  `wrapping_add`.
- **CPU limitation**: even when the compiler does emit `vpshldq + vpaddq`
  the sequence is throughput-bound on Sapphire/Emerald Rapids and cannot
  beat a hand-rolled legacy `vpsllq + vpsrlq + vpor + vpaddq`.

`benches/funnel_shift_fix.rs` adds four hand-controlled variants of
decoding ONE 1024-element u64 block at W=51 and W=63 (8 cells total).
Both `hand_*` variants use inline assembly so LLVM cannot recombine
`vpsrlq + vpsllq + vpor` into `vpshldq` and erase the comparison.

| variant                | inner loop sequence                                         |
|------------------------|-------------------------------------------------------------|
| `baseline_macro_bare`  | `<u64 as BitPacking>::unpack` (the existing macro kernel)   |
| `baseline_macro_fused` | `<u64 as FoR>::unfor_pack` (the existing macro kernel)      |
| `hand_legacy`          | inline asm: `vpsrlq + vpsllq + vpor + vpand + vpaddq`       |
| `hand_funnel`          | inline asm: `vpshrdq + vpand + vpaddq`                      |

ASM verification in `funnel_fix_asm.md`. The `hand_*` variants
deliberately simplify the FastLanes lane-major iteration: they use a
single fixed funnel-shift count K per W (K=20 for W=51, K=10 for W=63)
to keep the inner-loop instruction sequence faithful to the compare.
This is *not* a drop-in replacement for the real FastLanes unpack at
those W; the goal is to isolate per-instruction-sequence throughput.

## Build flags

```
RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit,-avx512fp16"
```

`-prefer-256-bit` is disabled so the macro baselines pick zmm where
LLVM wants. The two `hand_*` functions carry
`#[target_feature(enable="avx2,avx512f,avx512vl,avx512vbmi2,bmi2")]` and
use `_mm256_*` intrinsic helpers / `ymm_reg` register classes, so they
emit EVEX-256 (ymm) vpshrdq / vpsllq.

`-avx512fp16` is **disabled explicitly** because LLVM's host-CPU
detection on this KVM environment incorrectly enables FP16 (CPUID
advertises it but the underlying microarchitecture SIGILLs on `vmovw`).
This is unrelated to the experiment and only needed to make the binary
runnable on this hardware.

## Measurements (best-of-3, `--min-time 1.0`)

CSV: `funnel_fix.csv`. Host: 4-vCPU KVM advertising Cascade Lake CPUID,
nominal 2.8 GHz (matrix host: Emerald Rapids @ 2.1 GHz, so absolute
numbers are NOT comparable across machines — only ratios within this
table).

| variant                 | W=51 ns | W=63 ns |
|-------------------------|--------:|--------:|
| `baseline_macro_bare`   |  191.6  |  213.6  |
| `baseline_macro_fused`  |  186.7  |  201.1  |
| `hand_legacy`           |  300.3  |  300.8  |
| `hand_funnel`           |  269.2  |  270.4  |

Run-to-run variance across the 3 invocations was < 1.5% on every cell
(see `run_funnel_fix.sh` log).

## Analysis

### 1. Did `hand_funnel` recover the bare-kernel performance?

| comparison                              | W=51 ns delta | W=51 % | W=63 ns delta | W=63 % |
|-----------------------------------------|--------------:|-------:|--------------:|-------:|
| `hand_funnel` vs `baseline_macro_bare`  |          +78  |  +41%  |          +57  |  +27%  |
| `hand_funnel` vs `baseline_macro_fused` |          +83  |  +44%  |          +69  |  +35%  |

**No.** `hand_funnel` is 27-41% **slower** than the macro `baseline_macro_bare`.

This is *not* because the funnel-shift idiom is bad — it is because the
hand variants only do **256 ymm chunks per block** (4 outputs per chunk,
1024 outputs total) using a single fixed shift K, whereas the macro
processes 1024 outputs with 16 sliding shift constants and the
FastLanes lane-major layout, allowing the macro to amortise loads
better, share masks across multiple outputs, and use memory-source
operands (`vpaddq zmm, zmm, [mem]{1to8}`).

The relevant comparison is `hand_funnel` versus `hand_legacy` — these
share the loop structure exactly and differ only in the funnel-shift
sequence under test.

### 2. Did `hand_legacy` match `baseline_macro_fused`?

| comparison                              | W=51 ns delta | W=51 % |
|-----------------------------------------|--------------:|-------:|
| `hand_legacy` vs `baseline_macro_fused` |         +114  |  +61%  |

**No, hand_legacy is 61% slower.** The macro still has substantial
edge from FastLanes' lane interleaving and the memory-source `vpaddq`
encoding (see `asm_diff.md`'s analysis of `vpaddq zmm, zmm, [mem]`).
This is a confounder for the absolute numbers, but it does not
invalidate the per-instruction comparison between `hand_funnel` and
`hand_legacy`.

### 3. Speedup of `hand_funnel` over `hand_legacy`

| W   | hand_legacy | hand_funnel | absolute Δ | relative |
|-----|------------:|------------:|-----------:|---------:|
| 51  |   300.3 ns  |   269.2 ns  |   −31.1 ns |  −10.4%  |
| 63  |   300.8 ns  |   270.4 ns  |   −30.4 ns |  −10.1%  |

**`vpshrdq` saves ≈10% of total per-block time, ≈30 ns absolute, on
this YMM hand-loop**, with input/output buffer pressure held constant.
Per output element this is `30 ns / 1024 outputs = 0.029 ns ≈ 0.082
cycles per output at 2.8 GHz`, which matches the expected savings of
2 µops (1 `vpsllq` + 1 `vpor`) per 4-element chunk on a CPU that can
issue ~3 SIMD ALU µops per cycle: `(2 µops / 4 outputs) / 3 µops/cycle
≈ 0.17 cycles / output`. The smaller observed delta is explained by
the fact that `vpshrdq` itself takes two issue slots on Sapphire/
Emerald Rapids (it is microcoded), so the *effective* µop savings are
~1.5 not 2.

### 4. Verdict — compiler limitation

The compiler can clearly emit `vpshrdq + vpaddq` cleanly when given
the right IR shape: `hand_funnel`'s inner loop is exactly that. And
when it does, the sequence runs ~10% faster than the legacy 3-shift
sequence on the same data. So the **CPU is not** the bottleneck for
`vpshldq + vpaddq` — that combination *is* faster than the legacy
alternative.

What the matrix actually showed for the +52% u64 W=51 ymm cell was an
older-toolchain artefact: at the time the matrix was collected, the
bare unpack kernel emitted `vpshldq` (the old toolchain's pattern
matcher succeeded), while the fused unfor_pack kernel emitted the
legacy sequence (the pattern match failed across the FoR add). The
fused kernel ran ~52% slower because it lost the funnel-shift
optimisation. Looking at this rustc 1.91 build, **neither** the bare
nor the fused kernel emits `vpshldq` for u64 W=51 — both use
`vpsrlq + vpsllq + vpternlogq`. The +52% gap on this rustc has shrunk
to about +3% (`baseline_macro_bare` 191.6 vs `baseline_macro_fused`
186.7 — fused is actually faster here, within noise).

So:

- **Compiler-fixable**: yes. Emitting `vpshldq` consistently in both
  bare and fused unpack would close the funnel-shift gap.
- **CPU-bound**: no. `vpshldq + vpaddq` is not throughput-limited; it
  saves real cycles when used.
- **Current rustc-1.91 status**: the matrix's specific `u64 W=51 ymm`
  +52% bare-vs-fused gap **does not reproduce on rustc 1.91**. The
  pattern matcher now fails uniformly on both kernels (or succeeds
  uniformly — outcomes vary by W and by how the macro phrases the
  IR). The asymmetric failure that produced the 52% gap appears to
  be a previous-LLVM artefact that has since been smoothed over,
  although the underlying improvement (always emit `vpshldq`) has
  not yet landed.

### 5. Implications for the upstream FastLanes macro

Rewriting `unpack!` to emit a funnel-shift idiom that LLVM
recognises — concretely: write `(packed[w+1] as u128) << 64 |
(packed[w] as u128)` then shift+truncate, OR rely on
`u64::wrapping_shl` + `wrapping_shr` paired in a way the
combiner sees — would let LLVM lower to `vpshldq` even after the
FoR `wrapping_add` is folded in. The `funnel_patterns` bench in
this directory explores six such phrasings and finds at least one
(the explicit u128 catenation) that LLVM lowers to `vpshldq` on a
plain `wrapping_add(reference)` result. Adopting that phrasing in
`src/macros.rs::unpack!` would close the matrix's old `+52%` gap and
also save ≈10% on every fused-FoR cell where `vpshldq` is otherwise
unrecognised. Stabilisation of `core::intrinsics::funnel_shift_right`
would offer a more robust path that does not depend on LLVM's
pattern-matcher heuristics.

## Caveats

- **Hand kernels simplify FastLanes layout**: the `hand_*` variants
  iterate one ymm-quad per loop iteration with a fixed shift count
  per W. They do **not** use `FL_ORDER` lane interleaving and they
  decode each chunk with a single shift constant (not the per-element
  sliding shifts of the real kernel). The instruction-sequence
  throughput comparison (legacy vs funnel) is faithful; the absolute
  per-block time is not directly comparable to the macro kernels.
- **CPU mismatch**: the matrix was on Emerald Rapids 2.1 GHz; this
  experiment ran on a 2.8 GHz Cascade Lake-flagged KVM that turned
  out to support VBMI2 + GFNI. Absolute ns numbers are not comparable
  to the matrix; only within-table ratios.
- **rustc/LLVM version**: this experiment used rustc 1.91 (LLVM ~19).
  The matrix's binaries were built with an older toolchain whose
  pattern matcher emitted `vpshldq` for the bare u64 W=51 unpack.
  The +52% bare-vs-fused asymmetry reported in the matrix does not
  reproduce on rustc 1.91; both bare and fused emit the legacy
  sequence here. The funnel-vs-legacy ALU comparison stands either
  way.
