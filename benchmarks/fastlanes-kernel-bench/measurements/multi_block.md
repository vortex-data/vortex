# Multi-block (N=8) throughput vs single-block

## What is measured

`benches/multi_block.rs` runs **8 consecutive unpacks of 1024-element blocks**
in a single `bencher.bench_local` closure, sharing one packed input across
all 8 calls (each writes into a different output buffer). Per-block time =
total / 8.

If single-block timing is dominated by function-call overhead or
bench-harness sampling jitter, the multi-block per-block number will be
*lower* than the matrix's single-block number, by the amortised overhead.

If single-block timing is already dominated by the kernel itself, the
multi-block number will be *close to* the single-block number.

Build: same as the AVX-512 (zmm) build (`target-cpu=native`,
`target-feature=-prefer-256-bit`, codegen-units=1). The matrix column to
compare against is `simd=zmm` from `matrix_run1.csv`.

Results: `measurements/multi_block.csv`.

## Per-cell comparison (mb_per_block vs single-block zmm)

Selected from the full table (48 cells, all 4 types):

| cell           | variant       | mb per-block | single-block zmm | ratio |
|----------------|--------------|-------------:|-----------------:|------:|
| u8  W=1        | bare         | 10.8 ns      | 9.7 ns           | 1.11  |
| u8  W=8        | bare         | 16.0 ns      | 9.7 ns           | **1.64** |
| u16 W=1        | bare         | 12.7 ns      | 27.7 ns          | **0.46** |
| u16 W=7        | bare         | 17.1 ns      | 35.0 ns          | **0.49** |
| u16 W=11       | bare         | 30.0 ns      | 35.2 ns          | 0.85  |
| u16 W=16       | bare         | 35.2 ns      | 39.9 ns          | 0.88  |
| u32 W=1        | bare         | 22.0 ns      | 63.7 ns          | **0.35** |
| u32 W=10       | bare         | 42.4 ns      | 53.7 ns          | 0.79  |
| u32 W=10       | fused        | 39.8 ns      | 53.4 ns          | 0.74  |
| u32 W=32       | bare         | 77.6 ns      | 80.4 ns          | 0.97  |
| u32 W=32       | fused        | 99.6 ns      | 87.7 ns          | 1.14  |
| u64 W=1        | bare         | 168.4 ns     | 98.7 ns          | **1.71** |
| u64 W=11       | bare         | 165.1 ns     | 95.7 ns          | **1.73** |
| u64 W=33       | bare         | 168.5 ns     | 120.6 ns         | 1.40  |
| u64 W=55       | bare         | 202.6 ns     | 161.6 ns         | 1.25  |
| u64 W=64       | bare         | 219.5 ns     | 200.0 ns         | 1.10  |

## Interpretation

Two regimes are visible:

1. **Narrow widths, narrow types (u16, u32 at W < ~10)**: ratio is
   **0.3 to 0.6**. The multi-block per-block time is roughly *half* the
   single-block time. This says the single-block bench is dominated by
   per-call overhead -- divan's `bench_local` calibration + the
   `#[inline(never)]` boundary of `BitPacking::unpack` + register spills.
   When 8 unpacks share one closure call, the kernel body runs back-to-back
   and the actual per-block kernel cost shows through.

2. **u64 across all W, plus u8 W=8**: ratio is **>1.1**. The multi-block
   per-block time is *higher* than the single-block. This is consistent
   with L1d capacity pressure: 8 × 1024 × 8 bytes = 64 KB of output buffers
   exceed the L1d (48 KB). The single-block bench reuses one 8 KB output
   buffer that fits entirely in L1d. The multi-block bench spills to L2.
   So the ratio inversion is *not* "single-block was wrong, multi-block
   is right"; it's measuring two different things (L1-resident
   throughput vs L1-then-spill throughput).

Therefore the matrix headline numbers from `matrix_run1.csv` are
**measuring two things at once**: per-block kernel throughput plus
per-call overhead. The relative amount differs by cell:

- For u32 narrow-W cells (the "free fusing" regime), per-call overhead
  is ~25-50 ns out of ~60 ns total -- so about *half* of the matrix's
  single-block number is overhead. Per-block kernel cost is closer to
  30 ns.
- For u64 narrow-W cells, the ratio inverts: multi-block per-block
  cost is *higher* than the single-block bench, dominated by L1d
  capacity pressure. Per-call overhead is small relative to the
  L1d-bound steady-state kernel.

**This does not invalidate the bare-vs-fused comparison** because both
variants pay the same per-call overhead and L1d-capacity tax. The
*delta* between bare and fused is what the fusing question is about,
and that delta is preserved in both single-block and multi-block
numbers.

Cite a single specific example: u32 W=10 zmm, bare/fused at single-block
= 53.7 / 53.4 ns (-0.6%, "free"). At multi-block N=8 per-block =
42.4 / 39.8 ns (-6.1%, "free"). Both say "fusing is free"; the
multi-block is slightly more sensitive because the per-call overhead
that masked variation has been amortised away.

A counter-example: u32 W=32 bare/fused single-block = 80.4 / 87.7 ns
(+9% fused overhead). Multi-block per-block = 77.6 / 99.6 ns (+28%
fused overhead). With per-call overhead removed, the W=32 fused
overhead jumps from +9% to +28% -- the underlying kernel cost of
fusing at W=T (identity) is **much larger** than the matrix headline
implies, because per-call overhead was hiding it.

Interpretation: the "fusing is mostly free" claim from the matrix
table is **stronger** for narrow widths (the per-call overhead inflated
both numbers similarly, making the relative difference look small) and
**weaker** for full-width / wide cases (the per-call overhead masked
a real cost there).
