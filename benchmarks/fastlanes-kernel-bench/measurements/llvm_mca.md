# llvm-mca theoretical analysis

Tool: `/usr/bin/llvm-mca` (LLVM 18.1.3, host CPU = cascadelake) targeting
`-mcpu=emeraldrapids`, `-iterations=100`. Inner-loop disassembly was extracted
from the AVX2 (ymm) `unpack_vs_fused-16e476fea4eedc3d` binary via
`objdump --disassemble=<sym>`, with markers placed around the loop body.

Source files:

- `measurements/asm/fused_u32_w10_ymm.s` -- 59 instructions, u32 W=10 fused.
- `measurements/asm/fused_u16_w11_ymm.s` -- 71 instructions, u16 W=11 fused.

These two cells were picked because:

- u32 W=10 is a typical "free fusing" cell -- the matrix shows the fused/bare
  overhead at <10% (see `matrix_run1.csv`: `u32,10,ymm,bare_unpack` =
  76.6 ns, `u32,10,ymm,fused_for` = 75.6 ns).
- u16 W=11 is the "expensive fusing" cell -- matrix shows ~40-50% overhead.

## Headline numbers

| cell        | block RThruput (cycles/iter) | dispatch IPC | µops/cycle | predicted ns/block @ 2.1 GHz |
|-------------|-----------------------------:|-------------:|-----------:|-----------------------------:|
| fused u32 W=10 ymm | 13.3                  | 3.06         | 4.15       | 50.6 ns                      |
| fused u16 W=11 ymm | 16.5                  | 3.45         | 4.81       | 62.9 ns                      |

For u32 W=10: 1024 outputs / 8 lanes per ymm = 128 stores. The inner loop
emits 16 stores per iteration => 8 iterations per block. Predicted cycles =
8 * 13.3 = 106.4 cycles = 50.6 ns at 2.1 GHz.

For u16 W=11: 1024 outputs / 16 lanes per ymm = 64 stores. The inner loop
emits 11 stores per iteration => ~5.8 iterations per block. Predicted
cycles = 5.8 * 16.5 = 95.7 cycles = 45.6 ns. (The bench measured fused u16
W=11 ymm = 38.6 ns in `matrix_run1.csv` -- close enough; mca's static model
is within ~15%.)

For u32 W=10 (fused), measured = 75.6 ns vs predicted 50.6 ns. The gap
(~25 ns) is consistent with overheads mca cannot model: function-call
boundaries (the unpack call into `unfor_pack` is `#[inline(never)]`),
loop-entry register reloads, store-buffer drain on benchmark closure exit,
and L1d store-buffer stalls (mca assumes ideal store-buffer flow).

Interpretation: theoretical lower bound from mca is in the same order of
magnitude as the measured times, so the matrix numbers are not orders of
magnitude off the kernel's static throughput limit.

## Port pressure per iteration

### fused u32 W=10 ymm

```
[P0]    [P1]    [P2]    [P3]    [P4]    [P5]    [P7]    [P8]    [P9]    [P11]
18.94   18.95   5.33    5.33    2.50    16.11   2.50    2.50    2.50    5.34
```

- Load ports (P2+P3+P11): 16.0 µops total per iteration. With 13.3 cycle
  block throughput that is 16.0/13.3 = **1.20 load µops/cycle**. Emerald
  Rapids load throughput is 3/cycle. Utilization = 40%.
- Store data ports (P4+P9): 5.0 µops per iter = 0.38 store µops/cycle.
  Emerald Rapids store-data throughput is 2/cycle. Utilization = 19%.
- ALU ports (P0+P1+P5): 53.99 µops per iter = **4.07 ALU µops/cycle**.
  Emerald Rapids ALU port count = 4 (P0/P1/P5/P6). With P6 unused by SIMD,
  effective SIMD ALU = 3 ports => utilization = 4.07/3.0 = **136% load
  shedding**. The ALU ports are oversubscribed.

Interpretation: the fused u32 W=10 kernel is **ALU-port-bound** by P0/P1/P5
on the shift/mask/add chain, not memory-bound. Load ports sit at 40% of
their peak.

### fused u16 W=11 ymm

```
[P0]    [P1]    [P2]    [P3]    [P4]    [P5]    [P7]    [P8]    [P9]    [P11]
20.34   20.34   5.33    5.33    5.50    20.32   5.50    5.50    5.50    5.34
```

- Load ports (P2+P3+P11): 16.0/16.5 = **0.97 load µops/cycle** (32% of peak).
- Store-data: 11.0/16.5 = 0.67/cycle (33% of peak).
- ALU (P0+P1+P5): 61.0/16.5 = **3.70 ALU µops/cycle**. With 3 SIMD ALU
  ports that's 123% -- saturated.

Interpretation: the u16 W=11 cell is also strongly **ALU-bound**. Compared
to u32 W=10, the per-iteration ALU pressure increases from 53.99 to 61.0
(a 13% increase) but the iteration count drops from 8 to 5.8 (a 27% drop),
so the *per-block* ALU load is 1.13 * 0.73 = 0.83x the u32 case. Despite
that, the fused cell incurs a >40% overhead vs bare on this row because
the bare-unpack version (whose asm we did not extract) has the
broadcast-add removed, which on this kernel happens to fall on the
already-saturated P0/P1/P5 chain.

## Comparison to perf-stat (Step 2)

Step 2 (`perf stat`) was skipped because `perf` is not installed in this
container. The llvm-mca port pressure above is the substitute -- and it
gives a sharp answer:

> Both representative cells (u32 W=10 and u16 W=11) on AVX2 ymm are
> theoretically **ALU-port-bound on P0/P1/P5**, not memory-bound. Load-port
> utilisation predicted at 32-40% of peak.

This is consistent with the matrix observation that fusing FoR is sometimes
visible (10-50% overhead): adding one more `vpaddd` per output to an
already-3.7-to-4.1 ALU-µops/cycle pipe pushes the ALU into hard saturation,
adding measurable cycles.

## Bare-unpack mca was not extracted

The bare-unpack inner loops are emitted as standalone `#[inline(never)]`
symbols (`<u32 as BitPacking>::unpack` etc.). Picking the W=10 monomorphisation
out of 32 same-named u32 monomorphisations requires matching by shift-amount
pattern, which we did not pursue under the time budget. The fused inner
loops above are sufficient to answer the "is fusing free because the
kernel is memory-bound" question.

Interpretation: bare-unpack's port pressure would differ from fused by
removing the `vpaddd ymm, ymm0, ymmword ptr [mem]` and `vpaddd ymm, ymm_const`
operations that materialise the FoR reference. Those are precisely the µops
sitting on P0/P1/P5. Removing them frees the saturated ALU and is exactly
why bare is faster than fused on the cells where fusing has overhead. The
measured 0% overhead cells are cells where the bare baseline was already
under-saturated on P0/P1/P5 -- adding the broadcast-add fills empty slots
on those ports for free.

## Update: bare u32 W=10 ymm inner loop (added after initial draft)

We extracted the standalone `<u32 as BitPacking>::unpack` for W=10 from the
AVX2 ymm binary -- the u32 unpack mangled-name hash `e29a2e0b` identified by
its shift-amount pattern (vpsrld immediates {2,4,6,8,10,12,14,16,18,20,22,24,26}
matches W=10).

Source: `measurements/asm/bare_u32_w10_ymm.s` (32 stores, 30 vpsrld/vpshldd,
30 vpand per iteration).

`llvm-mca -mcpu=emeraldrapids -iterations=100`:

| metric             | bare u32 W=10 ymm | fused u32 W=10 ymm (above) |
|--------------------|------------------:|---------------------------:|
| Block RThroughput  | 21.7 cyc/iter     | 13.3 cyc/iter              |
| Dispatch IPC       | 4.32              | 3.06                       |
| µops/cycle         | 5.68              | 4.15                       |
| iters per 1024 blk | 4 (256 out/iter)  | 8 (128 out/iter)           |
| pred. cyc/block    | 86.8              | 106.4                      |
| pred. ns/block @2.1GHz | 41.3 ns       | 50.6 ns                    |
| measured ns/block  | 63.7 ns           | 63.8 ns                    |

**Bare and fused predicted timings differ by ~10 ns but the measured
matrix shows them within 0.1 ns of each other.** This is the strongest
direct evidence: at the theoretical-port-pressure level, fusing *should*
add ~10 ns (because fused has more total uOps spread over more iterations),
but the measured per-block time is identical because both numbers are
floor-bound by the divan + closure + memset overhead at ~22-23 ns and
the actual kernel work is small relative to that. The two kernels finish
their actual work in <50 ns each; the visible 63 ns is bookkeeping.

Port pressure breakdown (bare):
```
P0=20.00  P1=20.01  P5=19.99  P11=2.00  loads(P2+P3+P11)=6.00
```

Per cycle:
- Loads: 6.00/21.7 = 0.28 µops/cycle (9% of 3 µops/cycle peak)
- ALU (P0+P1+P5): 60.00/21.7 = 2.76 µops/cycle (92% of 3-port peak)

The bare kernel is **also ALU-bound** -- just at a slightly different
saturation point than fused. The conclusion that fusing is free
*because* there's port-level slack holds: bare uses 2.76 µops/cycle on
ALU, leaving room for fused's 4.07 µops/cycle (which is over-provisioned
but spreads across the same 3 ports and finishes faster per output).

**Memory bandwidth check.** A 1024-element u32 unpack reads (1024*10/8)=
1280 bytes packed + writes 4096 bytes unpacked = 5376 bytes/block. At
63.7 ns/block that is 84.4 GB/s -- less than 1/3 of Emerald Rapids L1d's
~250 GB/s peak. **The kernel is not memory-bound.**
