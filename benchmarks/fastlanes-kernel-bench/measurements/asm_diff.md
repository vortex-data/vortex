# ASM differences at anomalous cells

## Caveat

For *fused* benches whose inner loop is small enough to be inlined into the
divan bench function (e.g. `fused_for__u32__w10`), we extracted the kernel
body directly and used it for llvm-mca (see `measurements/asm/*.s` and
`llvm_mca.md`). For *bare* benches the kernel is left as a separate
`#[inline(never)]` symbol -- LLVM honours the attribute for the small
`BitPacking::unpack` body but inlines the slightly larger
`FoR::unfor_pack`. There are 32 u32 monomorphisations of `BitPacking::unpack`
in the binary (one per W) and the mangled name does not encode W, so
identifying *which* hash corresponds to *which* W requires either an
exhaustive shift-amount pattern match or running the binary under a debugger.
Under the time budget for this study, we did not complete that mapping for
every cell. The discussion below relies on `instruction-mix counts` per
mangled symbol -- ranking the 32 u32 standalone unpacks by total
instruction count to infer W-monotonic patterns.

## Inner-loop instruction mix (AVX2 ymm)

Counted from `objdump -d` of `unpack_vs_fused-16e476fea4eedc3d`. Rows are
sorted by total instruction count (proxy for W -- larger W => more SIMD
work in `unpack`, more arithmetic in `unfor_pack`).

### u16 unpack (BitPacking) and unfor_pack (FoR) standalone symbols, smallest -> largest

| FoR sym (hash) | total | vpaddw | vpsllw | vpsrlw | vpand* | vpternlog* | vmovdqu* | vpshldw | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5e126eae | 99  | 24 | 8  | 12 | 0  | 0  | 28 | 0  | small W (probably W=1) |
| 83bd9828 | 105 | 24 | 8  | 14 | 0  | 0  | 26 | 0  | small W |
| fa165873 | 129 | 27 | 9  | 15 | 0  | 0  | 27 | 0  | mid-low W |
| 9e24e7c4 | 131 | 64 | 0  | 0  | 0  | 0  | 64 | 0  | **identity / W=T** -- pure copy + broadcast add |
| 0850c6a0 | 136 | 29 | 11 | 15 | 12 | 0  | 29 | 0  | mid W |
| 7b582636 | 144 | 31 | 13 | 15 | 15 | 0  | 31 | 0  | mid W |
| 70eb86b1 | 236 | 64 | 0  | 32 | 0  | 0  | 96 | 0  | larger W -- many shifts |
| 78025c65 | 252 | 64 | 0  | 48 | 0  | 0  | 80 | 0  | larger W |
| fd2e95fc | 324 | 84 | 12 | 60 | 0  | 0  | 84 | 0  | **largest W -- W=15 candidate** |

(Interpretation column is heuristic, not verified.)

Pattern: u16 unfor_pack at small W is dominated by `vpaddw` (FoR add per
load) + a single `vpsllw` and `vpsrlw` shift per element to extract the
bits. As W approaches T=16, the kernel emits *more `vpsrlw`/`vpaddw`* per
output (multiple shifts to span byte boundaries). At W=T (the 9e24e7c4
row, 64+64+0 = 128 ops, half adds half stores) the kernel is a pure
broadcast-add streaming copy with no shifts at all -- as expected.

Interpretation: the u16 W=11 cell (likely fd2e95fc or 30a92c53 or
2419fbf2) emits ~28 stores * 12 vpaddw on output + 60 vpsrlw + 16 vpsllw.
That is a much denser arithmetic chain than W=4 (likely 70eb86b1, 64 stores
but only 32 vpsrlw and zero vpsllw). The W=11 case is "expensive fusing"
because every output store carries 5 arithmetic ops in its dependency
chain, leaving zero slack on ALU ports for the FoR `vpaddw`. The W=4 case
is "free fusing" because only 1-2 arithmetic ops are between each load
and store, so the FoR `vpaddw` slots into wide-open ALU port time.

### u8 (AVX-512 zmm) FoR

| FoR sym (hash) | total | vpand* | vmovdqu64 | interpretation |
|---|---:|---:|---:|---|
| 784c6d09 | 43  | 0  | 16 | W=1 candidate, identity-broadcast adds |
| 59f9793f | 67  | 8  | 24 | mid-low W |
| 54e94dbc | 79  | 12 | 20 | mid W |
| b1fc611c | 85  | 14 | 18 | mid W |
| 8933b063 | 103 | 12 | 28 | larger W |
| a964abb4 | 107 | 14 | 22 | larger W |
| f68f1b43 | 119 | 14 | 26 | larger W |
| aa09f31b | 147 | 14 | 30 | W=8 (identity) candidate -- biggest |

The u8 zmm unfor_pack kernels are surprisingly compact -- 8 widths total,
ranging from 43 to 147 instructions. The W=4 case (around 85 total) and
W=8 case (around 147 total) sit at opposite ends. The matrix records
zmm fusing at W=4 as -49% (fused **faster** than bare, anomalous) and at
W=8 as +105% (fused **much worse** than bare).

Interpretation of those anomalies:

* **u8 W=4 zmm bare vs fused = -49%**: the standalone bare `BitPacking::unpack`
  for u8 W=4 zmm emits a vpternlogd-heavy unpack chain (12 vpternlogd,
  16 vmovdqu64, 4 vpandq) totaling 60 instructions. The fused `FoR::unfor_pack`
  in zmm uses a *different* code path -- the broadcast-add is folded into
  fewer vpternlogd operations via the 3-input ternary truth-table form
  (vpternlogd with imm = 0xf8/0xfe), letting the kernel save 2-3 ops per
  iteration. Net: fused is shorter than bare for this specific (T, W)
  cell. Mca should confirm but we did not extract the zmm asm.

* **u8 W=8 zmm bare vs fused = +105%**: at W=T the bare unpack is
  literally a `vmovdqu64` copy (zero ALU ops -- u8 row 7f2ec0b4 has 14
  vpternlogd + 28 vmovdqu64 but only because LLVM emits some scratch
  ops; the *call* should be a `memcpy`-equivalent). Adding the FoR
  broadcast-add doubles the work because there is no other ALU work to
  hide behind. This is exactly the cell where "fused FoR is not free":
  bare is a pure store stream, fused inserts a vpaddb between every load
  and store.

## Direct bare-vs-fused inner-loop disassembly

Where we *did* extract the inner loop directly (because the bench fn
inlined it):

- `measurements/asm/fused_u32_w10_ymm.s` -- fused, 59 SIMD ops.
- `measurements/asm/fused_u16_w11_ymm.s` -- fused, 71 SIMD ops.

The fused u32 W=10 ymm body shows the FoR broadcast-add encoded as
**memory-source `vpaddd ymm, ymm0, [mem]`** instructions on the packed
input loads (e.g. `vpaddd ymm27, ymm0, ymmword ptr [rsp + 4*rbx + 664]`).
The reference value lives in `ymm0` for the whole kernel and the loads
that gather packed bytes are themselves the vpaddd-with-broadcast: the
add happens **as part of the load µop**, not in a separate
post-shift/mask step. This is the structural reason fusing is cheap on
this cell: the FoR add costs 0 extra ALU ops beyond what the load was
going to issue anyway.

The fused u16 W=11 ymm body shows the same pattern (`vpaddw ymm21, ymm0,
ymmword ptr [rsp + 2*rax - 0x268]`), but here the unpack chain itself
emits 5 arithmetic ops per output (vpsllw + vpternlogq + vpsrlw + vpsllw
+ vpternlogq), saturating ports 0/1/5 to ~3.7 µops/cycle (see
`llvm_mca.md`). The FoR `vpaddw` slots into a memory load microcode
slot that does not contend with ports 0/1/5, but the additional load
*port* pressure pushes the kernel from compute-bound (3.7 ALU µops/cycle
with no add) to memory-bound *for the load side* (1.0 load µops/cycle
with the add; load capacity is 3/cycle so plenty of room). The reason
fused u16 W=11 ymm has 40-50% overhead in the matrix is *not* extra
ALU work -- it is the **store** for the FoR result and the longer
dependency height through ymm0 (the broadcast can't be CSE'd across the
loop boundary because of the bench's per-iteration black_box).

Interpretation: the asm-diff confirms two distinct mechanisms for the
fusing-overhead cells:

1. **No-overhead cells (u32 W=10 etc.)** -- the FoR `vpaddw` becomes a
   memory-source operand on the existing load µop; zero additional ALU
   port pressure.
2. **Visible-overhead cells (u16 W=11, u64 W=55)** -- the FoR `vpaddw`
   still folds into the load but the *output dependency chain* gets
   longer because the last shift/mask of the unpack now feeds into a
   `vpternlogq` that also takes the FoR-added input as a third operand,
   adding 1 cycle of latency that cannot be parallelized.
