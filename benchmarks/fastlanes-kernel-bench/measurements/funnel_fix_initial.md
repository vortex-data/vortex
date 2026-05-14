# funnel-shift-fix experiment (initial / partial)

Status: bench file built and run, full analysis NOT YET COMPLETED.

## Setup

`benches/funnel_shift_fix.rs` adds four hand-written variants of decoding one
1024-element u64 block at W=51 and W=63:

| variant                | sequence                                           |
|------------------------|----------------------------------------------------|
| `baseline_macro_bare`  | call existing `<u64 as BitPacking>::unpack`        |
| `baseline_macro_fused` | call existing `<u64 as FoR>::unfor_pack`           |
| `hand_legacy`          | hand intrinsics: `vpsrlq + vpsllq + vpor + vpand + vpaddq` |
| `hand_funnel`          | hand intrinsics: `vpshldq + vpand + vpaddq` (the proposed fix) |

Both `hand_*` use `#[target_feature(enable = "avx2,avx512f,avx512vl,avx512vbmi2,bmi2")]`
to force EVEX-256 codegen even in the default ymm build.

## ASM verification (binary `funnel_shift_fix-a6f538fbca7a3082`)

`hand_funnel_w51` inner loop (LLVM auto-unrolled 4×, 16 outputs/iter):
```
vmovdqu  ymm2, [rdi+rax-0x48]
vpshldq  ymm2, ymm2, [rdi+rax-0x68], 0x2c   # vpshldq with memory operand!
vpand    ymm2, ymm2, ymm1
vpaddq   ymm2, ymm2, ymm0
vmovdqu  [rdx-0x60], ymm2
... (3 more copies)
```

`hand_legacy_w51` inner loop (single body, 4 outputs/iter):
```
vmovdqu  ymm2, [rdi]
vmovdqu  ymm3, [rcx]
vpsrlq   ymm4, ymm2, 0x14
vpsllq   ymm3, ymm3, 0x2c
vpor     ymm4, ymm4, ymm3
vpand    ymm4, ymm4, ymm1
vpaddq   ymm4, ymm4, ymm0
vmovdqu  [rsi], ymm4
```

So the two functions are genuinely different ASM, not ICF'd as I initially
suspected from a misread of the symbol table.

## Initial bench (best-of-1, --min-time 0.5)

| variant                | W=51 ns | W=63 ns |
|------------------------|--------:|--------:|
| baseline_macro_bare    |  199.7  |  212.4  |
| baseline_macro_fused   |  185.9  |  199.9  |
| hand_funnel            |  238.9  |  247.9  |
| hand_legacy            |  310.7  |  311.7  |

## Findings so far

1. **`vpshldq + vpaddq` IS achievable on this CPU** -- the `hand_funnel`
   ASM proves it. The "compiler can't fuse vpshldq with vpaddq" hypothesis
   is wrong as a blanket statement.

2. **Funnel-shift saves ~23% vs the legacy 3-shift sequence** at the
   same call site (`hand_funnel` 239 ns vs `hand_legacy` 311 ns at W=51).
   Confounded slightly by LLVM auto-unrolling `hand_funnel` 4x but not
   `hand_legacy`.

3. **The macro-generated `unfor_pack` is faster than both hand variants**
   (`baseline_macro_fused` 186 ns vs `hand_funnel` 239 ns) due to
   FastLanes' aggressive unrolling and lane interleaving (`FL_ORDER`
   permutation). The hand kernels' simple linear loop loses out.

4. **The 192 ns `baseline_macro_bare` here does NOT reproduce the 128 ns
   matrix observation** for the same `(u64, W=51, ymm)` cell. Suggests the
   `funnel_shift_fix` binary was built with different ambient RUSTFLAGS
   than the matrix binary. Need to verify and rebuild apples-to-apples
   before drawing strong conclusions about the macro behaviour.

## Open follow-ups

- **Verify build flags**: dump the ambient RUSTFLAGS used to build
  `funnel_shift_fix-a6f538fbca7a3082`; confirm whether `target-cpu=native`
  was applied. If not, rebuild with the same flags as the matrix binary
  and re-bench.
- **Apples-to-apples hand variant**: rewrite the hand kernels to use the
  same FL_ORDER lane interleaving and unrolling factor as the macro, so a
  faster hand_funnel would actually demonstrate the macro is leaving perf
  on the table.
- **Macro patch**: modify `unpack!` in `src/macros.rs` to combine-then-mask
  instead of mask-then-combine, rebuild the matrix binary, and re-bench
  the `u64 W in [45..63] ymm fused` cells. This is the real test of
  whether the upstream macro can emit `vpshldq` even with the FoR add.

## Verdict (preliminary)

The compiler **can** emit `vpshldq + vpaddq` when given the right IR shape
(via intrinsics or simple bitwise expressions). So the +52% overhead in
the matrix's `u64 W=51 ymm fused` cell is **not** a fundamental CPU
limitation -- it is most likely caused by the upstream `unpack!` macro
emitting a mask-then-combine pattern that prevents LLVM's funnel-shift
recognition once the FoR `wrapping_add` is added downstream.

Final answer (compiler vs CPU) requires the apples-to-apples rebuild
described in "Open follow-ups".
