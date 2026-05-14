# `funnel_shift_fix` ASM verification

Inner-loop disassembly extracted from
`target/release/deps/funnel_shift_fix-a6f538fbca7a3082`,
built with
`RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit,-avx512fp16"`
on rustc 1.91.0. Host: KVM x86_64 advertising Cascade Lake CPUID flags but
running on hardware that successfully executes AVX-512-VBMI2 (`vpshrdq`,
`vpermt2b`, `vpermt2q`) and GFNI (`vgf2p8affineqb`) — confirmed by
direct execution of those instructions. AVX-512-FP16 was disabled
(`-avx512fp16`) because LLVM's host-CPU detection emitted `vmovw`, which
SIGILLs on this host even though CPUID claims FP16 support.

The two baseline benches use the upstream macro-generated kernels as
compiled here. Note: with this rustc/LLVM (1.91.0) the macro-generated
**bare** unpack does *not* emit `vpshldq` for `u64 W=51`; it emits the same
`vpsrlq + vpsllq + vpternlogq` legacy sequence as the **fused** unfor_pack.
This differs from the matrix's older Rust toolchain, where the bare
kernel did emit `vpshldq`.

## 1. `baseline_macro_bare` — `<u64 as BitPacking>::unpack` for W=51 (zmm)

Inner loop (one iteration handles 8 ZMM lanes = 64 outputs per iter):

```asm
vmovdqu64  zmm20, [rdi+rax*8-0x1900]
vpandq     zmm21, zmm20, zmm0           # zmm0 = mask = (1<<51)-1
vmovdqu64  [rsi+rax*8-0x1900], zmm21
vmovdqu64  zmm21, [rdi+rax*8-0x1880]
vpsrlq     zmm20, zmm20, 0x33           # >> 51
vpsllq     zmm22, zmm21, 0xd            # << 13
vpternlogq zmm22, zmm20, [rip+...]{1to8}, 0xec   # OR + AND with const mask
vmovdqu64  [rsi+rax*8-0x1500], zmm22
... (~16 such blocks per iter, sliding shift counts 0x33,0x26,0x19,...)
```

ALU op-counts inside the loop:

```
vmovdqu64  : 17 (load+store)
vpsrlq     : 16
vpsllq     : 13
vpternlogq : 13   (OR + AND folded together)
vpandq     :  3   (when shift+W < 64 there is no funnel)
vpaddq     :  0   (no FoR step)
```

`vpshldq` (the funnel-shift) is **not** present. LLVM 1.91 chose the legacy
`vpsrlq + vpsllq + vpternlogq` lowering for the bare kernel as well.

## 2. `baseline_macro_fused` — `<u64 as FoR>::unfor_pack` for W=51 (zmm)

```asm
vmovdqu64  zmm22, [rdi+rax*8-0x1900]
vpandq     zmm23, zmm22, zmm1           # mask
vpaddq     zmm23, zmm23, zmm0           # zmm0 = reference broadcast (FoR add)
vmovdqu64  [rdx+rax*8-0x1900], zmm23
vpsrlq     zmm22, zmm22, 0x33
vmovdqu64  zmm23, [rdi+rax*8-0x1880]
vpsllq     zmm24, zmm23, 0xd
vpandq     zmm24, zmm24, [rip+...]{1to8}    # const mask of high contribution
vpaddq     zmm22, zmm22, zmm0           # FoR add to lo half
vpaddq     zmm22, zmm22, zmm24          # OR materialised as add (safe because masks are disjoint)
vmovdqu64  [rdx+rax*8-0x1500], zmm22
```

The **OR became a `vpaddq`** because the FoR add can chain straight onto
`vpsrlq` and the high-side `vpandq` already disjoint-mask-clears, so
`add(lo, hi)` is equivalent to `or(lo, hi)`. The cost: an extra `vpaddq`
per output element compared to the bare kernel's single `vpternlogq`.

ALU op-counts inside the loop:

```
vmovdqu64  : 17
vpsrlq     : 16
vpsllq     : 13
vpternlogq :  0   (replaced by vpandq + vpaddq)
vpandq     : 16   (per-element mask of shifted high half + final mask)
vpaddq     : 29   (FoR ref-add + shift-OR materialised as add)
```

So fused trades 13 `vpternlogq` ops for 13 extra `vpandq` + 29 `vpaddq` ops:
net **+29 ALU µops per iter** versus bare. This is the structural source of
the matrix's `u64 W=51 ymm fused` overhead.

## 3. `hand_legacy_w51` — inline-asm `vpsrlq + vpsllq + vpor + vpand + vpaddq` (ymm)

```asm
vmovdqu  ymm2, [rdi]                # lo word
vmovdqu  ymm3, [rcx]                # hi word
vpsrlq   ymm4, ymm2, 0x14           # >> 20
vpsllq   ymm3, ymm3, 0x2c           # << 44
vpor     ymm4, ymm4, ymm3
vpand    ymm4, ymm4, ymm1           # mask
vpaddq   ymm4, ymm4, ymm0           # FoR ref-add
vmovdqu  [rsi], ymm4
add      rax, 0x20                  # output stride
add      rdi, 0x18                  # input stride
cmp      rax, 0x2000
jne      <loop>
```

3 ALU ops on the funnel (`vpsrlq + vpsllq + vpor`) plus mask + add.

## 4. `hand_funnel_w51` — inline-asm `vpshrdq + vpand + vpaddq` (ymm)

```asm
vmovdqu  ymm2, [rdi]
vmovdqu  ymm3, [rcx]
vpshrdq  ymm4, ymm2, ymm3, 0x14     # funnel shift right by 20 (single µop)
vpand    ymm4, ymm4, ymm1
vpaddq   ymm4, ymm4, ymm0
vmovdqu  [rsi], ymm4
add      rax, 0x20
add      rdi, 0x18
cmp      rax, 0x2000
jne      <loop>
```

1 ALU op for the funnel + mask + add. Saves 2 µops per chunk compared to
`hand_legacy_w51`.

## 5. `hand_legacy_w63` and `hand_funnel_w63`

Identical structure to W=51 with shift constants `0xa` (=10) for vpshrdq /
vpsrlq and `0x36` (=54) for vpsllq. The mask is `(1<<63)-1`. ASM omitted
for brevity; the inner-loop instruction sequence is identical to (3) and
(4) modulo the immediate constants.

## Verification summary

| function           | `vpshldq`/`vpshrdq` | `vpsllq` (ymm/zmm) | `vpsrlq` | `vpaddq` | `vpor`/`vpternlogq` |
|--------------------|--------------------:|-------------------:|---------:|---------:|--------------------:|
| baseline bare W=51 |                  0  |                 13 |       16 |        0 |                  13 |
| baseline fused W=51|                  0  |                 13 |       16 |       29 |                   0 |
| hand_legacy W=51   |                  0  |                  1 |        1 |        1 |                   1 (vpor) |
| hand_funnel W=51   |                  1  |                  0 |        0 |        1 |                   0 |

The `hand_*` rows are per-loop-iteration inside a 256-iter loop; each iter
emits 4 outputs.
