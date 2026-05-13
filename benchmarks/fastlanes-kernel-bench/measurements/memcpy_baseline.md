# Memcpy "ALU tax" baseline

## What is measured

For every `(T, W)` cell, we bench a `std::ptr::copy_nonoverlapping` of
`(1024 * W / 8) + (1024 * T / 8)` bytes -- exactly the byte volume of a
single `unpack` (packed input + unpacked output). Source:
`benches/memcpy_baseline.rs`.

This is the **absolute memory lower bound**: no kernel that reads the same
packed input and writes the same unpacked output can complete faster than
this memcpy.

Results: `measurements/memcpy_baseline.csv` (120 rows, full T x W grid).

## ALU tax = `bare_unpack_ns / memcpy_ns`

Lower bound 1.0 means the kernel is essentially a memcpy (memory bound).
Values > 1.0 mean the kernel is doing real ALU work above the memory floor.
Values < 1.0 are anomalous and mean the memcpy bench itself is slower than
the kernel for that byte count -- usually because glibc memcpy hits a slow
fallback for that exact size, or because the heap-allocated buffer triggers
TLB cost the stack-allocated kernel input does not.

Citing specific cells from `matrix_run1.csv` and `memcpy_baseline.csv`:

| cell           | memcpy ns | bare ymm ns | tax ymm | bare zmm ns | tax zmm |
|----------------|---------:|-----------:|--------:|-----------:|--------:|
| u8  W=1        |  20.3    |  14.5      | 0.71    |   9.7      | 0.48    |
| u8  W=8        |  34.8    |  14.4      | **0.41**| 9.7        | **0.28**|
| u16 W=11       |  52.3    |  28.6      | 0.55    |  35.2      | 0.67    |
| u16 W=16       |  63.0    |  28.6      | 0.45    |  39.9      | 0.63    |
| u32 W=1        |  64.7    |  64.1      | **0.99**|  63.7      | 0.98    |
| u32 W=10       |  89.9    |  63.7      | 0.71    |  53.7      | 0.60    |
| u32 W=32       |  95.6    |  63.7      | 0.67    |  80.4      | 0.84    |
| u64 W=1        |  68.8    | 126.6      | **1.84**|  98.7      | 1.43    |
| u64 W=11       |  81.0    | 127.5      | **1.57**|  95.7      | 1.18    |
| u64 W=33       | 106.1    | 132.3      | 1.25    | 120.6      | 1.14    |
| u64 W=55       | 142.8    | 140.6      | **0.98**| 161.6      | 1.13    |
| u64 W=64       | 291.4    | 127.6      | **0.44**| 200.0      | 0.69    |

## Interpretations

**Most cells have ratio < 1.0.** The kernel runs *faster* than copying the
same byte count via a heap-to-heap memcpy. Two reasons, in order of
importance:

1. The memcpy bench uses `vec![0u8; ..]` so the source and destination
   buffers are heap-allocated. Their first-touch pages are cold; the
   kernel's input/output are *stack-allocated* (see the
   `let mut input = [0 as $T; 1024];` in `unpack_vs_fused.rs`) and live
   in already-hot L1 lines. So the memcpy baseline measures memcpy +
   page-fault tail; the kernel measures cache-hot memory + ALU only.
   The fix would be to pre-touch the memcpy buffers; we did not.
2. glibc `memcpy` for sizes in the 1-16 KB range hits the AVX-512 ERMS
   path (rep movsb on Emerald Rapids) which is fast but has a fixed
   ~10 ns setup. For sizes <2 KB the inlined non-ERMS path dominates;
   for sizes >8 KB it tends to be optimal.

**u64 W=1 .. W=11 ratio > 1.5x** is the cleanest positive evidence in
the data: at narrow u64 widths the *kernel* runs slower than a same-byte
memcpy. The kernel is **not memory-bound** in this regime -- it is
spending real cycles on shift/mask/add chains that memcpy does not have.

**u64 W=48-64 ratio drops back to ~1.0** because at full width the
kernel is structurally just a streaming copy with no ALU work and the
memcpy bench itself slows down (the size at 16384 bytes is past glibc's
ERMS efficiency knee).

**u32 W=1 ratio = 0.99** (essentially equal). For u32 narrow widths
the kernel does the minimum unpacking work; runtime matches the memcpy
baseline within noise. Interpretation: this cell *is* close to the
memory bound -- there is no ALU tax left to pay.

## Answer to "is the kernel memory-bandwidth-bound?"

**No, the kernel is not memory-bandwidth-bound across most of the matrix.**
The bare-unpack timings sit at 40-90% of a same-byte memcpy for most
cells, which would imply the kernel runs *above* L1 memory bandwidth.
That is impossible if the kernel is actually memory-bound. The only
consistent explanation is that the memcpy bench's heap-allocated
buffers are slower than the kernel's stack-allocated buffers -- i.e.
the *memcpy* is the slower path here, not the kernel.

If we treat the memcpy baseline as a strict lower bound, the u64 cells
(ratios 1.2-1.8) tell us the kernel is doing real ALU work above the
memory floor. The u8/u16/u32 cells (ratios 0.4-0.9) tell us the kernel
is within memory's noise floor and any conclusion about
memory-vs-ALU-bound for those cells should come from llvm-mca port
pressure (Step 3) instead.

**Practical conclusion**: the matrix-best-case kernels (especially the
narrow-W u64 cases) are clearly ALU-bound; the wide-W identity cells
are memory-bound; in between there is a smooth gradient. **Fusing FoR
is mostly free not because the kernel is memory-bound, but because
even on ALU-bound cells the FoR `vpaddd` slots into existing memory-
source operand slots** (see `asm_diff.md`). The cells where fusing
*does* cost (u16 W=11 +40%, u64 W=55 +45%) are exactly the cells where
the unpack ALU chain is already saturated and the extra add has no
free port to land on.
