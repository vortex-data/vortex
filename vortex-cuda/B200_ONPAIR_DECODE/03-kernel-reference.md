# Kernel reference

All kernels live in `vortex-cuda/kernels/src/onpair_shmem_*.cu`; `build.rs` compiles each to PTX.
Selection happens in `pick_auto_kernel` (`vortex-bench/src/onpair_bench.rs`). Variant kernels are
thin `#define` wrappers that re-`#include` a base kernel with a different launch config — so the
*body* is shared and only existing kernels' configs differ.

## Shipped (selected by `pick_auto_kernel`)

| kernel | role | notes |
|---|---|---|
| `4tpt_b128o12` | **B200 general default** | 128-thread blocks, `__launch_bounds__(128,12)` → 40 regs, 75% occ, no spill |
| `4tpt_split8read_b128o12` | **B200 high-`frac_le8`, dict ≤16384 entries** | 8 B `uint2` reads from `dict_s8` + >8 B tail from `dict_padded` |
| `4tpt_split8read` | GH200 high-frac bits12 | original Hopper winner (unchanged) |
| `4tpt` | GH200 general default | original (unchanged) |
| `const1`/`const2`/`s4l1_16tpt`/`s8_4tpt` | degenerate-dict fast paths | arch-independent early-exits |

## Evidence-only (in the registry for sweeps, never auto-selected)

New this work, each documents a mechanism or a negative result:

| kernel | what it demonstrates |
|---|---|
| `4tpt_b128`, `_o6`, `_b512o3`, `_b64`, `_b64o24`, `_wpb8_occ`, `_split8read_occ` | granularity/occupancy decomposition (granularity is the lever) |
| `4tpt_split4read_b128o12` | 4 B reads lose to 8 B (8 B is the optimal width) |
| `4tpt_lenbucket_b128` | quantized-width dict ties `b128o12` on bits16 |
| `4tpt_vwidth`, `_vwidth_b128` | exact variable-width dict — unaligned loads, −74% |
| `4tpt_vwidth4`, `_vwidth4_b128` | quantized 4-aligned variable-width — +6% bits12, −8 to −19% bits16 |
| `4tpt_cluster_dsmem` | dict in cluster distributed-shared-mem — byte-exact but −80% |
| `4tpt_shdict8` | `dict_s8` staged in shared (persistent grid) — −47 to −65% |
| `8tpt`, `8tpt_b128` | 8 tokens/thread — 4tpt is the amortization peak |
| `4tpt_ablate`, `_ablate_no{gather,emit,drain,scan}`, `_ablate_cfree` | **ablation NCU-proxies** (timing-only, not byte-exact) — found the emit is ~70% of runtime |

## Pre-existing kernels (from earlier sessions)

`2tpt`, `s4l1_*`, `s8_*`, `pdict`/`vdict`/`regcache` (persistent/shared dict variants), `ldcs`,
`split8`/`split8_wpb8*`, `tma`, `wpb8`. Mostly superseded by the b128o12/split8read winners or
kept as evidence.

## `KernelLayout` → kernel-arg mapping (in `launch_variant`)

Each layout selects the device buffers passed to the kernel. **Critical:** a wrong layout feeds
wrong buffers → `CUDA_ERROR_MISALIGNED_ADDRESS` / segfault. New layouts added this work:

| layout | kernel args | dict buffers |
|---|---|---|
| `Stride16` | codes, chunk_offsets, dict_padded, lens, output, total_tokens | 16 B padded dict (also used by 8tpt at chunk_size 256, and ablation kernels) |
| `SplitRead8` | + `dict_s8` | 8 B/entry dict + padded tail |
| `SplitRead4` | + `dict_s4` | 4 B/entry dict + padded tail |
| `VWidth` | dict_off32 (offset:24\|len:8) + dict_bytes | packed variable-width |
| `VWidth4` | dict_q4_dir (offset<<5\|len) + dict_q4 | 4-aligned quantized packed |
| `ClusterDsmem` | dict_padded + dict_entries | cluster-sharded in DSMEM |
| `ShDict8` | dict_s8 + dict_padded + lens + dict_entries | `dict_s8` staged in shared (persistent grid) |
| `LenBucket` | dict_lenbucket + lb_meta | requires `ONPAIR_DICT_REORDER=lenbucket` |

The directory buffers (`dict_off32`, `dict_q4`/`dict_q4_dir`) are built **decode-side from the
native dict bytes/offsets — no on-disk format change.**
