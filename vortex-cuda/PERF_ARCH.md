# OnPair `onpair_shmem` — projected throughput across NVIDIA architectures

Reference workload (kept constant across all projections):
10M URL rows, 4096-entry padded dict (64 KB), 51.2M tokens, ~11 B/token,
**584 MB byte-packed decoded output**, kernel-only timing, all inputs
HBM-resident. Measured A100 80GB: **503–525 GiB/s**, ncu DRAM
~712 GB/s read+write = **~51 % of A100's ~1.4 TB/s achievable HBM2e**
(1.555 TB/s spec × 0.9 STREAM efficiency).

## Summary

| Arch | Year | HBM peak | Effective HBM (~90%) | Projected GiB/s | Decode time (584 MB) | Bottleneck | Best published comparable (string/byte decomp) | Confidence |
|---|---|---|---|---|---|---|---|---|
| V100 SXM2 | 2017 | 0.9 TB/s | ~0.86 TB/s | **~290 GiB/s** | ~1.92 ms | DRAM + LSU mixed | nvCOMP LZ4 "up to 100 GB/s" general; no string-decode ≥100 on V100 | Medium |
| A100 80GB | 2020 | 1.555 TB/s | ~1.4 TB/s | **511 GiB/s (measured)** | 1.06 ms | DRAM at 51% + shmem RAW | **GSST 191 GB/s** (Vonk 2025); DietGPU ANS 250–410; nvCOMP LZ4 312–320 (column int) | — |
| H100 SXM | 2022 | 3.35 TB/s | ~2.85 TB/s | **~1040 GiB/s** | 0.52 ms | DRAM if bw-scales; possibly shmem/LSU | No published H100 string-decode in literature; nvCOMP H100 LZ4 ~1.4× A100 (~440 GB/s extrapolated, ratio not throughput) | Medium |
| H200 SXM | 2024 | 4.8 TB/s | ~4.1 TB/s | **~1490 GiB/s** | 0.36 ms | Likely DRAM | Same as H100 plus 43% more HBM3e bw; no published H200 string-decode | Medium-low |
| B200 SXM | 2024–25 | ~8 TB/s | ~6.8 TB/s | **~2470 GiB/s** | 0.22 ms | DRAM if SM/LSU scales; HW DE bypasses kernel entirely for nvCOMP formats | nvCOMP HW Decompression Engine **up to 600 GB/s** for LZ4/Snappy/Deflate (different mechanism) | Low |
| GB200 / B300 | 2025 | ~8 TB/s (192 GB HBM3e) | ~6.8 TB/s | **~2470 GiB/s** (single GPU) | 0.22 ms | Same as B200; NVLink doesn't apply to kernel-local case | Same DE 600 GB/s; B300 has 288 GB HBM3e | Low |

The projections assume the kernel stays bandwidth-bound. The next two
sections justify that assumption and call out where it likely breaks.

## How we project

A100 measurement: kernel writes 584 MB packed output and reads ~128 MB
of codes/lens/dict plus internal L1/L2 traffic; ncu observed
**~712 GB/s of effective DRAM** (663 GB/s write + ~50 GB/s read,
see `PERF_SEARCH.md` final ncu profile of `shmem`). That is
**~51 % of A100's ~1.4 TB/s achievable HBM** and ~46% of spec peak.

Naive bandwidth scaling: `new_throughput ≈ A100_throughput ×
(new_hbm / A100_hbm)`. With A100 at 1.555 TB/s and 525 GiB/s measured,
the per-TB/s constant is **~338 GiB/s of decoded output per TB/s of HBM**.
Multiply by each arch's spec peak:

- V100 0.9 TB/s → ~305 GiB/s (round to 290 because of older mem
  controller efficiency, see V100 note below).
- H100 3.35 TB/s → ~1130 GiB/s. Drop to ~1040 because the kernel
  has shared-mem RAW stalls that scale with SM clocks not memory.
- H200 4.8 TB/s → ~1620 GiB/s. Drop to ~1490 for the same reason
  (H200 SM count = H100 = 132; only HBM changes).
- B200 ~8 TB/s → ~2700 GiB/s. Drop to ~2470 because SM count went
  from 132 → 148 active (not 1.65×), so non-DRAM-bound parts of the
  kernel scale sub-linearly.

These are **guesses** based on one measurement; the constant could be
off by 1.3× either direction. See "honesty" below.

## Per-arch notes

### V100 (Volta, CC 7.0)
HBM2 900 GB/s, 80 SMs at 1.38 GHz, **128 KB unified L1/shmem per SM**
(vs A100's 192 KB). Differences that matter for this kernel:
- Same warp-level `__shfl_up_sync` semantics; the inclusive-scan
  works as-is.
- Same `uint4` aligned global stores (CC ≥ 6.0).
- `__stcs` works on Volta. `__launch_bounds__(256, 8)` requested
  occupancy may not be achievable — V100 has 64 warps/SM resident,
  same as A100, but smaller register file (256 KB vs 256 KB — equal
  actually); should be OK.
- DRAM utilisation on V100 is ~95% of HBM peak under good access
  patterns ([Choquette 2017](https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf)),
  so the "~90% achievable" assumption holds.
- **Risk**: 28 fewer SMs than A100 → fewer warps in flight → the
  ~37% "no eligible warps" cycles we saw on A100 could get worse.
  This could push V100 below the 290 GiB/s projection.

**Confidence: medium.** Bandwidth scaling is the dominant lever and
the kernel doesn't use Volta+ features. The unknown is whether the
smaller SM count hurts latency hiding for shmem RAW.

### A100 80GB (Ampere, CC 8.0) — measured reference
**525 GiB/s** at 1.04 ms (`shmem_sorted` best variant), 503–511 GiB/s
for the production `shmem` kernel. Ncu DRAM ~712 GB/s = **51 % of
achievable HBM**. Bottlenecks (from `PERF_SEARCH.md`):
- 30 % L1TEX scoreboard stall (dict_padded random reads).
- 43 % uncoalesced shared accesses (forced by byte-pack contract).
- The remaining slack is shared-mem RAW + warp-sync overhead.

Published comparables on A100:
- **GSST**: 191 GB/s, also string decode, FSST-style ([Vonk 2025](https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1)).
  We are **~2.95× ahead**, but on a strictly easier problem (dict
  lookup is much simpler than full FSST symbol expansion).
- **nvCOMP LZ4** on Mortgage column: 312–320 GB/s ([nvcomp Benchmarks.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md)).
  Different access pattern (LZ back-refs) but illustrative — we are
  ~1.7× ahead.
- **DietGPU ANS**: 250–410 GB/s ([repo](https://github.com/facebookresearch/dietgpu)).
  Different output shape (entropy-coded, fixed-width FP), not directly
  comparable.

### H100 SXM (Hopper, CC 9.0)
HBM3 3.35 TB/s (2.15×A100), 132 SMs (1.22×), 256 KB unified L1/shmem
per SM (1.33×). Hopper additions our kernel does **not** use:
- **TMA / `cp.async.bulk`** for async global↔shared (Hopper-only).
  Could be a big win for the dict-read side (eliminates the L1
  scoreboard stall that costs us 30 % cycles on A100), and could
  amortise output writes via `tma.store`. Not in the kernel.
- **DSMEM** (distributed shared memory across CTA cluster). Possibly
  useful for sharing the dict across blocks instead of refetching
  through L2. Not used.
- **Tensor Memory Async** — irrelevant; no matmul.

Naive bandwidth-scaling projection: 525 × 2.15 ≈ 1130 GiB/s.
Discount to ~1040 GiB/s for two reasons:
1. Hopper memory controller efficiency is ~85 % of peak on STREAM, vs
   ~90 % on A100 ([Wevolver HBM3 guide](https://www.wevolver.com/article/what-is-high-bandwidth-memory-3-hbm3-complete-engineering-guide-2025)),
   so achievable HBM3 is ~2.85 TB/s.
2. The shmem RAW + warp-sync overheads don't scale with HBM. SM count
   only grew 1.22×, and shmem clocks are similar; so the ~30 % of
   kernel time NOT in DRAM gets only 1.22× faster, not 2.15×.

**With TMA, the kernel could reach 1.3–1.6 TB/s effective DRAM**,
which would push throughput closer to the HBM ceiling (~1700 GiB/s
decoded equivalent). Total guess.

**Best published H100 string-decode**: I could not find one. nvCOMP
release notes claim "**up to 1.4× faster LZ4 on H100**" vs A100, and
"up to 1.3× Snappy on H100", but no raw GB/s ([nvCOMP release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html)).
If A100 LZ4 = 312 GB/s on column data, H100 LZ4 ≈ 440 GB/s; our
projection (1040) is ~2.4× ahead.

**Confidence: medium.** Bandwidth scales reliably; the unknown is
whether 30 % of A100 time spent in shmem/sync paths gets worse or
better on Hopper warp scheduler.

### H200 SXM (Hopper refresh, CC 9.0)
Identical SMs to H100, just HBM3e at 4.8 TB/s (1.43× H100). If the
kernel is HBM-bound, projection scales: **~1490 GiB/s**, 0.36 ms.
If shmem/sync is a real ceiling, the win is smaller — maybe 1.25× H100
instead of 1.43×.

**No published comparables** specific to H200 for string decompression.
H200 has no dedicated decompression engine (still Hopper).

**Confidence: medium-low.** Same SMs as H100 but more HBM; if H100 is
already shifting away from pure DRAM-bound (which is plausible since
A100 only spent 51% of cycles on DRAM), H200's extra bandwidth helps
less than 1.43×.

### B100 / B200 (Blackwell, CC 10.0)
HBM3e ~8 TB/s, ~148 active SMs per chiplet, **two chiplets per B200
package** unified via 10 TB/s C2C ([Allocomp](https://allocomp.com/nvidia-b200/)).
Whether our kernel sees ~8 TB/s aggregate or ~4 TB/s per chiplet
depends on how the launch is partitioned — for a single-GPU CUDA
context, the two chiplets present as one device.

Bandwidth-scaling: 525 × (8.0/1.555) ≈ 2700 GiB/s. Discount to ~2470
because:
1. SM-count ratio (132 → 148, 1.12×) is below HBM ratio (2.39×), so
   shmem/sync paths bottleneck.
2. Cross-chiplet shared memory and L2 are NOT free; if blocks land on
   different chiplets and the dict crosses the C2C link, dict-read
   latency goes up.

**Crucial caveat**: Blackwell ships a **fixed-function Decompression
Engine** at up to **600 GB/s** for LZ4/Snappy/Deflate/GDeflate
([NVIDIA blog](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)).
**It does NOT accelerate OnPair / dict-coded strings** — only those
specific formats. So on Blackwell, our SM-resident kernel runs as
usual; the DE is irrelevant to this kernel.

Blackwell features our kernel does not use:
- CTA-cluster shared memory (Hopper feature, extended on Blackwell).
- FP4 / FP6 / new tensor cores (irrelevant).
- TMA from Hopper (still available).

**Confidence: low.** The 2.39× HBM jump is large and the SM-count
mismatch is the largest yet; the projection could easily be 1.5× off.
We have not run this kernel on any Blackwell.

### GB200 / B300 (Blackwell Ultra)
GB200 is a 2× B200 + 1× Grace package; per-GPU specs match B200. B300
has more HBM3e (288 GB) and incrementally faster compute, similar
~8 TB/s bandwidth ([Slyd](https://slyd.com/hardware/nvidia-blackwell)).
**Per-single-GPU projection is the same as B200: ~2470 GiB/s.**
For pipelines that span multiple GPUs over NVLink (kernel does not
currently span), aggregate scales linearly with GPU count.

**Confidence: low**, for the same reasons as B200.

## Honesty section — what we don't know

### Things that could invalidate the projections

1. **The 525 GiB/s on A100 is the kernel-only number, NOT end-to-end.**
   For a PCIe Gen4 round-trip (compressed-in, decoded-out) on a host
   that lacks GPUDirect, the actual achievable rate is **30 ms total
   per `PERF_SEARCH.md` end-to-end accounting**, which is ~20 GiB/s
   amortised — worse than a single CPU core on packed bytes. The
   projection columns above are kernel-only and **assume the
   compressed bytes are already on-device**. For non-GPU-resident
   workloads, none of these numbers apply; PCIe dominates everything.

2. **We have not run this kernel on any GPU except A100.** The "30%
   of time spent in shmem RAW + dict scoreboard" decomposition from
   ncu may behave very differently on Hopper (smarter warp
   scheduler) or Blackwell (more shmem per SM, faster shmem ops).
   If shmem operations get relatively faster, our projection
   under-counts. If LSU per-SM throughput scales differently, the
   projection over-counts. **Guess: ±30 % is the realistic band.**

3. **HBM efficiency varies.** A100 at 51% of achievable HBM means
   the kernel is not actually DRAM-bound — it's "DRAM-aware but
   shmem-limited". As HBM bandwidth grows on newer arches without a
   matching shmem/SM bump, the kernel's effective HBM utilisation
   percentage will DROP, meaning projections that scale linearly
   with HBM are optimistic. The discount factors we applied
   (1130→1040, 1620→1490, 2700→2470) are guesses at how much.

4. **Driver / compiler / hardware errata could break things.** NVCC's
   PTX scheduling, CC-version-specific instruction availability
   (e.g. `__stcs` cache semantics differ), launch-bound register
   pressure heuristics all shift between toolchains. We confirmed
   PTX for sm_80 (A100); sm_90/sm_100 PTX is unverified.

5. **Could a hand-tuned per-arch impl beat us?** Yes, almost
   certainly. On Hopper, TMA-based prefetch of the dict + tma.store
   for output is the obvious lever (the dict reads cost us ~30 % of
   cycles on A100). Plausible Hopper-tuned ceiling: **1.5–1.8× our
   bandwidth-scaled projection, i.e. ~1500–1900 GiB/s on H100**.
   On Blackwell, the same plus CTA-cluster shared dict (dict fits in
   one cluster's combined shmem) could push another 1.2×, but this
   is speculation.

### Things we deliberately did not chase

- We did not try `cp.async` for the dict on A100 (the L1 hit rate is
  already 95 %; speculative win was <10 %). On H100 with TMA this
  becomes meaningfully more attractive.
- We did not test on V100, H100, H200, or any Blackwell. We don't
  have access. All non-A100 numbers above are projections, not
  measurements.

### Where 525 GiB/s on A100 is misleading

- **Kernel-only.** PCIe round-trip + decode is ~30 ms = ~20 GiB/s
  end-to-end (per `PERF_SEARCH.md`'s end-to-end accounting table),
  ~25× slower than a kernel-only number suggests for the round-trip
  case.
- **Dict-coded.** This kernel decompresses a *dictionary lookup*,
  not a full LZ stream or FSST symbol table. Comparing to GSST 191
  GB/s is fair on the access pattern axis but unfair on the algorithm
  axis — GSST does meaningfully more per-token work.
- **Single column / single dtype.** Real workloads multiplex many
  columns through the same SM. The 510 GiB/s assumes the GPU is doing
  nothing else.
- **Workload-shape sensitive.** 11 B/token is favourable. With shorter
  tokens (e.g. 3-byte average), the per-token overhead (warp scan +
  shmem write + sync) dominates, and we'd see lower decoded
  throughput. With longer tokens (e.g. 14-byte mean), we'd see higher.

## Bottom line

On A100: **measured 525 GiB/s, ~2.95× ahead of the only directly
comparable published string-decompressor** (GSST). On newer arches:
projections scale roughly with HBM bandwidth, discounted ~10–15 % for
shmem/sync paths that don't scale with HBM. **Confidence drops sharply
moving forward in time** — H100 medium, Blackwell low. A hand-tuned
per-arch implementation (using TMA on Hopper, CTA-cluster shmem on
Blackwell) could plausibly beat our projection by 1.3–1.8× on each.
None of this matters for non-GPU-resident workloads, where PCIe
dominates and a multi-core CPU wins.

## Sources

- [GSST (Vonk 2025) — TU Delft](https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1)
- [GPU-FSST (Anema 2025)](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf)
- [nvCOMP Benchmarks.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md)
- [nvCOMP release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html)
- [Blackwell Decompression Engine](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [Hopper TMA Deep Dive (PyTorch)](https://pytorch.org/blog/hopper-tma-unit/)
- [DietGPU](https://github.com/facebookresearch/dietgpu)
- [HBM3 efficiency, Wevolver](https://www.wevolver.com/article/what-is-high-bandwidth-memory-3-hbm3-complete-engineering-guide-2025)
- [Volta architecture, Choquette HotChips 2017](https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf)
