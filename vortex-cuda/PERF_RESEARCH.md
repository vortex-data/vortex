# OnPair CUDA decompress — algorithm research

Synthesis of the published GPU-decompression literature, NVIDIA docs, and
peer kernels, scoped to our `onpair_flat` bottleneck:
**~17 byte-granular `st.global.u8` per warp iter, LSU instruction queue
saturated, DRAM at 9 % utilisation, 142 GiB/s = 13.7 % HBM peak.**

Headline answer up front: every strong GPU string decompressor in the
public literature converges on the same trick — **stage the
variable-length output in shared memory, then drain to DRAM with
aligned wide stores**. GSST ([Vonk 2025][gsst], 191 GB/s on A100,
current Pareto front for GPU string decomp) explicitly identifies
"each byte written triggered a 32-byte cache-line transaction" as the
defining bottleneck and engineers around it with shared-memory staging
plus an aligned drain. The published [GPU-FSST source][fsst-gpu-src]
stages into a 2-D shared buffer indexed `[buf_words][THREAD_COUNT]`,
flushes word-aligned, then runs a separate
`transpose_no_bank_conflicts<32,8>` kernel to byte-pack the result —
i.e. the "strided intermediate + compaction" pattern executed in
shared memory rather than DRAM. Both lessons translate to us.

[gsst]: https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1
[fsst-gpu-src]: https://github.com/timanema/fsst-gpu/blob/main/src/compressors/compactionv5t/compaction-encode.cu

## 1. State of the art for GPU variable-length-output decompression

**nvCOMP (LZ4/Snappy/GDeflate/Bitcomp/ANS/Cascaded).** Closed source
since v2.3 ([docs][nvc-docs]). Public design ([NVIDIA blog][nvc-blog])
is consistent across formats: chunk the input, one CTA per chunk,
stage compressed input in shared mem, decode literals/back-refs in the
warp, drain. LZ4 quoted at ~60+ GB/s on string columns. Blackwell has
a fixed-function decompression engine for these formats at 600 GB/s
([FAQ][nvc-de]); Ampere is purely SM-resident.

**GSST — most relevant peer.** Same problem shape as us: random-length
symbol expansion to a byte-packed buffer. Their headline optimisation
(paper §4):

> "Introducing shared memory shifts the bottleneck to shared-memory
> stalls **and reveals misalignment issues, where each byte written
> triggered a 32-byte cache-line transaction**. Aligning memory
> accesses greatly increases memory and compute throughput."

The recipe: (1) each "split" decompresses into a shared-mem output
buffer; (2) before any global store, the output pointer is aligned to
an 8-B boundary; (3) the misaligned head is emitted as byte stores
**once per split** (not once per token); (4) the aligned body is
drained as wide aligned stores from shared mem. Their ablation
(paper fig 6) shows the alignment fix as the single biggest jump.

**GPU-FSST (Anema et al., ADMS/VLDB '25)** — open source. Their
compressor (same output shape as our decompressor) keeps a per-thread
column in `result[8][128]` shared mem, flushes one `uint32_t` per
thread per round to a layout that's coalesced (`dst[round*128 +
threadIdx.x]`), and then runs a separate transpose kernel to byte-pack
the column-major layout. 74 GB/s compression on RTX 4090.

**Sitaridi et al., [arXiv:1606.00519][sit]** — DEFLATE on GPU.
Relevant ideas: chunk-relative offsets + exclusive scan (same shape as
our `chunk_offsets`). The LZ-style "multi-round copy" trick doesn't
apply to dictionary decomp.

**UCCL-Zip** ([paper][uccl]) explicitly names the variable-length
output coalescing problem and notes the standard fix is a third
global-memory pass for compaction (which they then eliminate by
streaming directly into NCCL's FIFO — not applicable to us).

**DietGPU** ([repo][diet]) hits 250–600 GB/s on A100 but writes
fixed-width BF16/FP32 elements; no byte-packed output problem to
solve. Not directly applicable.

[nvc-docs]: https://docs.nvidia.com/cuda/nvcomp/
[nvc-blog]: https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/
[nvc-de]: https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html
[sit]: https://arxiv.org/pdf/1606.00519
[uccl]: https://arxiv.org/html/2604.17172v2
[diet]: https://github.com/facebookresearch/dietgpu

**Take-away.** Our exact problem (byte-packed, runtime-length ≤16 B
per element, no LZ back-refs) has been solved twice by peer kernels
at >130 GiB/s, both times by moving byte-granular work into shared
memory and only paying for aligned wide stores to DRAM. `onpair_flat`
does the byte stores directly against global memory — exactly what
GSST calls out as the worst case.

## 2. Known tricks for the LSU-issue ceiling

**Why byte stores to DRAM are so expensive.** Two compounding
effects: (a) **LG Throttle** ([NVIDIA forum][lg-throttle],
[GPUscout][gpuscout]) is a per-scheduler queue — many small stores
across many warps congest a shared resource. We sit at 38 %.
(b) Each unaligned byte store triggers a 32-B sector transaction in
L1 on CC 8.0. Our L1 ST sectors/request = 10.7 confirms this: every
1-byte write costs a full sector.

**Shared-mem staging + aligned drain — does it work in practice?**
Yes. GSST's ablation shows it as the largest single jump. Order of
magnitude estimate for us: ~17 stores/warp-iter today → ~2
stores/warp-iter (one aligned `st.global.v4.u32` per pair of lanes
plus head/tail) ⇒ 4–7× store-instruction reduction; mean output per
32-token chunk is 32 × 11 = 352 B ⇒ ~22 aligned u128 writes if drained
from shared memory, versus ~17 × N_warps today.

**Atomic / relaxed stores to bypass alignment.** Dead end. CUDA's
memory model ([guide][cuda-mm]) only guarantees atomicity for
naturally-aligned 1/2/4/8/16-B accesses. `atomicCAS` / `atomicOr`
synthesises byte writes via 32-bit RMW: strictly worse (RMW has
more instructions and serialises under contention). `red.global` is
just ATOMG without the return wire ([forum][red-atomg]). `.relaxed`
controls consistency, not addressability — `st.global.relaxed` is
still alignment-required.

[lg-throttle]: https://forums.developer.nvidia.com/t/long-scoreboard-stall-meanings/230738
[gpuscout]: https://www.ce.cit.tum.de/fileadmin/w00cgn/caps/vanecek/sv_gpuscout.pdf
[cuda-mm]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html
[red-atomg]: https://forums.developer.nvidia.com/t/difference-between-red-and-atomg-sass-instruction/203469

## 3. Candidate algorithms

### C1 — Warp shared-mem staging + 16 B drain (the GSST recipe)

```text
__shared__ uint8_t buf[CHUNK_OUT_MAX + 16];   // CHUNK_OUT_MAX ≈ 32*16=512 per warp
buf[excl .. excl+len] = token;                 // byte stores to shared mem
__syncwarp();
// Drain: warp emits aligned uint4 stores from buf to
// (output_bytes + chunk_offsets[c]). Lanes 0 / 31 handle ≤15-B
// head/tail with byte stores once per chunk.
```

- **Expected speedup:** 3–5× (→ ~500–700 GiB/s). Store-inst count
  drops from ~17/iter to ~2/iter. Eligible warps/sched should rise
  from 0.43 toward 4+.
- **Cost:** ~50 LOC patched into `onpair_flat.cu`.
- **Risks:** (a) chunk byte length = `chunk_offsets[c+1] -
  chunk_offsets[c]`, already known, so the drain is bounded.
  (b) per-chunk head misalignment ⇒ a few residual byte stores.
  (c) shared-mem footprint per warp ≈ 528 B; with 4 warps/block it's
  ~2 KB, far below the 192 KB unified L1/shared budget — occupancy
  unaffected.

### C2 — Per-chunk-pad to 16 B (output format change)

Pad each 32-token chunk's output region up to a 16-B multiple, so the
last lane's over-copy lands in the chunk's own pad bytes (disjoint
across chunks ⇒ no inter-warp race; the A2 failure mode disappears).

```text
chunk_padded_offsets[c+1] - chunk_padded_offsets[c]
    = align_up(sum_of_lens_in_chunk, 16)
all 32 lanes: aligned uint4 over-copy
```

- **Expected speedup:** 4–6× (one aligned 16-B store per lane, fully
  coalesced across the warp: 32 × 16 = 512 B = 4 sectors of 128 B).
- **Cost:** ~30 LOC.
- **Risks:** **changes the byte-packed output contract.** Mean blow-up
  ~352 → 368 B ≈ 4.5 %. Downstream would need a per-chunk offset
  remap or VarBinView with padded buffer offsets ([Arrow][arrow-vbv]
  supports this in principle, but it's a cross-team interface
  conversation).

### C3 — Stride-16 intermediate + compaction kernel

Pass 1 = current `onpair_padded_out.cu` (one aligned `uint4` per
token). Pass 2 reads the 815 MB strided buffer and produces 584 MB
packed, ideally using a C1-style shared-mem-drained kernel.

- **Expected speedup:** Pass 1 alone reaches the aligned-store
  ceiling (likely ≥500 GiB/s decoded-equivalent). Pass 2 reintroduces
  the byte-write problem to bytes already in HBM; if structured with
  C1, total ≈ 2 × DRAM traffic ÷ 1.555 TB/s ≈ 1.3 ms ⇒ ~450 GiB/s.
- **Cost:** ~80 LOC for pass 2.
- **Risks:** if pass 2 is implemented naively it rediscovers the
  142 GiB/s ceiling. Doubles HBM read traffic (write half is
  unchanged).

### C4 — Block-cooperative shared-mem drain

C1 scaled up: staging spans the whole CTA (4 warps × ~512 B ≈ 2 KiB),
drained by the full block after `__syncthreads()`. Drain step issues
32 aligned `uint4` stores per phase, perfectly coalesced (32 × 16 B =
one 4-sector L1 transaction). Requires CTA-level chunk-base offsets
(cheap host prefix-sum).

- **Expected speedup:** 5–8× (→ ~700–1000 GiB/s).
- **Cost:** ~80 LOC.
- **Risks:** `__syncthreads()` latency (mitigated by using
  `__syncwarp` for the staging phase, one block sync before the
  drain).

### C5 — Inverse-scan (output-driven gather), C6 — atomic byte writes

Both rejected. C5 needs per-output-chunk binary search through token
offsets, adds divergence, no peer kernel uses it. C6 is strictly
worse than current (see §2).

[arrow-vbv]: https://arrow.apache.org/docs/format/Columnar.html

## 4. Memory hierarchy considerations on A100 (CC 8.0)

- **Unified L1/shared, 192 KB/SM.** Per-CTA partition configurable;
  ample room for per-chunk staging.
- **`cp.async`** ([guide §4.11][cp-async]) is global→shared *loads*
  only. Doesn't help output. Could prefetch `dict_padded`
  asynchronously into shared, but L1 hit-rate is already 95 % — small
  expected gain.
- **TMA / `cp.async.bulk`** is Hopper-only (CC 9.0+). Not available
  on A100. Worth noting for a future port: TMA dramatically
  simplifies the C4 drain.
- **L2 residency control** (`cudaAccessPolicyWindow`) to pin
  `dict_padded` (64 KB) in L2 — small gain; L2 hit-rate already 98 %.
- **Cache hints**: annotate packed-output writes with `__stcs`
  (`st.global.cs`, single-use) — no reader-side reuse downstream.
  5–10 % expected from reduced L1 eviction pressure
  ([best-practices guide][bp-guide]).
- **Vectorised stores**: `uint4` (16 B) is Ampere's max store width.
  The drain *must* hit this width.

[cp-async]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
[bp-guide]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

## 4.5 Updated ranked next steps (post-C1 + WPB sweep)

After implementing C1 (`onpair_shmem`) + sweeping `WARPS_PER_BLOCK`, we
landed at **511 GiB/s = 33 % HBM peak**. Stuck at ~37 % "No Eligible
Warps" cycles, DRAM at 33 % peak. Workload is no longer
LSU-instruction-bound; the new bottleneck is some combination of
shared-mem RAW latency, dict-read scoreboard, and warp scheduler
overhead.

### Candidates not yet exhausted

| # | Idea | Why it might help | Risk |
|---|---|---|---|
| A13 | GPU-FSST column-major transpose drain. Each thread holds bytes for N chunks in `result[N][32]` shared; final transpose writes aligned bytes. | Eliminates the byte-granular shared-mem write (replaced by uint32 per chunk per thread). | Complex; bank conflicts on transpose. |
| A14 | Pre-compute per-TOKEN byte offsets host-side (in `compute_splits`). Kernel skips the warp scan entirely. | Removes 5 `__shfl_up_sync` per warp + the `__syncwarp` after shared writes. | 8 B per token × 51 M = 408 MB extra device traffic (kills the win unless we can fit in u32 — but 584 MB > 4 GB? Yes; could use u32 offsets if total_size < 4 GB). |
| A15 | `__ldcs` on dict_padded loads. Bypass L1 for dict (it's a streaming load pattern at the cache-line level since each lane hits a different line). | Frees L1 capacity for unaligned shared-mem reads. | Probably no win — dict L1 hit rate already 60-95 % depending on warp count. |
| A17 | Sort tokens by code on the host. After sort, all 32 lanes in a warp read 1-3 dict cache lines instead of ~32. | Kills the L1 sectors/load metric; might 2× the kernel throughput on the dict-read side. | Changes output row order. Requires a re-permutation pass downstream. Big API change. |
| A18 | 64-token warp, 2 tokens per thread, scan + drain larger amount of data per warp issue. | Better register-level latency hiding inside a single warp. | Failed once (A12 = 2 chunks per warp ran 20 % SLOWER). May work if implemented as "2 tokens per thread" instead of "2 chunks per warp" — less grid shrinkage. |
| A19 | Pure-register lane-shuffle drain. Each output 16-B chunk gathered via 16 `__shfl_sync` from various lanes' token registers, then written aligned. | No shared-mem step. Eliminates the RAW latency entirely. | 352 shfls per warp = high compute cost; binary-search-for-source-lane per byte adds divergence. |
| A20 | Larger dict cached in `__shared__`. Stage the full 64 KB padded dict at block startup. | All dict reads become shared-mem hits (1 cycle, no L1 pressure). | Eats 64 KB/block of shared; cuts blocks-per-SM from 8 to 2 (occupancy crash). Probably bad. |
| A21 | `cp.async` overlap of next-iter dict reads with current shared-mem drain. | Hides dict latency under the drain. | Requires multi-iter-per-warp restructure (failed once as A12). |
| A22 | Use `__nv_aligned_device_malloc(...)` for output, ensure 256-byte alignment to a sector. NVCC then knows the body's `+head` offset is sector-aligned. | Might unlock a wider store class. | Marginal. |
| A23 | Use `st.global.bypass` PTX hint to fully bypass cache. | All output writes go straight to HBM, no L1/L2 update. | Could be win if L1 contention is real. |
| A24 | Decompose into TWO kernels: 1) producer writes `(byte_count, uint4)` per token to a side-table; 2) compacting drain. | The drain reads aligned u128 input + writes packed output; less compute per token. | Adds an extra DRAM round trip (~+25 % traffic). |

### Recommended next attempts

1. **A14 (host-side per-token offsets)** — biggest expected win
   because it removes both the warp scan and the within-warp byte
   write race for the "destination position" calculation. Setup cost
   is 200 ms more host work; trivially amortized in production where
   the array is decompressed many times.
2. **A18 v2 (2 tokens per thread, NOT 2 chunks per warp)** — keeps grid
   size constant. Each thread reads two codes/lens/tokens and writes
   two outputs to shared, doubling the work per thread issue without
   changing warp count. Risk is still register pressure.
3. **A23 (st.global.bypass)** — one-line change, worth measuring.

## 5. Ranked recommendation

| # | Experiment | Expected gain | Cost | Confidence |
|---|---|---|---|---|
| **1** | **C1 — warp shared-mem staging + 16 B drain** | 3–5× (→ ~500–700 GiB/s) | ~50 LOC | High — GSST recipe |
| **2** | **C4 — block-cooperative shared-mem drain** | 5–8× (→ ~700–1000 GiB/s) | ~80 LOC | Med-high |
| **3** | **C3 — padded intermediate + C1-style compaction** | ~3× ceiling, ~450 GiB/s | ~80 LOC | Med |

**Why C1 first.** Single-kernel change. Tests directly whether GSST's
result reproduces on our workload. If it doesn't, we learn our
chunks are too small for shared-mem amortisation, and C4 is the
natural escalation. If C1 reaches ~500 GiB/s, we're DRAM-territory
and remaining levers are prefetch / cache hints — small relative
wins.

**Why not C2 first** (despite likely being highest absolute speedup):
it changes the output contract. VarBinView could in principle accept
padded buffers, but verifying that costs cross-team design. C1
chases the same ceiling without that conversation.

**What to measure after C1.** From ncu:

1. `LG Throttle` should drop from 38 % to <10 % of warp cycles.
2. `L1 ST sectors/request` should drop from 10.7 → ~4.0
   (32 × 16 = 512 B = 4 sectors).
3. Eligible warps/sched should rise from 0.43 toward 4–8.
4. DRAM utilisation should rise from 9 % toward 30–50 %.

If those metrics move and throughput rises >2×, the diagnosis is
confirmed and C4 is the next escalation. If throughput moves <1.5×
despite the metric shifts, we're in shared-mem-bank-conflict
territory — standard fix is 33-wide buffer padding
([NVIDIA blog][shared-mem]).

[shared-mem]: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

## Sources

- [GSST: Parallel string decompression at 191 GB/s on GPU (Vonk, Hoozemans, Al-Ars 2025)](https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1)
- [High Throughput GPU-Accelerated FSST String Compression (Anema et al. 2025)](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf)
- [fsst-gpu source — compaction-encode.cu](https://github.com/timanema/fsst-gpu/blob/main/src/compressors/compactionv5t/compaction-encode.cu)
- [Massively-Parallel Lossless Data Decompression (Sitaridi et al. 2016)](https://arxiv.org/pdf/1606.00519)
- [nvCOMP — decompression engine FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
- [Optimizing Data Transfer Using Lossless Compression with nvcomp](https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/)
- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels (NVIDIA)](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA Programming Guide §4.11 Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [Long scoreboard / LG Throttle stall semantics (NVIDIA forum)](https://forums.developer.nvidia.com/t/long-scoreboard-stall-meanings/230738)
- [GPUscout: Locating Data-Movement-related Bottlenecks on GPUs](https://www.ce.cit.tum.de/fileadmin/w00cgn/caps/vanecek/sv_gpuscout.pdf)
- [Difference between RED and ATOMG SASS instructions](https://forums.developer.nvidia.com/t/difference-between-red-and-atomg-sass-instruction/203469)
- [CUDA C++ Memory Model — atomicity of aligned loads/stores](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)
- [Apache Arrow Columnar Format — VarBinView spec](https://arrow.apache.org/docs/format/Columnar.html)
- [UCCL-Zip: Lossless Compression for GPU Communication](https://arxiv.org/html/2604.17172v2)
- [DietGPU repo](https://github.com/facebookresearch/dietgpu)
