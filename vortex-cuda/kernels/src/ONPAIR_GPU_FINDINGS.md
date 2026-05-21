# OnPair GPU decode — findings & experiment log

Working notes for the OnPair string-decompression CUDA kernels. Machine: single
**NVIDIA GH200 480GB** (Hopper, sm_90, 132 SMs, ~227 KB shared/SM with opt-in).
Bench binary: `onpair-chunk-bench gpu-decode-vortex`.

## Measurement methodology (important)

GPU clocks on this box are **not locked** (idle 345 MHz, boost to 1980 MHz;
locking via `nvidia-smi -lgc/-lmc` is blocked as a shared-infra change). The
bench warms up only 2 passes (~1.6 ms), far too short to ramp from idle.
Consequences:

- **Absolute `decode_ms` / GiB/s wander run-to-run** and are systematically
  optimistic in short (10-iter) runs that catch a transient boost spike.
- **Intra-run ranking is reliable**: within one process invocation every kernel
  runs back-to-back under the same drifting clock, so their *relative* order is
  stable. Always compare kernels from the *same* invocation, and use a high
  iteration count (>=100, ideally 300) so the timed window is dominated by
  steady-state clocks.
- **NCU section metrics are clock-robust**: SpeedOfLight %, MemoryWorkload,
  WarpState, Occupancy are ratios/counts normalised against achieved clock, so
  NCU is valid even without locked clocks (NCU also pins clocks to base by
  default).

## Corrected baseline: plain `4tpt` is the real winner

The prior handoff claimed `onpair_shmem_4tpt_wpb8_occ` (book-reviews, ps_comment)
and `onpair_shmem_4tpt_split8_wpb8_occ` (fineweb) as winners, beating plain
`onpair_shmem_4tpt`. **This does not reproduce.** Re-measured at 300 iters
(book-reviews) and 100 iters (fineweb), bits16, two runs each:

| kernel (bits16)            | book-reviews mean | fineweb (100it) |
|----------------------------|-------------------|-----------------|
| **onpair_shmem_4tpt**      | **0.877 ms**      | **2.000 ms**    |
| onpair_shmem_4tpt_split8   | 0.903 ms          | 2.057 ms        |
| onpair_shmem_4tpt_wpb8_occ | 0.905 ms          | 2.059 ms        |
| split8_wpb8_occ            | 0.913 ms          | 2.099 ms        |
| onpair_shmem_4tpt_wpb8     | 1.017 ms          | 2.316 ms        |
| onpair_shmem_4tpt_split8_wpb8 | 1.031 ms       | 2.365 ms        |

Plain `4tpt` is fastest on **both** datasets; every `wpb8`/`occ` variant is
slower. The `wpb8` (8 warps/block) variants cut occupancy on a kernel already
limited to ~50% theoretical occupancy, which hurts. **Conclusion: the
occ/wpb8/split8 line was chasing 10-iter measurement noise.** Treat plain
`4tpt` as the baseline going forward.

## NCU bottleneck (plain 4tpt, book-reviews bits16, base 1.53 GHz)

```
Memory Throughput (Mem Busy)   92.55 %
L1/TEX Cache Throughput        92.87 %   <-- limiter
DRAM Throughput                16.71 %   <-- NOT bandwidth bound
L2 Cache Throughput            66.71 %   (L2 hit 94.35 %)
Compute (SM) Throughput        34.27 %
Mem Pipes Busy                 32.47 %
L1/TEX Hit Rate                31.14 %
Warp Cycles / Issued Instr     19.74     (latency bound)
Avg Not-Predicated-Off / Warp  25.17 / 32 (~21% of byte-ladder stores wasted)
Theoretical Occupancy          50 % (register-limited; Block Limit Registers=2)
Achieved Occupancy             42.53 %
```

Interpretation: the kernel is **L1/TEX cache-request bound** (93%), not data-pipe
bound (Mem Pipes Busy only 32%) and not DRAM bound (17%). The dominant source is
the **uncoalesced 16-byte `uint4` gather into `dict_padded` per token** — the
dict is L2-resident (94% hit) but each random 16-byte gather still burns L1/TEX
request throughput. Secondary: register-limited occupancy and ~21% predicated-off
byte stores.

Two attack surfaces: (1) move the dict off the L1/TEX request path (into shared),
(2) lower per-token register footprint to lift occupancy.

## bits12 dictionaries fit in shared memory

| column        | bits12 dict_bytes | bits16 dict_bytes |
|---------------|-------------------|-------------------|
| fineweb/text  | 17,477 B          | 476,034 B         |
| book-reviews  | 19,844 B          | 526,906 B         |

So the **variable-length** bits12 dict (~17–20 KB) fits in the 48 KB default
shared carveout; the 16-byte **padded** bits12 dict (4096×16 = 64 KB) fits with
Hopper's opt-in larger carveout. bits16 dicts (~500 KB) do **not** fit — shared
dict is a bits12-only play. `dict_max_len > 8` on both columns, so the existing
`s8`/`s4` stride kernels are inapplicable.

## Prior art: "dict in shared" already tried (and why it failed)

From `onpair_shmem_tma.cu` history + `benches/onpair_real_data.rs`:

- **Cooperative load + `__syncthreads`**: regressed 22–33% on A100 and Hopper.
- **Per-thread `cp.async.cg`**: same regression, plus ILLEGAL_ADDRESS on
  max_len=16 columns.
- **TMA `cp.async.bulk` (`onpair_shmem_tma`, v3)**: gated behind
  `ONPAIR_ENABLE_TMA`; uses the 64 KB **padded** dict.

**Root cause of the regressions**: all kept the 1-block-per-1024-tokens launch
shape. A block loads the whole dict but emits only ~5 KB of output before
exiting, so the per-block load + barrier is never amortised. **Nobody tried the
variable-length packed dict, and nobody tried a persistent grid.**

## Experiments in progress

### Idea A — persistent grid + padded dict in shared (`onpair_shmem_4tpt_pdict`)

Launch a fixed persistent grid (~2 blocks/SM); each warp walks the chunk space
in a grid-stride loop. Dict cooperatively loaded into shared once per block
(single `__syncthreads`), reused across thousands of chunks → amortises the
load that killed prior attempts. Token expansion does aligned `uint4` reads from
shared (bypasses L1/TEX tag/sector pipeline). Padded 64 KB dict, opt-in shared.

*Status: implemented, validated (byte-exact vs CPU), NCU'd. **REGRESSED 27%***
(bits12 fineweb: 4tpt 1.83 ms vs pdict 2.33 ms). NCU diagnosis (bits12 fineweb):

| metric                     | 4tpt   | pdict      |
|----------------------------|--------|------------|
| shared-load bank conflicts | 115 K  | **82,000 K** (712×) |
| global-load sectors        | 549 M  | 21 M (gather gone) |
| global bytes/sector        | 31 %   | 92 %       |
| L1/TEX throughput          | 86 %   | 66 %       |
| achieved occupancy         | 46.9 % | **23.9 %** |
| duration                   | 2.30   | 2.87 ms    |

The global gather **was** eliminated (549M→21M sectors, 31%→92% coalescing) — the
hypothesis held — but two new costs swamped it: **82M shared bank conflicts** from
random 16-byte `uint4` reads, and **occupancy halved to 24%** because the 64 KB
padded dict allows only 1 block/SM. The padded layout is doubly wrong: too big
(occupancy) and 16-byte-aligned random reads conflict hard.

### Idea B — persistent grid + variable-length packed dict in shared (`onpair_shmem_4tpt_vdict`)

The packed ~17 KB dict in shared, (off,len) from global `dict_table`, byte-granular
shared->shared copy of `len` bytes/token, persistent grid.

*Status: implemented, validated, NCU'd. **REGRESSED 53%*** (bits12 fineweb:
4tpt 1.83 ms vs vdict 2.80 ms — worse than pdict). NCU (bits12 fineweb):

| metric              | 4tpt   | pdict      | vdict      |
|---------------------|--------|------------|------------|
| shared bank conflicts (ld) | 115 K | 82,000 K | 78,500 K |
| global-load sectors | 549 M  | 21 M       | 299 M (dict_table) |
| theoretical occ.    | 50 %   | 25 %       | **37.5 %** |
| achieved occ.       | 46.9 % | 23.9 %     | 24.9 %     |
| duration            | 2.30   | 2.87       | 3.39 ms    |

The smaller footprint **did** lift theoretical occupancy (25%→37.5%), confirming
the footprint intuition — but two costs dominate: (1) random byte-granular shared
reads still conflict ~78M times (a random gather conflicts regardless of element
width), and (2) the serial per-token byte-copy loop is slower than the uint4 path.

## Conclusion: dict-in-shared is a dead end for OnPair

Across padded (16 B uint4) and packed (byte) layouts, persistent-grid shared-dict
kernels regress 27–53%. Root cause is intrinsic: OnPair's dict access is a
**random gather** (32 lanes → 32 arbitrary dict entries per step). Shared memory
serialises that into bank conflicts (~700× baseline); L2 — where the dict already
lives at 94% hit — handles scattered access with hardware better suited to it.
Moving the dict to shared trades tolerable L1/TEX-request pressure for intolerable
shared-bank-conflict pressure. **The baseline `4tpt` is L1/TEX-saturated (86–93%)
on an essentially irreducible gather; DRAM is idle (17%).**

Remaining lever worth probing: **occupancy** (baseline is register-limited to 50%
theoretical). See Idea C.

### Idea C — lift occupancy via register reduction (rejected by analysis)

Baseline uses **exactly 64 registers/thread** (the cap for 2 blocks/SM at 512
threads). Reaching 3 blocks/SM (75%) needs <=42 regs — a 34% cut, infeasible
since 4tpt holds 4 tokens' state across the prefix-scan. Static shared (33 KB)
would also cap at 3 blocks. And at 86-93% L1/TEX the pipe is saturated, so more
occupancy can't help much. Lever closed; not implemented.

### Idea D — split-read dict (`onpair_shmem_4tpt_split8read`) — **WIN (+10%)**

Common-case token bytes read as `uint2` (8 B) from the 32 KB `dict_s8` array;
rare `len>8` high bytes from the 64 KB `dict_padded`. Same scan/drain as 4tpt,
standard grid, 64 regs.

*Status: implemented, validated, NCU'd. **10% faster*** (bits12 fineweb,
300 iters x2): 4tpt 1.833 ms vs split8read **1.650 ms**, reproducible to 0.001 ms.

| metric (bits12 fineweb)    | 4tpt   | split8read |
|----------------------------|--------|------------|
| L1/TEX Cache Throughput    | 86.3 % | **73.3 %** |
| global-load sectors        | 549 M  | 550 M (same) |
| sector hit rate            | 89.3 % | 91.0 %     |
| Compute (SM) Throughput    | 56.5 % | 58.2 %     |
| DRAM Throughput            | 17.2 % | 19.0 %     |
| Duration (NCU base clock)  | 2.30   | 2.08 ms    |

Mechanism: not fewer sectors (≈ same) and not really better hit rate — the win is
**less data through the L1/TEX data pipe**. Reading 8 B instead of 16 B for the
common short tokens (mean len ~6) halves the dict bytes the saturated L1/TEX pipe
must move, dropping its utilisation 86%→73% and freeing it to retire faster. This
is the first idea to actually *reduce* the bottleneck work rather than relocate
it.

## Full results: split8read vs 4tpt across all columns (GPU decode, 100 iters)

| dataset/bits        | decoded MB | 4tpt GiB/s | split8read GiB/s | speedup |
|---------------------|-----------:|-----------:|-----------------:|--------:|
| fineweb/bits12      | 1000.0     | 511.7      | 567.3            | **1.11x** |
| fineweb/bits16      | 1000.0     | 470.1      | 433.6            | 0.92x   |
| book-reviews/bits12 | 522.3      | 581.8      | 606.6            | **1.04x** |
| book-reviews/bits16 | 522.3      | 561.8      | 492.6            | 0.88x   |
| ps_comment/bits12   | 987.9      | 1117.3     | 1101.4           | 0.99x   |
| ps_comment/bits16   | 987.9      | 866.2      | 638.8            | 0.74x   |
| wikipedia/bits12    | 703.1      | 491.8      | 537.9            | **1.09x** |
| wikipedia/bits16    | 703.1      | 537.8      | 510.3            | 0.95x   |

All validate byte-exact. **split8read is a win on bits12 (≤4096-entry dict →
`dict_s8` is 32 KB, fits L1) and a loss on bits16** (65536-entry dict → `dict_s8`
is 512 KB, no L1 benefit, and the extra high-byte reads for `len>8` tokens cost
more — worst on ps_comment/bits16 at 0.74x, which has the most long tokens).
ps_comment/bits12 ties: at 1100 GiB/s it is already not L1-bound (short,
highly-compressible tokens), so there is no L1 pressure to relieve.

**Recommendation: select split8read only when the dict is small** (bits12 / dict
entries <= ~4096 so `dict_s8` <= 32 KB); fall back to plain `4tpt` for bits16.

## Chunk-size effects

GPU decode, fineweb bits12 (split8read vs 4tpt), per chunk_bytes:

| chunk   | n_chunks | 4tpt ms | split8read ms |
|---------|---------:|--------:|--------------:|
| 10 MB   | 96       | 2.614   | 2.427         |
| 100 MB  | 10       | 1.902   | 1.723         |
| 1000 MB | 1        | 1.821   | 1.640         |

Bigger chunks decode faster (fewer, larger kernel launches; 10 MB pays ~40% more
overhead). split8read keeps its bits12 edge at every chunk size.

Compression vs chunk size (one OnPair dict per chunk):

| dataset/bits     | 10 MB | 100 MB | 1000 MB | dict@1000MB |
|------------------|------:|-------:|--------:|------------:|
| fineweb/bits12   | 2.238 | 2.257  | 2.264   | 17 KB       |
| wikipedia/bits12 | 2.168 | 2.158  | 2.166   | 17 KB       |
| wikipedia/bits16 | 2.345 | 2.745  | 2.815   | 456 KB      |
| l_comment/bits12 | 4.07  | 4.18   | 4.16    | (1MB: 25 MB)|

**Dict saturation governs the chunk-size/compression tradeoff.** bits12 dicts
saturate at 4096 entries (~17 KB) within ~10 MB of text, so per-chunk dict size is
constant and compression ratio is flat across chunk sizes. bits16 (and
high-cardinality bits12 like l_comment) do NOT saturate, so smaller chunks
replicate a large dict many times: wikipedia/bits16 dict balloons 456 KB → 26 MB
going 1000 MB → 10 MB and ratio drops 2.815 → 2.345. **For bits16 prefer large
chunks for compression; for bits12 chunk size is free to pick for decode
parallelism.**

## Datasets

Added `wikipedia` (wikimedia/wikipedia, en 2023-11-01; `text`/`title`/`url`) to
`benchmarks/onpair-bench/columns.py`, mirroring fineweb. One ~420 MB en parquet
shard is fetched on first use.
