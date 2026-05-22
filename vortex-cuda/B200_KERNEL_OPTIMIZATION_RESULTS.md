# B200 OnPair decode kernel optimization — results (PRELIMINARY)

Applying the experiment plan in `B200_VS_GH200_ONPAIR_ANALYSIS.md` one track at
a time. **PRELIMINARY**: unlocked clocks (±~5%), single-invocation kernel ranking, NCU
blocked (`ERR_NVGPUCTRPERM`), clock-locking blocked. All variants validated **byte-exact**.

## Reproduce the evidence

`python3 vortex-cuda/onpair_b200_evidence.py` runs five controlled comparisons (one launch per
column, all kernels timed) and prints labeled tables, each isolating a single variable so the
mechanism is self-evident. Needs the CUDA bench built and the benchmark data generated. The five
demos and what they show on B200:

1. **Granularity, not occupancy** — 256→128-thread = +4%, 128t @50%→75% occ = +1% (noise),
   64t ties 128t (plateau). Block size moves it; target occupancy does not.
2. **8 B is the optimal dict read width** — 16 B→8 B = +26%, 8 B→4 B = −4%. Gather is
   transaction/MSHR-bound, not request-width-bound below the 32 B sector.
3. **bits16 wall** — `b128o12` 637, +L2-persist 637 (−0%, dict already L2-resident),
   `cluster_dsmem` 131 (−79%). The limiter is L2 latency hidden by high occupancy, not bandwidth.
4. **The `frac_le8` gate** — split8read Δ vs `b128o12` is monotonic in `frac_le8`
   (+26/+23/+5/−3/−4% for fineweb/wiki/URL/l_comment/ps_comment); 0.70 sits between URL (0.81,
   win) and l_comment (0.58, lose).
5. **Tiny columns are launch-bound** — dbtext (~0.6 MB) decodes in a near-constant ~15 µs at
   50 GiB/s vs fineweb's 637 GiB/s on the same kernel; GiB/s is only meaningful above ~tens of MB.

## Outcome: arch-aware `pick_auto_kernel`, B200 +6 to +46% over the old `4tpt` default

`pick_auto_kernel(chunks, cc_major)` now branches on compute capability so each GPU keeps
its own best kernel — **GH200 (sm_90) is unchanged**:

| arch | general case |
|---|---|
| sm_90 (Hopper/GH200) | `split8read` if small bits12 dict & `frac_le8≥0.90`, else `4tpt` (unchanged) |
| sm_100 (Blackwell/B200) | `split8read_b128o12` if small bits12 dict & `frac_le8≥0.90`, else `b128o12` |

### B200 decode: old shipped `4tpt` default → new auto-selected (GiB/s, byte-exact)

| dataset/col | bits | old `4tpt` | new auto kernel | new GiB/s | gain |
|---|---:|---:|---|---:|---:|
| fineweb/text | 12 | 550 | `split8read_b128o12` | 802 | **+46%** |
| wikipedia/text | 12 | 510 | `split8read_b128o12` | 747 | **+46%** |
| clickbench/URL | 12 | 717 | `split8read_b128o12` | 843 | **+18%** |
| tpch/l_comment | 12 | 1012 | `b128o12` | 1097 | +8% |
| tpch/ps_comment | 12 | 1174 | `b128o12` | 1259 | +7% |
| fineweb/text | 16 | 585 | `b128o12` | 638 | +9% |
| wikipedia/text | 16 | 573 | `b128o12` | 621 | +8% |
| tpch/l_comment | 16 | 849 | `b128o12` | 932 | +10% |
| tpch/ps_comment | 16 | 957 | `b128o12` | 1042 | +9% |
| clickbench/URL | 16 | 903 | `b128o12` | 957 | +6% |

## New kernels added (all `#define` launch-config variants of existing bodies)

| kernel | config | role |
|---|---|---|
| `4tpt_b128o12` | 128-thread blocks, `__launch_bounds__(128,12)` → 40 regs, **75% occ, no spill** | B200 general default |
| `4tpt_split8read_b128o12` | split8read body + same `(128,12)` bounds | B200 high-`frac_le8` bits12 default |
| `4tpt_b128`, `_o6`, `_b512o3`, `_b64`, `_b64o24`, `_split8read_occ` | granularity/occupancy sweep points | evidence |

## Track-by-track verdict

- **A (arch-aware selector):** shipped. The single highest-value, lowest-risk change. The
  sm_100 `split8read` gate is `small_dict && frac_le8 ≥ 0.70` (vs Hopper's 0.90). A B200
  per-column sweep of `frac_le8` vs the measured `split8read_b128o12` − `b128o12` delta showed
  the win extends down to clickbench/URL (`frac_le8` 0.81, **+5.5%**) while l_comment (0.58,
  −2.6%) and ps_comment (0.33, −4.8%) still regress, so 0.70 sits centered between the win and
  regression bands. bits16 is excluded automatically — its dicts have 65 k entries, failing
  `small_dict` (≤4096) regardless of `frac_le8`. Hopper left at 0.90 (no GH200 access to
  re-measure). Selector inputs (`frac_le8`, `dict_mean_len`, `dict_max_len`, `dict_entries_max`,
  `small_dict`) are now surfaced in the `gpu-decode-vortex` JSON for future gate tuning.
- **B/B″ (occupancy + block granularity):** decomposing the evidence-kernel sweep, the win
  is almost entirely **block granularity, not occupancy**: 256→128-thread at fixed 50% occ
  = +1–4%, but 50%→75% occ at fixed 128-thread = ~0% (noise). **Granularity plateaus at 128
  threads** (64-thread `b64` tied). So `b128o12`'s forced 75% occ is harmless but
  unnecessary — plain `b128` (56 regs, 50% occ) is equally fast and could be the default.
  See the decomposition table in `SESSION_HANDOFF_B200.md` §2.
- **B′ (register-lean scan-then-regather):** **subsumed** — forcing 75% occ via launch
  bounds already gets there with no spill and no algorithm change.
- **B+ (split8read + granularity):** the **standout**. `split8read`'s 8-byte (`uint2`)
  reads halve L1/TEX request width, but this only pays off at **128-thread granularity**:
  `split8read_occ`→`split8read_b128o12` (256→128-thread) goes **+5% → +26%** — granularity
  ~5×'s the win — while the 75% occ in `b128o12` contributes ~nothing. Net **+23–26%** on
  fineweb/wikipedia bits12 over `b128o12` (and +5.5% clickbench). This reverses the earlier
  "split8read is dead on Blackwell" conclusion, which only tested it at 256-thread — exactly
  the granularity where its win is suppressed.
- **split4read (4-byte reads) — tested at 128-thread, worse than split8read.** Natural next
  step of the request-narrowing mechanism: 4 B (`uint`) reads from a 16 KB `dict_s4` (half of
  split8read's 32 KB), aimed at the shortest-token columns (fineweb mean 4.2 B, 77% ≤4 B). The
  granularity rescue reproduces (`split4read` 512→128-thread: fineweb 522→615, +18%), but
  `split4read_b128o12` **loses to `split8read_b128o12` on every bits12 column** (fineweb −23%,
  wikipedia −20%, URL −8%, l_comment −19%, ps_comment −18%). **8 bytes is the optimal read
  width on B200:** the split8read win came from halving the forced 16 B-aligned padded-dict
  read to 8 B; going to 4 B is already below the 32 B sector, so it cuts no transactions and
  just adds the >4 B fallback cost. Kept as an evidence kernel (anchors the read-width sweep);
  not in the selector. This confirms the gather is **transaction/MSHR-bound, not
  request-width-bound below the sector** — consistent with the bits16 L2-persist no-op.
- **Tokens-per-thread amortization (8tpt):** **tested, 4tpt is the peak.** The sweep was
  2tpt < 4tpt; 8tpt (256 tokens/warp, halves the head/tail epilogue, 8 in-flight loads/thread)
  is the natural next step. New `8tpt` + `8tpt_b128` kernels are **byte-exact but slower
  everywhere**: bits12 −22% vs `split8read_b128o12` (−4% vs `b128o12`), bits16 −2 to −5% vs
  `b128o12`. 8tpt doubles register pressure (8 `uint4`+`uint32` live), cutting occupancy /
  forcing spills — costs more than the epilogue it saves. On bits16 the extra per-thread MLP
  does **not** help, confirming the kernel is occupancy-bound, not per-thread-ILP-bound.
  (Nuance: `8tpt_b128` > `8tpt` on bits16, so 128-thread granularity still helps — consistent
  with the granularity thesis.) Evidence kernels; not in the selector.
- **C (NCU profile):** **blocked** — no `CAP_SYS_ADMIN` in-container; clock-locking also
  blocked. The limiter is inferred from the granularity sweep (128-thread blocks win;
  occupancy does not), not directly measured. NCU would confirm whether the lever is
  scheduling/drain evenness across SMs vs. latency, and re-derive the launch bounds.
- **D (freq-ordered codes):** **dropped** — freq-ordering interferes with other parts of
  the pipeline (per project constraint). Was +4.6–7.5% on bits16; not pursued.
- **E (hot-dict shared cache):** **dead** — caching codes 0..N assumes freq-ordering; without
  it the hit rate is ~0 and it regresses −6 to −8%. Removed.
- **B‴ (persistent grid):** deprioritized — big columns already fill the device many waves,
  so launch/quantization tail is already amortized; would only help tiny columns.
- **Length-bucket dict @ 128-thread (bits16 L1-residency):** **tested, not shipped.** bits16 is the
  slow band (text ~635 GiB/s) and is walled by its 65 k-entry / ~1 MB dict not fitting L1;
  split8read can't help (dict too big for the 32 KB `dict_s8`). The freq-order-*independent*
  fix is the length-bucket layout (`ONPAIR_DICT_REORDER=lenbucket`): pack entries at per-width
  stride {4,8,12,16}, shrinking the working set 2–3×. The existing `lenbucket` kernel was only
  ever measured at 512-thread (−9.5% vs `b128o12`) — the same granularity trap that hid
  split8read's win. A new `lenbucket_b128` (128-thread) **confirms the granularity rescue**:
  512→128-thread is **578→646 GiB/s (+12%)** on fineweb bits16. But it only *ties* `b128o12`
  on high-`frac_le8` text (+0.6–0.9%, **within ±5% noise**) and regresses on long-token columns
  (l_comment −5.4%, ps_comment −6.1%, URL −1.9%) where the bucket-branch divergence outweighs
  the smaller working set. Kept as an evidence kernel; **not** in `pick_auto_kernel`.
- **L2 persistence for the bits16 dict (`ONPAIR_L2_PERSIST`):** **tested, no effect.** Pinning
  the ~1 MB bits16 dict in an L2 persisting access-policy window (`apply_l2_persist`) leaves
  decode unchanged on every bits16 column (fineweb 637→637, wikipedia 621→621, l_comment
  932→933 GiB/s). The identical numbers are the finding: B200's multi-MB L2 already keeps the
  1 MB dict resident during a single-column decode, so pinning buys nothing. **The bits16
  limiter is L1/TEX gather latency** (1 MB dict ≫ ~256 KB L1, random 16 B gathers), not L2
  eviction — neither L2-persist nor the lenbucket working-set shrink (net of divergence) moves
  it. bits16 is at the kernel/layout/cache ceiling for these columns; the only remaining lever
  (hot codes resident in L1) needs freq-ordering, which is forbidden (see Track E).
- **Cluster-DSMEM for the bits16 dict (Track G):** **tested, −75 to −80% — a clear dead end.**
  New `cluster_dsmem` kernel: a thread-block cluster of 8 co-resident blocks shards the ~1 MB
  padded dict across their shared memory (each holds 1/8), and per-token reads come from the
  owning block via `map_shared_rank` instead of L2. Output is **byte-exact** (the sharding +
  trailing `cluster.sync()` to keep all slices alive are correct), but throughput collapses:
  fineweb 637→131, wikipedia 621→128, l_comment 933→222, ps_comment 1038→259, URL 952→193 GiB/s.
  Two causes, both predicted: (1) the 128 KB dict slice forces ~1 block/SM → far too few warps
  to hide latency; (2) ~7/8 of reads are *remote*, so a random all-to-all DSMEM gather saturates
  the GPC SM-to-SM fabric (plus `cluster.sync` cost). **This is the decisive bits16 finding: the
  limiter is L2 latency hidden by high occupancy, not gather bandwidth** — trading occupancy for
  on-chip dict staging loses ~5×. `b128o12` at full occupancy is near-optimal. (Tested at 8
  warps/block; more warps recover some but cannot close a 5× gap — the remote-fabric gather is
  fundamental.) Evidence kernel; **not** in the selector. Three independent levers (working-set
  shrink, L2 pin, DSMEM stage) now all confirm bits16 is walled.

## Dict bit-width sweep: bits14 is the L1-residency sweet spot

The bits12-vs-bits16 gap is a decode-speed/compression-ratio tradeoff set by dict size. Sweeping
the OnPair code budget on fineweb (`run --bits 12,14,16`, all kernels timed):

| fineweb | dict entries | `dict_s8` | fits L1? | ratio | best kernel | decode GiB/s |
|---|---:|---:|:--:|---:|---|---:|
| bits12 | 4096 | 32 KB | yes | 1.7× | `split8read_b128o12` | 802 |
| **bits14** | **16384** | **128 KB** | **yes** | **2.3×** | **`split8read_b128o12`** | **664** |
| bits16 | 65536 | 512 KB | no | 2.9× | `b128o12` | 637 |

**`split8read` wins exactly while `dict_s8` (entries × 8 B) fits the ~256 KB L1 — and the win
decays as it approaches that size.** Full fineweb bit-width sweep (decode = best kernel GiB/s):

| bits | entries | `dict_s8` | ratio | decode | split8 Δ vs b128o12 |
|---:|---:|---:|---:|---:|---:|
| 10 | 1024 | 8 KB | 1.2× | 674 | (small dict) |
| 11 | 2048 | 16 KB | 1.4× | 759 | +33% |
| 12 | 4096 | 32 KB | 1.7× | **802** | +25% |
| 14 | 16384 | 128 KB | 2.3× | 678 | **+9%** |
| 15 | 32768 | 256 KB | 2.6× | 637 | +3% (noise) |
| 16 | 65536 | 512 KB | 2.9× | 637 | tie |

Two findings from the sweep: (1) **decode speed peaks at bits12** (smallest L1-resident dict,
802) and **falls as the dict grows, flattening at ~637 from bits15→bits16**; (2) **bits10/11 are
strictly dominated by bits12** (worse ratio, no speed gain — the dict is already L1-resident at
12). bits14 (128 KB) is the only intermediate sweet spot: +9% split8read at 2.3× ratio. bits14 is
the middle ground: ~2.3× ratio at L1-resident split8read speed.

**Half-filled 16-bit dict (bits15) does NOT help decode.** Capping a 16-bit space at 50%
(32768 entries, 256 KB `dict_s8`) decodes at **637 GiB/s — identical to the full 512 KB bits16
dict (637)**; halving the dict bought zero L2/decode improvement, only ratio loss (2.9→2.6×). At
256 KB the `dict_s8` exactly *fills* L1, leaving no room for the streaming codes/output, so
split8read's benefit collapses to +3% (noise). The cache win requires `dict_s8` *comfortably*
under L1 (≤128 KB / bits14). A richer "train bits16, keep top-32768, re-encode rare tokens via the
hot dict" scheme (a C++ trainer change) would land a 256 KB dict on the same flat part of the
curve, so it would not speed decode either — confirmed by the bits15 datapoint. **Selector change (sm_100 only):** the split8read dict-size gate was raised from
`entries ≤ 4096` (bits12) to **`entries ≤ 16384` (bits14)**; combined with the unchanged
`frac_le8 ≥ 0.70` short-token gate, this auto-selects split8read for bits14 high-`frac_le8` text
(fineweb +9%) while leaving bits16 and low-`frac_le8` columns (URL bits14 `frac` 0.66, where
split8 is only +1.5%) on `b128o12`. Hopper gate unchanged (`≤ 4096`). Verified: fineweb bits14
now auto-selects `split8read_b128o12`. Reproduce with evidence-script Demo 7 (`run --bits
12,14,16`). Whole-decompress note: bits16 still wins end-to-end (better ratio, transfer-bound),
so bits14 is the pick when on-device decode speed matters more than ratio.

## Why bits16 is walled: near-uniform dict access (the root cause)

bits16 decodes slower than bits12 on the *same* column (fineweb 637 vs 802) purely because the
65536-entry dict (~1 MB) overflows the ~256 KB L1 while the bits12 dict (4096 entries, 32–64 KB)
fits. The decisive question is whether the access is *concentrated* (a hot subset could be cached)
or *uniform* (the whole dict is needed). Measured (`distinct_codes`, `access_top4096_frac`):

| column | bits | distinct used | top-4096 coverage |
|---|---:|---:|---:|
| fineweb/text | 16 | 65462 / 65536 | **0.47** |
| wikipedia/text | 16 | 65386 / 65536 | 0.47 |
| tpch/l_comment | 16 | 65202 / 65437 | 0.43 |
| tpch/ps_comment | 16 | 65162 / 65390 | 0.51 |

**Access is near-uniform** — ~99.9% of entries are used and the 4096 hottest cover only ~47% of
accesses. So there is **no hot subset to cache**; the full 1 MB working set is genuinely needed.
This is the root cause behind every failed bits16 lever (hot-cache, lenbucket, L2-persist,
cluster-DSMEM, variable-width): there is no locality to exploit, and the working set can't be made
to fit L1.

### Variable-width dict (avoid the 16 B padding) — tested, doesn't beat `b128o12` on bits16

The padded dict is fixed 16 B/entry; mean entry is ~7 B, so >half is padding. Two un-padded
layouts, both **byte-exact**:
- **Exact 1..16** (`vwidth`: compact `offset:24|len:8` directory + packed bytes): **−69 to −79%.**
  Exact packing puts entries at arbitrary byte offsets → the 16 B load must be unaligned
  (`memcpy`), which is catastrophically slow. Exact packing is a non-starter for vector loads.
- **Quantized {4,8,12,16}, 4-byte-aligned** (`vwidth4`: entries padded to a multiple of 4 at
  aligned offsets, directory `offset<<5|len`, load `ceil(len/4)` aligned `uint32`s): alignment
  recovers most of the loss (fineweb 164→579) and it **beats `b128o12` by +6% on bits12** (where
  its 32 KB bytes + 16 KB directory both fit L1). But on **bits16 it loses −8 to −19%**: the
  ~768 KB working set (256 KB directory + ~512 KB bytes) still exceeds L1, and it adds a second
  L2-missing gather (directory) plus variable-load branches vs `b128o12`'s single aligned 16 B
  read. Evidence kernels (`vwidth`, `vwidth4` + `_b128`); not in the selector.

Takeaway: shrinking the dict footprint helps **only when it brings the working set under L1**
(bits12). For bits16 the access is uniform over a working set that can't fit L1, so no layout
change helps — and bits16's *worse decode is irrelevant end-to-end* (next section): it compresses
better, and whole-decompress is transfer-bound.

## Ablation NCU-proxy: the EMIT is ~70% of runtime (the real ceiling)

NCU is blocked, so the limiter was inferred by **ablation kernels** (`onpair_shmem_4tpt_ablate*`,
timing-only, not byte-exact): the full 4tpt/b128o12 decode minus one stage. Speedup when a stage
is removed = that stage's cost share. fineweb (GiB/s):

| stage removed | bits14 | bits16 |
|---|---:|---:|
| full decode (baseline) | 612 | 635 |
| − dict gather | 1010 (+65%) | 1089 (+71%) |
| **− emit (byte-staging to shared)** | **2060 (+237%)** | **2353 (+270%)** |
| − output drain | 668 (+9%) | 688 (+8%) |
| − warp prefix-scan | 813 (+33%) | 774 (+22%) |

**The emit dominates (~70% of runtime; removing it = 3.4× faster).** Gather is second (~40%),
scan ~20–25%, drain only ~8%. This is the key finding: all the dict-side tuning (split8read,
bits14, dict layouts) optimized the *second*-biggest cost; the **byte-ladder emit** (`for j<16:
s_buf[base+j]=tok[j]`, ~`len`≈7 scattered shared byte-stores/token, heavy bank conflicts across
the warp) is the real ceiling, with **1.6–3.4× headroom**. It also reconciles the `shdict8`
result: moving the dict to shared lost because the gather (~40%) was never the bottleneck.

**Why the emit is slow — conflict-free proxy:** an emit that writes the same `len` bytes/token
but to conflict-free addresses (each lane → distinct bank) recovers only **3%** of the emit cost
(fineweb bits14: full 614, conflict-free 651, no-emit 2008). So the emit is **store-count /
throughput bound, not bank-conflict bound** — the cost is the sheer number of byte-store
instructions (~`len`≈7/token × 128 = ~900 shared stores/warp). The fix must **cut store count**,
which rules out a swizzle layout and points at a **shuffle-based emit**: assemble aligned 16-byte
output chunks in registers via `__shfl` (off the LSU), then write ~`warp_total/16` *coalesced*
`uint4` stores — ~25× fewer store instructions. Justified by the ablation (emit 70%,
store-count-bound); the targeted next build. Ablation kernels (`_ablate`, `_ablate_no*`,
`_ablate_cfree`) are timing-only proxies, not in the selector.

## Attacking the L1/TEX-request limiter (GH200 NCU: 93% L1/TEX-request-bound)

The one hard profile we have (GH200 NCU, fineweb bits16) says decode is **L1/TEX cache-request
throughput bound** (93%), not DRAM (17%) or compute (34%), with the dict only 31% L1-resident.
Two request sources per token: the dict gather (~1) and the byte-ladder emit into shared (~`len`
≈ 7). Two attacks tried:

- **dict-in-shared (`shdict8`, persistent grid):** stage the 8 B/entry `dict_s8` in shared so the
  common-case dict read bypasses the L1 tag/sector pipeline (tail >8 B from global). **Byte-exact
  but −47 to −65%** (fineweb bits12 340 vs split8read 801; bits14 234 vs 678; bits16 inapplicable —
  512 KB `dict_s8` exceeds shared). The cooperative-load + `__syncthreads` + occupancy hit from
  large shared swamp any benefit (matches the older `pdict` loss). **Useful signal:** moving the
  dict *off* the L1/TEX path made decode *slower*, so the dict gather is **not** the bottleneck —
  its L1 reads are already cheap. By elimination, the **emit byte-staging is the dominant L1/TEX
  request consumer.**
- **shuffle-based emit (move the ~7 shared stores/token off the LSU via warp `__shfl`):** the
  logically-indicated lever, but a variable-length cross-lane byte scatter is a substantial,
  high-risk rewrite, and the redistribution may just move the bottleneck to the shuffle unit. Not
  yet attempted — recommend confirming with **NCU on B200** (blocked: needs `--cap-add SYS_ADMIN`)
  before investing, since the last several "reduce-requests" guesses (dict-in-shared, vwidth,
  cluster-DSMEM) all lost. `split8read_b128o12` (802 GiB/s bits12) sits at ~1/3 of the HBM
  bandwidth ceiling, request/latency-bound — likely near the practical ceiling for a per-token
  random-gather + variable-length-emit decode absent the emit rewrite.

## Whole-decompress (end-to-end) vs micro (kernel-only)

Everything above is **micro**: `time_kernel_variant` times only the kernel launches; the
compressed payload is staged H2D once in setup, before timing. The **whole-decompress** path
(copy the compressed column H2D, then decode) is a different regime, now measured
(`compressed_bytes`, `h2d_gib_s`, `whole_decompress_gib_s` in the JSON; `decoded_bytes /
(compressed/h2d + decode)` as an output rate):

| column | bits | ratio | decode GiB/s | h2d GiB/s (pageable) | whole GiB/s | whole/h2d |
|---|---:|---:|---:|---:|---:|---:|
| fineweb/text | 12 | 1.7× | 801 | 11.8 | 19 | 1.6× |
| fineweb/text | 16 | 2.9× | 637 | 12.2 | 34 | 2.7× |
| clickbench/URL | 12 | 2.3× | 845 | 10.9 | 25 | 2.3× |
| tpch/l_comment | 12 | 3.9× | 1100 | 9.2 | 35 | 3.8× |
| tpch/l_comment | 16 | 5.2× | 931 | 10.6 | 52 | 4.9× |
| tpch/ps_comment | 16 | 6.3× | 1040 | 11.0 | 66 | 5.9× |

**Whole-decompress is transfer-bound:** H2D (9–12 GiB/s pageable) is 60–100× slower than decode
(637–1100 GiB/s), so end-to-end time is dominated by the copy and `whole/h2d ≈ compression
ratio`. **Consequence: the decode-kernel wins above (+18–46%) are *not* the end-to-end
bottleneck for an H2D-then-decode pipeline — the compression ratio is.** They matter for
**on-device decode** (GPU query/scan where the compressed column already resides on the GPU, so
no H2D), which is the regime the kernel work targets. (h2d here is *pageable*; pinned host memory
is ~3–5× faster, but decode is still 15–100× faster than that — the transfer-bound conclusion
holds. Numbers PRELIMINARY.)

## Why dbtext / tiny columns look "slow" (not a dict-decode bug)

dbtext columns are 0.4–2 MiB and decode in a **near-constant ~17 µs** regardless of size —
fixed launch + grid-ramp overhead on a grid far too small to fill ~148 SMs. Same kernels
hit 1090+ GiB/s on l_comment. The GiB/s figure is meaningless below ~tens of MiB. (book-
reviews is not on the B200 — no public source; its GH200 607 GiB/s shows it's a fast large
column.) Only fix would be batching many small columns per launch — a harness change.

## Infra

Added `ONPAIR_FAST=1` env to `gpu-decode-vortex`: skips the slow reference kernel + bundled
nvCOMP runs, cutting a big-column kernel sweep from ~10 min to ~12 s. Kernel-only ranking
within one invocation (clock-noise-robust given unlocked clocks).
