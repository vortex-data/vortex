# Why B200 OnPair decode is only modestly faster than GH200 — analysis

**Question:** We ported OnPair GPU decode to a B200 (Blackwell, sm_100); the
numbers are not much better than the GH200 (Hopper, sm_90) baseline. Is it compile
flags, the launch/calling path, or the kernels?

**Short answer (revised against the measured B200 data in
`benchmarks/onpair-bench/B200_PRELIMINARY.md` / `b200_results.csv`):**

1. **Flags and the calling path are fine** — verified sm_100 PTX, CUDA 12.8, freshly
   built; flat grid that fills the device. Not the cause.
2. **"Barely faster" is real only for bits12 (+6–12%)** — and that *is* expected:
   the bits12 dict (~17 KB) is already fully cache-resident on GH200, so the kernel
   is L1/TEX-request + SM-throughput bound, and B200 adds only ~1.1–1.25× SMs at the
   same clock. That is the ceiling.
3. **bits16 is meaningfully faster (+14–34%)** — because B200's bigger L1/L2 relieves
   the ~500 KB-dict cache thrashing that crippled GH200 bits16. So the cache upgrade
   *does* help, but only where GH200 was cache-starved.
4. **The biggest, free win is a mis-tuned selector.** On B200 the optimal kernel
   flipped to `4tpt_wpb8_occ` (which *regressed* on GH200), but the shipped
   `pick_auto_kernel` still selects `4tpt`/`split8read` and never even considers it.
   That leaves **+6–11% on the table on every large column** — a big part of why
   B200 "looks not much better."

So: it's the **kernel selection + an arch-dependent occupancy/launch-config flip**,
not the flags or the call path. Details and evidence below.

---

## 1. Ruling out compile flags

`vortex-cuda/build.rs`: `nvcc -O3 -std=c++20 -arch=native --restrict --ptx ...`

Verified on this B200 box:

| Check | Result |
|---|---|
| Device | `NVIDIA B200, compute_cap 10.0, 183 GB` |
| Toolkit | `nvcc release 12.8` — supports `sm_100` |
| `-arch=native` here | `compute_100` (Blackwell) |
| Built PTX (`kernels/gen/release/onpair_shmem_4tpt.ptx`) | `.target sm_100`, `.version 8.7` |
| PTX vs `.cu` mtime | PTX `12:30` newer than `.cu` `10:58` — **rebuilt on the B200** |

The kernels are genuinely generated for Blackwell, not running stale `compute_90`
PTX. Shipping `--ptx` (JIT to SASS at runtime) costs a one-time cached JIT, not
steady-state throughput; the SASS is the same `sm_100` ptxas would emit offline.
No fast-math angle — decode is integer + byte-copy only.

**Verdict: not the flags.**

## 2. Ruling out the calling / launch path

Standard launch is a flat one-warp-per-chunk grid
(`launch_config`, `onpair_bench.rs:2180`): `grid = total_chunks / block_warps`,
`block = block_warps*32`. At chunk1000mb that's one launch over a grid that fills
the device many times over — no per-chunk host round-trip, no oversized blocks.
Launch overhead only bites the tiny dbtext/`s_comment` columns (<6 MB), which the
table already flags as "launch-bound"; it's amortised away on the big columns.

(Caveat to keep in mind, not a defect: `gpu-decode-vortex` also runs the slow
reference kernel + nvCOMP recompression with no skip flag — make sure the compared
numbers are per-kernel `decode_ms`, which they are in `summary.json`.)

**Verdict: not the calling path.**

## 3. It is the kernel — measured B200 vs GH200

### 3.1 The numbers (best kernel each platform, large columns, chunk1000mb, 100it)

| dataset/col | bits | GH200 (kernel) | B200 (kernel) | best-vs-best Δ |
|---|---:|---|---|---:|
| fineweb/text | 12 | 567 (split8read) | 614 (wpb8_occ) | **+8%** |
| fineweb/text | 16 | 470 (4tpt) | 630 (wpb8_occ) | **+34%** |
| wikipedia/text | 12 | 538 (split8read) | 588 (wpb8_occ) | **+9%** |
| wikipedia/text | 16 | 538 (4tpt) | 613 (wpb8_occ) | **+14%** |
| ps_comment | 12 | 1117 (4tpt) | 1248 (wpb8_occ) | **+12%** |
| ps_comment | 16 | 866 (4tpt) | 1043 (wpb8_occ) | **+20%** |

(GiB/s. GH200 rows carried from handover; B200 from `b200_results.csv`. Unlocked
clocks, single runs — treat ±~5% as noise; the patterns below are well outside it.)

**The "barely faster" feeling is a bits12 phenomenon (+8–12%). bits16 is +14–34%.**

### 3.2 Why bits12 is only ~+10% — the L1/TEX + SM ceiling

The GH200 NCU (`ONPAIR_GPU_FINDINGS.md`) found decode is **L1/TEX-request bound
(~93%) on the random dict gather, DRAM idle (~17%)**, register-capped at 64 regs →
50% theoretical / ~43% achieved occupancy, ~65% "no-eligible-warp". The bits12 dict
is ~17 KB and **already fully cache-resident** (94% L2-hit). So the bottleneck is a
per-SM resource (L1/TEX request rate), gated by `SM_count × SM_clock`:

| Resource | GH200 | B200 | Ratio | bits12 decode uses it? |
|---|---|---|---|---|
| HBM bandwidth | ~4 TB/s | ~8 TB/s | ~2.0× | No (DRAM idle) |
| L2 size | ~50 MB | ~126 MB | ~2.5× | No (17 KB dict already resident) |
| SM count | 132 | ~148¹ | **~1.12×** | **Yes** |
| SM boost clock | ~1980 MHz | ~1965 MHz (measured) | ~1.0× | **Yes** |

¹ Repo working figure (`PERF_ARCH.md`); B200 is dual-die, physical count higher.

A per-SM-bound kernel therefore scales ~1.1–1.25× — and bits12 measures +8–12%.
**This is the expected, fundamental ceiling, not a misconfiguration.** B200's two
big wins (HBM, L2) are exactly the resources the cache-resident bits12 case doesn't
bottleneck on.

### 3.3 Why bits16 is +14–34% — the cache upgrade *does* help here

The bits16 dict is ~500 KB. On GH200 it does **not** fit L1 and thrashes (the
freq-sort note records a 35% sector hit); bits16 was GH200's worst case (fineweb
b16 only 470 GiB/s). B200's larger L1/L2 relieves that thrashing, so the gather
miss penalty drops and bits16 sees super-SM-scaling gains (fineweb b16 +34%). So
the earlier "L2 is irrelevant" intuition holds only for bits12; for the
cache-starved bits16 case, B200's caches are the win.

### 3.4 The launch-config flip — what it actually is (and isn't)

**`4tpt_wpb8_occ` regressed on GH200 but is the best kernel in all 10 large-column
cells on B200.** It is tempting to call this an "occupancy win," but the PTX says
otherwise — and the real mechanism matters for what to try next:

| kernel | `.maxntid` × `.minnctapersm` | threads/SM | theoretical occ | reg budget |
|---|---|---|---|---|
| `4tpt` (baseline) | 512 × 2 | 1024 | 50% | 64 |
| `4tpt_wpb8_occ` | **256 × 4** | 1024 | **50% (same!)** | 64 (same) |
| `2tpt` | 512 × 4 | 2048 | **100%** | 32 |

So `wpb8_occ` does **not** raise occupancy or cut registers vs baseline — both target
1024 threads/SM at 64 regs. The only difference is **block granularity: 256-thread
blocks (4/SM) vs 512-thread blocks (2/SM)**. The kernel is warp-centric (1 warp = 1
chunk, `__syncwarp` only), so the work is identical; smaller blocks just pack and
drain more evenly across SMs → higher *achieved* occupancy via less quantization and
better tail behaviour. **Blackwell's scheduler rewards this finer granularity where
Hopper didn't.** That's a cheap, universal lever (smaller blocks ≈ free on B200) and
it has nothing to do with register pressure.

The genuinely informative result is **`2tpt`**, which *does* double the occupancy
target (100%, 32-reg budget by holding only 2 tokens of state) and **wins on B200**
for `l_comment`/`s_comment` b12 — where it did *not* dominate on GH200. That is
direct evidence the **B200 limiter has shifted toward occupancy/latency**, not the
L1/TEX-request *throughput* saturation that bound GH200. Which means:

**The bits12 "SM-count ceiling" (§3.2) is softer than it looks.** bits12 fineweb at
614 GiB/s is only ~8% of B200 HBM and far from any L2 bandwidth wall — it is purely
gather request/instruction-throughput-bound *per SM*. If B200 is now occupancy/
latency-bound (the 2tpt + granularity evidence), then **raising real occupancy can
beat +10%**, because more resident warps hide the gather latency the saturated-pipe
GH200 analysis assumed was irreducible. NCU (Track C) settles this; the behavioural
signs already point that way.

There is a clean tension to exploit: `4tpt` **amortizes** the per-warp fixed cost
(prefix-scan + head/tail drain epilogue) over 128 tokens but burns 64 regs → 50%
occ; `2tpt` halves the amortization but reaches 100% occ. The sweet spot is likely a
variant that keeps 4tpt's amortization **and** lifts occupancy — see Track B′/E′.

But `pick_auto_kernel` only ever chooses `split8read` (bits12, small dict,
`frac_le8 ≥ 0.90`) or `4tpt`. **It does not know about `wpb8_occ`.** So the shipped
path picks the wrong kernel on B200. Auto-vs-best gap on B200 (`summary.json`):

| dataset/col | bits | auto kernel · GiB/s | best kernel · GiB/s | gap |
|---|---:|---|---|---:|
| clickbench/URL | 12 | 4tpt · 717 | wpb8_occ · 794 | **+10.8%** |
| tpch/l_comment | 12 | 4tpt · 1012 | wpb8_occ · 1121 | **+10.8%** |
| tpch/l_comment | 16 | 4tpt · 849 | wpb8_occ · 933 | **+10.0%** |
| tpch/ps_comment | 16 | 4tpt · 957 | wpb8_occ · 1043 | **+8.9%** |
| tpch/s_comment | 16 | 4tpt · 356 | wpb8_occ · 411 | **+15.4%** |
| fineweb/text | 16 | 4tpt · 585 | wpb8_occ · 630 | **+7.6%** |
| wikipedia/text | 16 | 4tpt · 573 | wpb8_occ · 613 | **+7.0%** |

The shipped selector underperforms its own best kernel by **~6–11% on every large
column**. Auto-vs-auto (the comparison most likely to "look flat"): fineweb **b12**
567→602 = **+6%**, but fineweb **b16** 470→585 = **+24%**. So fixing the selector
recovers most of that gap *for free* — the kernels already exist and validate
byte-exact.

### 3.5 Context: vs Blackwell's hardware Decompression Engine

OnPair decode is **3–6× faster** than nvCOMP's Blackwell HW DE and usually at a
*better* ratio (e.g. ps_comment b12 1248 GiB/s · 6.23× vs LZ4-HW 211 · 2.55×;
clickbench URL 956 · 3.86× vs LZ4-HW 287 · 3.58×). The DE only does
LZ4/Deflate/Snappy (Zstd-HW unsupported, status 10) — it can't touch OnPair — so
this is favourable but orthogonal to the GH200→B200 question.

---

## 4. So: flags / calling / kernels?

- **Flags:** correct (sm_100, CUDA 12.8, rebuilt). Not the cause.
- **Calling:** flat grid fills the device; clean at large chunks. Not the cause.
- **Kernels:** **yes** — two effects:
  1. **bits12 is at the L1/TEX + SM-count ceiling (~+10%)** — fundamental; the dict
     is already cache-resident so B200's HBM/L2 don't help. Expected, not a bug.
  2. **The optimal kernel flipped to `4tpt_wpb8_occ` on B200, but the auto-selector
     doesn't pick it**, costing ~6–11% across all large columns. This is the single
     highest-value, lowest-risk fix.

---

## 5. What to do

1. **Re-fit `pick_auto_kernel` for sm_100: select `4tpt_wpb8_occ` for large
   columns.** Recovers +6–11% immediately; kernels exist and validate byte-exact.
   Gate on arch (keep Hopper's `split8read`/`4tpt` choice for sm_90). Highest value,
   lowest risk. *(Note: this is a Rust/host change — out of scope for this read-only
   pass; flagged for follow-up.)*
2. **Confirm the limiter mechanistically.** NCU is currently blocked in-container
   (`ERR_NVGPUCTRPERM`, matches the env note), so the L1-vs-occupancy story is
   indirect. The `wpb8_occ` win is strong evidence the B200 limiter is now occupancy
   / latency-hiding rather than pure L1/TEX request rate. Get one
   SpeedOfLight + MemoryWorkload + Occupancy profile (needs `CAP_SYS_ADMIN`) on
   `4tpt` vs `wpb8_occ`, bits12 and bits16, to settle it and re-derive
   `__launch_bounds__` for Blackwell.
3. **Re-fit the `split8read` thresholds.** It still wins some bits12 cells but
   `wpb8_occ` now usually beats it; check whether the larger B200 L1 lets
   `split8read` extend to bits16, and fold both into the arch-aware selector.
4. **Don't expect more from HBM/L2 on bits12** — it's SM-count-bound by
   construction. The only bits12 levers that move the number reduce L1/TEX request
   work (`split8read`, freq-ordered codes) or lift occupancy (the `wpb8_occ` flip,
   already realized). The Blackwell-only shot at the gather itself is cluster-DSMEM
   shared dict (untried; Hopper's per-block bank-conflict failure changes under
   cluster-distributed shared memory).

---

## 6. Experiment plan

### 6.0 Do-this-in-order runbook (the one-by-one path)

Do these **in sequence** — each step's result decides whether/how to do the next.
The §6.1 table below is the same work as a reference grid (hypotheses, effort, exit
criteria); use it for detail, use this list for order.

> **Step 0 — Setup (blocking, do first).** Get `CAP_SYS_ADMIN` so NCU runs, and lock
> clocks (`nvidia-smi -lgc`) if permitted. Without this you cannot trust any delta
> under ~10% (unlocked clocks ±5%, single runs). If clocks can't be locked, use
> ≥300-iter intra-run ranking and only compare kernels *within one invocation*.

> **Step 1 — Ship Track A (free win, no dependency).** Make `pick_auto_kernel`
> arch-aware: pick `4tpt_wpb8_occ` for large columns on sm_100; keep
> `split8read`/`4tpt` for sm_90. Recovers +6–11%, kernels already validate
> byte-exact. Independent — can land immediately, in parallel with everything else.

> **Step 2 — Run Track C (the gate).** One NCU profile (SpeedOfLight +
> MemoryWorkload + Occupancy) on `4tpt` vs `wpb8_occ` vs `2tpt`, bits12 and bits16.
> **This decides the whole branch:**
> - If **occupancy/latency-bound** (expected — `2tpt` & granularity wins point here)
>   → go to Step 3A (occupancy axis) first.
> - If still **L1/TEX-throughput-bound** → skip 3A, go to Step 3B (request-count
>   axis) first; occupancy won't help.

> **Step 3A — Occupancy/granularity axis (if Step 2 says latency-bound).** In order
> of cheapness: **B″** (`split8read_occ`, just `#define`s) → **B** (block×occupancy
> sweep, find the surface) → **B′** (register-lean scan-then-regather, the real bet
> for breaking the bits12 ceiling) → **B‴** (persistent grid, only if tail still
> shows in NCU). Stop early if one clearly wins and re-fold it into the Step-1
> selector.

> **Step 3B — Request-count axis (best for bits16, run after/with 3A).** **D**
> (re-measure `ONPAIR_DICT_REORDER=freq`; if ≥+5%, ship encoder ordering) → **E**
> (hot-dict shared cache — the headline new kernel; needs D's freq-order so low codes
> are hot). **F** (per-warp dedup) and **G** (cluster-DSMEM) are speculative
> last-resorts — only if C confirms request-count is the wall *and* B/E plateau.

> **Step 4 — Re-fit the selector.** Fold every winner into the arch-aware
> `pick_auto_kernel` (bits12 vs bits16, dict size, `frac_le8`) and re-run the full
> column sweep to confirm no regressions.

**Minimal critical path if you only do three things:** A (ship) → C (profile) →
whichever single track C points at (B′ if latency-bound, E if request-bound).

### 6.1 Track reference (detail for the steps above)

Independent tracks — each is a separate kernel/host change + bench run. Bench:
`onpair-chunk-bench gpu-decode-vortex --gpu-iters 300 --gpu-validate` on the big
columns (clickbench/URL, fineweb, wikipedia, tpch ps_comment/l_comment), bits12
**and** bits16. Compare kernels **within one invocation** (unlocked clocks); every
variant must validate byte-exact.

| # | Track | Hypothesis / mechanism | Build | Effort · Risk | Depends on | Success metric |
|---|---|---|---|---|---|---|
| **A** | **Arch-aware selector** | Optimum flipped to `4tpt_wpb8_occ` on sm_100; selector never picks it → 6–11% left on table | Host: `pick_auto_kernel` picks `wpb8_occ` for large cols when `arch==sm_100`; keep Hopper choice for sm_90 | XS · none | — | auto GiB/s → best GiB/s (recover +6–11%), all cells |
| **B** | **Block-granularity + occupancy sweep** | Two distinct B200 levers (§3.4): (i) **block size** — 256-thread blocks beat 512 at *same* occupancy (`wpb8_occ`); try 128 too; (ii) **true occupancy** — `2tpt` at 100% occ wins some cols. Sweep tpt∈{2,3,4} × block∈{128,256,512} × `.minnctapersm`∈{2,3,4,6} | existing + new bound macros (cheap; mostly `#define`s) | S · low | C (read achieved occ) | beat `wpb8_occ` on ≥1 large col; map the granularity×occ surface |
| **B′** | **Register-lean high-occ `4tpt` (NEW)** — "scan-then-regather" | Keep 4tpt's epilogue amortization but cut live registers to ≤42 (→3 blocks/SM, 75%) by scanning **lengths only** first (4 regs), then re-gathering token `uint4`s in the write phase instead of holding 4 across the scan. Trades a 2nd gather for occupancy — *bad* on throughput-bound GH200, plausibly *good* on latency-bound B200 | New `onpair_shmem_4tpt_lean.cu` | M · med | C (confirm latency-bound) | >50% achieved occ AND > `wpb8_occ` |
| **B″** | **`split8read_occ` combo** | split8read already wins bits12 *and* holds less per-token data (uint2=2 regs common case) → headroom to push occupancy/granularity it doesn't yet use (still `512×2`) | `#define` variants: `256×4`, `512×3` | XS · low | — | > `split8read` and > `wpb8_occ` on bits12 |
| **B‴** | **Persistent-grid plain `4tpt` (NEW)** | `wpb8_occ`'s granularity win implies tail/quantization matters; a persistent grid (fixed `N×SM` blocks, grid-stride over chunks) removes block launch/retire + quantization entirely. Note: `pdict`/`vdict` used persistent grid but failed on the *shared-dict* conflicts, not the grid — plain 4tpt persistent was never tried | New launch path + minor kernel loop | M · med | — | > `wpb8_occ`; clean tail |
| **C** | **Unblock NCU + lock clocks** | Mechanism is currently inferred, not measured (`ERR_NVGPUCTRPERM`); is B200 still L1/TEX-throughput bound, or now occupancy/latency bound (which `wpb8_occ` win implies)? | Run container with `CAP_SYS_ADMIN`; `nvidia-smi -lgc/-lmc` if permitted; NCU SpeedOfLight+MemWorkload+Occupancy on `4tpt` vs `wpb8_occ`, b12 & b16 | S · infra | — | one clean NCU section per kernel → names the B200 limiter |
| **D** | **Freq-ordered codes (re-measure)** | Low-code=hot improves L1/L2 locality (+8–13% b16 on GH200); may shrink with B200's bigger cache | `ONPAIR_DICT_REORDER=freq` (already wired) sweep on B200; if win holds, productionize in OnPair *encoder* | S(measure)/M(encoder) · low | — | b16 GiB/s uplift; if ≥+5%, ship encoder ordering |
| **E** | **Hot-dict shared cache (NEW)** | Zipfian dict → top-N codes cover 50–80% of tokens. Cache top-N (N≈256–512, ≈4–8 KB) in shared; `code<N` (= hot, under freq-order) served from shared (broadcast-friendly, *not* the full-dict random gather that made `pdict`/`vdict` conflict), `code≥N` falls back to global gather. Cuts L1/TEX **request count**, the actual limiter — distinct from the 32-entry `regcache` (too small, +5 regs) | New `onpair_shmem_4tpt_hotdict.cu`: cooperative load of top-N once/block, branch on code threshold | M · medium | D (needs freq-order so low codes are hot) | b16 GiB/s (biggest cache-starved win); beat `wpb8_occ` |
| **F** | **Per-warp gather dedup (NEW, speculative)** | Within 128 warp tokens many codes repeat (Zipf) → dedup unique codes, gather once, broadcast results; fewer L1/TEX requests | New kernel; warp-cooperative unique/scan over 128 codes | L · high | C (to confirm request-count is limiter) | net win after dedup overhead |
| **G** | **Cluster-DSMEM dict (Blackwell-only, speculative)** | Hopper dict-in-shared failed on per-block bank conflicts + occupancy halving; thread-block *cluster* splits the dict across N blocks' distributed shared mem (e.g. 64 KB/8 = 8 KB/block) → conflict profile changes | New kernel using cluster + DSMEM (`cluster.map_shared_rank`) | L · high | C | beat baseline on b16 (the cache-starved case) |

Notes:
- **A is the only zero-risk, ship-now item** and recovers most of the visible gap.
- **C should run first/in-parallel with everything** — it tells whether the B200
  limiter is still L1/TEX *throughput* (then only E/F/D — request reduction — help)
  or has shifted to *occupancy/latency* (then B's occupancy sweep is the lever, and
  the bits12 ceiling is softer than feared). The `wpb8_occ` win is a strong hint it
  shifted toward occupancy.
- **E is the highest-value new kernel idea** and is explicitly *not* a repeat of the
  `pdict`/`vdict` dead ends: those put the **whole** dict in shared (fully random 16 B
  reads → ~700× bank conflicts, occupancy halved). A small **hot-only** cache under
  freq-ordering is dominated by **broadcasts** (many lanes → same top code), a
  different access pattern, and at 4–8 KB it barely dents occupancy. Strongest on
  **bits16** (500 KB dict that thrashes L1), where the cold tail spills to B200's
  larger L2.
- Keep the GH200 dead ends closed (full dict-in-shared, lenbucket variable-stride,
  L2-persist, cp.async, split4) unless C shows the limiter changed.

## Bottom line

The B200 result is **+8–12% on bits12 and +14–34% on bits16** best-vs-best — not
flat. The "barely faster" perception comes from two things: (a) the shipped
**`pick_auto_kernel` picks the wrong kernel on B200** — the optimum flipped to
`4tpt_wpb8_occ`, and the selector (which only knows `4tpt`/`split8read`) leaves
**6–11% on the table on every large column**; and (b) bits12 sits near its per-SM
gather-throughput limit, where B200's 2× HBM / 2.5× L2 don't help because the tiny
dict is already cache-resident.

But the bits12 ceiling is **softer than first thought.** The `wpb8_occ` win is a
*block-granularity* effect, not extra occupancy (both are 50%/64-reg) — and the
separate **`2tpt`-at-100%-occupancy win** is direct evidence the B200 limiter has
shifted from L1/TEX *throughput saturation* (the GH200 wall) toward
*occupancy/latency*. At 614 GiB/s, bits12 is ~8% of HBM and far from any bandwidth
wall — it is algorithm/occupancy-bound, so **raising real occupancy (Tracks B/B′)
can plausibly beat +10%**, and request-reduction (D/E) attacks the other axis.

Flags (sm_100, CUDA 12.8) and the launch path are correct. Order of attack: ship
**A** (free), run **C** to confirm the limiter shifted (it gates everything), then
**B/B′/B″** (occupancy+granularity, now the most promising bits12 lever) and **D/E**
(request reduction, best for bits16) in parallel. (All B200 numbers preliminary:
unlocked clocks, single runs, NCU blocked — lock clocks / get `CAP_SYS_ADMIN` before
trusting sub-10% deltas.)
