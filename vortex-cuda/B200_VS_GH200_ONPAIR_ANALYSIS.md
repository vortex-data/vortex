# Why the initial B200 OnPair decode results looked only modestly faster than GH200 — analysis

**Question:** We ported OnPair GPU decode to a B200 (Blackwell, sm_100); the first
numbers were not much better than the GH200 (Hopper, sm_90) baseline. Is it compile
flags, the launch/calling path, or the kernels?

**Short answer (revised against the latest local B200 data in
`vortex-cuda/B200_KERNEL_OPTIMIZATION_RESULTS.md` plus
`benchmarks/onpair-bench/B200_PRELIMINARY.md` / `b200_results.csv`):**

1. **Flags and the calling path are fine** — verified sm_100 PTX, CUDA 12.8, freshly
   built; flat grid that fills the device on large columns. Not the cause.
2. **The original “barely faster” result was mostly a kernel-selection/configuration
   artifact.** The first B200 sweep topped out at `4tpt_wpb8_occ` (256-thread blocks),
   giving only +8–12% on bits12 text. The latest B200 sweep found 128-thread variants:
   `split8read_b128o12` for short-token bits12 and `b128o12` otherwise. That moves
   fineweb/wikipedia bits12 to **802/747 GiB/s**, about **+41%/+39% vs GH200**.
3. **bits16 remains meaningfully faster (+15–36% on shared large columns)** — likely
   because B200's larger L2 and changed cache/scheduler behaviour relieve the ~500 KB
   dict pressure that hurt GH200 bits16. The exact limiter still needs NCU.
4. **The selector is now arch-aware.** `pick_auto_kernel(chunks, cc_major)` keeps the
   GH200 rule (`split8read` for small bits12 dicts with `frac_le8>=0.90`, else `4tpt`)
   and uses the B200 rule (`split8read_b128o12` for small bits12 dicts with
   `frac_le8>=0.70`, else `b128o12`). The B200 win is mostly **128-thread block
   granularity**, not raw occupancy.

So: it is the **kernel shape + an arch-dependent launch-config/selector flip**, not
the flags or the call path. Details and evidence below.

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

## 3. It is the kernel — latest B200 vs GH200

### 3.1 Shared-column B200 vs GH200 (latest B200 auto-selected kernels)

| dataset/col | bits | GH200 (kernel) | B200 latest (kernel) | best-vs-best Δ |
|---|---:|---|---|---:|
| fineweb/text | 12 | 567 (split8read) | 802 (split8read_b128o12) | **+41%** |
| fineweb/text | 16 | 470 (4tpt) | 638 (b128o12) | **+36%** |
| wikipedia/text | 12 | 538 (split8read) | 747 (split8read_b128o12) | **+39%** |
| wikipedia/text | 16 | 538 (4tpt) | 621 (b128o12) | **+15%** |
| ps_comment | 12 | 1117 (4tpt) | 1259 (b128o12) | **+13%** |
| ps_comment | 16 | 866 (4tpt) | 1042 (b128o12) | **+20%** |

(GiB/s. GH200 rows carried from the handover / `b200_results.csv`; latest B200 rows
from `B200_KERNEL_OPTIMIZATION_RESULTS.md`. Unlocked clocks, single-invocation kernel
ranking — treat sub-10% deltas as noisy, but the bits12 text gains are far outside
that.)

**The original “barely faster” feeling is no longer true for short-token bits12 text.**
It was true for the first 256-thread B200 sweep (`wpb8_occ`: +8–12% on bits12), but
`split8read_b128o12` changes the story for fineweb/wikipedia. Long-token bits12
(`ps_comment`) still scales modestly (+13%).

### 3.2 What changed in the latest B200 optimization sweep

The latest sweep made the auto selector arch-aware and added 128-thread launch-config
variants of existing kernels. B200 gain over the old `4tpt` default:

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

All rows validated byte-exact. The B200 selector is now:

| arch | general-case selector |
|---|---|
| sm_90 (Hopper/GH200) | `split8read` if small bits12 dict and `frac_le8>=0.90`, else `4tpt` |
| sm_100 (Blackwell/B200) | `split8read_b128o12` if small bits12 dict and `frac_le8>=0.70`, else `b128o12` |

The lower B200 `frac_le8` gate captures clickbench/URL bits12 (`frac_le8=0.81`,
`split8read_b128o12` +5.5% over `b128o12`) while avoiding l_comment (`0.58`, -2.6%)
and ps_comment (`0.33`, -4.8%). bits16 fails the small-dict gate, so it always goes
to `b128o12`.

### 3.3 Mechanism: 128-thread granularity, not raw occupancy

The earlier analysis correctly spotted that `wpb8_occ` was a block-granularity effect,
but it stopped at 256-thread blocks. The latest sweep decomposes the B200 lever:

| comparison | what varies | B200 effect |
|---|---|---|
| `wpb8_occ` -> `b128` | 256 -> 128 threads/block, both 50% occ | **+1 to +4%** |
| `b128` -> `b128o12` | 50% -> 75% occ, both 128-thread | **~0% (noise)** |
| `b128` -> `b64`/`b64o24` | 128 -> 64 threads/block | **~0% (plateau at 128)** |
| `split8read_occ` -> `split8read_b128o12` | 256 -> 128-thread split8read | **+5% -> +26%** |

So the main Blackwell lever is **128-thread block granularity**. The forced 75%
occupancy in `b128o12` is harmless (ptxas cuts to ~40 regs, no spill), but the sweep
says occupancy itself buys little; plain `b128` is about as fast. NCU is still blocked
(`ERR_NVGPUCTRPERM`), so the low-level reason remains inferred: smaller CTAs appear to
schedule/drain more evenly across Blackwell SMs and let split8read's lower request width
pay off.

### 3.4 bits12 vs bits16 after the latest sweep

The GH200 NCU (`ONPAIR_GPU_FINDINGS.md`) found decode was **L1/TEX-request bound
(~93%) on the random dict gather, DRAM idle (~17%)**, register-capped at 64 regs ->
50% theoretical / ~43% achieved occupancy. B200 changes the best launch shape rather
than simply scaling that same kernel.

For bits12:
- short-token text (`fineweb`, `wikipedia`) now benefits from **both** split8read's 8 B
  common-case dict reads and 128-thread granularity: +39–41% vs GH200;
- long-token `ps_comment` does not benefit from split8read, so it gets the smaller
  `b128o12`/granularity win: +13% vs GH200.

For bits16, the dict is ~500 KB. It does **not** fit in the per-SM cache and thrashed
on GH200 (the freq-sort note recorded a 35% sector hit). B200's larger L2 and changed
cache/scheduler behaviour plausibly reduce the gather penalty; the latest shared-column
speedups are +15–36%. NCU should confirm whether that is L2 residency, L1/TEX behaviour,
latency hiding, or some mix.

### 3.5 Context: vs Blackwell's hardware Decompression Engine

OnPair decode is **~2.5–3.8x faster** than the best measured nvCOMP Blackwell HW DE
rows on the large columns. Against Deflate-hi (max-ratio HW preset), examples are
clickbench/URL b16 957 GiB/s · 3.86x vs 383 GiB/s · 6.44x, and ps_comment b12
1259 GiB/s · 6.23x vs 378 GiB/s · 5.67x. Against LZ4-HW, examples are
clickbench/URL b16 957 GiB/s · 3.86x vs 363 GiB/s · 3.70x, and ps_comment b12
1259 GiB/s · 6.23x vs 247 GiB/s · 2.56x. Ratio is mixed: Deflate-hi beats OnPair
ratio on clickbench/URL and l_comment, while OnPair wins decode throughput on every
measured large column. The DE only does LZ4/Deflate/Snappy (Zstd-HW unsupported,
status 10) — it cannot decode OnPair — so this is favourable but orthogonal to the
GH200 -> B200 question.

### 3.6 Complete available ratio and decode matrices

These tables are intentionally sparse: each numeric cell is present in the latest
local benchmark artifacts, and `--` means that GPU/technique/parameter combination
was not measured or has no numeric result. B200 ratio / nvCOMP rows come from
`benchmarks/onpair-bench/b200_results.csv` (mtime `2026-05-21 13:55 UTC`). B200 OnPair decode
for the 10 big-column cells is updated from
`vortex-cuda/B200_KERNEL_OPTIMIZATION_RESULTS.md` (mtime `2026-05-21 17:23 UTC`), because those
post-date `b200_results.csv`. GH200 OnPair rows come from the CSV plus
`benchmarks/onpair-bench/GPU_DECODE_SUMMARY.md` for `book-reviews` bits12/bits16.
Ratios are compression ratios (`decoded_bytes / compressed_bytes`). Decode speed is
kernel-only GiB/s over decoded bytes.

**Compression Ratio**

| dataset/column | B200 OnPair b12 | B200 OnPair b16 | B200 Deflate hi | B200 Deflate fast | B200 LZ4 | B200 Zstd L3 | B200 Zstd L-10 | GH200 OnPair b12 | GH200 OnPair b16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| book-reviews/text | -- | -- | -- | -- | -- | -- | -- | 2.60x | 3.29x |
| clickbench/URL | 2.89x | 3.86x | 6.44x | 1.45x | 3.70x | 5.64x | -- | -- | -- |
| dbtext/email | 2.08x | 1.95x | -- | -- | -- | -- | -- | -- | -- |
| dbtext/hex | 1.22x | 1.12x | -- | -- | -- | -- | -- | -- | -- |
| dbtext/l_comment | 3.82x | 3.65x | -- | -- | -- | -- | -- | -- | -- |
| dbtext/ps_comment | 5.50x | 5.14x | -- | -- | -- | -- | -- | -- | -- |
| dbtext/yago | 1.59x | 1.62x | -- | -- | -- | -- | -- | -- | -- |
| fineweb/text | 2.24x | 2.89x | 2.55x | 1.71x | 1.54x | 2.57x | -- | 2.24x | 2.89x |
| tpch-sf10/l_comment | 4.17x | 4.19x | 4.56x | 1.85x | 2.17x | 2.87x | 1.79x | -- | -- |
| tpch-sf10/ps_comment | 6.23x | 5.82x | 5.67x | 1.85x | 2.56x | 4.16x | -- | 6.23x | 5.82x |
| tpch-sf10/s_comment | 5.17x | 4.67x | -- | -- | -- | -- | -- | -- | -- |
| wikipedia/text | 2.15x | 2.80x | 2.70x | 1.67x | 1.64x | 2.74x | -- | 2.15x | 2.80x |

**Decompression Speed (GiB/s)**

| dataset/column | B200 OnPair b12 | B200 OnPair b16 | B200 Deflate hi | B200 Deflate fast | B200 LZ4 | B200 Zstd L3 | B200 Zstd L-10 | GH200 OnPair b12 | GH200 OnPair b16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| book-reviews/text | -- | -- | -- | -- | -- | -- | -- | 607.0 | 561.8 |
| clickbench/URL | 843.0 | 957.0 | 383.0 | 125.7 | 362.7 | 112.1 | -- | -- | -- |
| dbtext/email | 164.8 | 154.6 | -- | -- | -- | -- | -- | -- | -- |
| dbtext/hex | 74.4 | 72.0 | -- | -- | -- | -- | -- | -- | -- |
| dbtext/l_comment | 206.5 | 206.1 | -- | -- | -- | -- | -- | -- | -- |
| dbtext/ps_comment | 221.0 | 228.1 | -- | -- | -- | -- | -- | -- | -- |
| dbtext/yago | 140.9 | 137.0 | -- | -- | -- | -- | -- | -- | -- |
| fineweb/text | 802.0 | 638.0 | 169.8 | 125.6 | 188.3 | 8.3 | -- | 567.0 | 470.0 |
| tpch-sf10/l_comment | 1097.0 | 932.0 | 293.0 | 121.7 | 224.1 | 84.5 | 94.7 | -- | -- |
| tpch-sf10/ps_comment | 1259.0 | 1042.0 | 377.6 | 124.7 | 246.6 | 37.4 | -- | 1117.0 | 866.0 |
| tpch-sf10/s_comment | 409.1 | 411.0 | -- | -- | -- | -- | -- | -- | -- |
| wikipedia/text | 747.0 | 621.0 | 175.6 | 123.8 | 194.3 | 1.0 | -- | 538.0 | 538.0 |

---

## 4. So: flags / calling / kernels?

- **Flags:** correct (sm_100, CUDA 12.8, rebuilt). Not the cause.
- **Calling:** flat grid fills the device; clean at large chunks. Not the cause.
- **Kernels:** yes. The first B200 result used the wrong launch shape for Blackwell.
  The latest result is arch-aware and uses 128-thread kernels:
  1. **`split8read_b128o12` for small-dict, short-token bits12** — this is the big
     text-column win (+39–41% vs GH200 on fineweb/wikipedia, +46% vs old B200 `4tpt`).
  2. **`b128o12` elsewhere** — gives modest but consistent wins (+6–10% vs old B200
     `4tpt`, +13–36% vs GH200 on shared large columns).

---

## 5. Current Status / What To Do Next

1. **Selector work is done for the bench.** `pick_auto_kernel(chunks, cc_major)` is now
   arch-aware: GH200 keeps the old rule; B200 uses `split8read_b128o12` / `b128o12`.
2. **NCU remains the main blocker.** The mechanism is inferred from sweep decomposition
   because NCU is blocked in-container (`ERR_NVGPUCTRPERM`) and clock locking is not
   permitted. With `CAP_SYS_ADMIN`, profile `4tpt`, `wpb8_occ`, `b128`, `b128o12`, and
   `split8read_b128o12` on bits12/bits16.
3. **Possible simplification:** since `b128` ~= `b128o12`, consider using `b128` as the
   B200 general default after one locked-clock / high-iter confirmation. It avoids forced
   register reduction without giving up measured speed.
4. **Dropped/dead tracks:** freq-ordering is not pursued because it conflicts with other
   pipeline constraints; hot-dict shared cache depends on freq-order and regressed without
   it; persistent grid is low-value for large columns.
5. **Tiny columns remain launch-bound.** dbtext columns decode in roughly fixed launch/ramp
   time, so GiB/s is not meaningful there. Batching many tiny columns per launch would be a
   harness/product change, not an OnPair dict-decode kernel fix.

---

## 6. Updated Experiment Status

| Track | Status | Result |
|---|---|---|
| A: arch-aware selector | **Done** | B200 uses `split8read_b128o12` / `b128o12`; GH200 rule unchanged. |
| B/B-second: granularity + occupancy sweep | **Done** | 128-thread block granularity is the lever; forced 75% occupancy is ~noise. |
| B+: split8read + granularity | **Done / standout** | +23–26% on fineweb/wikipedia bits12 over `b128o12`; +5.5% on clickbench/URL bits12. |
| B-prime: register-lean 4tpt | **Subsumed** | `b128o12` reaches ~40 regs with no spill; no algorithm rewrite needed. |
| C: NCU + locked clocks | **Blocked** | Needs container/host permissions. |
| D: freq-ordered codes | **Dropped** | Conflicts with other pipeline constraints. |
| E: hot-dict shared cache | **Dead** | Requires freq-order; without it hit rate is poor and it regresses. |
| B-third: persistent grid | **Deprioritized** | Large columns already fill the device many waves; tail/launch quantization is amortized. |

## Bottom Line

The latest B200 result is no longer “only modestly faster” for the short-token bits12
text columns: `split8read_b128o12` reaches **802 GiB/s** on fineweb and **747 GiB/s**
on wikipedia, about **+41%/+39% vs GH200** and **+46% vs the old B200 `4tpt` default**.
For bits16, B200 is still meaningfully faster on shared large columns (**+15–36%**),
while long-token bits12 (`ps_comment`) remains a modest **+13%**.

The core lesson changed from “B200 only gives SM-count scaling” to “Blackwell needs a
different launch shape.” 128-thread CTAs unlock most of the B200 kernel win, and
split8read becomes valuable again at that granularity. Flags and the calling path were
not the issue. All latest numbers remain preliminary: unlocked clocks, single-run /
single-invocation rankings, and NCU blocked.
