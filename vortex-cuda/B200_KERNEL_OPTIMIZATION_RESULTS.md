# B200 OnPair decode kernel optimization — results (PRELIMINARY)

Applying the experiment plan in `B200_VS_GH200_ONPAIR_ANALYSIS.md` one track at
a time. **PRELIMINARY**: unlocked clocks (±~5%), single-invocation kernel ranking, NCU
blocked (`ERR_NVGPUCTRPERM`), clock-locking blocked. All variants validated **byte-exact**.

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
  the smaller working set. Kept as an evidence kernel; **not** in `pick_auto_kernel`. bits16 is
  effectively at the kernel/layout ceiling for these columns — remaining headroom is algorithmic
  (dict-cache approaches need freq-ordering, which is forbidden — see Track E).

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
