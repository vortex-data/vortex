# Findings — shipped wins + measured mechanisms

All numbers are B200, fineweb/text unless noted, GiB/s, PRELIMINARY (±~5% unlocked clocks).

## A. Shipped: arch-aware kernel selection

`pick_auto_kernel(chunks, cc_major)` dispatches by compute capability to two self-contained
selectors so neither architecture's tuning touches the other:

```
sm_100 (B200):  pick_general_blackwell(max_entries, frac_le8)
                  split8read_b128o12   if max_entries <= 16384 && frac_le8 >= 0.70
                  b128o12              otherwise
sm_90  (GH200): pick_general_hopper(max_entries, frac_le8)   # verbatim pre-B200 logic
                  split8read           if max_entries <= 4096  && frac_le8 >= 0.90
                  4tpt                 otherwise
```

Shared arch-independent early-exits (degenerate dicts) are unchanged: `const1`/`const2`
(all-1/all-2-byte), `s4l1_16tpt` (≤4 B), `s8_4tpt` (≤8 B).

**B200 gains over the old `4tpt` default (byte-exact):**

| column | bits | old `4tpt` | new auto | GiB/s | gain |
|---|---:|---:|---|---:|---:|
| fineweb/text | 12 | 550 | `split8read_b128o12` | 802 | **+46%** |
| wikipedia/text | 12 | — | `split8read_b128o12` | 747 | +46% |
| clickbench/URL | 12 | 717 | `split8read_b128o12` | 843 | **+18%** |
| tpch/l_comment | 12 | — | `b128o12` | 1097 | +8% |
| tpch/ps_comment | 12 | — | `b128o12` | 1259 | +7% |
| fineweb/text | 16 | 585 | `b128o12` | 638 | +9% |
| all bits16 | 16 | — | `b128o12` | — | +6–12% |

Two gate moves vs the original Hopper rule, both sm_100-only and validated byte-exact:
- **`frac_le8` 0.90 → 0.70:** captures clickbench/URL bits12 (frac 0.81, +5.5%) while excluding
  l_comment (0.58) / ps_comment (0.33), which regress with split8read. 0.70 sits centered between
  the win and regression bands.
- **dict-size `≤4096` → `≤16384`:** captures bits14 high-frac text (see §D).

## B. Granularity is the Blackwell lever, not occupancy

Controlled comparison on fineweb bits12 (each row holds everything fixed but one variable):

| kernel | block | occ | GiB/s |
|---|---|---|---|
| `4tpt_wpb8_occ` | 256 | 50% | 608 |
| `4tpt_b128` | 128 | 50% | 633 (+4%) |
| `4tpt_b128o12` | 128 | 75% | 636 (+5%) |
| `4tpt_b64` | 64 | 50% | 633 (plateau) |

256→128 threads jumps; 50%→75% occupancy at fixed 128 buys ~nothing; 64-thread ties 128
(plateau). **Block size moves the needle; target occupancy does not.** Theory: B200's dual-die /
~148-SM layout means more, smaller blocks load-balance the gather across both dies and their L2
partitions — which is why finer granularity helps on B200 but *regressed* on single-die GH200
(hence the arch gate).

## C. 8 bytes is the optimal dict read width

fineweb bits12, all 128-thread/75%:

| read width | kernel | GiB/s |
|---|---|---|
| 16 B (padded) | `b128o12` | 636 |
| **8 B (`uint2`)** | `split8read_b128o12` | **800 (+26%)** |
| 4 B (`uint`) | `split4read_b128o12` | 613 (−4%) |

Inverted-U with 8 B at the peak: halving the forced 16 B padded-dict read to 8 B cuts L2/TEX
transactions; going to 4 B is already below the 32 B sector, so it cuts no transactions and adds
a >4 B fallback. The gather is **transaction/MSHR-bound, not request-width-bound below the
sector.**

## D. Dict bit-width sweep — bits14 is the L1-residency sweet spot

`split8read` reads from `dict_s8` (entries × 8 B); it wins while that fits the ~256 KB L1.

| bits | entries | `dict_s8` | ratio | best decode | split8 Δ |
|---:|---:|---:|---:|---:|---:|
| 10 | 1024 | 8 KB | 1.2× | 674 | — |
| 11 | 2048 | 16 KB | 1.4× | 759 | +33% |
| 12 | 4096 | 32 KB | 1.7× | **802** | +25% |
| **14** | 16384 | 128 KB | 2.3× | 678 | **+9%** |
| 15 | 32768 | 256 KB | 2.6× | 637 | +3% (noise) |
| 16 | 65536 | 512 KB | 2.9× | 637 | tie |

- **Decode peaks at bits12** (smallest L1-resident dict) and **flattens at ~637 from
  bits15→bits16**.
- **bits10/11 are strictly dominated by bits12** (worse ratio, no speed gain — already
  L1-resident at 12).
- **bits14 (128 KB `dict_s8`) is the sweet spot:** +9% split8read at 2.3× ratio. The selector
  gate was widened to `≤16384` to capture it.
- **A half-filled 16-bit dict (bits15, 256 KB `dict_s8`) gives no decode gain** — 637, identical
  to full bits16 — because 256 KB exactly fills L1; the cache win needs `dict_s8` *comfortably*
  under L1 (≤128 KB).

## E. Why bits16 is walled — near-uniform dict access

Measured access distribution: bits16 columns use ~99.9% of all 65536 entries and the 4096
hottest cover only **~43–51%** of accesses. There is **no hot subset to cache**; the full ~1 MB
dict overflows L1 and the access is essentially uniform. This is the root cause behind every
failed bits16 lever (hot-cache, lenbucket, L2-persist, cluster-DSMEM, variable-width). The one
GH200 NCU profile corroborates: bits16 decode was L1/TEX-request bound at 93%, dict only 31%
L1-resident.

## F. Whole-decompress (H2D + decode) is transfer-bound

End-to-end output rate including the H2D copy of the compressed payload:

| fineweb | ratio | decode GiB/s | h2d GiB/s (pageable) | whole GiB/s | whole/h2d |
|---|---:|---:|---:|---:|---:|
| bits12 | 1.7× | 801 | ~11 | 19 | 1.6× |
| bits16 | 2.9× | 637 | ~11 | 34 | 2.7× |
| tpch/ps_comment b16 | 6.3× | 1040 | ~11 | 66 | 5.9× |

H2D (~10 GiB/s pageable; ~50 pinned on this PCIe5 box) is 60–100× slower than decode, so
end-to-end is dominated by the copy and `whole/h2d ≈ compression ratio`. **The decode-kernel wins
are not the bottleneck for an H2D-then-decode pipeline — compression ratio is.** They matter for
**on-device decode** (GPU query/scan where the column already lives on the GPU). Note bits16 wins
end-to-end on ratio even though it decodes slower.

## G. The headline discovery — the emit is ~70% of runtime (ablation NCU-proxy)

NCU is blocked, so the per-stage cost was measured by **ablation kernels** (full decode minus one
stage, timing-only). Speedup when a stage is removed ≈ its cost share. fineweb:

| stage removed | bits14 | bits16 |
|---|---:|---:|
| full decode | 612 | 635 |
| − dict gather | 1010 (+65%) | 1089 (+71%) |
| **− emit (byte-staging to shared)** | **2060 (+237%)** | **2353 (+270%)** |
| − output drain | 668 (+9%) | 688 (+8%) |
| − warp prefix-scan | 813 (+33%) | 774 (+22%) |

**The emit dominates (~70%; removing it is 3.4×).** Gather is second (~40%), scan ~20–25%, drain
~8%. A conflict-free-addressing emit proxy recovered only **3%**, so the emit is **store-count /
throughput bound** (~900 shared byte-stores/warp), *not* bank-conflict bound.

**Implication:** all dict-side tuning optimized the second-biggest cost. The remaining
**1.6–3.4× headroom is in the emit**, and the fix is to cut store count — a **shuffle/`byte_perm`
emit** (assemble aligned 16 B output chunks in registers off the LSU, then ~`warp_total/16`
coalesced `uint4` stores, ~25× fewer store instructions). Justified, not yet built.
