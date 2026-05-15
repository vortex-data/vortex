# LayoutPlan v2 — TPC-H baselines

Recorded after PR10 (streaming Let). Used to compare future
optimisations (struct/project mask pushdown, predicate pushdown,
sub-segment reads, etc.) against the current state.

## Environment

- **Date:** 2026-05-15
- **HEAD:** `7a25b2387` (`ngates/layoutv2-10`, "streaming Let — replace broadcast with TeeStream")
- **Hardware:** Darwin 25.4.0 / arm64 / 118 GB free RAM
- **Build:** `cargo build --bin datafusion-bench --release`
- **Allocator:** mimalloc (`vortex-bench` default)
- **Format:** `vortex-file-compressed` (single format; no parquet/arrow comparison here)

## How to reproduce

```bash
# V1 (default)
./target/release/datafusion-bench tpch \
  --formats vortex --iterations 3 --opt scale-factor=1.0 --hide-progress-bar

# V2 (LayoutPlan v2 + CSE + streaming Let)
VORTEX_LAYOUT_PLAN_V2=1 ./target/release/datafusion-bench tpch \
  --formats vortex --iterations 3 --opt scale-factor=1.0 --hide-progress-bar
```

## Numbers

3 iterations per query, scale factor 1.0. Times in microseconds (best of 3, as
reported by the bench harness). `Δ` is `(V2/V1 - 1) × 100`; positive means V2 is
slower.

| Query | V1 (μs) | V2 (μs) |    Δ% |
|------:|--------:|--------:|------:|
|     1 |  28 868 |  25 490 | -11.7 |
|     2 |   8 797 |  11 246 | +27.8 |
|     3 |  11 660 |  12 544 |  +7.6 |
|     4 |  11 387 |  12 563 | +10.3 |
|     5 |  25 230 |  27 680 |  +9.7 |
|     6 |   6 401 |   8 020 | +25.3 |
|     7 |  26 169 |  27 269 |  +4.2 |
|     8 |  18 514 |  22 131 | +19.5 |
|     9 |  22 432 |  31 196 | +39.1 |
|    10 |  19 103 |  22 506 | +17.8 |
|    11 |   6 494 |   8 003 | +23.2 |
|    12 |  12 673 |  13 658 |  +7.8 |
|    13 |  12 793 |  13 808 |  +7.9 |
|    14 |   7 632 |   8 507 | +11.5 |
|    15 |  13 047 |  14 106 |  +8.1 |
|    16 |   8 853 |   9 055 |  +2.3 |
|    17 |  37 220 |  35 388 |  -4.9 |
|    18 |  45 957 |  47 979 |  +4.4 |
|    19 |  11 900 |  12 559 |  +5.5 |
|    20 |  14 733 |  16 274 | +10.5 |
|    21 |  35 287 |  37 144 |  +5.3 |
|    22 |   5 770 |   7 370 | +27.7 |

## Observations

V2 is generally slower than V1 today. The two queries V2 wins on (Q1, Q17) are
both heavy aggregations where the per-row work dominates and the V2 pushdown
makes a small dent. Otherwise V2 pays for things the optimisations
*meant to land later in the stack* haven't shipped yet:

- **Q9, Q22** are the long-standing regression candidates. Q9 is +39%, Q22 is
  +28%. Both touch many fields and/or complex multi-conjunct filters. The
  expected fix is re-enabling `StructPlan`/`ProjectPlan` mask pushdown — was
  disabled in PR4 because per-field mask re-evaluation was paid every time;
  CSE + streaming Let should make re-enabling safe.
- **Q2, Q6, Q11** are smaller queries (< 12 ms) where overhead from the v2
  scan-construction path is a larger relative share.
- The "long pole" queries (Q17, Q18, Q21) are within ±5%, indicating per-row
  decode cost is similar; the gap is in scan setup / filter wiring.

## What's next (changes that should move these numbers)

1. **Re-enable `StructPlan` + `ProjectPlan` `try_pushdown_mask`** — should
   recover most of the Q9/Q22 regression once CSE collapses the per-field
   duplicate work.
2. **Lockstep mask consumption in `FlatPlan`** — should trim Q6/Q14 (filter-
   heavy single-column scans) by avoiding speculative reads beyond the mask.
3. **Predicate / projection-expression pushdown into a separate node** — the
   biggest expected win. Lets CSE see "read field a" as the shared subtree
   between filter and projection; today the leaf bakes the expression and CSE
   sees nothing to share.
4. **Sub-segment reads (`FuseFilterIntoFlat`)** — the I/O win. Reduces bytes
   read for selective filters; should help Q6/Q14/Q19.

## Re-running this baseline

After any change that should move numbers, append a new section below with the
date, HEAD, and a fresh table. Don't overwrite the historical baseline above —
later comparisons want the original numbers as a reference point.
