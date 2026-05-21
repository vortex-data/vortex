# ⚠️ PRELIMINARY — OnPair vs nvCOMP, GH200 vs B200

> **PRELIMINARY (2026-05-21).** Unlocked GPU clocks (±~1–5% on absolutes), single runs, NCU unavailable in-container (`ERR_NVGPUCTRPERM`). Re-measure with locked clocks before quoting.

> Source of truth: `b200_results.csv` / `.json` (regen `gen_b200_tables.py`). OnPair rows read live from `summary.json`; nvCOMP HW from `nvcomp_hw_bench.cu` (chunk 256 KiB).

> Decode/compress = GiB/s over uncompressed bytes, 100 iters. nvCOMP HW = Blackwell hardware Decompression Engine, byte-exact. Compression ratio is hardware-independent.

> **Each nvCOMP HW codec has two presets:** `hi` = max ratio (Deflate algo5); `fast` = best (de)compression throughput (Deflate algo0; LZ4 is single-pass, no level).


## 1. OnPair — B200, all columns (best kernel)

| dataset/column | bits | ratio | compress GiB/s | decode GiB/s | decode GB/s | kernel |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| clickbench/URL | 12 | 2.89× | 0.029 | 794.2 | 852.8 | 4tpt_wpb8_occ |
| clickbench/URL | 16 | 3.86× | 0.032 | 955.9 | 1026.4 | 4tpt_wpb8_occ |
| dbtext/email | 12 | 2.08× | 0.009 | 164.8 | 177.0 | 4tpt_wpb8_occ |
| dbtext/email | 16 | 1.95× | 0.007 | 154.6 | 166.0 | 4tpt |
| dbtext/hex | 12 | 1.22× | 0.005 | 74.4 | 79.9 | s4l1_8tpt |
| dbtext/hex | 16 | 1.12× | 0.004 | 72.0 | 77.3 | 4tpt_split8_wpb8_occ |
| dbtext/l_comment | 12 | 3.82× | 0.018 | 206.5 | 221.7 | 2tpt |
| dbtext/l_comment | 16 | 3.65× | 0.019 | 206.1 | 221.3 | 4tpt_wpb8_occ |
| dbtext/ps_comment | 12 | 5.50× | 0.024 | 221.0 | 237.3 | 4tpt |
| dbtext/ps_comment | 16 | 5.14× | 0.028 | 228.1 | 244.9 | 4tpt_ldcs |
| dbtext/yago | 12 | 1.59× | 0.006 | 140.9 | 151.3 | 4tpt_split8_wpb8 |
| dbtext/yago | 16 | 1.62× | 0.007 | 137.0 | 147.1 | 4tpt_split8read |
| fineweb/text | 12 | 2.24× | 0.021 | 613.7 | 659.0 | 4tpt_wpb8_occ |
| fineweb/text | 16 | 2.89× | 0.014 | 630.0 | 676.5 | 4tpt_wpb8_occ |
| tpch-sf10/l_comment | 12 | 4.17× | 0.040 | 1121.1 | 1203.8 | 4tpt_wpb8_occ |
| tpch-sf10/l_comment | 16 | 4.19× | 0.031 | 933.3 | 1002.1 | 4tpt_wpb8_occ |
| tpch-sf10/ps_comment | 12 | 6.23× | 0.072 | 1248.5 | 1340.6 | 4tpt_wpb8_occ |
| tpch-sf10/ps_comment | 16 | 5.82× | 0.074 | 1042.6 | 1119.5 | 4tpt_wpb8_occ |
| tpch-sf10/s_comment | 12 | 5.17× | 0.026 | 409.1 | 439.3 | 2tpt |
| tpch-sf10/s_comment | 16 | 4.67× | 0.026 | 411.0 | 441.3 | 4tpt_wpb8_occ |
| wikipedia/text | 12 | 2.15× | 0.014 | 587.7 | 631.0 | 4tpt_wpb8_occ |
| wikipedia/text | 16 | 2.80× | 0.016 | 612.7 | 657.9 | 4tpt_wpb8_occ |

## 2. nvCOMP hardware-engine presets — big columns (ratio · compress GiB/s · decode GiB/s)

| dataset/column | Deflate-hi (max ratio) | Deflate-fast (max throughput) | LZ4 (single) |
| --- | --- | --- | --- |
| clickbench/URL | 6.44× · 0.4 · 383 | 1.45× · 62.4 · 126 | 3.70× · 23.5 · 363 |
| fineweb/text | 2.55× · 0.5 · 170 | 1.71× · 64.4 · 126 | 1.54× · 10.9 · 188 |
| wikipedia/text | 2.70× · 0.5 · 176 | 1.67× · 80.6 · 124 | 1.64× · 8.7 · 194 |
| tpch-sf10/l_comment | 4.56× · 0.4 · 293 | 1.85× · 47.7 · 122 | 2.17× · 13.0 · 224 |
| tpch-sf10/ps_comment | 5.67× · 0.5 · 378 | 1.85× · 63.7 · 125 | 2.56× · 15.5 · 247 |

*Format: `ratio× · compress GiB/s · decode GiB/s`. Deflate-hi gives best ratio + fast decode but ~0.5 GiB/s compress; Deflate-fast compresses ~50–80 GiB/s but low ratio/decode; LZ4 is the balanced middle.*


## 2b. nvCOMP Zstd (CUDA backend — no HW path) presets (ratio · decode GiB/s)

| dataset/column | hi (level 3) | fast (level −10) | note |
| --- | --- | --- | --- |
| clickbench/URL | 5.64× · 112 | — | — |
| fineweb/text | 2.57× · 8 | — | frame-size artifact (long strings) |
| wikipedia/text | 2.74× · 1 | — | frame-size artifact (long strings) |
| tpch-sf10/l_comment | 2.87× · 84 | 1.79× · 95 | — |
| tpch-sf10/ps_comment | 4.16× · 37 | — | — |

*Zstd has no hardware-engine path (DE returns status 10). CUDA-backend decode is frame-size-sensitive: long-string columns (fineweb/wikipedia) collapse to <10 GiB/s because fixed values-per-frame makes huge frames. Compress is CPU-side (not comparable).*


## 3. Headline — OnPair vs best nvCOMP HW (big columns)

| dataset/column | OnPair best (ratio · decode) | Deflate-hi (ratio · decode) | OnPair decode advantage |
| --- | --- | --- | ---: |
| clickbench/URL | 3.86× · 956 (b16) | 6.44× · 383 | 2.5× |
| fineweb/text | 2.89× · 630 (b16) | 2.55× · 170 | 3.7× |
| wikipedia/text | 2.80× · 613 (b16) | 2.70× · 176 | 3.5× |
| tpch-sf10/l_comment | 4.17× · 1121 (b12) | 4.56× · 293 | 3.8× |
| tpch-sf10/ps_comment | 6.23× · 1248 (b12) | 5.67× · 378 | 3.3× |

*Deflate-hi beats OnPair on ratio for l_comment (4.56 vs 4.17) and clickbench URL (6.44 vs 3.86); OnPair wins ratio elsewhere and wins decode throughput everywhere.*


## 4. OnPair GH200 vs B200 (decode GiB/s)

| dataset/column | bits | ratio | GH200 | B200 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| book-reviews/text | 12 | — | 607 (4tpt_split8read) | not on B200 | — |
| fineweb/text | 12 | 2.24× | 567 (4tpt_split8read) | 614 | +8% |
| fineweb/text | 16 | 2.89× | 470 (4tpt) | 630 | +34% |
| tpch-sf10/ps_comment | 12 | 6.23× | 1117 (4tpt) | 1248 | +12% |
| tpch-sf10/ps_comment | 16 | 5.82× | 866 (4tpt) | 1043 | +20% |
| wikipedia/text | 12 | 2.15× | 538 (4tpt_split8read) | 588 | +9% |
| wikipedia/text | 16 | 2.80× | 538 (4tpt) | 613 | +14% |

## 5. Validation & known gaps

- **OnPair, 22/22**: read directly from `summary.json` (ratio=mem_ratio, decode=best, kernel=best) — no transcription.

- **nvCOMP HW**: all decodes byte-exact (`valid=YES`); ratios reproduce exactly, throughput ±~4% (clock noise).

- **GH200, 7/7**: match handover §5 exactly. book-reviews has no public source → not re-run on B200.

- **Deflate level matters**: SDK default algo=1 ("low ratio") understated ratio AND decode; presets above use algo5/algo0. Chunk size is not a ratio lever (Deflate window caps at 32 KiB).

- **Zstd CUDA** frame-size artifact on long-string columns; **Zstd HW unsupported** (status 10).

- **Caveat**: OnPair ratio = whole-column dictionary over 1000 MB chunks; nvCOMP ratio = batched 256 KiB chunks. Both realistic, different granularity.

- NCU mechanistic limiter analysis blocked (no `CAP_SYS_ADMIN`).

