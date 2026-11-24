<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# vx-analyze: reports over vx-lab datasets

`vx-analyze` reads a dataset written by `vx-lab materialize` and answers four questions: how much
better than today's compressor a per-chunk choice could be (headroom), where today's compressor
loses (regret), what fixed trial sets and cheap features can recover, and how that changes with
bandwidth.

```bash
cd benchmarks/compressor-lab/analysis
uv run vx-analyze report <dataset-dir> --out report.html      # one self-contained HTML file
uv run vx-analyze diff <base-dataset> <head-dataset>          # headroom and production cost, two commits
uv run vx-analyze oracle <dataset-dir> --out oracle.parquet   # per-site oracle picks: training labels
```

## Cost

```text
cost = compressed bytes / bandwidth + full decode time
```

Compress time is reported beside every result but never scored. A preset is a bandwidth, kept in
`src/vx_analyze/presets.toml` (`size` 100 MB/s, `s3-scan` 1 GB/s, `nvme` 4 GB/s, `memory` 32 GB/s),
and every report also sweeps 100 MB/s to 32 GB/s. `--presets <file>` swaps the set; `--primary`
picks the one used for regret, misses and features; `--exclude forced/pco` drops candidates.

Candidates that produce the same encoding (same size and tree) share one decode time, the median
of their samples, so timing noise never makes a duplicate of production look better than it.

## Report sections

1. **Data health:** coverage per source and dtype, failures, timing spread, and whether each
   `forced/<scheme>` candidate really put its scheme at the root.
2. **Headroom:** oracle vs production per preset, byte-weighted with per-chunk figures beside it,
   a 90% bootstrap interval over columns, ratio, decode throughput and compress time.
3. **Regret:** the distribution of production's regret per site and the columns that lose most.
4. **Why it misses:** production's root vs the oracle's root, weighted by regret, and the
   calibration of closed-form size estimates.
5. **Fixed trial sets:** production plus k greedily chosen candidates, keep the cheapest; scored
   held out.
6. **Features:** held-out gradient-boosted models per feature tier budget, permutation importance
   in cost terms, and a depth-3 tree to read.
7. **Frontier:** compression ratio vs decode throughput for each preset.

Held-out scoring is leave-one-source-out when a dataset has at least three sources, otherwise
five folds grouped by column.

## Inputs

Both layouts are read and normalised to one row per (chunk, site, candidate):

- today's CSVs: `rows-<tag>.csv`, `features-<tag>.csv`, `dataset.json`;
- Parquet: `rows/**/*.parquet` and `features/**/*.parquet` (hive partitions allowed), or
  `rows*.parquet` and `features*.parquet` at the top level.

Columns the CSVs lack default to a single root site: `site` (`root`), `depth` (0),
`parent_scheme`, and `dtype` (derived from `ptype`). When vx-lab writes per-site rows, add
`site`, `depth` and `parent_scheme` to both tables and every section splits by them with no
code change. Feature tiers are in `load.FEATURE_TIERS`; unknown features count as tier 5.

## Tests

```bash
uv run --group dev pytest
```
