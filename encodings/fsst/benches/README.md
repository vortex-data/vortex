<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# FSSTView benchmarks

`FSSTView` is a ListView-style FSST: it addresses its compressed codes with separate
`offsets` + `sizes` arrays instead of a single monotonic offsets array. That makes
`filter` / `take` / `slice` **metadata-only** (they rewrite only the small
offsets/sizes/lengths/validity arrays and reuse the compressed byte heap), whereas plain
`FSST` delegates those ops to `VarBin` and **rewrites the whole compressed code heap** each
time. The cost moves to a single canonicalization (decode → `VarBinViewArray`) at the end.

These benchmarks quantify that trade-off. All numbers are divan **medians**, 100 samples, on
one shared machine — treat them as directional; the relative ordering is stable. `fsst` =
stay in `FSST` (rewrite heap per op); `view` = convert to `FSSTView`, metadata-only ops,
decode once.

## 1. `fsst_view_compute` — synthetic shapes

Self-contained (no external data). ~2 MiB of synthetic strings in two shapes — `ManyShort`
(~12 B) and `FewLong` (~256 B) — with a clustered 10 % filter and a sorted take. Two
workloads, each ending in a `VarBinViewArray`:

- `single_filter_{fsst,view}` — one filter, then canonicalize.
- `chain_{fsst,view}` — convert once, then 5 alternating filter/take ops, canonicalize once
  (the case the view is designed for).

| workload | shape | fsst | view | speedup |
| --- | --- | --- | --- | --- |
| single_filter | ManyShort | 0.63 ms | 0.62 ms | ~1× |
| single_filter | FewLong | 65 µs | 53 µs | 1.2× |
| chain (5 ops) | ManyShort | 4.99 ms | 4.12 ms | 1.2× |
| chain (5 ops) | FewLong | 371 µs | 268 µs | 1.4× |

Takeaway: the gap widens with chain length, because each `fsst` op re-rewrites the heap while
the view stays metadata-only and defers the single decode.

## 2. `fsst_view_fineweb` — real columns

Two real columns from the HuggingFace FineWeb 10BT sample: `url` (200 k rows, ~72 B avg) and
`text` (40 k rows, ~3 KB avg). The ~2 GB sample is not downloaded; columns are extracted once
with DuckDB into length-prefixed dumps (see the bench module docs). No-ops unless `FINEWEB_URL`
/ `FINEWEB_TEXT` point at the files. Same two workloads as above.

| workload | column | fsst | view | speedup |
| --- | --- | --- | --- | --- |
| single_filter | url | 1.02 ms | 0.84 ms | 1.2× |
| single_filter | text | 5.81 ms | 4.38 ms | 1.3× |
| chain (5 ops) | url | 6.23 ms | 3.95 ms | 1.6× |
| chain (5 ops) | text | 44.2 ms | **5.16 ms** | **8.6×** |

Takeaway: on real data the view wins every case, and decisively for chained ops over long
strings — `fsst` rewrites the ~hundreds-of-MB code heap on every op; the view decodes once.

## 3. `fsst_view_fineweb_queries` — real query predicates

The actual `vortex-bench` FineWeb queries are `SELECT * FROM fineweb WHERE <predicate>`. Each
predicate is evaluated once in DuckDB against the real sample to produce an authentic per-row
selection mask (recipe: `benches/fineweb_queries_extract.py`); the bench applies that mask to
the FSST-compressed `url`/`text` column and decodes to a `VarBinViewArray`. This is the
materialization half of a real query. No-ops unless `FINEWEB_DIR` points at the dumps.

Mask shapes vary by predicate (over 200 k rows): `dump_eq` 7 %/177 runs and `date_prefix`
12 %/178 runs are clustered; `google_or` 2 %/4046 runs is scattered; `vortex`/`espn` are
~0.04–0.09 % and tiny.

| query (selectivity) | column | fsst | view | winner |
| --- | --- | --- | --- | --- |
| date_prefix (12 %) | text | 63.4 ms | 43.9 ms | view 1.4× |
| dump_eq (7 %) | text | 40.9 ms | 26.0 ms | view 1.6× |
| google_or (2 %) | text | 26.8 ms | 21.4 ms | view 1.25× |
| dump_eq (7 %) | url | 1.13 ms | 0.94 ms | view 1.2× |
| date_prefix (12 %) | url | 1.67 ms | 1.36 ms | view 1.2× |
| google_or (2 %) | url | 407 µs | 468 µs | fsst |
| google_and (0.19 %) | url | 30 µs | 164 µs | fsst |
| vortex (0.04 %) | url | 8 µs | 140 µs | fsst |

Takeaway — two regimes:

- **Bulk-ish selections, and anything on the long `text` column → view wins (1.25–1.6×)** by
  skipping the per-op heap rewrite. These are the queries that take tens of milliseconds.
- **Tiny, highly selective predicates on the short `url` column → fsst wins.** `fsst`'s filter
  rewrites an almost-empty heap (cheap), while the view pays a fixed ~130 µs floor:
  `fsstview_from_fsst` walks all 200 k offsets to derive the `sizes` array even though the
  predicate keeps <0.2 % of rows. Both are sub-millisecond there, so it rarely matters, but it
  is the view's one real weakness — converting the whole column ahead of a very selective
  filter is wasted work.

## How `Auto` chooses the decode

Canonicalization picks a decode strategy from the survivor layout (`FsstViewCompaction::Auto`):

- **Direct** — survivors are one contiguous run (untouched / sliced): one bulk decode, no copy.
- **RunDecode** — offsets still monotonic with few runs (clustered/range filters, sorted
  takes): decode each contiguous run straight into the element-ordered output, no gather copy.
- **GatherBulk** — scattered (shuffle take) or heavily fragmented (uniform-random filter):
  compact the live codes into one buffer, then a single bulk decode.

The threshold (`runs <= len / 4` → RunDecode, else GatherBulk) was calibrated with the
synthetic `fsst_view_compute` shapes.
