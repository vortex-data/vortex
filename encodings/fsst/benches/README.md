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

## 2. `fsst_view_fineweb_queries` — real query predicates

The actual `vortex-bench` FineWeb queries are `SELECT * FROM fineweb WHERE <predicate>`. Each
predicate is evaluated once in DuckDB against the real sample to produce an authentic per-row
selection mask (recipe: `benches/fineweb_queries_extract.py`); the bench applies that mask to
the FSST-compressed `url`/`text` column and decodes to a `VarBinViewArray`. This is the
materialization half of a real query. No-ops unless `FINEWEB_DIR` points at the dumps.

Mask shapes vary by predicate (over 200 k rows): `dump_eq` 7 %/177 runs and `date_prefix`
12 %/178 runs are clustered; `google_or` 2 %/4046 runs is scattered; `vortex`/`espn` are
~0.04–0.09 % and tiny.

The `view (before)` column is the original representation, which derived a full `sizes` array in
`fsstview_from_fsst` (one i64 per row, materialized over **all** 200 k rows regardless of
selectivity). The `view` column stores the per-element **end offset** instead — a zero-copy slice
of the FSST's existing monotonic offsets — so the conversion allocates nothing and a selective
predicate never pays to derive sizes for the rows it discards (see "Conversion is allocation-free"
below). `fsst` is unchanged by this work; its small run-to-run drift is machine noise (the two
measurement runs were back-to-back on a shared machine).

| query (selectivity) | column | fsst | view (before) | view | winner |
| --- | --- | --- | --- | --- | --- |
| date_prefix (12 %) | text | 69.3 ms | 41.0 ms | **41.4 ms** | view 1.67× |
| dump_eq (7 %) | text | 42.6 ms | 25.3 ms | **25.3 ms** | view 1.68× |
| google_or (2 %) | text | 23.9 ms | 23.7 ms | **19.8 ms** | view 1.2× |
| google_and (0.19 %) | text | 708 µs | 782 µs | **642 µs** | view |
| vortex (0.04 %) | text | 529 µs | 606 µs | **456 µs** | view |
| espn_and (0.08 %) | text | 284 µs | 407 µs | **271 µs** | view |
| espn_or (0.09 %) | text | 650 µs* | 418 µs | **281 µs** | view |
| date_prefix (12 %) | url | 1.68 ms | 1.39 ms | **1.25 ms** | view 1.34× |
| dump_eq (7 %) | url | 1.11 ms | 944 µs | **881 µs** | view 1.25× |
| google_or (2 %) | url | 398 µs | 478 µs | **331 µs** | view 1.2× |
| google_and (0.19 %) | url | 30.2 µs | 173 µs | **28.7 µs** | view |
| espn_and (0.08 %) | url | 14.5 µs | 146 µs | **14.9 µs** | ~tie |
| espn_or (0.09 %) | url | 16.4 µs | 152 µs | **16.0 µs** | ~tie |
| vortex (0.04 %) | url | 8.6 µs | 140 µs | **9.1 µs** | ~tie |

(divan medians. `*` `text/espn_or` `fsst` was noisy that run — fastest 283 µs, mean 578 µs.)

Takeaway:

- **The conversion floor is gone.** Every highly selective `url` predicate that previously trailed
  `fsst` by 9–16× — it paid a fixed ~140 µs to walk all 200 k offsets building `sizes` even when
  <0.2 % of rows survived — now matches `fsst` to within noise (`url/vortex` 140 µs → **9.1 µs**,
  `url/espn_and` 146 µs → **14.9 µs**). The same floor that quietly taxed the *short selective
  `text`* predicates (`text/vortex`, `text/espn_*`, `text/google_and`) is also gone, flipping each
  of those from an `fsst` win to a `view` win.
- **The winning cases do not regress.** The clustered/bulk selections the view was already built
  for hold or improve: `text/dump_eq` and `text/date_prefix` stay at ~1.67–1.68× (the decode, not
  the conversion, dominates them), while `url/date_prefix`, `url/dump_eq`, and both `google_or`
  columns get a touch faster because the conversion no longer allocates.

With the floor removed the view now wins or ties **every** query in this matrix.

## Conversion is allocation-free

`FSSTView` stores the per-element **end offset** (`codes_ends[i] = offset[i] + size[i]`) rather
than the size. A freshly converted heap is contiguous, so element `i` occupies
`offsets[i]..offsets[i + 1]`, which means **both** addressing arrays are zero-copy slices of the
FSST's existing monotonic offsets buffer: `codes_offsets = offsets[0..len]` and
`codes_ends = offsets[1..len + 1]`. `fsstview_from_fsst` therefore allocates and copies nothing —
in particular it never materializes a per-row `sizes` array, so a selective `filter`/`take` that
keeps a handful of rows no longer pays an O(rows) cost to derive sizes for the rows it discards.
The per-element size is recovered as `codes_ends[i] - codes_offsets[i]` only where it is needed
(canonicalize / `scalar_at`), over the survivors only. `filter`/`take`/`slice` stay metadata-only
and compose across a chain exactly as before — they now carry `codes_ends` alongside
`codes_offsets` instead of `codes_sizes`.

## How `Auto` chooses the decode

Canonicalization picks a decode strategy from the survivor layout (`FsstViewCompaction::Auto`):

- **Direct** — survivors are one contiguous run (untouched / sliced): one bulk decode, no copy.
- **RunDecode** — offsets still monotonic with few runs (clustered/range filters, sorted
  takes): decode each contiguous run straight into the element-ordered output, no gather copy.
- **GatherBulk** — scattered (shuffle take) or heavily fragmented (uniform-random filter):
  compact the live codes into one buffer, then a single bulk decode.

The threshold (`runs <= len / 4` → RunDecode, else GatherBulk) was calibrated with the
synthetic `fsst_view_compute` shapes.
