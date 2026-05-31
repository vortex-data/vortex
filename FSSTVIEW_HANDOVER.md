# FSSTView — Handover

## TL;DR

Added a new **`FSSTView`** array encoding to Vortex: a ListView-style FSST that addresses its
compressed codes with separate `offsets` + `sizes` arrays instead of one monotonic offsets array.
This makes `filter` / `take` / `slice` **metadata-only** (rewrite only small index arrays, reuse
the compressed byte heap), where plain `FSST` rewrites the whole compressed code heap per op. The
decode cost moves to a single canonicalization at the end.

- **Branch:** `claude/fsstview-array-listview-TdW45` (17 commits ahead of `develop`, pushed).
- **Status:** merge-ready. 107 tests pass, `clippy --all-targets --all-features` clean,
  `cargo +nightly fmt` clean, `vortex-file` builds, doc tests pass.
- **No PR opened yet** (was waiting on explicit request).
- **Scope:** additive, contained in `encodings/fsst/` plus 2 registration lines in `vortex-file`.

## What landed

New encoding `vortex.fsstview` in `encodings/fsst/src/fsstview/`:

| file | role |
| --- | --- |
| `array.rs` | encoding struct, `#[array_slots]` children (uncompressed_lengths, codes_offsets, codes_sizes, codes_validity), VTable, serde, `fsstview_from_fsst` conversion |
| `compute.rs` | metadata-only `FilterKernel` + `TakeExecute` |
| `ops.rs` | `scalar_at` |
| `slice.rs` | metadata-only `SliceReduce` |
| `from_fsst.rs` | `fsst_filter_to_view` / `fsst_take_to_view` helpers |
| `canonical.rs` | decode → `VarBinViewArray` / `VarBinArray`, with the `Auto` export strategy |
| `kernel.rs` / `rules.rs` | parent kernel + rule registration |
| `tests.rs` | conformance + agreement + nullable/gapped/RunDecode coverage |

Registered in `vortex-file/src/lib.rs` (`register_default_encodings`). Public API:
`FSSTView`, `FSSTViewArray`, `FsstViewCompaction`, `canonicalize_fsstview_with`,
`canonicalize_fsstview_to_varbin`, `fsst_filter_to_view`, `fsst_take_to_view`, `fsstview_from_fsst`.

## Canonicalization strategy (`FsstViewCompaction::Auto`)

After metadata-only ops the survivors are scattered in the original heap; `Auto` picks how to
decode from the survivor layout:

- **Direct** — one contiguous run (untouched / sliced): single bulk decode, no copy.
- **RunDecode** — offsets monotonic, few runs (clustered/range filters, sorted takes): decode each
  contiguous run straight into the element-ordered output, no gather copy. Threshold:
  `runs <= len / 4`.
- **GatherBulk** — scattered (shuffle take) or fragmented (uniform-random filter): compact live
  codes into one buffer, single bulk decode.

`RunDecode` and the gather coalescing came from the optimization work; `PerElement` and
`RunCoalesce` were explored, proven worse, and removed before merge.

## Benchmarks & results

Three benches in `encodings/fsst/benches/` (full write-up in `benches/README.md`). All numbers are
divan **medians**, 100 samples, single shared machine — directional, relative ordering stable.

1. **`fsst_view_compute`** — synthetic, no external data. ~2 MiB strings, ManyShort (~12 B) /
   FewLong (~256 B). Single filter and a 5-op chain → VarBinView.
   - chain FewLong: fsst 371 µs → view **268 µs** (1.4×); chain ManyShort 4.99 ms → **4.12 ms**.

2. **`fsst_view_fineweb`** — real FineWeb `url` (200k × ~72 B) and `text` (40k × ~3 KB) columns.
   - single_filter text: 5.81 ms → **4.38 ms** (1.3×)
   - chain text: 44.2 ms → **5.16 ms** (**8.6×**) ← headline
   - chain url: 6.23 ms → **3.95 ms** (1.6×)

3. **`fsst_view_fineweb_queries`** — the real `vortex-bench` query predicates (`dump = ...`,
   `date LIKE '2020-10-%'`, `url/text LIKE '%google%'`, `'% vortex %'`, espn filters), evaluated
   in DuckDB to authentic per-row masks, then materialize the column → VarBinView.
   - text/date_prefix (12%): 63.4 ms → **43.9 ms** (1.4×)
   - text/dump_eq (7%): 40.9 ms → **26.0 ms** (1.6×)
   - url/vortex (0.04%): fsst **8 µs** vs view 140 µs

**Two regimes:** the view wins everywhere the work is non-trivial (long `text` column, chained
ops, bulk selections) — up to 8.6×. It loses only on tiny highly-selective predicates over the
short `url` column, where it pays a fixed ~130 µs floor (the conversion walks all 200k offsets to
build `sizes` even though <0.2% survive). Those cases are all sub-millisecond.

### Reproducing the FineWeb benches

The ~2 GB sample is **not** downloaded by the benches. Extract columns + query masks once:

```bash
pip install duckdb
python3 encodings/fsst/benches/fineweb_queries_extract.py     # writes /tmp/fw_*.bin
FINEWEB_DIR=/tmp cargo bench -p vortex-fsst --bench fsst_view_fineweb_queries
# for the column bench:
FINEWEB_URL=/tmp/fw_url.bin FINEWEB_TEXT=/tmp/fw_text.bin \
  cargo bench -p vortex-fsst --bench fsst_view_fineweb
```

Benches no-op (CI-safe) when the env vars are unset.

## Known limitation / next step

The view's one weakness is the **fixed conversion cost on highly selective filters**:
`fsstview_from_fsst` derives the full `sizes` array (`offsets[i+1] - offsets[i]` over all rows)
even when a predicate keeps <1% of rows. Confirmed with samply + cachegrind: the conversion is the
top wall-clock cost on the `url`-selective queries (~130 µs floor), and the loop is already
SIMD-vectorized and memory-bandwidth-bound (it streams `len * 8` bytes for i64 offsets/sizes).

Possible follow-ups, **not done** (would need care + their own benchmarks):
- Defer / lazily represent `sizes` so a selective filter doesn't materialize it for discarded rows.
- Store `sizes` in the narrowest int width (values are small; offsets are i64), cutting the
  conversion's memory traffic.

Both touch the representation that `filter`/`take` operate on, so they are not drop-in.

## Verification commands

```bash
cargo nextest run -p vortex-fsst          # (or cargo test -p vortex-fsst) — 107 pass
cargo clippy -p vortex-fsst --all-targets --all-features
cargo clippy -p vortex-file
cargo +nightly fmt --all
```

## Methodology notes (for whoever continues)

- `perf` is unavailable in the dev sandbox (kernel mismatch). Use **samply** (set
  `/proc/sys/kernel/perf_event_paranoid` to 1) for wall-clock sampling and **cachegrind** for
  cache/instruction modeling. Build the profiled example with
  `RUSTFLAGS="-C force-frame-pointers=yes -C debuginfo=2"` and resolve addresses with `addr2line`.
- Caution learned the hard way: **instruction count is not time.** A 12× instruction-count
  reduction in the conversion barely moved wall-clock; always confirm with a sampling profiler and
  a realistic workload (real FineWeb columns, real query masks), not synthetic micro-loops.
