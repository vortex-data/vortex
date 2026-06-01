# FSSTView — Handover

## TL;DR

Added a new **`FSSTView`** array encoding to Vortex: a ListView-style FSST that addresses its
compressed codes with separate per-element `offsets` + `ends` arrays instead of one monotonic
offsets array. This makes `filter` / `take` / `slice` **metadata-only** (rewrite only small index
arrays, reuse the compressed byte heap), where plain `FSST` rewrites the whole compressed code heap
per op. The decode cost moves to a single canonicalization at the end.

Storing the per-element **end offset** (rather than the size) makes the `FSST` → `FSSTView`
conversion allocation-free — both addressing arrays are zero-copy slices of the FSST's existing
offsets — which **eliminated the conversion floor** that previously made the view 9–16× slower than
`fsst` on tiny highly selective `url` predicates (see "Conversion floor — resolved" below).

- **Branch:** `claude/fsstview-conversion-floor-kRAeg` (built on the original
  `claude/fsstview-array-listview-TdW45`).
- **Status:** merge-ready. 105 tests pass, `clippy --all-targets --all-features` clean,
  `cargo +nightly fmt` clean, `vortex-file` builds, doc tests pass.
- **No PR opened yet** (was waiting on explicit request).
- **Scope:** additive, contained in `encodings/fsst/` plus 2 registration lines in `vortex-file`.

## What landed

New encoding `vortex.fsstview` in `encodings/fsst/src/fsstview/`:

| file | role |
| --- | --- |
| `array.rs` | encoding struct, `#[array_slots]` children (uncompressed_lengths, codes_offsets, codes_ends, codes_validity), VTable, serde, allocation-free `fsstview_from_fsst` conversion |
| `compute.rs` | metadata-only `FilterKernel` + `TakeExecute` |
| `ops.rs` | `scalar_at` |
| `slice.rs` | metadata-only `SliceReduce` |
| `from_fsst.rs` | `fsst_filter_to_view` / `fsst_take_to_view` helpers |
| `canonical.rs` | decode → `VarBinViewArray` / `VarBinArray`, with the `Auto` export strategy |
| `kernel.rs` / `rules.rs` | parent kernel + rule registration |
| `tests.rs` | conformance + agreement + nullable/gapped/RunDecode coverage + zero-copy conversion guard |

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

Two benches in `encodings/fsst/benches/` (full write-up in `benches/README.md`). All numbers are
divan **medians**, 100 samples, single shared machine — directional, relative ordering stable.

1. **`fsst_view_compute`** — synthetic, no external data, **runs in CI**. ~2 MiB strings, ManyShort
   (~12 B) / FewLong (~256 B). Single filter and a 5-op chain → VarBinView. The chain is where the
   view's advantage compounds (each `fsst` op re-rewrites the heap; the view stays metadata-only):
   - chain FewLong: fsst 371 µs → view **268 µs** (1.4×); chain ManyShort 4.99 ms → **4.12 ms**.

2. **`fsst_view_fineweb_queries`** — the real `vortex-bench` query predicates (`dump = ...`,
   `date LIKE '2020-10-%'`, `url/text LIKE '%google%'`, `'% vortex %'`, espn filters), evaluated
   in DuckDB to authentic per-row masks, then materialize the column → VarBinView. Numbers below
   are a same-machine before/after (old `sizes` representation → new `ends` representation):
   - text/date_prefix (12%): fsst 69.3 ms vs view **41.4 ms** (1.67×; was 41.0 ms — held)
   - text/dump_eq (7%): fsst 42.6 ms vs view **25.3 ms** (1.68×; was 25.3 ms — held)
   - url/vortex (0.04%): fsst 8.6 µs vs view **9.1 µs** (was view 140 µs — floor removed)
   - url/espn_and (0.08%): fsst 14.5 µs vs view **14.9 µs** (was view 146 µs)
   - text/espn_and (0.08%): fsst 284 µs vs view **271 µs** (was view 407 µs — flips to a view win)

With the `ends` representation the view now **wins or ties every query** in the matrix: the bulk /
clustered / long-`text` cases still win by skipping the per-op heap rewrite (up to 1.68× here), and
the tiny highly selective predicates that used to lose to the conversion floor now match `fsst` to
within noise. Full table in `benches/README.md`.

### Reproducing the FineWeb queries bench

The ~2 GB sample is **not** downloaded by the bench. Extract columns + query masks once:

```bash
pip install duckdb
python3 encodings/fsst/benches/fineweb_queries_extract.py     # writes /tmp/fw_*.bin
FINEWEB_DIR=/tmp cargo bench -p vortex-fsst --bench fsst_view_fineweb_queries
```

The bench no-ops (CI-safe) when `FINEWEB_DIR` is unset.

## Conversion floor — resolved

The view's one previous weakness was a **fixed conversion cost on highly selective filters**: the
original `fsstview_from_fsst` derived a full `sizes` array (`offsets[i+1] - offsets[i]` over all
rows) even when a predicate kept <1% of rows. Samply + cachegrind had pinned this as the top
wall-clock cost (~130–150 µs floor) on the `url`-selective queries — a memory-bandwidth-bound loop
streaming `len * 8` bytes.

**Fix (this branch): store the end offset, not the size.** `codes_sizes` was replaced by
`codes_ends`, where `codes_ends[i] = codes_offsets[i] + size[i]`. Because a freshly converted heap
is contiguous (element `i` occupies `offsets[i]..offsets[i+1]`), **both** addressing arrays are now
zero-copy slices of the FSST's existing monotonic offsets buffer
(`codes_offsets = offsets[0..len]`, `codes_ends = offsets[1..len+1]`). The conversion allocates and
copies nothing; no per-row `sizes` array is materialized, so a selective `filter`/`take` never pays
to derive sizes for the rows it discards. The per-element size is recovered as
`codes_ends[i] - codes_offsets[i]` only at canonicalize / `scalar_at`, over the survivors only.

This keeps `filter`/`take`/`slice` metadata-only and composable across a chain (they carry
`codes_ends` alongside `codes_offsets`); the conversion is **not** fused into the filter. Measured
result (same-machine before/after, `fsst_view_fineweb_queries`): `url/vortex` 140 µs → **9.1 µs**,
`url/espn_and` 146 µs → **14.9 µs**, and the previously winning clustered cases (`text/dump_eq`,
`text/date_prefix`) held flat. The view now wins or ties every query in the matrix.

A regression guard (`conversion_shares_offsets_buffer_zero_copy` in `tests.rs`) asserts the
structural invariant the fix relies on: a freshly converted view's `codes_ends` slice begins exactly
one element past `codes_offsets` in the *same allocation*. This catches a silent revert to a
size-materializing conversion — which the value/agreement tests would not, since the decoded values
would still match — without depending on the FineWeb bench (gated out of CI).

The alternative follow-up (store `sizes` in the narrowest int width) was considered and rejected:
it only halves the *write* traffic, leaving the unavoidable full read of the offsets — whereas the
`ends` representation removes the whole O(rows) pass. Narrowing widths is orthogonal and can still
be layered on the file layer's compression if desired.

## Verification commands

```bash
cargo nextest run -p vortex-fsst          # (or cargo test -p vortex-fsst) — 105 pass
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
