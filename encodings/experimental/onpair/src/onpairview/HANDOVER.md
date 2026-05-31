# OnPairView — Handover

Branch: `ji/onpair-view` (off `onpair-encoding`, **not** `develop`/`claude/onpair-encoding-features-NWETX`).
Last commit at handover: `cd939bd`. All work committed + pushed; tree clean.

> ⚠️ The designated branch `claude/onpair-encoding-features-NWETX` is just a copy of
> `develop` with **no OnPair code**. The real OnPair encoding lives on `onpair-encoding`
> at `encodings/experimental/onpair/`, which is what this work is based on.

---

## What this is

`OnPairView` — the **ListView-shaped sibling of the `OnPair`** string encoding
(`encodings/experimental/onpair/src/onpairview/`). Both are Utf8/Binary encodings
backed by the same OnPair dictionary + flat `codes` token stream; the difference
is how a row addresses its tokens:

| | `OnPair` (List-shaped) | `OnPairView` (ListView-shaped) |
|---|---|---|
| row → tokens | one monotonic `codes_offsets` (`codes[off[i]..off[i+1]]`) | `codes_offsets` **+** `codes_sizes` (`codes[off[i]..off[i]+size[i]]`) |
| windows | contiguous, sorted | arbitrary, out-of-order, may overlap |
| filter/take | **rebuild** surviving codes — O(tokens) | **metadata-only** — rewrite tiny per-row children, share `codes` — O(rows) |
| cost model | pays compaction **eagerly** every op | defers it to **one gather** at export |

Slots are declared with `#[array_slots(OnPairView)]`:
`dict_offsets, codes, codes_offsets, codes_sizes, uncompressed_lengths, validity?`.

## Public API (`pub use` from the crate)

- `OnPairView`, `OnPairViewArray`, `OnPairViewArrayExt`, `OnPairViewArraySlotsExt`
- `OnPairView::from_onpair(&OnPairArray, ctx)` — cheap reshape List→ListView.
- `OnPairView::try_new(...)` / `new_unchecked(...)`
- `canonicalize_with(view, OnPairViewDecodeMode, ctx)` → `VarBinViewArray` (canonical).
- `canonicalize_to_varbin(view, ctx)` → `VarBinArray` (compact, no per-row views).
- `compact(view, ctx)` → dense `OnPairViewArray` (drops dead codes; `ListView::rebuild` analog).
- `onpair_take_compact(&OnPairArray, indices, ctx)` — compacting OnPair baseline (in `convert.rs`).
- `OnPairViewDecodeMode { Auto, SpanWithDead, Gather }` — `Auto` = always gather; others opt-in.

Kernels implemented: `FilterKernel`, `TakeReduce`/`TakeExecute`, `SliceReduce`,
`CastReduce`, `scalar_at`, `execute` (canonicalize), serde (`OnPairViewMetadata`).

## Decode strategy (export to VarBinView) — `canonical.rs`

`onpair::decompress_into` decodes one **contiguous** codes slice; `build_views`
splits the output into rows from `uncompressed_lengths`. So export must feed the
decoder the row-ordered concatenation of live tokens:

1. `analyze` — one O(rows) scan → `Layout { span_decodable, base, end, live_tokens }`.
2. `decode_compact_bytes` / `compact_span_codes` (shared with `compact`):
   - **contiguous** windows → slice `codes[base..end]`, decode directly (no copy);
   - **fragmented** (filtered/reordered) → gather live windows into a contiguous
     buffer, decode once.
3. `build_views` → views.

**`Auto` always gathers.** The "decode span + dead values" path (`SpanWithDead`)
loses at every gap density and is kept only as an opt-in experiment.

## Benchmarks — `benches/view_compute.rs`

`cargo bench -p vortex-onpair --bench view_compute` (divan; filter `-- <name>`).
Corpora ≈ 2 MB: `ManyShort` (~493k short rows) and `FewLong` (~18k long rows).
Phases: `*_ops` (filter/take), `*_canonicalize` (export), `filter_export_{varbin,varbinview}`.

## Measured results (median, current code)

**Compute — OnPairView wins everywhere, grows with payload size:**

| op (ManyShort) | OnPair | OnPairView | speedup |
|---|---|---|---|
| Filter non-sel (90%) | 3.48 ms | 0.79 ms | 4.4× |
| Take non-sel (all) | 5.22 ms | 0.15 ms | ~32× |
| Take shuffle | 11.3 ms | 1.66 ms | 6.8× |
| Filter→Take | 4.46 ms | 0.81 ms | 5.5× |

(Direct-children filter/take — commit `29cd352` — gave ~2.7× on take vs the old
ListView round-trip.)

**Export — OnPair wins (it pre-compacted); OnPairView gathers:**
OnPair ~1.4–2.3× faster, widest for shuffled/scattered results.

**End-to-end (compute+export):** OnPairView wins for chained/large/shuffled
(Filter→Take **3×**, Take shuffle 1.23×); OnPair wins single-op-then-export.

**VarBin vs VarBinView export** (after `new_unchecked`, `9b08f55`): VarBin is
~7–35% faster for short strings (no per-row 16-byte views).

## When to use which (the decision rule)

- **OnPairView**: chained filter/take, non-selective/shuffle over large payloads,
  results consumed without full materialization (filter→count), cheap lazy views.
- **OnPair**: a single (esp. selective) op immediately exported; smallest
  materialized footprint by default; repeated reads of a result (or `compact()` the view).

## Negative results — DO NOT RE-TRY (all measured + reverted)

1. **Span-decode with dead values** (`SpanWithDead`): loses to gather at every gap
   density (gather is O(live); span pays O(span) dead-token decode + a `byte_at`
   prefix). Disabled in `Auto` (`93085d6`).
2. **Streaming per-run decode, no gather buffer** (`a5e5361`→revert `5092780`):
   **3.5–16× slower**. `onpair::decompress_into` is a throughput decoder with high
   per-call cost; many small calls destroy amortization. The gather memcpy is far
   cheaper than the per-call overhead it would save.
3. **Cache tiling decode+build** (`22937fe`→revert `fdf4b9d`): **+14–44%**, and
   **flat across 1k–32k codes** (no cache sweet spot). The decoded bytes are
   accessed sequentially and prefetched; the dominant cost is per-row `make_view`
   CPU, which tiling can't reduce.

**Root cause across all three:** export is bound by (a) `build_views` per-row CPU
(`make_view`) and (b) the decoder's throughput design — neither is helped by
streaming/tiling/locality tricks.

## The one cross-cutting optimization landed: `make_view` (`9814809`, `cd939bd`)

`vortex-array/src/arrays/varbinview/view.rs` — `BinaryView::make_view`. The inline
path was a `match value.len()` over 13 lengths → on x86 a **jump-table indirect
branch** that mispredicts on every length change (hot for short strings, used by
**every** string encoding's canonicalization, FSST included). Replaced with
`copy_inline` (overlapping fixed-width copies): **no jump table, no `memcpy` call**.
Verified via `--emit asm`: inline fill 108→~40 instructions, indirect jump removed.
This is the highest-leverage export change because `build_views` is ~60% of export
for short strings. (Removed the now-dead `Inlined::new::<N>`.)

## Where the remaining export ceiling is

`build_views` in `vortex-array` (~60% of short-string export) is per-row
`make_view` CPU. The only further lever is a **bulk/SIMD `build_views`** in
`vortex-array` (compute all offsets, then a vectorized inline-vs-ref split) — a
cross-crate change that speeds up all string encodings. Not yet done.

## NOT done (intentionally, not requested)

- Session registration in `vortex-file` behind `unstable_encodings` + a serde
  round-trip test (would make OnPairView a first-class registered encoding rather
  than direct-kernel/`canonicalize_*`-only).
- The bulk `build_views` in `vortex-array`.

## Verify

```bash
cargo build -p vortex-onpair
cargo test  -p vortex-onpair --lib          # 25 pass
cargo test  -p vortex-array  --lib varbinview  # 60 pass
cargo clippy -p vortex-onpair --all-targets
cargo clippy -p vortex-array  --lib
cargo bench -p vortex-onpair --bench view_compute -- --sample-count 12 --sample-size 4
```

Sandbox notes: no `perf`/`samply` (used env-gated phase timers + `rustc --emit asm`
for the analysis); `RUSTC_WRAPPER=` needed if sccache errors; install nightly
rustfmt for `cargo +nightly fmt`.
