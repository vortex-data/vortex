# OnPair decode-path optimization

This documents two changes to the OnPair **decompression** (canonicalisation)
path in `src/canonical.rs` + `src/decode.rs`, and the benchmark evidence for
them.

## What changed

The `OnPair → VarBinViewArray` path (`onpair_decode_views`) used to:

1. `execute` the per-row `uncompressed_lengths` child,
2. `OwnedDecodeInputs::collect` — materialise **all four** integer children
   (`dict_offsets`, `codes`, `codes_offsets`, `uncompressed_lengths`),
3. call `onpair::decompressed_len` to size the output buffer,
4. `decompress_into`, then `build_views`.

Two redundancies were removed:

1. **No second size pass.** `onpair::decompressed_len` re-walks *every token*
   in the `codes` stream, doing a random `dict_offsets` lookup per token, only
   to recompute the total output size. That size is exactly the sum of the
   per-row `uncompressed_lengths` we already materialise, so we now sum that
   (a sequential, auto-vectorisable pass over a buffer already in cache)
   instead.
2. **No unused child materialisation.** The contiguous decoder
   (`onpair::decompress_into`) walks the flat `codes` array directly and never
   reads the per-row `code_boundaries` (`codes_offsets`). A new
   `FullDecodeInputs` skips collecting that child entirely (which, for a
   cascaded/bit-packed `codes_offsets`, also avoids an extra child `execute`).

`OwnedDecodeInputs` is retained unchanged for `scalar_at`, which *does* decode
per row and needs `code_boundaries`.

## How it was measured

* `--bench decode` (`canonicalize_to_varbinview`) — synthetic corpora.
* `--bench real_data` — real string data: TPC-H columns generated in-memory via
  `tpchgen` (`l_comment`, `o_comment`, `c_comment`, `p_name`) and a ClickBench
  column (synthetic URL fallback, or a real parquet via `ONPAIR_BENCH_PARQUET`).

Before/after numbers come from running the *same* benchmark binary against the
`onpair-encoding` baseline source and the optimised source. All numbers are
divan medians, 100 samples, `bench` profile.

## Results — synthetic (`canonicalize_to_varbinview`)

| Case            | baseline | optimised | speedup |
| --------------- | -------- | --------- | ------- |
| HighCard, 100k  | 2.716 ms | 1.639 ms  | 1.66×   |
| Long, 100k      | 3.620 ms | 2.238 ms  | 1.62×   |
| Short, 100k     | 1.445 ms | 1.372 ms  | 1.05×   |
| UrlLog, 100k    | 2.004 ms | 1.408 ms  | 1.42×   |
| UrlLog, 1M      | 25.19 ms | 19.45 ms  | 1.30×   |

The gain tracks tokens-per-row: token-heavy corpora (HighCard, Long) win most
because the eliminated `decompressed_len` pass was proportional to total
tokens; `Short` (few tokens/row) barely moves.

## Results — real data (`real_data`, `onpair_decode` vs `fsst_decode`)

OnPair decode median, before vs after the optimization, with the FSST fast
baseline for comparison:

| Column     | baseline OnPair | optimised OnPair | OnPair speedup | FSST baseline | optimised OnPair vs FSST |
| ---------- | --------------- | ---------------- | -------------- | ------------- | ------------------------ |
| c_comment  | 1.633 ms        | 1.180 ms (4.61 GB/s) | 1.38×      | 1.707 ms      | **1.45× faster**         |
| clickbench | 6.697 ms        | 5.097 ms (3.29 GB/s) | 1.31×      | 6.630 ms      | **1.30× faster**         |
| l_comment  | 9.610 ms        | 7.787 ms (2.15 GB/s) | 1.23×      | 9.708 ms      | **1.25× faster**         |
| o_comment  | 6.664 ms        | 4.951 ms (3.39 GB/s) | 1.35×      | 6.629 ms      | **1.34× faster**         |
| p_name     | 1.520 ms        | 1.202 ms (2.72 GB/s) | 1.26×      | 1.466 ms      | **1.22× faster**         |

Takeaway: before the change, OnPair decode was roughly tied with (and on
`p_name`/`o_comment`/`clickbench` slightly behind) the FSST fast baseline on
every real column. After the change, OnPair decode is **1.22–1.45× faster than
FSST** on every column tested.
