# Vortex EntropyBins

A leaf encoding for integers. Each value is split into a bin id and an offset inside that bin,
using the bins pco's optimizer trains for every chunk of 2^18 values. Ids are entropy-coded with
16-lane interleaved tANS; offsets are bit-packed in row order at their bin's width. Values are
stored in independently decodable blocks of 1K to 16K rows.

The encoding is a leaf: transforms such as frame of reference, GCD scaling, ALP, Dict or RunEnd
are parent encodings chosen by the cascading compressor. The one transform it fuses is the
difference from the row `lag` back (lag 0 to 8), because the decoder folds its prefix sum into
the merge, and differences of adjacent rows entropy-code better than a transposed delta's
residuals.

## Compute

Compare and between against constants run on the bin ids: a bin covers a contiguous range of
values, so most blocks are answered from their ids without reading offsets (lag 0 only). Filter
and take decode only the blocks holding a selected row; sparse selections decode ids only up to
the last selected row of a block. Random access decodes a block's ids up to the row, and a
repeated probe caches the block it touched twice.

## Dials

`EntropyBinsConfig` sets the layout planned for an array; the BtrBlocks `EntropyBinsScheme` adds
`min_gain_percent`. Presets: `FAST`, `BALANCED` (default) and `SMALLEST`.

| Dial | Trade-off measured | FAST | BALANCED | SMALLEST |
|---|---|---|---|---|
| `max_block_values` | Each doubling halves the remaining per-block bytes and doubles one-off random access. Dumped columns, all blocks forced: 1K → 4K → 16K is −1.35% / −1.69% bytes (64-bit) and −9.7% / −12.2% (narrow); random access 0.7 → 2.4 → 9.4 µs. Decode speed is flat. | 1K | 4K | 16K |
| `larger_block_gain_percent` | How much smaller the largest allowed block must make an array before the planner leaves 1K blocks. | 0 | 15 | 1 |
| `narrow_word_gain_percent` | 8-bit refill words save ~8 B per block (−0.42% 64-bit, −1.9% narrow) for a 2–5% slower decode. | never | ≥ 1% | always |
| `lags` | Each lag tried costs one estimate when compressing. Lag 0 and 1 cover most columns; lags 2–8 serve interleaved series. | 0, 1 | 0–4, 8 | 0–4, 8 |
| `level` | pco level used to train bins. Level 4: +8.7% / +12% bytes for 5–14% faster encodes; level 10: −0.2% (64-bit), +1.4% (narrow). | 8 | 8 | 8 |
| `min_gain_percent` (scheme) | How much the scheme must beat the cascade's other choices. 10 / 25 / 40%: 64-bit 6.32 / 6.11 / 5.98x at 7.6 / 8.1 / 8.4 GB/s decode. | 25 | 10 | 0 |

In the BtrBlocks cascade with 1 MB file blocks (serialized bytes, 77 64-bit and 68 narrow
columns), BALANCED reaches 7.159x / 19.30x and SMALLEST 7.183x / 19.79x, both decoding at
~6.0 / ~3.6 GB/s.

## Design choices

Kept, with what each was measured against:

| Choice | Alternatives measured |
|---|---|
| pco's bins, at most 64 per chunk | 32 bins: +2.3% bytes. 128 bins: −0.43% (almost all one Dict-shaped column) but ids no longer fit the 64-entry vector tables. |
| 2^18 values per chunk | 2^16: −0.4% (64-bit), flat on narrow; 2^20: +0.9% / +2.6%. Not worth a per-array setting. |
| tANS with 16 lanes and state log ≤ 7 | Decode tables fit two registers. A per-chunk state log saves 0.02% / 0.19%. |
| Free initial lane state (FSE_initCState2) | −0.28% / −0.48%, no decode cost. |
| 3-bit per-lane stop field | A wider field saves at most 0.04% / 0.45%. |
| u16 block header, packed block lengths, 64 B tail padding | −0.2% (64-bit) / −1.7% (narrow at 1K blocks). |
| Bin lowers serialized as gaps; block lengths and seeds frame-of-reference packed | −0.35% / −0.46% of serialized bytes with 1 MB file blocks. |
| Lag-k differences fused into the merge (k ≤ 8) | Against FastLanes Delta stacked on a lag-0 leaf (1 MB arrays, deltas zigzagged, its best case): 5.76x vs 5.28x (64-bit) and 17.1x vs 14.4x (narrow); per array the better of lag 0 and stacked Delta reaches only 5.60x / 16.0x. On the arrays that pick a lag, fused decodes at 6.3 / 2.9 GB/s against 3.2 / 1.6 GB/s stacked, which pays a second pass. Lags 2–8 add +0.7% / +0.5% over lag 0/1 only, mostly interleaved series. |
| Sign bit flipped for signed values and differences | Keeps latents in value order, which the id compare and between kernels need. Zigzag instead: +1.26% bytes (64-bit) / +0.46% (narrow), likely because zigzag interleaves positive and negative values, so a bin over a skewed range spends a bit on sign parity. |
| Per-block vector merge chosen from the widths a block uses | Rare 63–64 bit bins no longer force a chunk onto the portable merge (hits.UserID decodes 1.6x faster). |

Rejected or not yet worth it:

| Idea | Why |
|---|---|
| Order-1 id context | Upper bound 1.9% / 3.8% of bytes, but serial across lanes, so no SIMD decode. |
| Same-lane context, per-block tables, entropy-coded top offset bits | Upper bounds 0.4–1.3% each, before the cost of the extra tables. |
| Per-block bin-id ranges for pruning | 2 B/block; prunes about as well as existing 8K zone maps. |
| Mid-block checkpoints | A restart needs each lane's buffered bits as well as its state, so it costs about a block boundary. |
| Per-block or per-group layout choices | Measured on 1 MB arrays at their planned layout (64-bit / narrow): lag 0 or 1 per block −0.26% / −0.53%; adding a per-block base −0.27% / −0.49%; 2 or 4 weight tables per chunk with a block selector −0.16% / −0.87% and −0.22% / −0.93%; per-bin offset trimming −0.07% / −0.14%; per-block trimming 0%. Offsets are 79% / 53% of the bytes and the chunk's bins already fit them, so per-block choices mostly move the ids, and switching tables per block would break the four-block lockstep id decode. |
| Trial-encoding the two best lags | Fixes the estimator's lag misses on 3 narrow columns (+1.54% → +0.23% vs an exhaustive search) but those columns go to RunEnd in the cascade; +16–33% compression time for no cascade gain. |

Gaps that belong to parent encodings, not this leaf: pco alone is smaller on 35 of 50 64-bit
integer columns (7.46x vs 6.31x for BtrBlocks + EntropyBins), almost entirely from its
integer-multiple and higher-order delta modes (timestamps with whole-second precision, prices,
quantities).
