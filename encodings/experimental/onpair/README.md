# Vortex OnPair

A Vortex Encoding for Binary and Utf8 data that uses the
[OnPair][onpair] short-string compression algorithm. OnPair is a
dictionary-based encoder with fast per-row random access.

The trainer / encoder lives in the standalone [`onpair`][onpair-crate]
crate; this crate wraps the resulting column as a Vortex array with
cascading-compressor support on every integer child.

## Compute

Like the FSST encoding, this crate provides `cast` and `filter`
pushdown. Other operators fall back to ordinary decompression.

## Default Configuration

The default training preset is **dict-12**: 12 bits per token,
dictionary capped at 4 096 entries. Token codes are stored as a
`PrimitiveArray<u16>`; downstream `FastLanes::BitPacking` losslessly
narrows the child to exactly `bits`-bit codes on disk.

## Layout

- Buffer 0 — `dict_bytes`: dictionary blob built by the OnPair trainer,
  padded with `MAX_TOKEN_SIZE` trailing zero bytes so the over-copy
  decoder can read 16 bytes past the last token.
- Slot 0 — `dict_offsets`: `PrimitiveArray<u32>`, len `dict_size + 1`.
- Slot 1 — `codes`: `PrimitiveArray<u16>`, length `total_tokens`.
- Slot 2 — `codes_offsets`: `PrimitiveArray<u32>`, length `num_rows + 1`.
- Slot 3 — `uncompressed_lengths`: integer `PrimitiveArray`, length
  `num_rows`.
- Slot 4 — optional validity child.

All four integer slot children flow through the standard cascading
compressor pipeline (FoR / BitPacking / RunEnd / etc.).

[onpair]: https://arxiv.org/abs/2508.02280
[onpair-crate]: https://github.com/spiraldb/onpair
