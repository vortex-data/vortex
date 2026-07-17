# Vortex OnPair

A Vortex Encoding for Binary and Utf8 data that uses the
[OnPair][onpair] short-string compression algorithm. OnPair is a
dictionary-based encoder with fast per-row random access.

The trainer / encoder lives in the standalone [`onpair`][onpair-crate]
crate; this crate wraps the resulting column as a Vortex array with
cascading-compressor support on every integer child.

## Compute

Like the FSST encoding, this crate pushes down common operations over the
encoded representation. It supports `cast`, `filter`, byte length, and
constant equality / inequality. Unsupported operators fall back to ordinary
decompression.

## Default Configuration

The default training configuration uses OnPair's default dictionary budget and
a fixed seed. Vortex stores token codes as an integer child array; downstream
integer compression may narrow or bit-pack that child independently.

## Layout

- Buffer 0 — `dict_bytes`: dictionary blob built by the OnPair trainer,
  including the read padding required by the decoder.
- Slot 0 — `dict_offsets`: integer child, len `dict_size + 1`.
- Slot 1 — `codes`: integer child, length `total_tokens`.
- Slot 2 — `codes_offsets`: integer child, length `num_rows + 1`.
- Slot 3 — `uncompressed_lengths`: integer child, length `num_rows`.
- Slot 4 — optional validity child.

All four integer slot children flow through the standard cascading
compressor pipeline (FoR / BitPacking / RunEnd / etc.).

[onpair]: https://arxiv.org/abs/2508.02280
[onpair-crate]: https://github.com/spiraldb/onpair
