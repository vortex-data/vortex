# vortex-onpair

A Vortex string array backed by the [OnPair][onpair] short-string compression
library. OnPair is a dictionary-based encoder with fast per-row random access
and **compressed-domain predicate evaluation** for `=`, `LIKE 'prefix%'` and
`LIKE '%substring%'` — pushdown is wired through the standard Vortex compute
kernels.

The default training preset is **dict-12**: 12 bits per token, dictionary
capped at 4 096 entries. Token codes are stored as a bit-packed stream inside
the OnPair column blob (see `vortex-onpair-sys`).

Layout (mirroring `vortex-fsst`):

- Buffer 0: serialised `OnPairColumn` (`ONPAIR01` magic + dictionary +
  packed token stream).
- Slot 0: `uncompressed_lengths` primitive child, used during canonicalisation
  to build `VarBinView` offsets without re-decoding sequentially.
- Slot 1: optional `codes_validity` child for nullable arrays.

[onpair]: https://arxiv.org/abs/2508.02280
