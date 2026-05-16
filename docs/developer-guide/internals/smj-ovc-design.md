# Sorted Merge Join with Offset-Value Coding — Design Notes

Working notes for an eventual SMJ implementation. Not a spec.

## Approach

- Produce ord-bytes (memcmp-equivalent normalized keys) per side, per column.
- Run a single uniform merge driver over ord-byte rows, optionally with column-level
  or byte-level OVC.
- Encoding awareness lives in the **producer**, not the merge driver. Each
  `(target, physical_encoding)` pair has a small kernel that writes the column's
  contribution into the row buffer.

The row form is scoped to the operator: only sort-key columns are materialized,
the payload stays columnar, and join output is reassembled by `take` on the
originals.

## Producer matrix (sketch)

Targets:
- `PrimitiveBE(N)` — sign-flipped big-endian bytes for fixed-width primitives.
- `RankBE(width)` — fixed-width integer rank into a shared cross-side rank space.
- `EscapedUtf8` — variable-width escape-encoded bytes for strings/lists.

Each target has an `OrdEncode` impl per physical encoding (`PrimitiveArray`,
`DictArray`, `FSST`, `RunEnd`, `FoR`, `ALP`, `ConstantArray`, ...). The planner
picks a target per column once; both sides encode into it.

## Future optimization: dict stays in rank space (n-way friendly)

For dict-encoded columns participating in the sort key:

1. Build one rank plan across **all n inputs' dictionaries** before the merge.
   - n-way merge over the sorted dict values, `O(Σ |D_i|)`.
   - Produces one `rank_i: Buffer<u32>` per side mapping codes to a shared rank.
2. Encoder writes a fixed-width BE rank per row into the row buffer.
3. Merge driver compares ranks as integers — no string materialization.
4. Output preserves dict encoding via `take` on the original `DictArray`.

This composes cleanly with n-way merge because dict is a **value-level
compression** (per-row code), not a structural one. The merge driver doesn't
need to know the column was dict-encoded.

Contrast with run-end: it's a **structural compression** (physical rows ≠
logical rows). Run-end-aware merging works at n=2 but doesn't scale to n-way
without significant complexity (cross-side run alignment, partial-consumption
bookkeeping). For n-way, expand run-end columns into the ord-byte rows and let
OVC collapse the redundant comparisons. The bandwidth cost of writing repeated
bytes is recovered by OVC's offset-based short-circuiting.

Rule of thumb:
- Encoding-as-producer (dict, FSST, ALP, FoR, primitive) — keep in compressed
  form into the encoder. Composes with n.
- Encoding-as-topology (run-end) — expand into the row form for n-way. Optional
  run-aware fast path for n=2 with very long runs.

## Open questions

- Constant and common-prefix columns: framework for non-uniform ord codes
  (column-level structural info + per-row payload). See chat discussion;
  capture once stable.
- Planner heuristics for `RankBE` vs `EscapedUtf8` target selection on
  asymmetric (dict-on-one-side) string columns.
- Threshold for run-end expand vs run-aware merge at n=2.
- Interaction with chunked arrays where per-chunk encodings differ.
