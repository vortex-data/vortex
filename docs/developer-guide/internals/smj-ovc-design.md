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

## Row layout

Ord-bytes are laid out **row-major**: row `i`'s contributions from every sort
column appear contiguously, then row `i+1` begins. Within each column the
bytes are big-endian, sign-flipped where necessary, so a byte-wise `memcmp` on
two rows matches the logical sort order.

```
stride = Σ width(col_k)                  (fixed-width case)

row 0: [ col0_ord_bytes | col1_ord_bytes | col2_ord_bytes | ... ]
row 1: [ col0_ord_bytes | col1_ord_bytes | col2_ord_bytes | ... ]
...
```

Variable-width key columns use a side `offsets: Buffer<u32>` table so row `i`
is the slice `data[offsets[i] .. offsets[i+1]]`. Within that slice, columns
are still concatenated in sort order.

Access pattern is **sequential**, not random:
- The merge driver compares row vs row by scanning from byte 0 forward until
  it finds the first differing byte (memcmp / OVC).
- It never reaches into the middle of a row.
- Columns are not stored columnarly inside the row buffer — that would force
  scatter-gather per compare. Row-major is what makes memcmp the whole compare.

## Basic algorithm (uniform-type sketch)

Assume every sort column is a fixed-width primitive for the sketch. Ord-bytes
become a flat `Buffer<u8>` of `n_rows * stride`. Merge of two pre-sorted sides:

```
fn merge(left: OrdRows, right: OrdRows, stride: usize) -> Vec<(usize, usize)> {
    let (mut l, mut r) = (0, 0);
    let mut out = Vec::new();
    while l < left.n_rows && r < right.n_rows {
        let lb = &left.bytes[l * stride .. (l + 1) * stride];
        let rb = &right.bytes[r * stride .. (r + 1) * stride];
        match lb.cmp(rb) {                       // memcmp
            Ordering::Less    => l += 1,
            Ordering::Greater => r += 1,
            Ordering::Equal   => { out.push((l, r)); l += 1; r += 1; }
        }
    }
    out
}
```

OVC layers on top by caching, per side, `(offset, value_at_offset)` for the
current head row against the most recently emitted row, and comparing those
two `(offset, value)` pairs as integers instead of running `memcmp` from byte 0
each time. The byte layout above doesn't change.

## Non-uniform ord codes framework

Some columns don't fit the "every row contributes the same number of bytes"
model. Constant columns contribute zero per-row bytes. Common-prefix columns
elide a shared prefix. Dict columns substitute codes for full values. These
all factor the same way:

```
OrdContribution = (header, payload)

  uniform fixed-width prim   header = ∅                     payload = N bytes/row
  uniform varlen             header = ∅                     payload = (offsets, data)
  constant                   header = the bytes             payload = ∅
  common-prefix              header = the shared prefix     payload = suffix bytes/row
  dict (rank-aligned)        header = rank table            payload = codes
  run-end                    -- expanded by encoder, not a runtime case --
```

`header` is per-column-per-side, computed once at encode time. `payload` is per
row. The merge driver runs a small decision per column:

1. Both sides' header fully determines the value (constant + constant):
   compare headers once; result is global for the column. Often the planner
   drops the column (equal headers) or short-circuits the join (unequal).
2. One side constant, other varies: compare each varying row against the
   constant header. The constant side has no per-row work for this column.
3. Common-prefix on both: if prefixes differ, header decides globally. If
   equal, compare suffix payloads only — the per-row stride for this column
   shrinks by `|prefix|`.
4. Dict (rank) on both: header is the rank table (cross-side aligned at plan
   time), payload is codes. Per-row compare is two indexed loads and an
   integer compare.
5. Mixed (dict on one, plain on the other): the producer matrix decides
   whether to put both into rank space or both into byte space; one of the
   symmetric cases above then applies.
6. Plain on both: empty headers, payload-vs-payload compare.

The driver iterates columns in sort order, using case (1)-style header
decisions as fast exits and falling through to per-row payload work only when
needed.

OVC over this framework: the `offset` position space spans column indices and
bytes within payloads. Columns whose header fully decides the answer either
contribute zero offset bytes (constant) or one fixed chunk (common-prefix
header, dict rank). The merge driver still sees a single sequential byte
position counter — non-uniformity is hidden in the header/payload split.

## Open questions

- Planner heuristics for `RankBE` vs `EscapedUtf8` target selection on
  asymmetric (dict-on-one-side) string columns.
- Threshold for run-end expand vs run-aware merge at n=2.
- Interaction with chunked arrays where per-chunk encodings differ.
- Concrete representation of `header` and `payload` for varlen payloads
  when a column also has a non-empty header (e.g. common-prefix + varlen
  suffixes).
