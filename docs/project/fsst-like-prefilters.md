# FSST LIKE Pre-filter Design Notes

Companion notes for `ji/fsst-like-paper-2`. Captures the design space for pre-filter
algorithms used by the FSST contains DFA, the trade-offs between them, and what is
actually being implemented now versus deferred.

## Setting

The DFA matchers in `encodings/fsst/src/dfa/` evaluate `LIKE '%needle%'` (and related
patterns) directly on FSST code streams. Each chunk has its own learned 256-symbol
dictionary plus the universal escape code `0xFF`. A pre-filter narrows down which code
positions are candidate match starts so the DFA is not run from every position.

Two existing pre-filters in `folded_contains.rs`:

- **1-byte progressing bitset.** A PSHUFB-Mula nibble-table bitset over `all_bytes`
  marking positions whose code progresses state 0. Up to 8 codes via single pass, more
  via multi-pass OR-merge.
- **Cartesian 2-byte pair bitset.** Two independent 1-byte bitsets (`c1_set`, `c2_set`),
  AND'd with a 1-position shift. Selectivity `|c1| · |c2| / 65536`.

The Cartesian path leaves performance on the table: it admits cross-product
false-positive pairs `(c1_a, c2_b)` for `a ≠ b`.

## Selectivity comparison

For 3 valid `(c1, c2)` pairs with disjoint c1's and c2's:

| Variant   | Selectivity   | False-positive pairs |
|-----------|---------------|----------------------|
| Cartesian | 9 / 65536     | 6                    |
| Teddy     | 3 / 65536     | 0                    |

3× sparser bitset → 3× fewer DFA verifier dispatches at the same SIMD cost. On real
FSST-trained URL data the cross-product / valid-pair ratio is typically 5–10× because
several c1's share c2's.

## Pre-filter taxonomy

Categorized G (general string-matching), F (FSST-specific), G→F (general adapted to
FSST code streams).

### Position-level anchors (already in branch)

| Tech                                      | Class | In branch? |
|-------------------------------------------|-------|------------|
| `memchr1/2/3`                             | G     | yes (skip.rs) |
| PSHUFB-Mula 1-byte bitset                 | G     | yes (anchor_scan.rs) |
| Multi-pass PSHUFB OR-merge                | G     | yes |
| Cartesian 2-byte pair bitset              | G     | yes |
| Bucketed-Cartesian Teddy (this PR)        | G→F   | this PR |
| Match-path enumeration → anchor union     | F     | yes (implicit) |
| Escape anchor `0xFF` for escape-only bytes| F     | not yet |

### Position-level alternatives (deferred)

| Tech                                            | Class | Notes |
|-------------------------------------------------|-------|-------|
| Best-window pair selection (suffix-anchor variant) | G→F   | needs reversed DFA verifier |
| Best-window pair selection (arbitrary interior) | G→F   | needs bidirectional verifier |
| 3-byte fingerprint (Teddy-3)                    | G     | extension of this PR's bucket scheme |
| Shift-Or in code space                          | G→F   | competitive on short needles |
| BNDM in code space                              | G→F   | research-y under FSST nondeterminism |

### Multi-pattern (deferred)

| Tech                              | Class | Notes |
|-----------------------------------|-------|-------|
| Slim Teddy across patterns        | G→F   | Hyperscan-style; one bucket per pattern |
| Fat Teddy (16 buckets)            | G→F   | AVX2 lanes |
| Aho-Corasick over code sequences  | G→F   | state blow-up under multi-path |
| Wu-Manber block-hash              | G→F   | scales to many patterns |

### Chunk-level filters (deferred)

| Tech                                       | Class    | Fires on this paper's eval? |
|--------------------------------------------|----------|-----------------------------|
| Symbol-presence bitmap per chunk           | F-only   | mostly no on ASCII English  |
| Decoded-byte reachability per chunk        | F-only   | mostly no                   |
| Length min/max per chunk                   | G        | mostly no on planned needles|
| q-gram bitset                              | G        | mostly no                   |
| Per-chunk symbol-code histogram for anchor | F        | yes — informs anchor choice |

Decision: defer all chunk-level filters from this PR. Reasoning: on the planned
ClickBench / TPC-H / TPC-DS columns the byte distribution is dense enough that per-chunk
filters skip ~0% of chunks. They become useful on multilingual or sort-clustered data
that is not in the paper's eval set. See "Where they actually fire" below.

#### Where they actually fire

- Categorical / low-cardinality columns with sort-clustering (zone-map territory).
- Multilingual columns where chunks cluster by script.
- Mixed-content columns (JSON / URL / empty) when chunks cluster by row type.
- Truly rare bytes (e.g., emoji, control chars, non-ASCII) — not in mined workloads.
- Long needles (≥ 100 bytes) on columns with many short rows — not in mined workloads.

The position-level zero check that *does* fire on real workloads is
`SkipStrategy::find_next_progressing` returning `None` per row — already in branch.

## What this PR adds: bucketed Cartesian Teddy

Generalizes the existing 2-byte Cartesian path. Each Teddy bucket holds one
`(c1_subset, c2_subset)` Cartesian sub-product; the overall bitset OR's the buckets.
Same SIMD instruction sequence as Cartesian (4 PSHUFB + AND + movemask), with bucket
bits stored across the 8 bits of each nibble-table entry.

### Selectivity

```
P(candidate) = Σ_b (|c1_b| · |c2_b|) / 65536
```

vs. pure Cartesian's `(Σ |c1_b|) · (Σ |c2_b|) / 65536`. The cross terms in the sum are
exactly the false-positive pairs the bucketing eliminates.

### Bucketing strategy: shared-c1

For the valid pair set, group pairs by their `c1`. Bucket `b` = `({one c1}, {all c2's
following that c1})`. With ≤ 8 distinct c1's this fits in slim Teddy with **zero false
positives**: if every pair in a bucket shares the same c1, the bucket's check
"input[i] = c1 AND input[i+1] ∈ c2_set" admits no spurious cross-pairs.

Falls back gracefully:

- Every pair has unique c1 → degenerates to plain Teddy (one pair per bucket).
- All pairs share one c1 → degenerates to Cartesian (one bucket = whole cross-product).
- Smooth interpolation between the two.

Symmetric shared-c2 clustering is equivalent; pick whichever side has fewer distinct
codes.

### Multi-pass spillover

For > 8 distinct c1 values, chunk the buckets and OR-merge. Cost = `ceil(distinct_c1 /
8)` walks over `all_bytes` — same trick as `build_progressing_bitset_unbounded`.

### Tail case

If one bucket's c2 set exceeds 8 codes (rare for needle-derived pair sets), that bucket
becomes a Cartesian over-approximation within itself — a small selectivity loss, no
extra passes.

### Fallback ladder

The scan path picks the first applicable mode:

1. **Bucketed Teddy (single pass)** if distinct c1's ≤ 8 and each bucket's c2 ≤ 8.
2. **Bucketed Teddy (multi-pass)** if distinct c1's > 8.
3. **Cartesian pair bitset** (existing) if c1 union and c2 union each ≤ 8 but bucketing
   doesn't help.
4. **1-byte progressing bitset** (existing) as the universal fallback.

## Deferred work

The following were considered and deferred. See section above for selectivity / cost
notes.

1. **Best-window pair selection.** Pick the rarest 2-byte window in the needle, not
   the prefix. Suffix-anchor variant is straightforward (build reversed DFA); arbitrary
   interior anchors need bidirectional verification and per-chunk byte-frequency stats.
   Listed as future work in the paper's Limitations section.
2. **Chunk-level filters.** Symbol-presence bitmap, decoded-byte reachability, length
   min/max, q-gram bitset. Mostly do not fire on the planned eval workloads but compose
   trivially with the existing matcher. Worth one paragraph in Future Work / Engine
   Integration as standard zone-map composition.
3. **Multi-pattern Teddy.** `LIKE '%a%' OR LIKE '%b%' OR …`. Naturally bucketed (one
   pattern per bucket); reuses the same SIMD core as this PR. Out of scope until the
   single-pattern path is settled.
4. **Long-needle support (> 254 bytes).** `u16` state ids extend the DFA cap to ~65 KB
   needles. Factor-then-verify can support arbitrarily long needles via 254-byte sub-
   factors. Currently long needles fall back to decoded LIKE.
5. **Escape anchor `0xFF`.** When the needle contains a byte the chunk dictionary only
   emits via escape, `(0xFF, b)` is a high-selectivity pair anchor. Folds naturally
   into bucketed Teddy when integrated.

## What the original FSST paper covers

For citation alignment in the paper, the original FSST paper (Boncz/Neumann/Leis,
VLDB 2020) covers:

- 256-symbol learned dictionary with 8-byte symbol cap.
- Escape code `0xFF` for literal bytes.
- Greedy multi-round symbol training with sampling.
- Block-grouped dictionaries (~128 KB shared dict per block).
- Lexicographic order preservation.
- Random-access decoding.
- LZ4/ZSTD layering for compression ratio.

It does **not** cover:

- DFA matching on code streams.
- Anchor scans (1-byte or pair).
- Teddy-style multi-pattern matching.
- Per-chunk metadata bitmaps for predicate skipping.
- Long-needle pushdown.

These are contributions of this paper.
