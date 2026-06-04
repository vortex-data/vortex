# CodeBigramBloom: OnPair-aware skip index for substring containment

## Problem

We want a per-chunk skip index that answers `LIKE '%needle%'` using
OnPair's dictionary structure, aiming for tighter pruning than
`TrigramBloom` (variant B) which is encoding-agnostic.

## Background: why byte trigrams plateau

`TrigramBloom` inserts every 3-byte window of the raw string data.
For URL-shaped data with 8K-row chunks, common trigrams like `://`,
`www`, `.co`, `com` appear in every chunk, saturating the bloom.
At 16 bits/row, ClickBench URL gets +24.7pp vs_floor — about 29%
of chunks skipped.  More bits don't help because the trigrams
genuinely exist.

OnPair code bigrams are sparser: with 4096 tokens averaging ~2-3
bytes each, a chunk of 8K rows has ~35K code bigrams vs ~700K byte
trigrams.  The bloom saturates 20x later.

## Core idea

**Build phase:** identical to `TokenPairBloom` — insert all
consecutive code pairs `(codes[i], codes[i+1])` within each row.

**Query phase:** for `LIKE '%needle%'`, enumerate all possible
alignments of the needle within the token stream.  For each
alignment, derive the deterministic token sequence via greedy LPM
and check whether the code bigrams are in the bloom.

## Alignment enumeration

The needle is a substring of some row.  In that row's token stream,
the needle is covered by:

```
  [ entry_token ][ t0 ][ t1 ]...[ tk ][ exit_token ]
  ←── cover ──→ ←──── remainder ─────→
```

- `entry_token` is partially consumed (the needle starts `cover`
  bytes before its end).  `cover ∈ {0, 1, ..., max_token_len}`.
- `t0..tk` are fully within the needle.
- `exit_token` is partially consumed (the needle ends before its
  end).

**Key property:** greedy LPM is deterministic from any aligned
position.  Once we know `cover`, `tokenize_needle(needle[cover..])`
gives the unique interior token sequence.

## The exit-token problem

`tokenize_needle(needle[cover..])` produces tokens `[t0, t1, ..., tk]`
that fully consume `needle[cover..]`.  But the actual row has bytes
after the needle.  At each position P in the remainder, the row's
greedy tokenizer sees `remainder[P..] + bytes_after_needle`, while
ours sees only `remainder[P..]`.  The row might pick a **longer**
token that extends past the needle.

The affected token is whichever token's coverage crosses the needle's
end boundary.  For all **interior** tokens (fully within the needle),
both tokenizers see the same bytes and agree.

## Precise safety check

A token starting at position P in the remainder is **safe** iff no
dict entry at P can extend past the needle:

```rust
fn is_safe_position(dv, index, remainder, pos) -> bool {
    let remaining = &remainder[pos..];
    if remaining.is_empty() {
        return false;  // at needle end, always unsafe
    }
    for id in index.range_for(remaining[0]) {
        let entry = dict_entry(id);
        if entry.len > remaining.len()
            && entry.bytes.starts_with(remaining) {
            return false;  // longer entry exists
        }
    }
    true
}
```

**Why this works:** `tokenize_needle` at position P picked the
longest dict entry matching `remainder[P..]`.  The actual row's
tokenizer at P also picks the longest matching `row[P..]`.  If no
entry of length > `remaining.len()` starts with `remaining`, then
the longest match at P is ≤ `remaining.len()` bytes — identical in
both tokenizations.

**Why this is tight:** unlike the pessimistic `MAX_TOKEN_SIZE = 16`
cutoff, this checks the *actual dictionary*.  Most positions are
safe because dict entries rarely happen to start with the exact
trailing bytes of the needle.  The lex-sorted dict + first-byte
index makes this check O(candidates_per_byte) per position.

A bigram `(t_i, t_{i+1})` is **reliable** iff both `start_of(t_i)`
and `start_of(t_{i+1})` are safe positions.  Only reliable bigrams
are checked against the bloom.

## Algorithm summary

```
might_contain(needle):
  // Case: needle fits in a single dict token
  for each present token containing needle as substring:
    return true

  for cover in 0..=MAX_TOKEN_LEN:
    if cover > 0:
      find entry tokens: present tokens ending with needle[0..cover]
    remainder = needle[cover..]
    toks = tokenize_needle(remainder)

    // Determine which positions are safe
    for each token position: compute is_safe_position()

    // Check reliable bigrams
    for each bigram (t_i, t_{i+1}) where both positions safe:
      if !bloom.contains(pair_hash(t_i, t_{i+1})):
        this cover fails → try next

    // Check entry bigram if safe
    if cover > 0 and position 0 is safe:
      need some entry_token with bloom.contains(pair_hash(entry, t0))

    if all reliable bigrams pass and entry ok:
      return true  (this alignment could match)

  return false  (no alignment works → skip this chunk)
```

## Comparison with TrigramBloom (B)

| Property | TrigramBloom (B) | CodeBigramBloom (E) |
|----------|------------------|---------------------|
| Build input | raw string bytes | OnPair code stream |
| Insert count per chunk | ~raw_bytes - 2 | ~n_tokens - n_rows |
| Bloom saturation | high (common trigrams) | low (code pairs are sparser) |
| Query: alignment issue | none (byte-aligned) | must enumerate ≤17 covers |
| Query: unsafe positions | none | checked per-position |
| Best for | any encoding | OnPair, long needles |

## Expected benefit

For needles where most positions are safe (i.e., the trailing bytes
of the needle don't coincidentally match a longer dict entry), E
checks O(needle_len / avg_token_len) code bigrams.  Because code
bigrams are ~20x sparser than byte trigrams, the bloom has much lower
saturation and can prune chunks that B cannot.

For very short needles (< 5 bytes) or needles whose trailing bytes
match common dict prefixes, E has few reliable bigrams and provides
little signal.  In practice, B and E can be combined: skip iff
**both** say skip.

## Lexicographic dict ordering

The OnPair dictionary is sorted lexicographically.  This enables:

1. **O(1) first-byte lookup** via `DictIndex::range_for(byte)` —
   already used by `tokenize_needle`.
2. **Efficient `is_safe_position`** — only scan entries in the
   first-byte range, not the full dict.
3. **Prefix range queries** — "all entries starting with prefix P"
   is a contiguous range, findable by binary search within the
   first-byte range.  Useful for the entry-token search (find
   entries ending with a given suffix: reverse the problem and
   scan, but the first-byte index helps for the exit check).

Future work: build a **suffix-first-byte index** (group dict entries
by their last byte) to accelerate the entry-token search from
O(dict_size) to O(candidates_per_last_byte).

## BitFunnel-style frequency-conscious extensions

### Motivation: saturation on high-diversity columns

Even with code bigrams (which are ~20× sparser than byte trigrams),
high-diversity columns like FineWeb URL still saturate the bloom.
URLs from the entire web share boilerplate bigrams (`(http,://)`,
`(://,www)`, `(www,.)`) that appear in **every chunk's bloom**, so
their bloom probes always return "yes" and contribute zero pruning
signal. They consume `k=3` bits per insertion but add no information.

This is exactly the problem BitFunnel ([Goodwin et al., SIGIR 2017])
addressed for Bing's signature-based news index. Their insight:
**common terms saturate every signature equally — allocate fewer
bits to them, and more bits to rare terms.**

### `UbiquitousBigrams` — binary skip/keep

The simplest application: identify bigrams appearing in `> X%` of
chunks ("ubiquitous"), and **skip them on both insert and probe**.
At query time, ubiquitous bigrams are treated as "uninformative": they are not
used as negative evidence. This is sound even when a skipped bigram is absent
from a specific chunk, because the index simply declines to prune on that
bigram.

Build (once per column, at write time, when codes are already in
memory):
```
for each chunk c:
  collect unique bigrams in c
  for each bigram b in c:
    count[b] += 1
ubiquitous = { b : count[b] > X% * n_chunks }
```

Storage: a sorted `Vec<u32>` of packed `(a << 16) | b` IDs.
Typically 1 KB – 200 KB per column depending on data diversity.
Amortized 0.01 – 1.7 B/row across all chunks.

### `BigramTiers` — variable `k` per bigram (full BitFunnel)

A more aggressive variant: classify bigrams into 4 frequency tiers
and use a different `k` (hash count) per tier:

| Tier | Frequency | k | Bits per insertion |
|------|-----------|---|--------------------|
| 0    | > 50% chunks | 0 | 0 (skip entirely) |
| 1    | 25–50% chunks | 1 | 1 |
| 2    | 10–25% chunks | 2 | 2 |
| 3    | ≤ 10% chunks (default) | 3 | 3 |

This concentrates bloom bits on the rare bigrams that actually
carry pruning signal. Storage cost is higher than `UbiquitousBigrams`
(needs `(u32, u8)` per entry, ~5 B per classified bigram) but the
precision win is also larger.

### Query soundness

Skipping a bigram at probe time is sound because skipped or lower-tier bigrams
are treated as weaker evidence, never as proof of absence. A `k=0` bigram is
ignored entirely; `k=1`/`k=2` bigrams can only false-positive more often than
`k=3` bigrams.

The bloom never reports a false negative for an inserted bit. Skipping an
*insertion* simply means we cannot use that bigram as pruning evidence.

### When does this help?

The win depends on the saturation ceiling of the underlying bloom:

- **Low saturation** (TPC-H comments, ClickBench Title): code bigrams
  already prune well at 16 bits/row. Ubiquity table adds metadata
  overhead with minimal precision gain.
- **Medium saturation** (ClickBench URL): F (ubiq-skip) gets 1-4pp
  tighter than E at 16-32 bits.
- **High saturation** (FineWeb URL): F gets 3-5pp tighter than E.
  G (tiered) can squeeze another 1-2pp at the cost of a much larger
  metadata table.

## Cross-dataset results

Pareto-frontier `vs_floor` (substring workload, lower = better)
on 8192-row chunks. `F` uses `--ubiq-pct=75` (or 50 for FineWeb);
`G` uses fixed tiers (top=50%, common=25%, medium=10%).

| Dataset | Floor% | @1 B/row | @2 B/row | @4 B/row | @8 B/row |
|---------|--------|----------|----------|----------|----------|
| ClickBench URL | 60.6 | B:+25.93 | F:+22.60 | F:+19.26 | E:+17.64 |
| ClickBench SearchPhrase | 30.5 | B:+35.47 | E:+34.93 | E:+34.87 | E:+34.86 |
| ClickBench Title | 64.2 | B:+18.43 | B:+18.36 | (dominated) | (dominated) |
| TPC-H l_comment | 83.5 | F:+15.70 | G:+14.42 | F:+13.54 | F:+13.29 |
| TPC-H o_comment | 91.8 | B:+7.79 | (dominated) | G:+7.58 | F:+7.32 |
| FineWeb url | 30.2 | B:+63.09 | F:+57.87 | G:+42.69 | E:+35.60 |
| FineWeb text | 60.8 | B:+39.21 | B:+39.13 | B:+39.06 | B:+38.96 |

Column-level metadata overhead (amortized across 122 chunks):

| Dataset | `UbiquitousBigrams` (F) | `BigramTiers` (G) |
|---------|-------------------------|-------------------|
| ClickBench URL | 14 KB (+0.11 B/row) | 445 KB (+3.65 B/row) |
| ClickBench SearchPhrase | 4 B (~0) | 4 KB (+0.03 B/row) |
| ClickBench Title | 2 KB (+0.02 B/row) | 311 KB (+2.55 B/row) |
| TPC-H l_comment | 1.4 KB (+0.01 B/row) | 160 KB (+1.61 B/row) |
| TPC-H o_comment | 5.5 KB (+0.05 B/row) | 307 KB (+3.08 B/row) |
| FineWeb url | 172 KB (+1.72 B/row) | 1.2 MB (+12 B/row) |
| FineWeb text | 3.3 MB (+33 B/row!) | 11 MB (unaffordable) |

The Pareto winners are:
- **Low diversity** (TPC-H, ClickBench Title): floor is too high for
  any variant to help much; differences ≤ 0.5 pp.
- **Medium diversity** (ClickBench URL): F is the clear winner at
  every bit budget — small metadata, real precision win.
- **High diversity** (FineWeb URL): F at 2 B/row, G at 4 B/row.
  G's tier table justifies its cost only at higher bit budgets.
- **Pathological** (FineWeb text): metadata explodes; skip indexes
  are the wrong tool. Use a per-row inverted index instead.

## Recommended configuration

| Column profile | Recommended variant | `--ubiq-pct` |
|----------------|---------------------|--------------|
| Low-cardinality (p_type) | A (DictPresence) | n/a |
| Structured/templated (TPC-H comments) | E (code bigrams) | n/a |
| URL-like, single-site (ClickBench URL) | F | 75-90 |
| High-diversity (FineWeb URL) | F | 40-50 |
| Long free-text (FineWeb text) | — (no skip index helps) | n/a |

For automatic tuning: at write time, build the ubiquity table at
`--ubiq-pct=50`, measure how many bigrams it captures, and choose:
- < 5K entries → ship as `UbiquitousBigrams` (cheap)
- 5K–50K entries → switch to `BigramTiers` if extra B/row budget exists
- > 50K entries (extreme diversity) → consider larger chunk_size or
  document-level indexing instead

## Open questions

1. **Auto-tuned threshold.** The optimal `ubiq-pct` differs by column
   (75-90 for ClickBench, 40-50 for FineWeb). A self-tuning approach
   could measure per-chunk FPR on a synthetic workload and pick the
   threshold that minimizes total bits-per-pruned-chunk.

2. **Per-chunk-size vs. global tables.** Currently the ubiquity table
   is per-chunk-size. For a fixed dataset chunk_size this is fine;
   for variable chunking it would need to be rebuilt.

3. **Higher-order tiers.** Could we extend `BigramTiers` to also
   include rare *code trigrams* with `k=3+`? Adds insertions but
   may further reduce FPR on the long tail.

4. **Persistence.** The ubiquity table is column-level metadata. The current
   integration plan stores it in an OnPair-local auxiliary skip-index layout so
   reads can use it without recomputing while keeping this crate Vortex-free.
