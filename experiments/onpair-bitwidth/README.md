# OnPair bitwidth selection — experiment

Branch: `claude/onpair-bitwidth-selection-jMg3S`

Goal: push the OnPair (arXiv 2508.02280) compression ratio beyond the paper's
fixed 16-bit token width without sacrificing random-access decode. OnPair fixes
2 bytes per stream token and stops the dictionary at 65 536 entries; this
experiment shows ~5–8% better ratio by (a) optionally pruning intermediate
junk tokens during training (Picky-BPE style) and (b) replacing the flat
16-bit stream code with a bit-packed two-tier code whose short-tier width is
auto-selected from per-dataset token-frequency statistics.

This work is exploratory and lives entirely under `experiments/`. Nothing in
the rest of Vortex is touched.

## Results

All numbers are measured against the upstream reference implementation
(`gargiulofrancesco/onpair_rs`) on a single AVX2 machine, with the same
ratio accounting (`space_used = packed_stream_bytes + dict_bytes +
4 * (n_tokens + 1)`; string offsets excluded — matches `OnPair::space_used`
in upstream).

Roundtrip-verified on every variant.

| Dataset (real)                    | Input MB | OnPair  | Opt (best)        | Δ        |
| --------------------------------- | -------: | ------: | ----------------: | -------: |
| English words (dwyl/english-words)| 3.49     | 1.594×  | **1.694×**        | +6.27%   |
| Tranco top-200k domains           | 2.58     | 1.510×  | **1.607×**        | +6.40%   |
| IMDB primaryTitle (first 500k)    | 9.15     | 2.032×  | **2.140×**        | +5.30%   |
| Wikipedia ns0 titles (first 500k) | 15.54    | 4.049×  | **4.389×**        | +8.39%   |

The "Opt (best)" column is Picky training (τ=0.80, min_unigram=4, 1–3 passes)
plus the two-tier bit-packed code with auto-chosen short-tier size `k` and
widths `b1 = log2(k)`, `b2 = ceil(log2(N - k))`. In all four datasets the
auto-selected `k` was 2 048 or 4 096.

## What changed

Three pieces, none touching the upstream parser hot path:

1. `LongestPrefixMatcher::remove` (small patch — see `lpm-remove.patch`).
   Hashmap removal for short tokens; `id = None` on the trie leaf for long
   tokens. Required for slot recycling during Picky training.

2. `src/onpair_picky.rs` — Picky-BPE training pass. Maintains unigram counts
   per token id. When merge `(a,b) → c` triggers, evict either side if its
   `IoS = freq(a,b) / unigram(a) ≥ τ` AND `unigram(a) ≥ min_unigram`.
   Evicted ids are pushed onto a free-list and reused for the next merge,
   so the 16-bit dictionary can admit more high-utility merges than the
   single-shot vanilla pass.

3. `src/onpair_opt.rs` — the full pipeline. Trains (with or without Picky),
   parses, then:

   - Ranks live tokens by stream frequency.
   - Sweeps `log2(k) ∈ [1, 15]` and picks the `k` minimising exact packed
     bytes: `stream_bits = N + cov(k)·log2(k) + (N - cov(k))·log2(D - k)`.
   - Re-encodes the stream as `[1-bit flag][b1 or b2 bits]` per token.
     `flag = 0` → top-tier rank `< k` at `b1` bits;
     `flag = 1` → bottom-tier offset at `b2` bits.

   String boundaries are stored as bit offsets so random-access decode is
   still O(string length) per string (same access model as upstream).

## Why this works

- The token-rank histogram on every real dataset I tried has heavy enough
  skew that ~44% of stream mass falls in the top 2 048 ids (11-bit code).
  A 1-bit flag plus 11-bit short / 16-bit long code beats flat 16-bit on
  every dataset. Two-tier byte-aligned codes (1-byte short + 0xFF-escape
  long) are *worse* than flat 16-bit because the rare path costs 3 bytes;
  bit alignment is what makes this win.

- Picky pruning gives a smaller, second-order improvement (~0–1.5% extra
  ratio on top of two-tier). It helps more when the dictionary saturates
  with marginally-useful intermediate merges (domains: yes; words: barely).
  Multi-pass training hurts on small datasets because dict bytes grow
  faster than the stream shrinks.

- No change to the LPM hot path, no change to decode complexity per token
  beyond two bit reads and a flag-conditional shift width.

## What does NOT improve ratio (and why)

- Just dropping unused dictionary tokens after parsing: ~+1% only.
  Most live tokens get used at least a few times; dead weight is small.
- Flat bit-pack at the smallest power-of-two that addresses the dictionary:
  no help on saturated dictionaries (`ceil(log2 65536) = 16`).
- 1-byte short / 0xFF-escape two-tier code: worse than 16-bit flat (rare
  tokens become 3 bytes; rare set is too big to amortise).
- Multi-pass training without Picky: it just refills slots without raising
  the merge quality bar, so dict overhead grows while stream barely shrinks.

## What I did not try (and why)

- **Adaptive bitwidth at training time** (the branch name): when the dict
  doesn't fill, you could code at `ceil(log2(N_used))` bits flat. Useful
  only when N_used drops well below the next power of two; in practice the
  dictionary fills on every real dataset I tested. The two-tier code
  subsumes this case (it picks `k` to minimise total bits, and when the dict
  is undersized it just shifts mass to the short tier).
- **ANS / Huffman over the token stream**: best ratio possible, but kills
  random access. Not pursued.
- **Aho-Corasick parser**: this is a *speed* lever (parse is OnPair's
  dominant cost: 133 s vs 1.4 s training on the paper's L_URL), not a
  ratio lever — out of scope for this experiment.

## Reproducing

The experiment lives outside the upstream tree because it touches
`onpair_rs` directly. To rerun:

```bash
git clone https://github.com/gargiulofrancesco/onpair_rs.git /tmp/onpair_rs
cd /tmp/onpair_rs
patch -p1 < $VORTEX/experiments/onpair-bitwidth/lpm-remove.patch
cp $VORTEX/experiments/onpair-bitwidth/src/onpair_picky.rs src/compressor/
cp $VORTEX/experiments/onpair-bitwidth/src/onpair_opt.rs   src/compressor/
cp $VORTEX/experiments/onpair-bitwidth/src/bench.rs        examples/
# add `pub mod onpair_picky; pub mod onpair_opt;` and re-exports to
# src/compressor/mod.rs and src/lib.rs.
RUSTFLAGS="-C target-cpu=native" cargo build --release --example bench
./target/release/examples/bench <dataset.txt> ...
```

Datasets used:

- `words.txt`: <https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt>
- `domains_200k.txt`: first 200 000 lines of Tranco top-1M
  (<https://tranco-list.eu/top-1m.csv.zip>)
- `imdb_titles.txt`: first 500 000 non-`\N` `primaryTitle` rows from
  <https://datasets.imdbws.com/title.basics.tsv.gz>
- `wiki_titles.txt`: first 500 000 lines of
  <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz>

## Files

- `README.md` — this file
- `lpm-remove.patch` — adds `LongestPrefixMatcher::remove`
- `src/onpair_picky.rs` — Picky-BPE training, vanilla 16-bit code
- `src/onpair_opt.rs` — Picky training + two-tier bit-packed code
- `src/bench.rs` — benchmark + roundtrip harness

Upstream OnPair is MIT (Gargiulo and Venturini, 2025).
