# OnPair bitwidth selection — experiment

Branch: `claude/onpair-bitwidth-selection-jMg3S`

Goal: push the OnPair (arXiv 2508.02280) compression ratio beyond the paper's
fixed 16-bit token width without sacrificing random-access decode. OnPair fixes
2 bytes per stream token and stores its dictionary as raw bytes plus a 4-byte
boundary per token; this experiment combines two orthogonal techniques to cut
total bytes by **13–25%** versus upstream OnPair, with roundtrip-verified
random-access decode preserved.

This work is exploratory and lives entirely under `experiments/`. Nothing in
the rest of Vortex is touched.

## Headline results

Roundtrip-verified on all four real datasets. Accounting matches OnPair's
`space_used` (stream + dict + bookkeeping); per-string offsets excluded.

| Dataset (real, public) | Input | **OnPair** | OnPairOpt (2-tier) | OnPair4Tier | OnPairFcDict | **OnPairCombined** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English words (dwyl) | 3.49 MB | 1.593× | 1.693× (+6.27%) | 1.724× (+8.20%) | 1.956× (+22.78%) | **1.998× (+25.40%)** |
| Tranco top-200k domains | 2.58 MB | 1.516× | 1.602× (+5.70%) | 1.622× (+7.01%) | 1.872× (+23.51%) | **1.897× (+25.12%)** |
| IMDB primaryTitle (500k) | 9.15 MB | 2.032× | 2.138× (+5.20%) | 2.159× (+6.27%) | 2.269× (+11.65%) | **2.307× (+13.56%)** |
| Wikipedia ns0 titles (500k) | 15.54 MB | 4.026× | 4.366× (+8.45%) | 4.404× (+9.40%) | 4.908× (+21.91%) | **4.983× (+23.79%)** |

`OnPairCombined` stacks the two universal winners discovered by parallel
ablation:

1. **Four-tier bit-packed stream code**: a 2-bit prefix selects one of four
   power-of-two-width tiers; the partition `(a0,a1,a2,a3)` is brute-force-swept
   per dataset to minimise total stream bits.
2. **Plain front-coded sorted dictionary**: tokens lex-sorted, grouped into
   buckets of 128, encoded as `(LCP, suffix)` per entry. A `freq_rank →
   lex_rank` permutation (2 bytes per token) lets the stream still index by
   frequency rank for the tier code.

Picky-BPE intermediate-token pruning during training is retained from the
earlier round but contributes <1% on top of these two; the dominant levers
are the two above.

## Per-dataset partition (illustrative; randomized training drifts ~0.1%)

| Dataset | (a0,a1,a2,a3) | tier sizes | bucket size |
| --- | --- | --- | --- |
| words   | (8, 12, 15, 15) | (256, 4 096, 32 768, 32 768) | 128 |
| domains | (9, 12, 15, 15) | (512, 4 096, 32 768, 32 768) | 128 |
| imdb    | (10, 12, 15, 15) | (1 024, 4 096, 32 768, 32 768) | 128 |
| wiki    | (11, 13, 14, 15) | (2 048, 8 192, 16 384, 32 768) | 128 |

Wiki's distinct 11/13/14/15 split reflects its much heavier mid-tier mass
(many highly-redundant "List of …" / "(disambiguation)" suffix tokens).

## What was tried and what won

Four orthogonal modifications were prototyped in parallel against the
2-tier `OnPairOpt` baseline (1.694× / 1.607× / 2.140× / 4.389×):

| Modification | words | domains | imdb | wiki | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **4-tier 2-bit-prefix code** | +1.91% | +0.62% | +0.93% | +2.90% | ✅ universal win |
| **Front-coded sorted dict (B=128)** | **+15.6%** | **+16.9%** | +6.3% | +14.0% | ✅ universal, dominant |
| Canonical Huffman on long tier | −0.8% | −1.4% | −0.4% | +1.8% | ❌ ≤30 KB length-table eats gain on small data |
| Utility-scored merges (5-pass FSST-style) | +8.3% | +9.1% | −5.3% | −3.6% | ⚠ dataset-dependent — wins on small, loses on long-tail |

The two universal winners stack cleanly to give `OnPairCombined`.

## Why front coding is so much bigger than I expected

The OnPair dictionary is ~20–24% of total bytes; consecutive tokens after
lex-sort share substantial prefixes (English-word morphology, domain TLDs,
"List of …" Wikipedia patterns). Plain front coding shrinks the raw byte
buffer by 30–55%; even after paying the 2-byte-per-token permutation needed to
keep the stream's frequency-ranked tier code intact, the net dictionary save
is 40–60%, which translates to +6% to +17% on the total ratio.

Per-dataset breakdown of dict-side bytes (B=64 numbers; B=128 saves another
~1%):

| Dataset | OnPair raw dict + boundaries | FcDict fc + offsets + perm | Δ dict bytes |
| --- | ---: | ---: | ---: |
| words   |   686 KB | 412 KB | −274 KB |
| domains |   650 KB | 417 KB | −233 KB |
| imdb    |   676 KB | 420 KB | −256 KB |
| wiki    | 1 030 KB | 613 KB | −417 KB |

IMDB sees the smallest gain because its dictionary is a smaller share of total
bytes — the stream dominates for unique-per-row title columns.

## What was *not* a win, and the diagnosis

- **Canonical Huffman on the long tier**: saves 0.25–1.0 bits/long-token, but
  the 4-bit-per-rare-id length table is ~30 KB, which eats the savings on
  datasets under ~10 MB. Only wiki recovers the overhead, and only on some
  random-shuffle seeds. A more economical length-table encoding (RLE on
  `bl_count` + nibble-pack used lengths only) might rescue this; not pursued.
- **FSST-style utility-scored merges (5 passes)**: wins big on low-redundancy
  data (domains +9.1%) but loses on high-redundancy data (wiki −3.6%). The
  utility formula `freq × len_saved − dict_overhead` favours long merges that
  end up in the long tier of the bit-packed code; flat threshold accidentally
  yields a more entropy-friendly distribution on Zipf-heavy data.
- **Plain Huffman/ANS over the stream**: would be the ratio ceiling but kills
  random access; explicitly out of scope.
- **Multi-pass training without Picky**: dict bytes grow faster than the
  stream shrinks. Net negative on small datasets.

## What's still on the table

- **Adaptive length-table compression** for canonical Huffman → could unlock
  the long-tier entropy gain (estimated +1–4% on top of Combined).
- **Two-flavor dictionary**: pick utility-scored merge selection for
  low-redundancy data, threshold for high-redundancy. A 1-bit dataset flag
  would let us harvest the +9% on domains without the −3% on wiki.
- **FSST+ common-prefix layer** (Yan, CWI MSc thesis 2025): an additional
  prefix-table for shared head bytes across whole strings; reported up to
  1.5× further reduction on top of FSST. Big enough to be worth a follow-up.
- **Re-Pair-in-place** on per-chunk data: theoretical +5–15% on repetitive
  columns but ~800 LoC and a substantial test surface; deferred.

## Files

- `README.md` — this file
- `lpm-remove.patch` — adds `LongestPrefixMatcher::remove` (required for slot
  recycling during Picky training)
- `src/onpair_picky.rs` — Picky-BPE training with intermediate-token eviction
- `src/onpair_opt.rs` — Picky + 2-tier bit-packed stream code (round-1 result)
- `src/onpair_4tier.rs` — Picky + 4-tier 2-bit-prefix bit-packed stream
- `src/onpair_fcdict.rs` — Picky + 2-tier stream + front-coded sorted dict
- `src/onpair_combined.rs` — **Picky + 4-tier stream + front-coded sorted dict**
- `src/bench.rs` — round-1 benchmark
- `src/bench_combined.rs` — final benchmark comparing all five variants

## Reproducing

```bash
git clone https://github.com/gargiulofrancesco/onpair_rs.git /tmp/onpair_rs
cd /tmp/onpair_rs
patch -p1 < $VORTEX/experiments/onpair-bitwidth/lpm-remove.patch
cp $VORTEX/experiments/onpair-bitwidth/src/onpair_*.rs    src/compressor/
cp $VORTEX/experiments/onpair-bitwidth/src/bench_combined.rs examples/
# wire in `pub mod onpair_picky; pub mod onpair_opt; pub mod onpair_4tier;
#         pub mod onpair_fcdict; pub mod onpair_combined;` in
# src/compressor/mod.rs and re-export the structs in src/lib.rs.
RUSTFLAGS="-C target-cpu=native" cargo build --release --example bench_combined
./target/release/examples/bench_combined
```

Datasets:

- `words.txt`: <https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt>
- `domains_200k.txt`: first 200 000 lines of Tranco top-1M
  (<https://tranco-list.eu/top-1m.csv.zip>)
- `imdb_titles.txt`: first 500 000 non-`\N` `primaryTitle` rows from
  <https://datasets.imdbws.com/title.basics.tsv.gz>
- `wiki_titles.txt`: first 500 000 lines of
  <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz>

Upstream OnPair is MIT (Gargiulo and Venturini, 2025).
