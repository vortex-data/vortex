# Workshop Paper Plan: LIKE Predicate Pushdown on FSST-Compressed Strings via DFA

## Working Title

**"Scanning Compressed Strings: DFA-Based LIKE Pushdown for FSST-Encoded Columnar Data"**

---

## 1. Core Contribution (Thesis)

LIKE pattern matching (`prefix%` and `%needle%`) can be evaluated **directly on
FSST-compressed code streams** by constructing a DFA over the symbol table,
avoiding decompression entirely. This yields significant speedups for
string-heavy analytical queries on columnar stores.

---

## 2. Interesting Things to Include in the Paper

Beyond just "we built a DFA and it's fast," here are the dimensions that make
this technically interesting and give reviewers something to think about.

### 2.1 The Multi-Byte Symbol Problem (core novelty)

This is not just "run a DFA on bytes." The hard part is that FSST symbols are
1–8 bytes long, so a single code byte can advance the DFA by multiple byte-level
states. The key insight:

- **Stage 3 (symbol simulation)**: For each `(state, symbol_code)` pair, we
  feed the symbol's bytes one-at-a-time through the byte-level KMP table and
  record the resulting state. This is the step that doesn't exist in standard
  DFA literature.

- **Cross-symbol boundary matching**: The pattern "abcdef" might straddle a
  3-byte symbol "abc" and a 3-byte symbol "def". Or it might straddle a 2-byte
  symbol "ab", a 3-byte symbol "cde", and need the 'f' from the next symbol.
  The multi-byte simulation handles this naturally.

- **KMP fallback across symbols**: Consider needle "abab" and a 4-byte symbol
  "abac". Processing this symbol from state 0: a→1, b→2, a→3, c→? KMP failure
  at state 3 falls back to state 1 (matched "a"), then c mismatches again → 0.
  This is critical for correctness and happens transparently in stage 3.

**Paper angle**: Walk through a worked example showing how the same pattern can
be matched via different symbol decompositions, and why the staged construction
is correct for all of them.

### 2.2 Escape Handling: The Sentinel Architecture

The escape code (255) is FSST's way of encoding literal bytes that don't have
a symbol. The DFA uses a **sentinel state** approach:

- In the fused table, code 255 → sentinel value (not a real state)
- Scanner detects sentinel, reads next byte, looks up in separate byte-level table
- This keeps the **hot path** (symbol codes 0-254) branch-free: one table lookup
- The **cold path** (escapes) adds one branch + one extra lookup

**Paper angle**: Quantify how escape frequency affects throughput. The sentinel
architecture means the DFA gracefully degrades as escape rate increases, rather
than failing entirely. Plot the throughput curve as a function of escape rate.

### 2.3 State Space Fits in u8

The DFA uses `u8` state IDs, so the flat table is exactly `n_states × 256`
bytes. For a 10-byte needle, that's `12 × 256 = 3072 bytes` — fits in L1 cache.
Even the maximum 254-byte needle only produces `65,536 bytes` = 64KB.

**Paper angle**: Compare with general-purpose regex engines that use `u32` or
pointer-based NFAs. The u8 constraint limits pattern length but gives guaranteed
cache-resident execution.

### 2.4 Prefix vs Contains: Fundamentally Different DFA Shapes

The prefix DFA is trivial — any mismatch goes to a sticky FAIL state (no
backtracking). The contains DFA needs KMP failure transitions because the needle
can start at any position in the string.

**Paper angle**: Show that prefix matching is ~free (almost always faster than
checking a single byte in decompressed form), while contains matching is where
the interesting performance tradeoffs live.

### 2.5 The Compression–Pushdown Tradeoff

Better FSST compression means:
- Fewer codes per string → fewer DFA transitions → faster DFA
- Fewer escape codes → fewer branches → faster DFA
- But also: faster decompression (fewer symbols to emit bytes for)
- The **speedup ratio** (DFA / decompress-then-match) may not be constant

**Paper angle**: Is the DFA speedup multiplicative with compression ratio, or
does it saturate? On very well-compressed data, decompression is already fast,
so the DFA's advantage might shrink. On poorly compressed data, there are more
codes to scan but also more escapes. Map out this tradeoff empirically.

### 2.6 Interaction with Zone-Map Pruning

In the full Vortex pipeline, zone maps prune entire zones before the DFA runs.
The DFA only scans zones that survive pruning. This means:

- For **high-selectivity** queries (e.g., `url LIKE 'https://specific-domain%'`),
  zone maps do most of the work and the DFA only scans a few zones
- For **low-selectivity** queries (e.g., `text LIKE '%the%'`), zone maps can't
  prune much and the DFA scans everything

**Paper angle**: Show the combined benefit. Zone maps reduce the *volume* of
data; the DFA reduces the *cost per byte* of scanning what remains.

### 2.7 What the DFA Avoids

Beyond avoiding decompression, the DFA avoids:
- **Memory allocation**: Decompression needs a buffer for the uncompressed
  strings. The DFA scans in-place on the compressed codes buffer.
- **Cache pollution**: Decompressed strings are typically 2-5x larger, evicting
  other data from cache. The DFA works on the smaller compressed representation.
- **Memcpy**: Even if decompression is fast, writing those bytes to a buffer
  and then reading them back for matching costs bandwidth.

**Paper angle**: Measure cache miss rates and memory bandwidth usage, not just
wall clock time. Use `perf stat` to show L1/L2/L3 cache misses and branch
mispredictions.

---

## 3. Detailed Experiments

### 3.1 Experiment A: Real-World Symbol Table Distributions

**Motivation**: Understand what FSST symbol tables actually look like on real
data, since this determines DFA performance characteristics.

**Data sources** (use actual data, not synthetic combinations):
- **ClickBench** `URL` column (~100M real web URLs from Yandex.Metrica)
- **ClickBench** `Title` column (page titles, mixed languages)
- **ClickBench** `SearchPhrase` column (real search queries, very diverse)
- **FineWeb** `url` column (Common Crawl URLs)
- **FineWeb** `text` column (web page content, English prose)
- **TPC-H** `l_comment` column (synthetic but realistic business text)

**Metrics per dataset**:
- Number of active symbols (out of 255 max)
- Symbol length distribution histogram (1-byte through 8-byte)
- Mean symbol length (key predictor of DFA performance)
- Escape rate (% of codes that are escape+literal pairs)
- Code entropy (Shannon entropy of the code stream)
- Effective alphabet size (2^entropy)
- Symbol frequency skew (Gini coefficient or p50/p90/p99 coverage)
- Compression ratio

**Why this matters**: Reviewers will want to know if the DFA's assumptions hold
on real data. Specifically: are escape rates low enough? Are symbol tables
skewed enough that the DFA's cache behavior is good?

### 3.2 Experiment B: DFA vs Decompress-then-Match Throughput

**Motivation**: The core performance claim. Measure speedup across real datasets.

**Method**: For each dataset, compress with FSST, then time:
1. **DFA path**: Build DFA, scan compressed codes
2. **Decompress path**: Decompress all strings, run Arrow LIKE kernel

**Varying**:
- Pattern type: `prefix%` vs `%needle%`
- Pattern length: 4, 8, 16, 32, 64, 128 bytes
- Selectivity: choose patterns that match ~0.01%, ~1%, ~10%, ~50% of rows

**Use real patterns** on real data:
- ClickBench URLs: `'https://www.google%'`, `'%yandex%'`, `'%utm_source=%'`
- FineWeb URLs: `'https://en.wikipedia%'`, `'%github.com%'`
- FineWeb text: `'%however%'`, `'%the%'`, `'%Artificial Intelligence%'`

**Metrics**: GB/s of compressed data scanned, speedup ratio, ns per string.

### 3.3 Experiment C: Escape Rate Sensitivity (Controlled Sweep)

**Motivation**: Escape rate is the DFA's key weakness. Quantify the degradation.

**Method**: Generate synthetic data with controlled escape rates by manipulating
the FSST symbol table training process:
1. Train on N rows, then compress N+M rows (the M extra rows introduce novel
   byte patterns that become escapes)
2. Alternatively: take a real dataset, corrupt K% of strings with random bytes
   to force escapes
3. Or: train symbol tables on progressively smaller subsets of the data

**Sweep**: Escape rates from 0% to 50% in increments of 5%.

**Metrics**: DFA throughput, decompress throughput, speedup ratio — all as a
function of escape rate. Expect a roughly linear degradation in DFA throughput
(each escape adds one branch + one table lookup).

**Why continuous**: This gives us a clean curve, not just scattered bar charts.
Lets us extrapolate to data regimes we haven't tested.

### 3.4 Experiment D: Mean Symbol Length vs Throughput

**Motivation**: Mean symbol length determines how many DFA transitions per
string. Longer symbols → fewer transitions → faster scan.

**Method**: Use the same sweep approach — train on subsets of varying size,
or on data with different entropy levels. Measure mean symbol length and
throughput simultaneously.

**Plot**: Scatter of mean_sym_len (x) vs throughput_GB/s (y) for DFA and for
decompress. Include real datasets as named points on the same plot.

### 3.5 Experiment E: Pattern Length Scaling

**Motivation**: DFA table size grows linearly with pattern length. Does this
cause cache pressure for long patterns?

**Method**: Fix a dataset (e.g., FineWeb text). Vary needle length from 1 to
200 bytes using real substring extracts from the data.

**Metrics**: DFA construction time (ns), scan throughput (GB/s).

**Hypothesis**: Throughput should be roughly constant until the table exceeds
L1 cache (~32KB), then degrade slightly. Construction time should be
microseconds for all practical lengths.

### 3.6 Experiment F: DFA Construction Cost

**Motivation**: Show that one-time construction is cheap relative to scan.

**Method**: Time DFA construction for various patterns and symbol tables.
Compare against the scan time for 1K, 10K, 100K, 1M rows.

**Expected result**: Construction is <10μs, scan of 10K rows is >100μs,
so the construction cost is amortized after a few hundred rows.

### 3.7 Experiment G: End-to-End File Scan with Zone Pruning

**Motivation**: The DFA doesn't operate in isolation. Show the combined effect
of zone-map pruning + DFA scan in the full Vortex file reader.

**Method**: Write ClickBench/FineWeb data to Vortex files, then run LIKE
queries through the full reader pipeline with instrumentation.

**Queries**:
```sql
SELECT count(*) FROM clickbench WHERE URL LIKE 'https://www.google%'
SELECT count(*) FROM clickbench WHERE URL LIKE '%yandex%'
SELECT count(*) FROM fineweb WHERE url LIKE '%wikipedia%'
SELECT count(*) FROM fineweb WHERE text LIKE '%machine learning%'
```

**Measure**: Total query time, zones pruned, zones scanned, bytes read, DFA
scan time as a fraction of total.

### 3.8 Experiment H: Comparison with DuckDB

**Motivation**: DuckDB also uses FSST. Does it do LIKE pushdown on compressed
data? If not, we can show the gap. If yes, we compare approaches.

**Method**: Load the same ClickBench data in DuckDB. Run identical LIKE queries.
Compare wall-clock times.

**Note**: DuckDB may have other optimizations (SIMD, different scan strategy)
that make this not an apples-to-apples comparison of the DFA technique alone.
Frame carefully.

### 3.9 Experiment I: perf counters (Micro-architectural Analysis)

**Motivation**: Explain *why* the DFA is fast, not just *that* it's fast.

**Method**: Run the DFA scan and decompress-then-match under `perf stat`.
Compare:
- L1/L2/L3 cache misses
- Branch mispredictions
- Instructions per cycle (IPC)
- Memory bandwidth consumed

**Expected**: DFA has dramatically fewer cache misses (working set is the
transition table + compressed codes, both small) and fewer branch mispredictions
(the hot path is branchless for non-escape codes).

---

## 4. Datasets

| Dataset | Source | String columns | Why include |
|---------|--------|----------------|-------------|
| ClickBench `hits_0` | R2 bucket (112MB partition) | URL, Title, SearchPhrase, Referer | Real web traffic data, diverse URLs |
| FineWeb sample | HuggingFace (parquet) | url, text | Real web crawl, long English prose |
| TPC-H `lineitem` | dbgen SF=1 | l_comment | Standard benchmark, predictable |

For controlled experiments (C, D), use **FineWeb text** as the base and
perturb it rather than generating from scratch.

---

## 5. Key Figures

1. **Summary table**: Symbol table characteristics across all real datasets
2. **Bar chart**: DFA speedup vs decompress-then-match, per dataset and pattern type
3. **Line chart**: DFA throughput (GB/s) vs escape rate (0-50%), smooth curve
4. **Line chart**: DFA throughput vs mean symbol length, with real datasets overlaid
5. **Line chart**: DFA throughput vs pattern length (1-200 bytes)
6. **Stacked bar**: End-to-end query time breakdown (zone prune + DFA scan + other)
7. **Table**: DFA construction time vs scan time at various row counts
8. **perf counters table**: Cache misses, branch mispredicts, IPC for DFA vs decompress

---

## 6. Paper Outline (6-8 pages, double column)

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 1 | Motivation, FSST everywhere, LIKE is a bottleneck |
| 2. Background | 0.75 | FSST encoding, LIKE semantics, KMP |
| 3. DFA Construction | 1.5 | Four stages, worked example, escape handling, correctness |
| 4. Integration | 0.5 | Vortex pushdown framework, zone-map composition |
| 5. Evaluation | 2.5 | Experiments A-I, tables, figures |
| 6. Related Work | 0.5 | FSST, compressed matching, dictionary pushdown |
| 7. Conclusion | 0.25 | Summary, future work |

---

## 7. Target Venues

- **DaMoN** (Data Management on New Hardware) @ SIGMOD — perfect fit, micro-architectural angle
- **ADMS** (Accelerating Analytics) @ VLDB — good fit, systems focus
- **CIDR** — if we expand to full Vortex pushdown story
- **DBTest** — if we emphasize the fuzzing/correctness angle

---

## 8. Open Questions

1. Should we compare against Parquet? (No FSST, but readers may expect it)
2. Does DuckDB push LIKE into FSST? Need to check.
3. Can we get access to a production ClickBench-scale dataset (100M rows)?
4. Do we want to show the escape-folded DFA variant mentioned in the TODOs?
