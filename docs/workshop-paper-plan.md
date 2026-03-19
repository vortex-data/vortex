# Workshop Paper Plan: LIKE Predicate Pushdown on FSST-Compressed Strings via DFA

## Working Title

**"Scanning Compressed Strings: DFA-Based LIKE Pushdown for FSST-Encoded Columnar Data"**

Alternative: *"Zero-Decompression Pattern Matching on FSST-Compressed Strings"*

---

## 1. Core Contribution (Thesis)

LIKE pattern matching (`prefix%` and `%needle%`) can be evaluated **directly on
FSST-compressed code streams** by constructing a DFA over the symbol table,
avoiding decompression entirely. This yields significant speedups for
string-heavy analytical queries on columnar stores.

---

## 2. Paper Outline

### Abstract
- FSST is a lightweight string compression scheme used in columnar databases
- LIKE predicates are expensive because they typically require full decompression
- We present a technique to compile LIKE patterns into DFAs over FSST symbol tables
- Enables direct scanning of compressed code streams
- Demonstrate X× speedup over decompress-then-match on real-world datasets

### 2.1 Introduction & Motivation
- String columns dominate many analytical workloads (logs, URLs, user agents, names)
- FSST achieves good compression ratios with fast decompression
- But LIKE predicates still force decompression → bottleneck
- Opportunity: FSST's symbol table is a finite, known alphabet → amenable to DFA construction
- Key insight: compile the pattern + symbol table into a flat transition table, scan codes directly

### 2.2 Background
- **FSST compression**: symbol table of up to 255 multi-byte symbols (1-8 bytes each), code 255 = escape for literal bytes
- **LIKE semantics**: SQL LIKE with `%` (any substring) and `_` (single char) wildcards
- **KMP algorithm**: failure function for efficient substring search
- **Predicate pushdown in columnar stores**: zone maps, dictionary pushdown, etc.

### 2.3 DFA Construction (Technical Core)

#### 2.3.1 Pattern Classification
- `prefix%` → prefix DFA (accept state is sticky, mismatch → fail)
- `%needle%` → contains DFA (KMP-based failure transitions, accept is sticky)
- Unsupported shapes fall back to decompression

#### 2.3.2 Four-Stage Construction Pipeline
1. **KMP failure function** over pattern bytes
2. **Byte-level transition table**: `(state × byte) → state`
3. **Symbol-level transition table**: simulate multi-byte symbols through byte table
4. **Flat u8 table**: fused `transitions[state * 256 + code]` for branchless lookup

#### 2.3.3 Escape Handling
- Code 255 (escape) maps to sentinel state
- Separate `escape_transitions` table for literal byte fallback
- Keeps hot path (non-escape codes) branch-free

#### 2.3.4 Correctness Argument
- KMP guarantees no missed matches for patterns with internal repetition
- Symbol simulation correctly handles cross-symbol pattern boundaries
- Fuzzing validates bit-for-bit equivalence with decompress-then-match

### 2.4 Integration into Vortex
- Brief description of Vortex's layout reader / predicate pushdown framework
- How FSST LIKE pushdown fits as a `LikeKernel` implementation
- Composability with zone-map pruning (prune zones first, then DFA scan survivors)

### 2.5 Experimental Evaluation (see §3 below)

### 2.6 Related Work
- FSST (Boncz et al., VLDB 2020)
- Predicate pushdown in Parquet, ORC, Delta Lake
- Dictionary-based LIKE evaluation (e.g., DuckDB dict pushdown)
- Compressed pattern matching literature (Navarro & Raffinot, compressed text indexing)
- SIMDified string matching (Hyperscan, Teddy)
- Column sketch approaches (approximate filtering)

### 2.7 Limitations & Future Work
- Currently limited to prefix% and %needle% patterns
- Suffix patterns (%suffix) possible with backward scan
- Escape-folded DFAs for branchless execution on short patterns
- SIMD-vectorized multi-row DFA evaluation
- Extension to regex pushdown (finite pattern subset)
- ILIKE (case-insensitive) support

### 2.8 Conclusion

---

## 3. Experiments to Run

### 3.1 Micro-Benchmarks (DFA scan throughput)

**Goal**: Measure raw scanning speed of DFA vs. decompress-then-match.

| Variable               | Values                                                        |
|------------------------|---------------------------------------------------------------|
| Pattern type           | `prefix%`, `%needle%`                                         |
| Pattern length (bytes) | 1, 4, 8, 16, 32, 64, 128, 253                                |
| String length dist     | Short (10-20B), medium (50-100B), long (200-1000B)            |
| Selectivity            | 0.01%, 1%, 10%, 50% (fraction of rows matching)              |
| Dataset                | Synthetic random, English text, URLs, log lines, UUIDs        |

**Metrics**: Throughput (GB/s of compressed data), speedup over baseline.

**Baseline**: Decompress all → run Arrow LIKE kernel.

### 3.2 Symbol Table Quality Impact

**Goal**: Understand how FSST symbol table quality affects DFA performance.

| Variable                   | Values                                       |
|----------------------------|----------------------------------------------|
| Symbol table training size | 100, 1K, 10K, 100K rows                     |
| Compression ratio achieved | Measure and correlate with DFA speedup       |
| Escape byte frequency      | Low (<5%), medium (10-20%), high (>30%)      |

**Hypothesis**: Higher compression ratio → fewer codes per string → fewer DFA
transitions → faster scan. High escape rates degrade performance due to
sentinel branching.

### 3.3 End-to-End Query Benchmarks

**Goal**: Measure impact on realistic analytical queries in the Vortex file reader.

**Queries** (on real/semi-real datasets):
1. `SELECT count(*) FROM t WHERE url LIKE 'https://example.com%'` (prefix, high selectivity filter)
2. `SELECT * FROM t WHERE message LIKE '%error%'` (contains on log data, low selectivity)
3. `SELECT user_agent, count(*) FROM t WHERE user_agent LIKE '%Chrome%' GROUP BY user_agent` (contains + aggregation)
4. `SELECT * FROM t WHERE path LIKE '/api/v2%' AND status > 400` (compound predicate with LIKE + numeric)

**Datasets**:
- **ClickBench** (hits table: URL, Title, SearchPhrase columns)
- **GitHub Archive** event payloads
- **Common Crawl** URL dataset
- **Synthetic log data** (structured log lines with varying entropy)

**Measurements**:
- Wall-clock query time (with and without DFA pushdown)
- Bytes read from storage (to show I/O reduction from zone pruning + DFA)
- Decompression time saved

### 3.4 Comparison with Alternative Approaches

**Goal**: Position DFA pushdown against other string filtering strategies.

| Approach                       | Description                                     |
|--------------------------------|-------------------------------------------------|
| **Baseline (decompress+match)**| FSST decompress → Arrow LIKE kernel             |
| **DFA pushdown (this work)**   | Direct DFA scan on compressed codes             |
| **Dictionary pushdown**        | Evaluate LIKE on dict values, map through codes  |
| **Zone-map only**              | Prune zones using min/max string stats           |
| **Uncompressed scan**          | No FSST at all, scan raw strings                 |

Measure each on the same datasets. Dictionary pushdown is the closest
competitor — it avoids per-row decompression but requires materialized
dictionary. DFA pushdown works on FSST which doesn't have an explicit
dictionary in the same way.

### 3.5 Construction Cost Amortization

**Goal**: Show that DFA construction is negligible relative to scan time.

- Measure DFA construction time vs. number of rows scanned
- Find break-even point (how many rows before construction cost is amortized)
- Typical construction: O(pattern_len × 256) for symbol table, should be microseconds

### 3.6 Scalability

**Goal**: Show behavior at scale.

- Vary number of rows: 1K, 10K, 100K, 1M, 10M, 100M
- Measure throughput stability (should be constant GB/s)
- Memory footprint of DFA (flat table = 256 × num_states bytes, always < 64KB)

---

## 4. Datasets to Prepare

| Dataset          | Source                        | String columns        | Approx size  |
|------------------|-------------------------------|-----------------------|--------------|
| ClickBench       | clickhouse.com/benchmark      | URL, Title, etc.      | ~75GB        |
| GitHub Archive   | gharchive.org                 | repo name, event type | ~50GB/month  |
| Common Crawl URLs| commoncrawl.org               | URL                   | Variable     |
| Synthetic logs   | Generate with templates       | message, path, UA     | Configurable |
| TPC-H            | Standard benchmark            | c_comment, o_comment  | Scale factor |

---

## 5. Key Figures to Produce

1. **Bar chart**: Speedup of DFA pushdown vs. decompress-then-match, grouped by pattern type and dataset
2. **Line chart**: Throughput (GB/s) vs. pattern length
3. **Line chart**: Throughput vs. escape byte frequency
4. **Stacked bar**: Query time breakdown (I/O, decompression, matching, other) with and without DFA pushdown
5. **Table**: End-to-end query times on ClickBench queries
6. **Scatter plot**: Compression ratio vs. DFA speedup (to show correlation)
7. **Bar chart**: DFA construction time vs. scan time for various row counts

---

## 6. What We Need to Build/Prepare

### Code
- [x] Core DFA implementation (done)
- [x] LikeKernel integration (done)
- [x] Fuzzer for correctness validation (done)
- [ ] Benchmark harness for micro-benchmarks (partially done in `encodings/fsst/benches/fsst_like.rs`)
- [ ] End-to-end query benchmark scripts
- [ ] Comparison benchmark with dictionary pushdown disabled/enabled
- [ ] Benchmark with varying escape byte frequencies (synthetic symbol tables)
- [ ] DFA construction time measurement

### Data
- [ ] Download/prepare ClickBench dataset in Vortex format
- [ ] Generate synthetic log datasets with controlled characteristics
- [ ] Prepare Common Crawl URL sample

### Writing
- [ ] Draft introduction and motivation
- [ ] Technical section with algorithm description and diagrams
- [ ] DFA transition table diagram (visual)
- [ ] Experimental results and analysis
- [ ] Related work survey

---

## 7. Target Venues

- **VLDB Workshop** (e.g., ADMS - Accelerating Analytics and Data Management Systems)
- **SIGMOD Workshop** (e.g., DAMON - Data Management on New Hardware)
- **CIDR** (industry track, if expanded with more Vortex integration story)
- **DaMoN** (standalone workshop at SIGMOD)

---

## 8. Timeline Estimate

| Phase                    | Tasks                                          |
|--------------------------|-------------------------------------------------|
| **Phase 1: Benchmarks**  | Build harness, run micro + e2e benchmarks       |
| **Phase 2: Analysis**    | Analyze results, identify story                 |
| **Phase 3: Writing**     | Draft paper, produce figures                    |
| **Phase 4: Polish**      | Internal review, revise, submit                 |

---

## 9. Open Questions

1. Should we include Parquet LIKE performance as a baseline? (It doesn't have FSST, so it's apples-to-oranges, but readers may expect it)
2. Do we want to benchmark against DuckDB's FSST LIKE handling (if any)?
3. Should the paper scope include the broader Vortex pushdown framework, or focus narrowly on the DFA technique?
4. Is there a compelling real-world query workload we can cite from production use?
