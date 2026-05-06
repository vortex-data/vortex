# FSST LIKE Paper Draft

## Working Title

Exact LIKE Evaluation over FSST-Compressed Strings

## One-Paragraph Summary

This paper argues that a useful subset of SQL `LIKE` predicates can be evaluated exactly on
FSST-compressed strings without first decompressing them. The core idea is to compile literal
pattern shapes such as `prefix%`, `%suffix`, `%needle%`, and ordered multi-segment patterns like
`%seg1%seg2%` into automata that operate directly on FSST code streams, while falling back to the
ordinary decoded path for unsupported patterns. The goal is not to support all of `LIKE`, but to
show that common string filters can stay in compressed space, preserve exact semantics, and reduce
scan cost on realistic analytical string columns.

## Brief Paper Outline

### 1. Introduction

Compressed string columns are common in analytical storage, but string filtering often forces
decompression. We show that several common `LIKE` predicates can be evaluated exactly over
FSST-encoded data.

### 2. Background

Explain FSST encoding, symbol codes, escapes, and why symbol-aware matching is harder than raw
substring matching on decoded bytes.

### 3. Technique

Describe the matcher construction from byte-level pattern matching to symbol-aware automata over
FSST code streams. Emphasize exactness, conservative fallback, and the supported pattern classes.

### 4. Correctness

Show that encoded-space evaluation is semantically identical to decoded `LIKE` for supported
patterns, and that unsupported patterns safely fall back.

### 5. Evaluation

Measure whether encoded-space `LIKE` improves scan performance, how results vary across datasets
and pattern types, and when the compressed-space path wins over decompression.

### 6. Limitations and Future Work

Current scope excludes `_` wildcards, case-insensitive matching, and fully general `LIKE`.
Potential extensions include more pattern classes, planner integration, and end-to-end query
evaluation.

## Draft Section Summaries

- **Introduction**: motivate the problem by showing that compressed string columns are common, but
  string predicates often force decompression. State that the paper targets exact compressed-space
  evaluation for a practical subset of `LIKE` over FSST-compressed strings, with scan-path
  pushdown left as follow-up work.
- **Background**: explain FSST symbol encoding, escape bytes, and why matching on encoded symbols
  is different from matching on raw decoded bytes.
- **Technique**: describe how supported `LIKE` patterns are parsed, compiled into symbol-aware
  automata, and evaluated directly on FSST code streams, with conservative fallback for unsupported
  cases.
- **Correctness**: show that supported encoded-space evaluation is exact and that unsupported
  patterns retain correctness by falling back to the ordinary decoded path.
- **Evaluation Setup**: define the datasets, columns, mined workloads, baselines, and
  measurement protocol used in the experiments.
- **Evaluation Results**: report correctness, kernel-level performance, sensitivity to column and
  pattern structure, and practical relevance on mined benchmark queries.
- **Limitations**: clarify the current scope, including unsupported pattern classes and the fact
  that broader planner-level pushdown is separate from the compressed-space mechanism itself.
- **Future Work**: outline extensions such as more `LIKE` forms, richer workload mining, stronger
  engine integration, and full end-to-end query evaluation.

## Evaluation Goals

The evaluation section should answer four questions.

### RQ1: Correctness

Does encoded-space evaluation produce exactly the same boolean results as decoded `LIKE` on the
same strings?

### RQ2: Kernel Performance

How much faster is FSST-space `LIKE` than:

- scanning raw uncompressed strings
- decompressing FSST then running `LIKE`

### RQ3: Sensitivity

How do results vary with:

- pattern class: prefix, suffix, contains, multi-contains
- selectivity
- string length distribution
- compression ratio
- dataset structure

### RQ4: Practical Relevance

How often do real analytical string columns contain needle patterns that are both:

- common enough to matter in benchmark workloads
- supported by the encoded matcher

## Scope of the First Evaluation

The first version of the paper should explicitly be a compressed-space evaluation paper, not a full
scan-path or query-engine paper.

That means:

- we measure kernel-level `LIKE` cost on arrays
- we compare encoded-space matching against decoded baselines
- we use realistic columns and realistic mined patterns
- we do not require scan-level evidence for the core claim
- we leave scan-path or SQL-planner integration to follow-up work

If we later wire the experiment into a Vortex scan path or a DataFusion/DuckDB path, that becomes a
follow-up systems paper or extension rather than the core story here.

## Recommended Positioning

The paper should be positioned as a compressed-space evaluation paper, not as a pushdown paper.

That means the core story is:

- primary mechanism: exact compressed-space evaluation for a useful subset of `LIKE` over
  FSST-encoded strings
- follow-up systems direction: this mechanism can later be integrated into a scan path to enable
  pushdown for supported patterns
- required evidence: realistic kernel-level evaluation on benchmark-derived workloads
- supporting evidence: kernel-level measurements that explain where the wins come from

This matters because "pushdown" can mean several different things:

- evaluation on encoded arrays without decompression
- evaluation inside a file scan
- evaluation pushed through a SQL planner

Without scan-level measurements, we should avoid making pushdown the main empirical claim. The
cleaner claim is exact compressed-space `LIKE` evaluation, with pushdown framed as future
integration work.

## Dataset Plan

We should use multiple datasets because FSST gains depend heavily on string structure. The first
wave should start with TPC-H, TPC-DS, and ClickBench, using multiple string columns per suite.

### ClickBench

Candidate columns already fit the current benchmark shape well:

- `URL`
- `Referer`
- `Title`
- `SearchPhrase`
- `Params`

These give a good mix of:

- long structured strings
- repeated web prefixes and host fragments
- short search-like text
- high and low redundancy columns

### TPC-H

TPC-H is valuable because it is standard, easy to regenerate, and has several string-heavy fields.
We should not limit ourselves to only `lineitem.l_comment`.

Candidate columns:

- `lineitem.l_comment`
- `orders.o_comment`
- `part.p_name`
- `part.p_comment`
- `customer.c_comment`
- `supplier.s_comment`
- `customer.c_mktsegment`

This gives both:

- free-text-like comments
- categorical or semi-structured short strings

### TPC-DS

TPC-DS adds a different mix of descriptive and categorical string data. It is useful because it
has more tables and more heterogeneous text-like attributes than TPC-H.

Candidate columns:

- `item.i_item_desc`
- `item.i_brand`
- `item.i_class`
- `customer.c_email_address`
- `customer_address.ca_city`
- `customer_address.ca_state`
- `reason.r_reason_desc`
- `promotion.p_promo_name`
- `store.s_market_desc`
- `web_site.web_name`

These should give:

- short categorical strings
- descriptive medium-length strings
- structured strings such as email-like values

## Column Selection Rules

The column set should not look hand-picked for FSST. For each suite, we should deliberately cover
three column families where possible:

- comment-like or descriptive text
- structured strings such as URLs, email addresses, or path-like values
- short categorical or label-like strings

The point is not to maximize speedup. The point is to show where the method helps, where it does
not, and how those outcomes correlate with column structure.

For each chosen column, we should report at least:

- row count
- total raw bytes
- average string length
- approximate distinctness or uniqueness rate
- FSST compression ratio

That metadata gives the reader a way to interpret speedups instead of treating each column as an
opaque benchmark case.

## Dataset Scaling

For the first pass, we should keep scale modest enough to iterate quickly, but large enough to
expose real scan behavior.

Suggested starting point:

- TPC-H SF1 for development, SF10 for final numbers
- TPC-DS SF1 for development, SF10 if generation/runtime are manageable
- ClickBench using one standard shard first, then larger coverage if needed

The final paper should include at least one larger-scale setting, because compressed-space gains
matter more when scan cost dominates overhead.

Suggested evaluation tiers:

- development tier: TPC-H SF1, TPC-DS SF1, single ClickBench shard
- paper tier: at least one larger setting such as TPC-H SF10 and TPC-DS SF10
- stretch tier: broader ClickBench coverage or multiple shards if runtime is still acceptable

It is fine if not every suite reaches the same size, as long as we are explicit about it.

## Workload Construction

The workload should not be a set of hand-picked nice cases. We should derive patterns from the
data itself, then organize them into controlled categories.

### Pattern Classes

For each column, generate workloads for:

- `prefix%`
- `%suffix`
- `%needle%`
- `%seg1%seg2%`
- `%seg1%seg2%...%segN%`

The first three are required. Multi-contains is important because it is closer to realistic
SQL `LIKE` usage than pure prefix-only filtering, and the evaluation should include both:

- 2-segment patterns as the primary multi-contains case
- 3+ segment patterns as an extended case when good mined examples exist

### Realistic and Pathological Workloads

We should separate two workload families instead of mixing them together.

#### Realistic workloads

Patterns mined from the data itself. These justify practical relevance and should be the main
results in the paper.

#### Pathological workloads

We should include a small, clearly separated subsection for cases such as:

- near-maximum supported pattern byte lengths
- zero-match controls
- strings with many escape-coded bytes
- awkward multi-contains shapes

These are not the main workload. They exist to document edge behavior and implementation limits.

### Needle Mining

For each dataset column:

1. Sample a subset of strings.
2. Extract candidate prefixes, suffixes, and substrings at several byte lengths.
3. Count support in the sample.
4. Keep a diverse set of candidates across high, medium, and low selectivity.
5. Add a few control patterns that do not occur.

Suggested initial byte lengths:

- prefixes: 3, 5, 8, 12, 16, 20
- suffixes: 3, 5, 8, 12, 16, 20
- contains needles: 3, 5, 8, 12
- multi-contains segments: 3-8 byte literals, composed into ordered pairs or longer chains

For multi-contains, the mining strategy should support both short and longer ordered patterns:

- mine frequent substrings
- enumerate ordered co-occurring pairs for the main workload
- optionally extend to triples when the same row sample yields stable ordered co-occurrence
- keep a small number of representative patterns per selectivity bucket

For suffixes, the same mining process should be run on reversed strings so that suffix generation
is symmetrical with prefix generation rather than an afterthought.

### Selectivity Buckets

Each pattern should be labeled by estimated match fraction:

- high selectivity: `> 10%`
- medium selectivity: `1% - 10%`
- low selectivity: `< 1%`
- zero-match controls

These thresholds are good enough for the first paper draft, even if we later tune them.

### Supported vs Unsupported Patterns

We should keep two groups separate:

- supported encoded-space patterns
- unsupported patterns that fall back

The main evaluation should focus on supported literal `LIKE` patterns. Unsupported patterns still
matter, but mainly to show safe fallback behavior rather than performance wins.

### Query Budget per Column

To keep the workload balanced, we should set a fixed target budget per column instead of letting
some columns dominate because they happen to generate many candidates.

A reasonable starting budget is:

- 4 to 6 prefixes
- 4 to 6 suffixes
- 4 to 6 contains needles
- 2 to 4 two-segment multi-contains patterns
- 1 to 2 three-plus-segment multi-contains patterns
- 2 unsupported controls

Within each family, choose queries from multiple selectivity buckets where possible.

## Baselines

The most important comparison is not just raw strings versus FSST. We need three paths.

### Baseline A: Raw Strings

Run `LIKE` on an uncompressed string array.

This answers:

- does compressed-space evaluation beat the obvious decoded representation?

### Baseline B: FSST + Decompress + LIKE

Compress the column with FSST, then decode before running `LIKE`.

This answers:

- how much of the gain comes specifically from avoiding decompression?

### Baseline C: FSST Encoded-Space LIKE

Run the matcher directly on the FSST code stream.

This is the proposed method.

### Optional Baseline D: End-to-End Scan

If we later integrate the benchmark into file scans or query-engine execution, we can add:

- Vortex file scan with encoded matcher
- Vortex file scan with fallback/decode path
- Parquet or raw-engine reference path

This is follow-up work rather than part of the core evaluation plan for the first paper.

## Evaluation Tiers

To avoid over-claiming, the evaluation section should be structured in tiers.

### Tier 1: Core kernel evaluation

This remains mandatory because it explains the mechanism:

- arrays only
- mined workloads plus a small pathological subsection
- exact correctness checks
- raw versus encoded-space versus decompress-then-like baselines

### Tier 2: Scan-level evaluation

This is follow-up work:

- read Vortex files rather than rebuilding arrays from sidecar strings
- measure whether the kernel benefit survives scan overheads
- evaluate the mechanism through the scan path on benchmark-derived columns

### Tier 3: Query-engine evaluation

This is the stretch goal:

- DataFusion or DuckDB filter execution through a real scan stack
- planner-visible pushdown story

If Tier 2 and Tier 3 do not happen, the paper can still be strong because the core claim is the
compressed-space mechanism itself.

## Metrics

Each result row should include:

- query pattern
- pattern class
- dataset and column
- estimated and actual selectivity
- row count
- average string length
- FSST compression ratio
- raw scan time
- FSST encoded-space scan time
- FSST decompress-then-like time
- speedup over raw
- speedup over decompress-then-like
- match count

For plots, the most useful derived metrics are:

- median latency per query
- rows/sec
- relative speedup
- speedup versus compression ratio
- speedup versus match selectivity

Recommended aggregate views:

- median speedup by dataset
- median speedup by column family
- median speedup by pattern class
- win/loss count versus the decompress baseline
- speedup as a function of pattern length
- speedup as a function of average string length

## Correctness Methodology

Correctness must be part of the evaluation, not just the implementation section.

For every generated pattern:

- run decoded `LIKE`
- run encoded-space `LIKE`
- compare result bitmaps exactly

In addition:

- retain fuzzing as randomized evidence
- include unsupported-pattern fallback checks
- include long-pattern boundary cases at the implementation limits

The paper should state clearly that supported patterns are exact, not approximate.

## Experimental Protocol

The first protocol should be simple and reproducible.

- Build arrays from the same source strings.
- Train one FSST model per column.
- Run warmup iterations before timing.
- Report median of repeated timed runs.
- Run on an otherwise idle machine.
- Use release builds for final numbers.

Suggested initial settings:

- warmup: 3 iterations
- timed runs: 10 iterations
- report median latency

For final paper numbers, we should also record:

- CPU model
- memory size
- compiler version
- build profile

To reduce noise, final measurements should also record:

- whether data was cold or warm in the page cache
- whether the process was pinned or otherwise isolated
- whether FSST training time is included or excluded

My recommendation is to exclude training time from the main kernel results and report it
separately, because the paper is about predicate evaluation on already-compressed data.

## Threats to Validity

The evaluation section should discuss likely reviewer objections directly.

### Mining bias

If we mine needles from the same data we evaluate on, we may over-represent frequent easy cases.
This is acceptable as long as:

- we state that the goal is realistic workload construction
- we include pathological workloads separately
- we report zero-match and low-selectivity controls

### Column-selection bias

If we only show URL-like columns or only show comment fields, the results will not generalize.
This is why the benchmark should deliberately span multiple column families.

### Scale bias

Single-shard ClickBench or only-SF1 TPC-H numbers are useful for development, but not strong final
evidence on their own. The final paper should include at least one materially larger setting.

### Claim inflation

Kernel-level speedups do not automatically imply planner-level or query-level speedups. The paper
must separate these claims. For this paper, pushdown should be discussed as future integration work
unless scan-level evidence is actually added.

## Expected Tables and Figures

The methodology becomes much clearer if we know the intended outputs early.

### Table 1: Dataset inventory

For each dataset-column pair:

- rows
- raw bytes
- avg string length
- compression ratio
- column family

### Table 2: Workload inventory

For each pattern family:

- number of queries
- selectivity distribution
- supported versus unsupported count

### Figure 1: Speedup by pattern class

Compare encoded-space `LIKE` against:

- raw strings
- decompress-then-like

### Figure 2: Speedup versus compression ratio

Show whether high compression correlates with higher evaluation wins.

### Figure 3: Speedup versus string length or selectivity

Useful for explaining where the method wins and where it flatlines.

## What We Need to Add Next

The current branch already has a reasonable start for:

- ClickBench columns
- TPC-H `lineitem.l_comment`
- mined prefix and contains queries
- raw vs FSST vs decompress baselines

The next methodology work should be:

1. add more TPC-H columns
2. add TPC-DS columns
3. add suffix mining
4. add multi-contains mining
5. record dataset and column metadata in benchmark output
6. leave scan-path integration as follow-up work

## Immediate Next Steps

1. Lock the first-pass column matrix for ClickBench, TPC-H, and TPC-DS.
2. Extend dataset preparation so those columns can all be extracted reproducibly.
3. Add suffix mining to match the current prefix and contains mining flow.
4. Add mined multi-contains generation for ordered pairs, then extend to 3+ segments where viable.
5. Extend benchmark output so every result row includes dataset, column, pattern class,
   selectivity, compression ratio, and string-length metadata.
6. Keep a small pathological subsection with zero-match controls, near-limit patterns, and
   escape-heavy rows.
7. Leave scan-path and SQL-engine integration as follow-up work after the core paper shape is
   stable.

## Recommended Defaults

If we want to move quickly, these are the defaults I would use unless we decide otherwise.

1. Main claim: exact compressed-space `LIKE` evaluation on FSST-encoded strings.
2. Main benchmark unit: benchmark-derived mined queries over real columns.
3. Dataset coverage: at least 3 to 5 columns per suite, spanning different column families.
4. Workload mix: mined realistic literals first, with a very small pathological subsection.
5. Multi-contains: include both 2-segment and 3+ segment cases, but keep 2-segment as the main
   representative case if longer patterns are sparse.

## Open Questions

These are the main questions to resolve before locking the evaluation section.

1. Which first-pass column matrix should we lock for TPC-H, TPC-DS, and ClickBench?
2. How large should the final paper tier be for each suite?
3. How many mined queries per column do we want in the main workload table?
4. Do we want 3+ segment multi-contains in the main paper figures, or mainly in a secondary
   subsection?
