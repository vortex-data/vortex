# FSST LIKE Paper Outline

## Purpose

This is the short companion note for the FSST `LIKE` paper. It captures:

- the section-by-section paper story
- the current evaluation framing
- the immediate implementation next steps

## Core Claim

- The core mechanism is exact compressed-space evaluation for a useful subset of `LIKE` over
  FSST-encoded strings.
- A follow-up systems direction is to integrate this mechanism into a scan path to enable predicate
  pushdown.
- Kernel-level measurements explain why the mechanism wins.
- Scan-level integration is follow-up work, not part of the first paper.

## Section Overview

- **Introduction**: motivate the problem by showing that compressed string columns are common, but
  string predicates often force decompression. State that the paper targets exact compressed-space
  evaluation for a practical subset of `LIKE`, with scan-path pushdown left as follow-up work.
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

## Evaluation Framing

- Main mechanism: compressed-space `LIKE` evaluation, not decompression-first evaluation.
- Main paper claim: exact compressed-space evaluation for supported patterns.
- Supporting evidence: array-level kernel benchmarks.
- Optional follow-up evidence: scan-level evaluation over Vortex files or one SQL-engine tier.

## Workload Plan

- Use real benchmark-derived mined patterns as the main workload.
- Cover `prefix%`, `%suffix`, `%needle%`, `%seg1%seg2%`, and longer `%seg1%seg2%...%segN%` cases.
- Include both 2-segment and 3+ segment multi-contains patterns.
- Keep a very small pathological subsection with:
    - zero-match controls
    - near-limit patterns
    - escape-heavy rows
    - awkward multi-contains cases

## Dataset Plan

- **ClickBench**: start with columns such as `URL`, `Referer`, `Title`, `SearchPhrase`, `Params`.
- **TPC-H**: expand beyond `lineitem.l_comment` to include fields like `o_comment`, `p_name`,
  `p_comment`, `c_comment`, `s_comment`, and `c_mktsegment`.
- **TPC-DS**: include a mix of descriptive, categorical, and structured string columns such as
  `i_item_desc`, `i_brand`, `i_class`, `ca_city`, `ca_state`, `r_reason_desc`, and `web_name`.

## Baselines

- **Raw strings**: run `LIKE` on the uncompressed string array.
- **FSST + decompress + LIKE**: compress, decode, then evaluate `LIKE`.
- **FSST encoded-space LIKE**: evaluate directly on the FSST code stream.

## Immediate Next Steps

1. Lock the first-pass column matrix for ClickBench, TPC-H, and TPC-DS.
2. Extend dataset preparation so those columns can all be extracted reproducibly.
3. Add suffix mining to match the current prefix and contains mining flow.
4. Add mined multi-contains generation for ordered pairs, then extend to 3+ segments where viable.
5. Extend benchmark output so every result row includes dataset, column, pattern class,
   selectivity, compression ratio, and string-length metadata.
6. Keep a small pathological subsection with zero-match controls, near-limit patterns, and
   escape-heavy rows.
7. Leave scan-path or SQL-engine integration as follow-up work after the core paper shape is
   stable.

## Next steps

... 