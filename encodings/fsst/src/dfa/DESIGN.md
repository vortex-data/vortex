# FSST LIKE Pre-filter — Bucketed/Streaming Teddy

Working notes from the `ji/fsst-like-paper-2` branch session covering the
bucketed Cartesian Teddy pre-filter for `FoldedContainsDfa`, its
streaming AVX-512 evolution, and the measured impact on real workloads.
This is a "merge and refine" document: the implementation lives in
`encodings/fsst/src/dfa/`; this note explains *why* and *what's still
loose*.

## TL;DR

- **What we built**: a streaming bucketed Teddy (2-byte and 3-byte
  fingerprint) pre-filter for `%needle%` contains-shape LIKE on
  FSST-compressed columns. Inline DFA verify in a single AVX-512 pass
  over `all_bytes`. Replaces a per-string 1-byte memchr + DFA verifier
  loop.
- **End-to-end win**: ClickBench Q20 (`COUNT(*) WHERE URL LIKE
  '%google%'`) on the 100-shard partitioned dataset (~100M rows):
  **187 ms** vs 350 ms no-pushdown (1.89× faster) and 236 ms via the
  existing `ClassifiedDfa` default path (1.26× faster).
- **Where the win comes from**: dramatically tighter candidate density
  (1 candidate per ~22 KB of `all_bytes` for Teddy-3 vs 1 per ~106 bytes
  for the 1-byte path) → ~25× fewer DFA verifier dispatches.
- **Caveat**: only applies when LIKE is selective (<1% match rate).
  For dense patterns (`%htt%` on URLs matching every row), raw varbin
  scan beats FSST pushdown — neither this work nor any prefilter can
  rescue that regime.

## What changed in this branch

### Production code path

The `FoldedContainsDfa::scan_to_bitbuf` ladder (in
`encodings/fsst/src/dfa/folded_contains.rs`) is:

1. **Streaming Teddy-3** (`fused_teddy_triple_scan`) — preferred when
   `accept_state ≥ 3` and progressing c1's chain through two
   intermediate non-accept normal states.
2. **Streaming Teddy-2** (`fused_teddy_pair_scan`) — fallback for short
   needles or escape-anchored c1's.
3. **1-byte progressing bitset** (`build_progressing_bitset_unbounded`
   + `scan_with_anchor_bitset`) — final fallback when the pair/triple
   paths don't apply.

Both Teddy variants implement *real* Hyperscan-style Teddy: the
per-32/64-byte candidate mask lives in a register, never in memory, and
the DFA verifier is called inline when the movemask is non-zero. This
is the second iteration — the first version (still buildable for
benches) materialized a `Vec<u64>` bitset and walked it separately.

### AVX-512

`teddy_triple_pass_avx512` uses 64-byte loads + `vpternlogq` (3-way
fused AND) + `vpcmpneqb` → `__mmask64`. ~2× the throughput of the
equivalent AVX2 path on the test machine (Xeon 6975P-C):
4.28 GB/s aggregate vs 2.74 GB/s.

### Bucket collection

`anchor_scan::collect_bucketed_pair_codes` /
`collect_bucketed_triple_codes` walk the folded transition table and
return one bucket per distinct c1, with each bucket holding the per-c1
advancing-c2 set (and, for Teddy-3, the union of advancing-c3 codes
across that c1's c2 set). This is *shared-c1 bucketing*: cross-bucket
false positives are eliminated by construction, and the within-bucket
nibble-cross FPs in the c2/c3 sets are accepted as a small selectivity
loss in exchange for keeping each bucket to one bit.

### Builder optimizations

`FoldedContainsDfa::new` was made ~1.5× faster across three changes:

| Phase | Before | After | Note |
|---|---|---|---|
| `kmp_byte_transitions` | 4.5 µs/chunk | 0.76 µs/chunk | Rewrote with the standard KMP recurrence (each row = failure row + one override) instead of running the iterative fallback per cell. |
| `build_symbol_transitions` | 8.2 µs/chunk | 6.7 µs/chunk | Swapped loop order to expose independent loads per byte step, manual 4-way unroll, raw-pointer access. |
| fused-table fill + collect | 7.0 µs/chunk | 4.4 µs/chunk | `copy_from_slice` instead of per-cell loop; skip `bucketed_pair_codes` when Teddy-3 applies. |

Single-thread aggregate `build_fsst_matcher` cost dropped from
70.8 ms → 57.6 ms on the partitioned Q20 (1.23×). Wall-clock impact is
in the noise (~2-3 ms) because build is only ~11% of LIKE work at this
shape.

## Empirical results

### ClickBench Q20, partitioned (100 shards, ~100M rows), median µs of 20 iters

| Configuration | Median | vs no-pushdown |
|---|---|---|
| No FSST pushdown (`DISABLE_LIKE_PUSHDOWN=1`) | 354.0 ms | 1.00× |
| Default routing (`ClassifiedDfa`) | 236.2 ms | 1.50× |
| **FoldedContains + AVX-512 Teddy-3 streaming** | **187.4 ms** | **1.89×** |

### Per-component breakdown of LIKE on the streaming AVX-512 path

| Component | Total (1 iter) | µs/chunk | Share |
|---|---|---|---|
| `scan_to_bitbuf` (Teddy + inline verify) | 503 ms | 109 | **79%** |
| `build_fsst_matcher` | 58 ms | 12.4 | 9% |
| `offsets_to_prim` | 47 ms | 10.2 | 7% |
| `validity` | 6 ms | 1.2 | 1% |

### Streaming Teddy-3 internals

| Metric | Value |
|---|---|
| Chunks per query | 4,632 |
| `all_bytes` scanned | 2,640 MB |
| Candidates emitted | 115,343 (1 per 22,895 bytes) |
| Aggregate throughput | 4,278 MB/s |
| Per-chunk total | 133 µs |

For comparison, the 1-byte memchr path on `ClassifiedDfa` emits
24.7M candidates over the same `all_bytes` (216× more) and verifies
99.96% of them as false positives. The 503 ms `scan_to_bitbuf` budget
is what's left after that 216× candidate reduction.

### Mined-query benchmark (string-filter-bench)

Across mined patterns from ClickBench URL/SearchPhrase columns:

| Pattern selectivity | FSST DFA vs raw varbin | FSST DFA vs decomp+like |
|---|---|---|
| <1% (selective) | **3–6× faster** | 5–20× faster |
| 1–10% | ~even | 2–5× faster |
| >10% (dense) | 3–6× slower | comparable |

The FSST DFA always beats decompress-then-LIKE (the duckdb "no pushdown"
path). Whether it beats raw varbin is selectivity-conditional.

## Where this work lights up across benchmarks

LIKE queries in the four benchmark suites that *don't* project the
LIKE'd column (where pre-filter improvements translate directly to
wall-clock):

| Query | Pattern shape | Column | Workload size | This work helps? |
|---|---|---|---|---|
| **ClickBench Q20** | contains `%google%` | URL | ~100M rows | **Yes — primary target** (1.89×) |
| TPCH Q9 | contains `%green%` | p_name (SF=10) | ~2M rows | Build path runs but join dominates (~1% LIKE budget) |
| TPCH Q2 | suffix `%BRASS` | p_type | small | Different DFA (`SuffixMatcher`) |
| TPCH Q13 | multi-contains `%special%requests%` | o_comment | ~60M rows (SF=10) | Different DFA (`MultiContainsDfa`); needs port |
| TPCH Q16 | multi-contains `%Customer%Complaints%` | s_comment | small | Same — needs port |
| TPCH Q20 | prefix `forest%` | p_name | small | Different DFA (`FlatPrefixDfa`) |
| TPC-DS Q91 | prefix `Unknown%` | hd_buy_potential | small | Different DFA |

So the streaming Teddy work *directly* helps exactly one query in the
mined benchmark suite at scale (ClickBench Q20). The next most useful
investment is porting the same scheme into `MultiContainsDfa` (TPCH
Q13 / Q16) and into `ClassifiedDfa` (the default routing path for
single-segment contains).

## What's still open

### Hot
- **Port streaming Teddy into `ClassifiedDfa`** — the default routing
  uses a per-byte memchr + DFA loop and doesn't benefit from any
  bucketed-Teddy work. Mechanical port; expected to lift the *default*
  Q20 number from 236 → ~190 ms (matching the `DISABLE_CLASSIFIED`
  number).
- **`MultiContainsDfa` Teddy port** — enables TPCH Q13/Q16. Requires
  some new scaffolding for cross-segment verify.
- **SSA-merge Teddy path** — today `scan_to_bitbuf` skips Teddy
  entirely when `single_step_accept_codes.is_some()` and falls to the
  1-byte progressing bitset (folded_contains.rs:355). On URL data this
  fires for short ambient needles like `%htt%` / `%ear%` because the
  FSST trainer mints SSA symbols (e.g. `http`, `https`, `https://`,
  `search`) — any such symbol is a one-step accept from state 0, so
  the pair scheme can't anchor on it. The 1-byte fallback then runs
  on a dense bitset and verifies most positions, producing the 6×
  regression vs raw varbin reported in
  `fsst_prefilter_compare`. Proposed scheme: build Teddy-2/3 over the
  **non-SSA** progressing codes, separately build a 1-byte PSHUFB
  bitset over the **SSA** codes, and OR the two candidate streams
  inside `fused_teddy_*_scan`. Verifier still inline, candidate set =
  "starts an SSA symbol" ∪ "matches the c1·c2(·c3) fingerprint";
  correctness preserved, candidate density drops to roughly
  `|SSA| / 256 + Teddy-density`. Expected to flip `%htt%` / `%ear%`
  from "6× slower than raw varbin" to "comparable or faster" without
  touching the selective regime.
- **Dense-pattern short-circuit** — even with SSA-merge Teddy, a
  needle whose SSA codes alone cover most bytes (`%htt%` on URL data)
  will produce a near-saturated candidate bitmap. After building the
  SSA 1-byte bitset, popcount-sample it; if the candidate density
  exceeds a threshold (e.g. >25% of `all_bytes` are SSA bytes), skip
  the DFA entirely and either return constant-true (when the SSA code
  alone implies the full needle, which is the SSA definition) or fall
  through to a materialize-and-memmem path. The raw-varbin 7.4 ms
  floor on `%htt%` is the lower bound we're trying to match here, not
  beat — the goal is "don't pay 45 ms to compute what memmem could
  tell us in 7".

### Medium
- **Build-time variant counters** — `FoldedContainsDfa::new` already
  collects `progressing_codes`, `single_step_accept_codes`,
  `bucketed_pair_codes`, `bucketed_triple_codes` but doesn't surface
  their cardinalities. Adding a one-line print under a trace flag
  (counts of each set, plus `accept_state`) makes it cheap to confirm
  *why* a given pattern took the path it did — e.g. proving the
  `%htt%` regression is the SSA-fallback branch rather than the DFA
  itself. Required to validate the SSA-merge Teddy work above.
- **`offsets_to_prim` allocation** — 10 µs/chunk, ~47 ms single-thread
  on Q20. The conversion materializes a fresh primitive array per call.
  Could be elided by reading offsets in-place.
- **Layout-chunk size** — 570 KB/chunk × 4632 chunks. Per-chunk fixed
  costs (matcher build + offsets conversion + dict scatter) total
  ~22 µs/chunk = ~100 ms single-thread. Bigger chunks halve this.
- **Dict scatter cost** — invisible in current traces. Each filter
  result is wrapped through `vortex.dict(bool?) → vortex.bool` to
  scatter from dict values back to row positions. Likely 10-30 µs/chunk;
  could be elided by keeping the result dict-encoded.

### Lower priority
- **Fat Teddy (16 buckets)** for multi-pattern LIKE OR — enables
  ClickBench Q23.
- **Bucket-bit-aware verifier dispatch** — Hyperscan-style per-bucket
  verify. Marginal for single-needle case.
- **NEON / WASM SIMD** — portability, not raw perf.

## Code surface summary

| File | Lines | Role |
|---|---|---|
| `dfa/anchor_scan.rs` | 2,467 | Pre-filter primitives — Teddy variants, bucket collection, AVX2/AVX-512/scalar dispatch |
| `dfa/folded_contains.rs` | 755 | `FoldedContainsDfa` matcher; `scan_to_bitbuf` dispatch ladder; bench accessors |
| `dfa/mod.rs` | 582 | `FsstMatcher` enum dispatcher; KMP + symbol-transition builders |
| `dfa/multi_contains.rs` | 308 | Multi-segment `%seg1%seg2%` DFA (not Teddy-ised yet) |
| `dfa/flat_contains.rs` | 188 | Long-needle (≥128 bytes) contains; not Teddy-ised |
| `dfa/{prefix,suffix,skip}.rs` | ~480 | Other LIKE shapes |
| `dfa_compressed/mod.rs` | ~1,200 | `ClassifiedDfa` (default routing, not Teddy-ised) |
| `benches/fsst_prefilter_compare.rs` | 280 | Microbench A/B-ing the prefilter variants |

### Test/debug surface (consider pruning)

12 environment variables across the FSST crate:

**Keep** (useful for production debugging / A/B):
- `VORTEX_FSST_DISABLE_LIKE_PUSHDOWN` — pushdown kill switch
- `VORTEX_FSST_DISABLE_CLASSIFIED` — route through `FoldedContains`
- `VORTEX_FSST_LIKE_TRACE` — per-component timing on each LIKE call

**Probably drop** (debugging instrumentation that served its purpose):
- `VORTEX_FSST_BUILD_TRACE` — DFA-build phase timing (used to identify
  KMP/sym_trans bottlenecks; not needed in production)
- `VORTEX_FSST_BUCKET_BUILD_TRACE`, `VORTEX_FSST_STREAM_TRACE` — fine-
  grained Teddy internals
- `VORTEX_FSST_TEDDY_DEEP_TRACE`, `VORTEX_FSST_TEDDY_TRACE` — overlap
- `VORTEX_FSST_FORCE_ONE_BYTE`, `VORTEX_FSST_FORCE_TEDDY_PAIR` —
  experimental controls superseded by the cascade
- `VORTEX_FSST_DFA_TRACE` — `ClassifiedDfa` internal counters; keep
  while ClassifiedDfa is on the default path
- `VORTEX_FSST_LIKE_TIMELINE` — per-thread wall-clock; rarely used

### Dead code in production (no callers without `_test-harness`)

- `anchor_scan::build_progressing_bitset` (bounded variant; replaced by
  unbounded)
- `anchor_scan::range_has_hit` (not called)
- `anchor_scan::collect_progressing_codes` (bounded; replaced by
  unbounded)
- `anchor_scan::build_bucketed_{pair,triple}_bitset` and their fills —
  only the bench uses them
- `anchor_scan::build_pair_bitset`, `fill_two_bitsets`, `collect_pair_codes`
  — legacy Cartesian path, only the bench A/B's it

## Open simplifications

See the chat thread for the running list. Currently considering:

1. Delete pure dead code (`build_progressing_bitset`, `range_has_hit`,
   `collect_progressing_codes`).
2. Choose: keep the materialized Teddy variants under `_test-harness`
   for the bench, OR rewrite the bench against the streaming variants
   and delete the materialized ones entirely.
3. Collapse the 12 trace knobs to 3–4.
4. Split `anchor_scan.rs` into per-strategy files.
5. Move bench accessors off `FoldedContainsDfa` into a separate
   test-harness module so the matcher type itself has a narrower public
   surface.
