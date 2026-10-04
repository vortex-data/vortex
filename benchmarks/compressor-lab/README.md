# Learned compression scheme selection: plan and learnings

Status: **design parked.** The feasibility study in `benchmarks/compressor-feasibility` found
~7.7% one-step headroom on integers, mostly cascade-estimation misses, and a large win from a
decode-aware objective at high bandwidth. See its README before building the full system.

## Goal

For each dtype, choose a compression scheme at each node of the cascade by learning a small set
of stats and decision points. The selection should optimise size and also decode and pushdown
cost. The aim is to beat today's hand-set thresholds and to avoid most of the ~1% sample
compression that the compressor does today.

## How the compressor chooses today

- `CascadingCompressor` (`vortex-compressor`) picks a scheme per node in two passes
  (`compressor/select.rs`):
  - **Pass 1:** each scheme's `expected_compression_ratio` returns `Skip`, `AlwaysUse` or
    `Ratio` from cheap stats.
  - **Pass 2:** schemes that return `Deferred` compress a stratified ~1% sample
    (`compressor/sample.rs`) or run a callback (Delta, Sequence).
- The chosen scheme recurses into its children through `compress_child`, up to
  `MAX_CASCADE = 3`. Ancestor and descendant exclusion rules apply along the way.
- The integer schemes are FOR, ZigZag, BitPacking, Sparse, Dict, RunEnd, Sequence, RLE, Delta
  and Pco (Pco is excluded by default).
- Most Pass 1 checks are hand-set caps:
  - Dict: distinct values > 50%.
  - Sparse: top value < 90%.
  - RunEnd and RLE: run-length thresholds.
  - FOR vs BitPacking: bit width.

## What others do

| System | Approach | Lesson |
|---|---|---|
| OpenZL (Meta) | Offline trainer: greedy stream clustering, plus ACE, an NSGA-II genetic search over codec graphs that gives a speed/ratio Pareto front. At runtime: rule-based selectors, and a small neural "Compression Transformer" that scores one codec at a time and recurses on its output streams. It dropped an earlier GBT selector (4.5 MB of generated C). | Decide one step at a time and recurse. Codegen the model. Keep a deterministic fallback. |
| CodecDB (SIGMOD '21) | An MLP ranks encodings from sampled features: cardinality, sortedness, entropy, length and repeated words. | 96% / 87% accuracy on strings / ints. **Head (contiguous) sampling beats random sampling**, because delta and RLE need locality. |
| LEA (aiDM '21) | Per-encoding random-forest regressors for size and scan speed, trained on synthetic data. The 1% sample's encoded size is a feature. | Real data alone didn't generalise. Optimising size alone caused query regressions. |
| BtrBlocks, DuckDB, Parquet | Sampling, analyse-then-pick, or rules. | These are the baselines the learned approaches beat. |

## Design decisions reached

1. **The model predicts one step, not a whole tree.** It picks the scheme for this node, and the
   compressor recurses as it does today. Each step is labelled with the *final* cost of its
   subtree.
2. **The oracle is exhaustive search.** For bytes, the cost adds up across children, so keeping
   only the best child is exact (dynamic programming). Only hard constraints filter candidates:
   `matches`, exclusions, the cascade budget and errors. Heuristic `Skip`s are recorded but
   never obeyed.
3. **Explore more paths where costs don't add up.** Keep the top k (or ε-Pareto) alternatives per
   child once decode and pushdown time are objectives. Audit the compressor's own rules
   (`MAX_CASCADE`, exclusions, accept-if-smaller) by relaxing them one at a time. Scheme
   parameters are always the latest version.
4. **Objective:** one time-based cost,
   `bytes / bandwidth + Σ w_op · pushdown_time + survivors · decode_time`. Store the
   multi-objective raw data and apply weights late, as presets (`size`, `balanced`, `scan`).
5. **Model:** start with a shallow, regret-weighted decision tree per dtype, refine its
   thresholds with CMA-ES against cached oracle results, and codegen it into Rust. Use an MLP
   scorer only if needed. When the model is unsure, fall back to sampling the top m schemes.
6. **A genetic algorithm only when the space stops being enumerable,** for example once schemes
   get tunable parameters. Exhaustive search plus dynamic programming is cheaper and exact today.

## Integer features (tiers by cost)

- **Tier 0 (context):** ptype bits, signedness, log length, cascade depth, parent scheme, child
  index, cascade budget left.
- **Tier 1 (`IntegerStats`):** null fraction, `bits_bp`, `bits_for`, `bits_zz`, `for_gain`,
  log average run length.
- **Tier 2 (distinct):** distinct fraction, `bits(distinct)`, top-1 fraction, top-k coverage,
  entropy.
- **Tier 3 (new single pass):** delta bit width, sorted fraction, step-mode fraction (for
  sequences), bit-width percentiles (p50, p90, p99), exception fraction at p90, common trailing
  zeros.
- **Tier 4 (derived):** log ratio of each scheme's closed-form size estimate to the canonical size.
- **Tier 5 (optional):** each scheme's 1% sample compressed size.

## System architecture (if built)

- **Rust owns arrays:** chunks, features, search, measurement, inference. **Python owns
  tables:** labels per preset, regret, splits by dataset, models, export.
- **One seam in the compressor:** a `SelectionStrategy` trait used where `choose_best_scheme`
  is called today. It owns choosing and compressing, so it can run the default, forced,
  exhaustive or learned strategy. Every level of the cascade goes through it.
- **Registries:** `ChunkSource`, `FeatureExtractor`, `Measurement`. Features come from the same
  code at runtime and in the lab.
- **Data flow:**
  1. Plan.
  2. Chunk.
  3. Search, in parallel. This records bytes, compression ratio and the tree, and serialises
     every candidate encoding into a content-addressed blob store.
  4. Verify: decode and pushdown results are checked against hashes from the canonical data.
  5. Plan the timing.
  6. Time sequentially on a clean, isolated machine: decode and pushdown probes, with
     calibration.
  7. Assemble the dataset.
- **Checkpointing:**
  - A deterministic plan is a DAG of tasks keyed by
    `hash(kind, inputs, params, vx_version, git_commit)`.
  - **Facts** are upserted by key, and re-running a task must reproduce them exactly.
  - **Observations** (timings) are appended per `(key, machine_id, run_id)`, so re-runs add
    samples and other architectures stay separate.
  - Shard with `hash(key) mod n`. Done markers live in a ledger directory.
- **Unit of parallelism and checkpointing:** `(chunk, dtype)`. Parallelising per scheme happens
  inside a task, because cascades couple schemes.
- A parked draft of the planner lives in `src/` on this branch. It isn't a workspace member and
  hasn't been built.

## Open questions checked before building it

1. Is there much to gain over the current compressor?
2. Can we choose without sampling-based estimates?
3. Do the hand-set caps cost anything?
4. Does combining decode speed with compression ratio change the choices?
