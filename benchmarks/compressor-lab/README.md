# vx-lab: data generation for learned compression scheme selection

`vx-lab` generates training data for choosing compression schemes. It is built to run for a long
time, across machines, commits and datasets, and to grow with new schemes, features and dtypes.

```bash
# Everything, resumable: plan, facts in 4 shards, timings, training tables, models.
benchmarks/compressor-lab/pipeline.sh benchmarks/compressor-lab/specs/int-v1.toml /tmp/lab 4
```

`specs/int-v1.toml` reads Parquet from `data/` next to the spec. It covers ClickBench, NYC taxi,
TPC-H SF1 and synthetic grids, with integers only and default encodings only.

## How it works

1. **Spec → plan.** `vx-lab plan create` expands a TOML spec into tasks. Each task key is a hash
   of its inputs plus the code identity (Vortex version, git commit, dirty flag). The same spec,
   data and commit produce a byte-identical plan; re-creating it reports `unchanged`. Parquet
   files are identified by content hash, not path, so moving the data keeps the keys.
2. **Facts.** `vx-lab run` executes pending tasks:

   | Stage | Output |
   |---|---|
   | `chunk` | The canonical chunk, as a blob. |
   | `features` | Keyed by `FEATURES_VERSION`. |
   | `candidates` | Every candidate's encoding as a blob, plus its size and tree. Keyed by the candidate set and `CANDIDATES_VERSION`. |

   - Facts are upserted by key, and a done marker in `ledger/` makes re-runs skip them.
   - `--shard i/n` splits work across processes or machines. A chunk's tasks always share a
     shard, so no shard waits on another.
   - `--verify` re-runs done tasks and reports any output that changed (nondeterminism).
3. **Observations.** `vx-lab run --stages time` times every candidate: compression (and checks
   that re-compressing reproduces the stored encoding byte for byte) and decode. Each run
   *appends* a file per task under `obs/time/machine=<id>/`. Repeated runs add samples, and
   machines with a different CPU or architecture get a different `machine_id` and stay separate.
   The time stage reads only blobs, so a store can be copied to a quiet machine without the
   source data.
4. **Materialize.** `vx-lab materialize` joins one plan's facts with one machine's pooled
   timings into `rows-*.csv`, `features-*.csv` and `dataset.json`. That's the layout
   `benchmarks/compressor-feasibility/train.py` reads to fit and export models.

```text
store/
  ledger/<kk>/<task key>.json           done markers (written last)
  facts/<kind>/<kk>/<task key>.json     fact outputs; rewriting = upsert
  blobs/<hh>/<sha256>                   content-addressed arrays (chunks and encodings)
  obs/time/machine=<id>/<task key>/<run id>.json
  machines/<id>.json
```

`vx-lab plan show --store` reports progress per stage and timing runs per machine.
`vx-lab plan tasks --kind candidates --shard 3/16 --pending --store` lists work to hand out.

## Extending it

| To add… | Change | What re-runs |
|---|---|---|
| A dataset | A `[[sources]]` entry in a spec (`parquet`, `tpch` or `synthetic`) | Only the new chunks |
| A synthetic generator | `synthetic.rs` | New chunks only |
| A source kind (e.g. object store, Vortex files) | `spec::SourceSpec`, `source.rs` (resolve), `load.rs` (rows), `plan::ChunkIdentity` | New chunks only |
| A feature | `features.rs`, mirrored in `train.py`'s `add_estimates`; bump `FEATURES_VERSION` | `features` only |
| A candidate (scheme, policy or variant) | `candidates.rs`; it appears in plans via the scheme selection | `candidates` and `time` |
| A change to how candidates are built | Bump `CANDIDATES_VERSION` | `candidates` and `time` |
| A measurement (pushdown compare, filter, take) | `runner::run_time`, `TimeObservation`, `materialize` columns | `time` only |
| A dtype (floats, strings) | `features.rs` and `candidates.rs` for that dtype; the plan's `dtypes` already filters | New chunks only |
| A search strategy (exhaustive DP) | `spec::SearchStrategy`, `runner` (the strategy is part of the candidates key) | `candidates` only |

## Running it long term

- **Commits.** Every key includes the commit, so facts from different code never mix. A new
  commit means a new plan; the old store data stays valid for the old plan. To reuse timings
  across commits, add an explicit policy at materialize time (for example, reuse when an
  encoding's hash is unchanged), never by changing keys.
- **Machines.** Time on quiet, dedicated machines: copy the store there, run
  `--stages time` one or more times, copy `obs/` back. Materialize with `--machine <id>` per
  machine, and train per machine or architecture.
- **Storage.** The `int-v1` run (1,582 chunks, 11 candidates) uses about 1.1 GB, mostly blobs.
  Blobs are shared across plans by content hash. Garbage collection (keep the blobs referenced
  by the plans you keep) isn't implemented yet.
- **Scale.** Facts are JSON files, one per task: fine to around 10⁵ chunks on a local
  filesystem. Beyond that, move facts to Parquet partitions and the store to object storage.
  The layout is already path-per-key, so that is a storage swap, not a redesign.

---

# Design notes: plan and learnings

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
- This is now implemented as `vx-lab` (above).

## Open questions checked before building it

1. Is there much to gain over the current compressor?
2. Can we choose without sampling-based estimates?
3. Do the hand-set caps cost anything?
4. Does combining decode speed with compression ratio change the choices?
