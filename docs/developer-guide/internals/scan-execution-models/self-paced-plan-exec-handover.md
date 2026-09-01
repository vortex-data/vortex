# Self-Paced Plan Execution Handover

This document is the implementation handover for the restricted self-paced plan execution
experiment on branch `ji/self-paced-fair-natural-splits`. It describes the code as it exists at
the end of the coordinator-sharding experiment (2026-08-23), which followed the segment-streamed
demand experiment. It is not a proposal to merge this executor as a production scan path.

The companion [findings report](self-paced-plan-exec-findings.md) contains the broader benchmark
record. The [learning ledger](self-paced-plan-exec-learnings.md) keeps provisional conclusions that
may turn out to be incomplete or wrong.

## Current question

The experiment asks whether an explicit plan execution graph can improve a scan by:

- scheduling reads and predicate work at segment granularity;
- publishing progressively smaller row demand between predicates;
- sharing a decoded segment when filter and projection use the same field; and
- choosing between dependency-driven and parallel predicate execution.

It deliberately supports only a highly restricted serialized layout. Do not generalize its
results to all Vortex layouts or SQL execution.

## Fair comparison contract

The current comparison has these non-negotiable properties:

- The input is serialized once and reopened by both paths. The measured FineWeb Q06 file is
  1,669,473,052 bytes with stable hash `0x886de969ce96c930`.
- The allowed layout is exactly `Struct(Chunked(Flat))`. Unsupported layouts and planning
  rewrites are disabled by the experiment's layout strategy.
- V1 runs normally over every real natural split. It is never given self-paced morsels and must
  not fall back to a fixed row split.
- Self-paced merges 16 consecutive natural splits into each outer morsel. A morsel can cross a
  chunk boundary and is never smaller than any natural split it contains.
- Both paths scan the same rows, query object, serialized segments, warm fixture, and output. A
  stable row count and ordered hash are checked before a timing is accepted.
- Both are capped at concurrency 16 and the process is pinned to CPUs 0-15. If fewer than 16
  self-paced morsels exist, self-paced concurrency is capped to its morsel count and the result
  must call that out.
- Fixture construction and Parquet ingestion are outside the timed region. Timed runs consume all
  output and alternate executor order.

For the final FineWeb Q06 run, all 15 local Parquet files produced 14,868,862 rows, 157 ingestion
chunks, 1,823 natural splits, and 116 merge-16 self-paced morsels. Both sides therefore had enough
independent work to occupy 16 workers.

## Implementation map

The experimental implementation is concentrated in these files:

- `vortex-layout/src/plan/exec/model.rs` defines operations, completions, cached predicate
  coverage, policies, metrics, and trace events.
- `vortex-layout/src/plan/exec/graph.rs` defines shared resource nodes and their predicate and
  projection consumers.
- `vortex-layout/src/plan/exec/reactor.rs` owns the mutable execution graph, fragment state,
  resource deduplication, task readiness, completion adoption, and metrics.
- `vortex-layout/src/plan/exec/evaluate.rs` performs reads, flat decoding, sparse predicate
  evaluation, selection, packing, and final fragment-mask concatenation.
- `vortex-layout/src/plan/exec/baseline.rs` is the concurrent driver. It owns one `Execution`,
  admits tasks, runs inline operations, and sends other operations to the worker pool.
- `vortex-layout/src/plan/exec/tests.rs` checks result parity, fragment streaming, reduced demand,
  empty demand, sharing, scheduling, and trace behavior.
- `vortex-file/benches/self_paced_vs_v1.rs` builds the serialized fixtures, enforces the comparison
  contract, runs the suites, and prints traces and metrics.
- `vortex-layout/src/plan/exec/baseline.rs` also owns the sharded runner
  (`run_self_paced_sharded`, `VORTEX_SELF_PACED_SHARDS=N`): N coordinator threads over contiguous
  morsel groups, one shared worker pool, static per-shard admission `concurrency / N`.
- `VORTEX_SELF_PACED_SHARD_MODE=owned` is the best-performing mode: 16 threads (matching the
  concurrency budget) each run `run_self_paced_single` over their own morsel group, coordinating
  and evaluating inline with no pool, no channel, and no dispatch. It wins 25 of 28 workloads on
  the measurement host and is the recommended configuration for further work.
- The harness enforces a per-iteration no-caching invariant (`assert_cold_scan_io`): self-paced
  I/O must equal its cold warmup exactly, and both engines must re-read at least the warmup's
  unique-segment floor (V1 gets a 1% counting allowance for dropped duplicate in-flight reads).
- `vortex-layout/src/plan/exec/pipeline.rs` (`VORTEX_SELF_PACED_SHARD_MODE=pipeline`) is the
  extensible successor and the fastest mode: the scheduler sees only `dyn MorselPipeline`, demand
  compute is a pluggable `DemandPolicy` (`VORTEX_SELF_PACED_DEMAND=cascade|eager`), children may
  have arbitrary unaligned chunk boundaries (root-row-space cutting via `overlapping_chunks`),
  and a per-thread decoded-chunk cache preserves filter/projection sharing. FineWeb geometric
  mean ~0.32 versus V1; remaining weak spots are thread-tail imbalance on few-morsel workloads
  (TPC-H Q6 0.94, ClickBench Q40/Q41) — work stealing is the next fix.
- Coordinator phase timing (`VORTEX_SELF_PACED_PHASE_TIMING=1`) fills the `coordinator_*` and
  `completion_queue_dwell_*` metrics and the harness prints a `phase_timing` line per self-paced
  run. Keep it off for reported comparisons.
- `vortex-file/examples/fineweb_split_audit.rs` regenerates the physical split catalogs
  (`VORTEX_SPLIT_AUDIT_MODE=fineweb|tpch|clickbench`); the FineWeb and TPC-H catalogs reproduce
  the previously documented split counts (1,823/2,527 and 458), the ClickBench one does not
  (21-column audit files versus the 105-column production files).

## Execution flow

Each outer self-paced morsel is split internally at serialized chunk boundaries. These fragments
are mask-progress units, not new output morsels.

1. A fragment begins with logical all-true demand and the first predicate conjunct.
2. The plan executor attaches that conjunct and the current demand coverage to the fragment's
   segment read/decode task.
3. A worker reads and decodes the segment, then evaluates that predicate only for demanded rows.
4. Completion returns the decoded array plus a cached predicate value, its evaluated-row bitmap,
   input true count, and evaluation time.
5. The coordinator adopts the result directly into every waiter whose captured coverage is valid.
   A later waiter may reuse it only if its current demand is a subset of the evaluated coverage.
6. Reduced fragment demand can unblock the next predicate or projection segment before sibling
   fragments finish. Fragments of one morsel may progress independently and in parallel.
7. Once all fragments seal, `MergeDemandFragments` concatenates their bit buffers in row order.
   Projection selection consumes that single outer-morsel mask and preserves the output contract.

`SegmentId` is the current physical resource identity. One resource node can have predicate and
projection consumers, and the decoded array is reused when both refer to that resource. This is an
accepted experiment restriction: a production identity will also need source/layout context.

## Scheduler ownership

Fragment masks, predicate dependencies, cache coverage, and readiness belong to plan execution.
The scheduler should only decide which ready work to admit under CPU, I/O, concurrency, and byte
budgets. It should not interpret or combine row masks.

The experiment currently has one orchestration thread because `run_self_paced_concurrent` owns one
mutable `Execution`. Its loop drains completions, advances morsels, discovers ready tasks, claims
them, performs inline transitions, and queues outputs. Only claimed evaluation work is parallel.
This avoided locks and made leases, deduplication, cancellation, and traces deterministic, but it
places thousands of small transitions and bitmap publications on one critical path.

A likely production direction is to shard plan execution by morsel or small morsel groups. Resource
completion events would be published to the owning shards; a shared scheduler would retain global
admission and byte accounting. Cross-shard segment deduplication needs an explicit resource owner
or concurrent registry rather than accidental coordinator serialization.

## Metrics added

The trace separates physical work from execution machinery:

| Group | Important metrics | Meaning |
| --- | --- | --- |
| I/O | `requests`, `unique_segments`, `bytes` | Physical segment requests and returned bytes |
| Sharing | `shared_resources`, `shared_read_bytes`, `shared_decode_reuse_hits` | Filter/projection resource overlap and reuse |
| Graph | `transitions`, `nodes_inspected`, `tasks_*` | Coordinator and scheduling work |
| Fragments | `demand_fragments`, `fragment_predicates_completed`, `fragment_demand_updates` | Progressive mask state |
| Unblocking | `fragment_projection_reads_unblocked` | Projection work exposed before the outer mask seals |
| Fused work | `segment_predicates_fused`, `fragment_cached_predicate_hits` | Predicate work completed with read/decode and then adopted |
| Reduced demand | `reduced_demand_predicates`, `reduced_demand_input_rows`, `reduced_demand_skipped_rows` | Sparse predicate applications and avoided row visits |
| CPU estimates | `segment_predicate_eval_ns`, `fragment_demand_adoption_ns`, `fragment_merge_elapsed_ns` | Aggregate measured operation time, not wall time |

Q06 uses disjoint filter and projection fields, so its sharing metrics correctly remain zero. Other
tests and query shapes demonstrate reuse when a field appears in both.

## Final measured state

Owned coordination (16 self-coordinating threads, `VORTEX_SELF_PACED_SHARD_MODE=owned`) on a
16-core, 30 GB host wins 25 of 28 workloads: FineWeb 9/9 (geometric mean ~0.63, Q06 at `0.79`),
TPC-H 3/3 (0.56-0.69), ClickBench 13/16 (~0.75; losses are dashboard/Q40 at 1.07 and Q41 at
1.23). The intermediate pooled-shard results (4 shards, `1.40x` on Q06, down from `2.50x`
single-coordinator on this host) are retained in the findings report along with I/O parity
evidence and caveats.

The earlier single-coordinator five-iteration FineWeb Q06 comparison (2026-08-22 host) was:

| Executor | Median |
| --- | ---: |
| V1 natural splits | 22.240 ms |
| Self-paced merge-16 | 48.695 ms |
| Self-paced/V1 | 2.190x |

The last detailed trace had nearly equal physical work: self-paced issued 10,918 unique requests
and returned 714,536,112 bytes, compared with about 10,931 V1 requests and 714.6 MB. Self-paced
performed 5,461 fused segment predicates over 1,823 fragments and 116 morsels. Of the later
predicate applications, 3,638 used reduced demand: they evaluated 24,957 requested rows and
skipped 29,689,351 row applications. Aggregate predicate CPU fell from about 18.8 ms in the
all-row fused version to about 10.9 ms.

That work reduction did not improve wall time. The demand-aware path still publishes and adopts
thousands of partial masks through the single coordinator. The outside mask merge was only about
0.57 ms, so optimizing final concatenation alone is unlikely to close the gap.

## Experiment history

The progression on full FineWeb Q06 is useful when deciding what not to repeat:

| Variant | Approximate self-paced/V1 | Result |
| --- | ---: | --- |
| Per-fragment CPU predicate tasks | 2.299x | Correct streaming, too many tiny tasks |
| All predicates fused into read/decode | 2.105x | Fewer tasks, but evaluates rows later masks reject |
| Completion-side adoption | 2.047x best sample | Removed a redundant transition class |
| Separate sparse CPU task after decode | 2.431x | Saved predicate work but restored task overhead |
| Fused sparse predicate with per-bit demand assembly | 2.922x | Coordinator bitmap construction dominated |
| Byte-copy demand assembly | 2.212x sample | Recovered most of the per-bit regression |
| Final coverage-safe demand-aware path | 2.190x | Less predicate CPU, orchestration still dominates |

Trace collection perturbs short timings. Use traces to explain task and byte counts, and use
non-traced alternating runs for performance comparisons.

## Priority work

### Done: coordinator cost established and sharding prototyped

Phase timing showed the single coordinator busy ~89% of the Q06 run (advance 34%, completion
handling 28%, dispatch 24%) with workers starving behind it (~17 us average completion dwell).
Two and four shard prototypes reduced Q06 from 2.50x to 1.64x and 1.40x on the measurement host;
eight shards regressed slightly (admission 2 per shard). Batching fragment transitions, batching
resource joins, allocation-free adoption counts, and the speculative-pass skip were worth only
~8% combined: work reduction does not shorten a serialized critical path.

### Done: owned coordination removed the coordinator entirely

Sixteen threads each own a morsel group and both coordinate and evaluate inline
(`run_self_paced_single` per thread). No pool, channel, dispatch, or dwell remains; thread count
matches V1's 16 workers. This flipped 25 of 28 workloads to self-paced wins, including every
FineWeb and TPC-H shape.

### P0: harden owned mode

- Work stealing or dynamic morsel assignment: a thread that finishes its group idles while
  stragglers run; ClickBench Q40/Q41/dashboard (the three remaining losses) have few, uneven
  morsels per thread.
- Build per-thread `Execution` state from only the chunks overlapping the owned rows; every
  thread still pays the full plan-wide resource table inside the timed region.
- Cross-thread resource sharing is currently only avoided because morsel groups end on natural
  splits; segments spanning group boundaries in general layouts need an explicit shared resource
  registry (or acceptance of bounded duplicate reads).
- Blocking reads are acceptable for the in-memory source only. Object-store latency needs either
  a small per-thread async read-ahead or a return to pooled I/O while keeping owned CPU work.
- Preserve the per-iteration cold-scan I/O invariant in every future comparison.

### P0: protect correctness

- Preserve exact output row count/hash checks for every benchmark.
- Keep explicit cache-coverage tests for a resource shared by fragments or morsels with different
  demand. Never treat a partially evaluated predicate as a full-segment cache.
- Extend fragment tests across empty masks, nullability, multiple chunks, and resources spanning
  more than one outer morsel before broadening supported layouts.

### P1: stream morsel output (pipeline done; reactor and measurement remaining)

- Done in the pipeline: `MorselPipeline::execute` takes a batch sink and emits ordered
  dense-prefix `ExecBatch` values, one per chunk-boundary span shared by every projected field.
  Each span's decoded chunks are released at emission and the scheduler clears the per-thread
  cache between morsels, so executor-retained decoded memory is bounded by the working set
  instead of growing with the scan. The hash gates are boundary-insensitive, so parity checks
  are unchanged; batch order is restored by (morsel, emission) index.
- Remaining: the reactor's `AdvanceResult` returns the prefixes sealed by that call (`Retired`
  meaning the final prefix was emitted; its single batch remains the valid degenerate stream),
  and the harness starts measuring time-to-first-batch and peak retained output — both currently
  unmeasured.
- Streaming complements, but does not replace, the Q6 makespan work: downstream consumers
  overlap with a morsel's tail instead of waiting on whole-morsel batches, while intra-morsel
  parallelism still governs the scan's own critical path.

### P1: reduce mask machinery

- Represent unresolved demand as bit buffers and versions in execution state. Materialize a
  `BoolArray` only at an evaluator or public array boundary.
- Batch several completion adoptions and advance affected fragments once per batch.
- Keep no-op all-true and unchanged masks symbolic. Do not allocate a full all-true `BoolArray`
  for every morsel or fragment.
- Re-evaluate fragment size and natural-split rollup using estimated bytes and CPU work, not only a
  fixed split count.

### P1: make policy adaptive

- Estimate the CPU saved by waiting for the preceding predicate from observed input/output true
  counts and per-row predicate cost.
- Compare that saving with observed dependency wait and coordinator publication latency. Run
  independent predicates in parallel when waiting is expected to cost more than the avoided work.
- Preserve predicate-order feedback, but distinguish ordering from parallelism: the cheapest or
  most selective predicate can be launched first while another is admitted concurrently.

### P1: benchmark representative shapes

- Re-run all ClickBench scan shapes, TPC-H single-table scans, and all local FineWeb data after any
  scheduler change, always with the fair contract above.
- Report filter/projection overlap, selectivity, projected field count, bytes, natural split count,
  morsel count, and whether 16-way parallelism was available.
- Include broad/select-all scans, highly selective scans, expensive predicates, shared filter and
  projection fields, many/few columns, and both I/O-heavy and CPU-heavy layouts.

### P2: production coverage

Add compressed encodings, nullable arrays, general expressions, object-store latency, cancellation,
memory pressure, and source-aware segment identity only after the control path is competitive on
the restricted layout.

## Reproduction

The final focused command was:

```bash
taskset -c 0-15 env \
  VORTEX_FINEWEB_PARQUET=<dir with the 15 FineWeb 10BT sample shards> \
  VORTEX_FINEWEB_SPLIT_CATALOG=<fineweb-natural-splits.json> \
  VORTEX_SELF_PACED_COMPARE_ITERATIONS=5 \
  VORTEX_SELF_PACED_COMPARE_WORKLOAD=fineweb_q06 \
  VORTEX_SELF_PACED_SHARDS=4 \
  target/release/deps/self_paced_vs_v1-<hash>
```

The executable hash is build-specific; rebuild the `self_paced_vs_v1` benchmark and use the
resulting binary. For diagnosis, add `VORTEX_SELF_PACED_COMPARE_TRACE=1` or
`VORTEX_SELF_PACED_PHASE_TIMING=1`, but do not compare either timing with non-traced medians.

Everything needed to reproduce from a clean host is scripted or documented:

- FineWeb: the 15 `sample/10BT/000..014_00000.parquet` shards from
  `huggingface.co/datasets/HuggingFaceFW/fineweb` at revision `v1.4.0` (~29 GB).
- TPC-H: `duckdb -c "INSTALL tpch; LOAD tpch; CALL dbgen(sf=10); COPY lineitem TO
  'lineitem_sf10.parquet' (FORMAT parquet);"`, passed via `VORTEX_TPCH_LINEITEM_PARQUET`.
- ClickBench: `hits_0..99.parquet` from the ClickBench `parquet_many` mirror (~14 GB), passed via
  `VORTEX_CLICKBENCH_PARQUET_DIR` and `VORTEX_CLICKBENCH_MAX_FILES`.
- Catalogs: `cargo run --release --example fineweb_split_audit -p vortex-file` with
  `VORTEX_SPLIT_AUDIT_MODE` and `VORTEX_SPLIT_CATALOG_OUT`. The FineWeb catalog reproduces the
  documented 1,823/2,527 split counts exactly and the serialized fixture byte length matches
  (1,669,473,052 bytes); the serialized hash differs across hosts even at identical length, so
  treat the hash as host-specific rather than a portable fixture identity.
- Memory: the 100-file ClickBench fixture needs more than 22 GB resident; a 30 GB host runs at
  most ~20 files together with the per-workload rechunked copy.

The final verification before handover was:

```text
cargo test -p vortex-layout plan::exec
cargo clippy -p vortex-layout -p vortex-file --all-targets --all-features -- -D warnings
cargo +nightly fmt --all
```

The targeted execution tests passed 30/30 and Clippy completed without warnings.
