# Morsel Prototype: P1 Findings

Measured results for the P1 spine of the
[morsel-based plan execution design](morsel-based-plan-execution.md), implemented in
`vortex-morsel`. This document records what was built, what was measured, and — importantly —
which parts of the [prototype plan's](morsel-prototype-plan.md) evaluation matrix could **not** be
evaluated in this environment and why.

## What was built

`vortex-morsel` implements the P1 surface from the prototype plan:

```rust
trait ExecNode: Send {
    fn reset(&mut self, range: Range<u64>);
    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll>;
    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll>;
    fn retire(&mut self, cx: &mut RetireCx<'_>);
    fn children(&self) -> &[NodeId];
}
```

The five operators are `FlatExec`, `ChunkedExec`, `StructExec`, `ConjunctExec` (cascade and
parallel behind one policy flag) and `FilterExec`.

Design points that survived contact with the code:

- **Nodes never perform IO.** `next_plan` registers `IoUse`s keyed to whole stored units against
  the `IoPlane` and receives tickets; `execute` may only wait on tickets its own planning stream
  emitted, and consuming a ticket a node never named is an error rather than an inline read. The
  `source_range`, `extent`, `producer` and `estimated_bytes` fields are carried and stamped but
  not yet *consulted* — nothing reads them until P2's admission loop exists.
- **Emit-once planning.** Planning is budget-bounded (`PLAN_BUDGET = 64` uses) and resumable:
  chunked keeps a cut cursor, struct and conjunct keep field cursors, so a node that exhausts the
  quantum yields `PlanItem::Plan` and resumes where it stopped rather than restarting.
- **Immutable plan, per-thread state.** The graph model's objection to stateful nodes (§9 of
  [the graph model](scan-execution-graph-model.md)) — that a node shared by every unit cannot hold
  per-morsel state — is answered by splitting the two: `ExecPlan` is one immutable blueprint per
  scan, and each driving thread instantiates its own arena of mutable node state, reset per morsel.
  Nothing is allocated per morsel and nothing on the hot path is shared between threads.
- **The arena take/put trick** is what lets a node hold `&mut self` while recursively driving its
  children: the driver removes the node from its slot, hands the rest of the arena to the child
  poll, and puts it back. The tree shape guarantees a node is never reachable from its own
  subtree, so a taken slot is never observed empty; the debug path panics if it ever is.
- **Unsupported shapes are build errors.** Nested structs, non-struct roots, nullable root
  structs and non-flat/non-chunked columns fail in `build_plan` rather than falling back, so an
  unsupported query cannot be timed as if the prototype had executed it.
- **Retention is derived from demand, never from a budget.** An earlier revision carried a
  per-thread decoded-chunk cache; it was removed because a budget-and-eviction cache is state V1
  does not have and its numbers measured the cache, not the executor. What replaced it is the P1
  slice of P2's keyed cells: **leased shared decoded cells** (`cells.rs`). Before the scan
  starts, the driver counts — from the morsel cut and the plan's flat nodes alone — exactly how
  many (node, morsel) pairs will touch each stored unit. The first morsel to decode a unit
  publishes the array into its cell; every retiring morsel releases its lease whether it used
  the cell or not; the last release drops the array. Nothing is held speculatively, nothing has
  a budget, nothing survives the scan, and the lease ledger is asserted to drain to zero. A
  morsel whose planning finds the cell already populated skips issuing the read entirely — its
  own unreleased lease guarantees the value survives until it retires. The `no-reuse`
  configuration disables the layer completely and holds no state across morsels at all; it is
  kept as the fairness row and as the chaos check (`decodes + reuses` in a sharing run must
  exactly equal `decodes` in a non-sharing run, asserted per query).
- **The cell map is sharded 16 ways.** The first cut used one mutex, and the wide-numeric
  workload at 4 threads got *slower* than 1 thread (0.53 vs 0.50 vs V1): 4,560 lease touches on
  one lock serialised the scan. Sixteen shards restored 0.34. The measured lesson for P2: lease
  traffic scales with (nodes × morsels), so the cell index must be sharded or lock-free from the
  start.

One deviation from the sketch worth recording: rather than rewriting expressions to push
predicates onto individual fields, each conjunct and the projection are re-bound against the
*narrowed* struct dtype of exactly the top-level fields they reference. This achieves the same
column pruning using only public expression API, and keeps the executor's semantics identical to
V1's by construction (the same `apply_bound` on the same assembled struct).

## Correctness

18 differential tests, all passing. Every one uses the V1 `LayoutReader` as the oracle and
asserts equal row counts and equal ordered content over 8 query shapes:

| Property | Test |
|---|---|
| Agrees with V1 at 1, 2 and 4 threads | `matches_v1_oracle` |
| Misaligned chunking is invisible | `misaligned_chunks_match_aligned_reference` |
| The document's `[0,3,10)` vs `[0,6,10)` case, and its split set | `document_misalignment_case` |
| Result independent of morsel size (1, 7, 128, 4096 rows, and per-split) | `independent_of_morsel_size` |
| Cascade and parallel conjunct policies observationally identical | `conjunct_policy_is_not_observable` |
| Shared cells change no output, at 1 and 4 threads, and account exactly | `shared_cells_are_not_observable` |
| Straddled chunks are decoded exactly once per scan | `shared_cells_reuse_straddled_chunks` |
| Every read was named by a planning stream | `every_read_was_planned` |
| All-false filter emits nothing | `empty_filter_emits_nothing` |
| Unsupported layouts are build errors | `rejects_unsupported_layouts` |

The evaluation binary re-runs the oracle check for **every** configuration on **every** query
before any timing happens; a configuration that disagrees is reported as a failure and excluded
from the timing table. All 105 configuration-query pairs in the run below matched.

## What could not be evaluated, and why

The prototype plan's gate E1 reads: *D within 5% of C's rerun across suites; ordering
D ≈ C < B(owned) < B(coordinator) reproduced.* **Gate E1 as specified was not evaluated.** Three
reasons, all environmental rather than results anyone should read past:

1. **Rows B and C do not exist in this repository.** The self-paced graph/reactor and pipeline
   executors that the [findings document](self-paced-plan-exec-findings.md) reports 2.53x → 0.41x
   for are not present at any commit reachable here — a search of the tree for `self_paced`,
   `morsel`, or a `vortex-scan-v2` crate finds nothing. Only rows A (V1) and D (this prototype)
   could be run. Without C, "within 5% of C" is unmeasurable, and so is the ordering claim.
2. **The named suites need multi-gigabyte downloads.** FineWeb's sample is ~2 GB of Parquet and
   ClickBench's `hits` is far larger; this host has 4 cores and 15 GB of RAM, and the harness
   holds segments in memory. TPC-H SF10 needs a generator that is not vendored.
3. **P0's latency-injection IO source and chaos mode are not built.** They gate E2 and E3, which
   are P2 work and out of scope for P1 anyway.

What was measured instead is a set of **shape-matched synthetic workloads**: struct-of-chunked-flat
columns whose per-column chunk boundaries deliberately disagree, scanned under conjunctive filters
of varying selectivity with narrow and wide projections. These reproduce the structure the plan
says the real suites lower to, and they exercise exactly what E1 is about — the executor's own
scheduling-unit cost. They do **not** exercise encoding-specific decode costs (FSST, ALP-RD,
dictionary), and their absolute wall times are not comparable to the recorded suite numbers.

## Results

Host: 4 logical cores, segments in memory, 1M rows per workload (250k for the string-heavy one,
which has far wider rows), 5 alternating iterations, median reported. Reproduce with:

```bash
cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval
```

Ratios are against **A: V1 single-threaded**, which is the apples-to-apples baseline for a
one-thread morsel run — the harness drives V1 on `SingleThreadRuntime`, which runs every task on
the calling thread. Row A' gives V1 a multi-threaded Tokio runtime with the same core count, which
is how DataFusion actually drives it.

Geometric means over all 15 queries:

| Row | Geomean vs V1(1) | Range |
|---|--:|---|
| A  V1, 1 thread | 1.000 | — |
| A' V1, tokio x4 | 0.743 | 0.31 – 1.63 |
| D  morsel, 1 thread, per-split morsels | **0.539** | 0.30 – 0.81 |
| D  morsel, 1 thread, sharing disabled (no-reuse) | 0.644 | 0.36 – 1.02 |
| D  morsel, 4 threads, per-split morsels | 0.366 | 0.22 – 0.84 |
| D  morsel, 4 threads, 64k-row morsels | **0.249** | 0.12 – 0.60 |
| D  morsel, 4 threads, parallel conjuncts | 0.340 | 0.22 – 0.66 |

Per workload (geomean vs V1(1)):

| Row | string-heavy | wide-numeric | narrow-analytic |
|---|--:|--:|--:|
| A' V1, tokio x4 | 0.545 | 0.958 | 0.833 |
| D  morsel, 1 thread | 0.594 | 0.494 | 0.531 |
| D  morsel, 1 thread, no-reuse | 0.816 | 0.546 | 0.558 |
| D  morsel, 4 threads | 0.384 | 0.343 | 0.379 |
| D  morsel, 4 threads, 64k morsels | 0.303 | 0.188 | 0.296 |

The full table, every query and every counter, is in
[`morsel-prototype-p1-eval.md`](morsel-prototype-p1-eval.md).

### What the numbers say

**The leased cells recover the cross-morsel decode reuse the removed cache had shown, this time
by construction rather than by budget.** On string-heavy `SH1 select-all`, the no-reuse row
decodes 310 times; the sharing row decodes 121 times — exactly once per chunk — and serves the
other 189 from cells, moving 0.87 to 0.46 single-threaded. The counters give the mechanism its
own audit: in every sharing run, `decodes + reuses` equals the non-sharing run's `decodes`
exactly (asserted per query in the tests), and the lease ledger is asserted empty at scan end.
Because a plan-time cell hit skips the read as well, requests fall with decodes: the sharing row
issues 121 requests where the no-reuse row issues 310. Retention peaked at the scan's active
window — the morsels overlapping one unit are consecutive indices off a monotone cursor, so a
cell lives from the first of them to the last.

**Even with sharing disabled, the executor beats V1 on the same cut** (0.644 geomean): no future
per evaluation, no task per split. That row is the state-for-state comparison with V1 and is the
floor the sharing mechanism builds on, not a number sharing inflates.

**Coalescing morsels remains worth more than sharing on wide tables.** `WN1 select-all` at 4
threads: per-split morsels with sharing 0.24; 64k-row morsels 0.12 with zero reuses — one morsel
spanning sixteen chunks slices each chunk once, so there is nothing left to share. Sharing and
coalescing are substitutes on wide numeric data and complements on misaligned string data, where
even coalesced morsels straddle the small text chunks.

**The one-lock version of the cells was a measured failure.** With a single mutex, wide-numeric
at 4 threads ran *slower* than at 1 (0.53 vs 0.50): thousands of tiny lease operations serialised
the scan. Sixteen shards restored 0.34. This is the admission-plane lesson E2 is designed to
catch — a shared structure touched per (node, morsel) must never be a single point of
serialisation — surfaced early by the lease ledger.

**Cascade and parallel conjuncts remain within noise of each other** on these cheap-predicate
workloads (0.366 vs 0.340 at 4 threads); the expensive-conjunct case that should separate them
is still not in the fixtures.

### Two honest caveats in D's favour, to discount

- **The 4-thread rows spawn threads per run.** Sub-millisecond queries show D at 4 threads
  losing ground to D at 1 thread because ~200 µs of thread spawn dominates. A real
  implementation uses a pool. Read the 4-thread rows only on queries above a few milliseconds.
- **Time-to-first-batch is not directly comparable.** D's is measured from the first morsel a
  thread completes, V1's from the first item off the stream. D's numbers are much better
  (0.55 ms vs 8.7 ms on `SH1`) and the direction is real — D emits as soon as one morsel
  finishes rather than after the pipeline fills — but the two clocks are not measuring quite
  the same event.

## Where this leaves the phase order

P1's spine is built, correct against the V1 oracle, and faster than V1 on every workload measured
at equal thread count. What P1 cannot do is answer the question E1 was written to answer, because
the executor E1 compares against is not in this repository.

Two things would need to happen before the gate means anything:

1. **Locate or rebuild rows B and C.** If the self-paced experiment exists on a branch, running
   its pipeline mode on this host against these same fixtures makes the 5% comparison meaningful
   in one afternoon. If it does not exist, the bar has to be restated against something that does.
2. **Decide whether the shape-matched fixtures are enough.** They isolate scheduling-unit cost
   well, which is E1's actual subject, but the recorded 0.33/0.6 geomeans came from real
   encodings. Comparing a synthetic-fixture ratio against a real-suite ratio is not sound, and
   this document does not do it.

The leased shared cells built here are the first P2 slice landed ahead of schedule: the ~2x
cross-morsel decode reuse on misaligned string layouts is now delivered by demand-derived
retention (leases counted from the morsel cut, drained to zero by retirement) rather than by a
cache, with the no-reuse configuration retained as the state-for-state fairness row. What P2
still owes on top: sharing the *bytes* cells across threads the same way, verdict-driven
cancellation of unissued uses, and the latency-grid experiments (E2) that decide when
registration should be bypassed entirely.
