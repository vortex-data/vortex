# Morsel Prototype: API and Experiment Plan

Status: **implementation plan (2026-08-27)** for prototyping the model in
[morsel-based plan execution](morsel-based-plan-execution.md): a stateful `execute` interior per
morsel, plus a scheduler-visible IO plane (`next_plan` request revelation, demand cells, an
admission policy). The plan is structured so every phase lands with an experiment over real
queries, reusing the [self-paced findings](self-paced-plan-exec-findings.md) harness, fixtures,
and fair-comparison contract.

## Design decisions carried in

These were argued from the measured executor progression (single coordinator 2.53x -> pooled
shards 1.40x -> owned 0.79x -> pipeline 0.41x on FineWeb Q06; Q09's byte-identical 41% win as
scheduling-unit attribution) and from working through lists, ALP-RD patches, and demand
refinement:

1. **Stateful execute spine.** One `execute` call drives a morsel's operator state machines
   inline. Suspensions that resume in nanoseconds on the same thread stay implicit (a program
   counter), never reified as tasks. Reify a wait only when it is long (IO), when another thread
   should resume it (CPU above the granularity floor), or when the scheduler must price it.
2. **IO plane as the only materialized graph.** `next_plan` reveals requests once; wait sets are
   the dependency edges; wakes are event-driven (ticket -> parked `(morsel, node)` cursor), never
   rediscovered by scanning.
3. **Emit-once planning with the completion invariant.** Planning may only rest in
   `Blocked(triggers)` or `Complete`, and `Complete` forfeits refinement: any event that could
   still improve this node's IO must be parked on explicitly. Refinement before emission comes
   from deferral; refinement after emission only through sampled state; replacement is
   cancel-plus-new-use and exceptional.
4. **Demand plane closed over scalars.** Monotone bit cells with block summaries (any/count) and
   a version counter; the scheduler reads only verdicts over `(cell, range)`. Each `IoUse`
   carries a `source_range` stamped at emission (the inverse image of its extent under the map
   known then); the owning leaf may re-stamp unissued uses on map-version bumps. Maps (offsets,
   patch indices, dictionaries) never cross into the scheduler.
5. **Derived demand as guarded memoization.** A gated map node caches
   `(output, input_version, input_true_count)`; consumers recompute only at IO-decision points
   and only when the input true-count drop crosses a threshold. Stale output is a sound superset.
6. **Inline bypass below the floor, for IO too.** A frontier read against fast local storage with
   sealed demand and no sharing potential may skip registration and issue inline
   (pipeline-style). Registration is reserved for requests that are speculative, shareable,
   cancelable, or high-latency.
7. **Policy, not code.** Cascade versus eager, per-conjunct ordering, speculation horizon, and
   re-cut thresholds live behind one `IoPolicy` object over sampled `IoFacts`; operators never
   encode scheduling.

## API under test

Condensed to the seams the experiments must exercise; the full contracts live in the design doc.

~~~rust
trait ExecNode {
    fn next_plan(&mut self, cx: &mut PlanCx) -> VortexResult<PlanPoll>;
    fn execute(&mut self, cx: &mut ExecCx) -> VortexResult<ExecPoll>;
    fn retire(&mut self, cx: &mut RetireCx);
}

enum PlanPoll { Item(PlanItem), Blocked(WaitSet), Complete }   // Complete forfeits refinement
enum ExecPoll { Value(ValueBatch), Blocked(WaitSet), Yield(Progress), Done }
enum Wait { Io(IoTicket), Fact(FactTicket), Cpu(CpuTicket), Credit(CreditTicket) }

struct IoUse {
    key: IoKey,                 // whole stored unit; straddling morsels join one cell
    extent: Extent,             // bytes/rows this use covers, frozen at emission
    cell: DemandCellId,         // producer-domain bit cell
    source_range: Range<u64>,   // inverse image of extent; leaf re-stamps on map upgrades
    producer: ProducerId,       // provenance for per-conjunct weighting
    estimated_bytes: usize,
}

// Sampled at decision points; never pushed.
struct IoFacts { demand: DemandVerdict, pending: PendingRefiners, cost: IoCost, unlocks: Unlocks }
enum Unlocks {
    Frontier { morsel: MorselId, distance_rows: u64 },  // 0 == a parked Blocked(Io) named it
    Refines  { cell: DemandCellId, producer: ProducerId },
    Gate     { reveals_est_bytes: u64 },                // facts that unlock planning (offsets)
}

trait IoPolicy {
    fn priority(&self, f: &IoFacts, est: &Estimator) -> Priority;
    fn admit(&mut self, budgets: &Budgets);
}
~~~

`Estimator` is scan-wide state (per-producer selectivity and refinement velocity, per-device
latency), updated from observed masks; demand cells are morsel-owned and die at retire.

## Implementation phases

Each phase has a gate experiment; do not start the next phase until the gate passes or the
failure is written up in the findings doc.

### P0: harness

Port the self-paced comparison contract unchanged: V1 as semantic oracle (row counts and ordered
hashes validated before timing), five alternating iterations, cold-scan IO invariants, the
existing FineWeb Q00-Q17, TPC-H SF10 (Q1, Q6, V1-friendly), and ClickBench (20-file, 21 shapes)
fixtures. Add two harness capabilities:

- **Latency injection**: a wrapper IO source with configurable per-request latency (0, 1 ms,
  10 ms, 50 ms) and a bounded in-flight window, to stand in for object storage.
- **Demand-plane chaos**: run with the out-of-band plane disabled and maximally delayed;
  results must be byte-identical (the commutation-law differential from the demand design).

### P1: morsel exec spine (no IO plane)

`FLAT`, `CHUNKED`, `STRUCT`, `FILTER`, `CONJUNCT_PARALLEL` as arena-owned state machines,
threads self-scheduling morsels off one shared cursor, per-thread decoded-chunk cache, inline IO
bypass only. This is a re-expression of the winning pipeline mode through the new trait; the
deterministic unit tests from the design doc's correctness properties (order-varying IO
completion and polling) come with it.

**Gate (E1).**

### P2: IO plane

Use registration, keyed shared cells with leases, demand cells + verdicts, parked `(morsel,
node)` wakes, `IoPolicy` with the required/speculative split, cascade and eager as policy
objects, the floor bypass. **Gate (E2, E3).**

### P3: gated planning

List offsets (gated open, element-domain child, derived demand with the memo guard), sub-segment
cuts for bit-packed buffers and ALP-RD patches (static affine cuts plus the indices-gated
patch-value cut), dictionary referenced-values. **Gate (E4).**

### P4: adaptive policy

Per-conjunct estimator-driven admission weighting, just-in-time speculation horizon from measured
latency and frontier velocity, pending-refiner discounts, re-cut thresholds. **Gate (E5).**

## Evaluation matrix

The headline evaluation is a same-host, same-fixture comparison of four executors over the
restricted real layout node set — **FLAT, CHUNKED, and STRUCT only, plus FILTER and
CONJUNCT_PARALLEL** (every suite query already lowers to struct-of-chunked-flat columns with
conjunct predicates, so no query changes are needed; list, dictionary, and ALP-RD gated planning
stay out of the matrix and are evaluated separately in E4):

| Row | Executor | Role |
| --- | --- | --- |
| A | V1 `LayoutReader` | semantic oracle and baseline; validates row counts and ordered hashes |
| B | Self-paced **graph/reactor** (existing experimental code, single-coordinator and owned modes) | the dependency-graph-as-data comparator |
| C | Self-paced **pipeline** mode | the fastest recorded stateful executor; the bar the new API must not regress |
| D | **This prototype** (stateful execute spine + IO plane) | the system under test |

Rows B and C are rerun on the measurement host, not quoted from the findings doc, so all four
rows share hardware, fixtures, and iteration discipline. Every experiment below reports the full
matrix unless it names a subset.

All experiments record, per run: wall time, requests issued, bytes read, bytes cancelled
pre-issue, speculative bytes wasted, demand-plane microseconds, per-morsel use counts, wake
counts and parked-wake latency, and time-to-first-batch. Comparisons are five-iteration medians
under the fair contract.

### E1: overhead parity on local NVMe (gate for P1)

*Hypothesis:* the trait seam costs nothing; the stateful spine reproduces pipeline-mode
performance and preserves the measured ordering D ≈ C < B(owned) < B(coordinator), with V1
between B's two modes.

Run the full 42-workload suite (18 FineWeb, 3 TPC-H, 21 ClickBench) across the whole matrix.
Success: D's geometric means within 5% of C's same-host rerun (~0.33 FineWeb, ~0.6 ClickBench vs
V1 in the recorded results), and a Q09 rerun (byte-identical IO by construction across all four
rows) reproduces the scheduling-unit attribution — D and C beat A and B with equal physical
work. A miss localizes to the trait dispatch or arena layout and must be profiled before
proceeding.

### E2: does the IO plane earn its overhead? (gate for P2)

*Hypothesis:* on injected latency the registered IO plane beats inline-blocking reads by an
amount that grows with latency, and on NVMe the floor bypass keeps it at E1 parity.

Grid: {inline-only, IO plane with bypass, IO plane forced (no bypass)} x {0, 1, 10, 50 ms} over
a latency-sensitive subset (FineWeb Q06, Q09, Q10; TPC-H Q6; ClickBench dashboard plus two
selective shapes), with rows A and B run at each latency point as external references — the
graph row is the interesting comparator here, since request visibility is the one thing it
bought. Success: forced-plane at 0 ms costs <5% vs inline (bounds the reification tax);
with-bypass at 0 ms is at parity; at 10 ms+ the plane wins materially on every shape with
overlappable IO, on both wall time and time-to-first-batch, and D at 10 ms+ is at least at
parity with B — showing the IO plane recovers the graph's latency-hiding without its CPU
bookkeeping. Also record queue dwell and admission loop occupancy to confirm no
coordinator-style serial section reappears (admission busy <10% of one core).

### E3: demand value and speculation pricing (with P2)

*Hypothesis:* cancellation and prefetch pricing reproduce the measured selective-shape wins, and
policy choice is genuinely swappable.

Shapes: Q12 (empty result), Q13 (narrow selective), Q10 (shared filter/projection), Q01/Q02, and
the ClickBench selective additions. Sweep {cascade, eager, adaptive} x latency {0, 10 ms}.
Measure bytes cancelled, first-predicate-only request counts on Q12 (target: match the recorded
1,823 vs 7,292), and speculative waste against the budget. Success: adaptive is within noise of
the best static policy per shape, never the worst; demand-plane time <1% of run time; the chaos
run stays byte-identical.

### E4: gated planning and sub-segment reads (gate for P3)

*Hypothesis:* row-to-byte cuts pay for themselves and gated facts sequence correctly under
latency.

Fixtures: a list-heavy synthetic (variable lengths, empty/null/giant lists, nested one level)
plus TPC-H `l_comment`-style strings and an ALP-RD float fixture with a measured patch rate;
predicates on list length and on scalar columns so element demand seals late. Measure bytes
saved by static cuts (bit-packed left/right parts) and gated cuts (patch values, element runs)
against whole-segment reads; per-morsel planning microseconds (target <1% of morsel time, uses
per morsel bounded by the run-count guard); offsets rushed as `Gate` priority (verify offsets
never queue behind bulk data). Differential: V1 list oracle cases (empty, null, straddling,
fallible expressions on demanded rows only).

### E5: adaptive conjunct ordering (gate for P4)

*Hypothesis:* estimator-driven ordering matches the best static order without being told it, and
beats static on skew.

Shapes: Q11 and ClickBench Q45 (five-conjunct chains), plus a synthetic where selectivity
inverts halfway through the file (clustered predicate). Success: on stationary data, adaptive is
within noise of the best static conjunct order; on the inverting fixture it beats every static
order; estimator overhead is unmeasurable.

### E6: microbenchmarks (continuous)

Criterion benches pinned in CI for the seams the design keeps promising are cheap: use
registration/retire round trip, verdict sample (with and without an offsets inverse map), parked
wake to re-entry latency, derived-demand guard hit and miss, one 128K mask intersect with
summary maintenance. These are the budget table backing every "this is noise" claim; regressions
fail the run.

## Exit criteria

The prototype graduates to a replacement plan when: E1 and E2 gates pass; every measured
workload is at parity or better with the pipeline mode on NVMe and strictly better under 10 ms
injected latency; the chaos differential and V1 oracle have no divergences; and the findings doc
records per-experiment tables in the same format as the self-paced reports. Anything that fails
gets written up with a phase-timing breakdown before the design doc is amended — the reactor's
lesson is that architecture verdicts come from attributed measurements, not totals.
