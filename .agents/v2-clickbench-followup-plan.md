# V2 ClickBench Follow-Up Plan

## Problem Space

The central problem is not simply that V2 is "streaming" and V1 is "pull-based". It is that
runtime information used to skip work can arrive too late in a push-oriented plan. V1 can carry a
mask/future down the tree before a child decides whether to fetch, build, or execute values. V2 can
end up discovering an all-false mask while returning batches upward, after child streams have
already scheduled I/O or forced CPU work.

Be precise about work types. Vortex arrays are lazy: building a Vortex `ArrayRef`, applying
`array.filter(mask)`, or applying an expression is not necessarily full decompression or
canonicalization. Distinguish these stages when interpreting logs and profiles:

- segment materialization: awaiting segment bytes and constructing a lazy Vortex array tree;
- predicate mask execution: forcing a boolean expression to `Mask`, e.g. string LIKE/FSST DFA work;
- lazy filter/projection construction: creating filtered/applied array graphs that may execute
  later;
- canonicalization/decompression/downstream execution: work forced by `execute`, Arrow conversion,
  aggregation, grouping, sorting, or string kernels.

### Demand Domains

`RowDemand` should remain an exact ordinal-mask abstraction inside row-domain-preserving scan and
layout subtrees. In that domain, row `i` still means row `i` in the file or partition, so nodes can
skip real value production and preserve cardinality with placeholders. Dense masks are cheap to
slice, intersect, count, and answer "is this segment/window fully dead?"

This does not generalize cleanly to every physical query operator. After joins, grouping,
repartitioning, or unordered exchanges, row ordinals may no longer be stable or meaningful. The
query-engine-level abstraction is runtime filtering / SIP over an explicit domain: key ranges,
bloom filters, hash sets, sorted-key intervals, partition ids, or some other identity. A scan may
lower those filters into `RowDemand` only when it has a mapping from that domain back to file row
coordinates.

The useful split is:

- `RowDemand`: scan-local, row-domain-preserving, exact ordinal masks;
- key/partition/SIP demand: query-plan-level, domain-typed runtime filters;
- translation points: scan/index/statistics operators that lower SIP demand into row ranges or
  row masks when possible.

Attaching `row_idx` and joining later may be useful at boundaries where row identity must leave the
scan domain, but it is probably too late for scan-local late materialization. It converts a cheap
mask/range skip question into tuple flow and join/filter work.

### Preventing Streams From Outrunning Demand

Every node that can avoid expensive work should check demand before scheduling I/O or CPU and again
after each await, because another predicate or runtime filter may have advanced while the task was
suspended. This is necessary but not sufficient.

The missing piece is work admission control. A push stream needs a bounded speculative window so it
cannot run far past the frontier where useful demand information is expected to arrive. The parent
or coordinator should own dynamic permits/windows for children. Demand-producing work should get a
larger budget when it is likely to save more downstream work than it costs; value/projection work
should be held closer to the known-demand frontier when its rows are likely to be discarded.

Useful runtime information for that scheduler:

- current coverage frontier and version of each demand/SIP resource;
- selectivity, all-false rate, and output density by range/window;
- elapsed time, rows/sec, and allocation pressure for each predicate or projection;
- estimated downstream bytes/CPU that could be skipped if the resource is refined;
- confidence/staleness of the estimate and observed queue/backpressure state.

The scheduling question is value-of-information: should an operator wait for a demand resource,
drive the demand producer further ahead, or proceed speculatively? A fixed timeout or fixed row
window is only a crude approximation. The better long-term shape is adaptive row/byte/time budgets
that expand for cheap/selective demand producers and shrink for expensive consumers when demand is
still uncertain.

### Design Constraints

- Do not blindly copy V1. V1 is the performance baseline, but V2 should keep the extra physical
  plan structure if it lets us express demand resources, stream alignment, and future query-engine
  operators more cleanly.
- Do not make `RowDemand` the universal query-plan abstraction. Keep it exact and row-domain
  oriented; use domain-typed SIP/runtime filters elsewhere.
- Do not over-interpret "decode" in profiles or logs. Verify whether work is segment
  materialization, lazy array graph construction, predicate mask execution, or downstream forced
  execution.
- Avoid solutions that only help one query shape. Classify results by predicate count, predicate
  type, selectivity, projected column width, chunk/window shape, and downstream aggregation/sort
  behavior.
- The target is adaptive execution: reach the good cases for Q21/Q22-style selective string
  predicates without regressing single-conjunct filters, cheap numeric predicates, or no-filter
  aggregation queries.

## Solution Proposal: Demand-Aware Stream Scheduling

Keep the V2 physical plan model, but stop treating all child streams as equally entitled to run
ahead. The immediate implementation can stay stream-based. Spawned producer tasks and custom
futures can emulate the flow inversion we eventually want from push-based pipelined execution:
demand/SIP producers publish information first, while expensive consumers receive bounded
speculative permits.

The proposal has three layers:

1. domain-typed runtime information resources;
2. demand-aware admission control for streams;
3. scan-local lowering rules that exploit `RowDemand` without making it the general query-engine
   abstraction.

### 1. Runtime Information Resources

Generalize the current `Resource` / `DemandSource` shape into a small family of runtime information
resources. `RowDemand` remains the exact scan-local ordinal-mask resource. A future SIP resource can
cover key ranges, bloom filters, partition ids, or sorted intervals.

For scheduling, a resource needs more than `mask_for(range)`. It should expose enough metadata for
the caller to decide whether waiting is valuable:

- `version()`: has the observable information changed?
- `coverage(domain_range)`: unknown, partial, or complete coverage for the requested domain.
- `estimate(domain_range)`: estimated selectivity, all-false probability, confidence, and expected
  saved bytes/CPU.
- `refine(domain_range, budget)`: optional async work to produce more information for the range.
- exact answer when available, e.g. `RowDemand::mask_for(range)`.

For correctness, unknown demand must be treated as "still demanded". For scheduling, unknown is a
separate state: an operator may choose to wait for more coverage if the expected saved work is high.
This avoids overloading `Mask::new_true` to mean both "known all true" and "not known yet".

Current `PublishedMaskDemand` is a useful prototype but should grow a coverage/frontier API. It can
publish mask batches as they are produced by a filter/conjunct stream. Consumers that ask about a
covered all-false range can skip work; consumers that ask about uncovered ranges can either wait,
drive the publisher, or proceed speculatively.

### 2. Flow-Controlled Streams

Replace fixed-depth child producer channels with explicit per-child permits. Today
`AlignedArrayStream` spawns every child and lets each run ahead by `CHILD_BUFFER_DEPTH` chunks. That
is simple, but it hides the decision that matters: which child should spend I/O/CPU next?

Introduce a `FlowControlledAlignedStream` or evolve `AlignedArrayStream` with a scheduler:

- each child producer owns its source stream but must acquire a permit before polling it;
- permits are row/byte/time budgets, not just channel slots;
- the parent/coordinator updates permits after every emitted batch and after every resource version
  change;
- producers still run concurrently, but only within their granted speculative window;
- dropping the aligned stream cancels producers and outstanding work as today.

The first prototype can use coarse permits:

- demand-producing children: buffer depth `N` or row budget `64k`;
- expensive value/projection children: buffer depth `0..1` until relevant demand coverage exists;
- cheap/narrow children: small speculative budget;
- all budgets adjust using observed selectivity and elapsed time.

Longer term, use a value-of-information score:

```text
score = expected_downstream_work_saved / expected_resource_cost
```

Give more permits to work with a high score. Shrink permits when a child produces low-value
information, when downstream consumers are backpressured, or when a range is already covered.

This keeps streams as the public execution shape for now. Internally, spawned producers plus permit
futures act like pipeline drivers: demand producers can be pushed forward, consumers can be held at
the frontier, and all tasks are still cancellable by dropping the parent stream.

### 2b. Bounded Out-of-Order Dataflow With In-Order Retirement

The better long-term shape is a bounded, backpressured streaming dataflow pipeline with
out-of-order execution and in-order retirement. Row offset should be an ordering and correctness
constraint, not the primary scheduling priority.

Each produced unit should look more like a morsel than an iterator batch:

- `domain`: ordinal row domain, keyed domain, sorted-key domain, or translated scan-local domain;
- `order_key`: sequence number, ordinal range, key range, or sorted interval used for retirement;
- `range_lineage`: input ordinal/key ranges covered or proven empty;
- `estimated_bytes`: queued output bytes plus scratch/mask bytes;
- `estimated_cpu_ns`: cost estimate from prior observations, stats, or encoding metadata;
- `operator_role`: demand producer, demand consumer, value producer, projection, combiner, sink;
- `info_value`: estimated downstream bytes/CPU that could be skipped if this morsel refines demand;
- `age`: time since the morsel became schedulable, used to avoid starvation.

Ordered streams then have two separate mechanisms:

- a scheduler chooses any ready morsel whose memory/work budget allows progress;
- a resequencer retires completed morsels only when the next `order_key` is ready or proven empty.

This matches the CPU analogy: execute out of order, retire in order. It also matches the scan
problem: a selective conjunct can run ahead of a projection even when its row offsets are later,
because its result may make earlier or later projected ranges cheaply skippable. The output edge
still cannot emit rows past its retirement cursor until all preceding ranges are accounted for.

Priority should therefore be a composite score, not just row offset:

```text
priority =
  readiness/dependency state
  + information value / estimated cost
  + lead credit for demand-producing work
  + age credit
  - memory pressure
  - downstream backpressure
```

The "lead credit" is the useful version of a frontier adjustor. It lets a predicate or SIP producer
run ahead when it is cheap, selective, or likely to make other work skip. It shrinks when the
producer stops being selective, has high cost per useful bit, fills its reorder buffer, or downstream
work is already saturated. The unit should not be "rows" alone. Rows are useful for alignment and
retirement, but admission should be bounded primarily by bytes, queued results, and estimated CPU
time. A wide string morsel and a narrow integer morsel with the same row count should not consume the
same budget.

The anti-deadlock rule is that row order never becomes the only grant currency. A scheduler may
prefer demand producers, but it must keep a work-conserving escape hatch:

- every active ordered edge gets a small byte/work credit floor when it is on the critical path to
  the retirement cursor;
- if no high-priority morsel can run and the next retirement key is blocked, grant the smallest
  dependency needed to prove the next key live, dead, or still waiting on I/O;
- memory backpressure can stop speculative lead work, but it should not stop the unique work item
  needed to retire the oldest blocked range;
- completed all-false/empty morsels retire just like value morsels, so dropping rows cannot block
  the sequencer.

For `ConjunctPlan`, this means the conjunct scheduler should release and prioritize morsels by
estimated value of information:

- cheap/selective conjuncts receive lead credit because they publish demand for siblings;
- expensive conjuncts receive little or no lead until upstream demand says their range is likely
  live;
- lead is recomputed from observed `ns/input_row`, selectivity, all-false rate, bytes avoided, and
  backlog;
- sibling conjuncts may execute different ranges at the same time, but combined masks retire in
  ordinal order.

For `ChunkedPlan`, the equivalent is an ordered readahead edge:

- child chunks may be polled/spawned out of order or ahead of the current output chunk;
- each child result is keyed by chunk ordinal plus local range;
- a bounded reorder buffer preserves output order;
- the memory budget controls how many later chunks may be ready while an earlier chunk is still
  running.

This is different from the current row-frontier prototype. The current `OutputFrontier` proves that
leaf plans can accept backpressure, but manually releasing concrete row frontiers inserts round
trips and overfits to ordinal rows. The next prototype should keep `OutputFrontier` as a leaf-facing
admission API, but replace "release rows to child X" with a scheduler-owned morsel queue,
byte/work budgets, priority scoring, and per-edge resequencing.

### 2c. Information-Prioritized Query Engine Design

The query engine should treat runtime information as a first-class output of operators, not as an
incidental side effect of data batches. SIP is often back-propagated from joins or aggregations to
scans, while conjunct masks and row demand are more local to adjacent scan/filter/projection
operators. Both are the same scheduling problem: useful information has a producer, a domain, a
coverage frontier, a cost, and consumers that can skip work if they see it soon enough.

Model execution as three interacting planes:

- data plane: morsels carrying values, masks, or empty-range proofs through operator edges;
- information plane: versioned resources such as row demand, key filters, bloom filters, sorted
  ranges, partition filters, and statistics-derived zone decisions;
- control plane: scheduler budgets, priorities, admission, cancellation, and ordered retirement.

Operators should declare information contracts at planning/lowering time:

- domains they preserve exactly, transform, destroy, or can lower back to scan-local ordinals;
- information resources they can produce, with estimated cost and coverage granularity;
- information resources they can consume, with estimated bytes/CPU/I/O saved;
- output ordering requirements and retirement keys;
- memory footprint of queued outputs, scratch buffers, and in-flight I/O.

Examples:

- a scan over a sorted key can lower a key-range SIP resource into ordinal row ranges;
- a hash join build side can publish a keyed runtime filter consumed by probe-side scans;
- a selective conjunct can publish ordinal row demand consumed by sibling conjuncts and projected
  columns;
- a projection over wide strings consumes row demand but usually produces little information;
- a grouping or unordered exchange may destroy row ordinals and therefore cannot consume
  scan-local `RowDemand` directly.

The runtime should not repeatedly walk the physical tree to ask for work. Lower the plan into
long-lived work sources and information resources. A work source owns its operator state and
registers ready morsels with the scheduler when dependencies, budgets, and resource versions make
progress possible. The scheduler arbitrates among ready morsels; operators update resources as work
completes.

An implementation sketch:

```text
PhysicalPlan
  -> WorkSource[]        // long-lived drivers for operators or fused pipelines
  -> InfoResource[]      // row demand, SIP filters, zone decisions, coverage maps
  -> OrderedEdge[]       // resequencers and merge edges
  -> Scheduler           // budgets, priorities, cancellation, fairness
```

Each morsel carries:

```text
morsel_id
source_id
domain
order_key
input_lineage
required_resources
produced_resources
estimated_cpu_ns
estimated_io_bytes
estimated_memory_bytes
priority_class
```

The scheduler's central policy is value of information under bounded speculation:

```text
info_gain =
  probability_resource_refines
  * probability_consumers_overlap_coverage
  * estimated_consumer_work_saved

cost =
  estimated_cpu_ns
  + estimated_io_wait_penalty
  + estimated_memory_pressure
  + opportunity_cost_from_blocked_retirement

priority = info_gain / cost + lead_credit + age_credit - backpressure_penalty
```

`lead_credit` is not a row-count frontier. It is an allowance for information producers to stay
ahead of consumers in the units that can actually exhaust the runtime: output bytes, scratch bytes,
I/O bytes, CPU time, and reorder-buffer occupancy. Row ranges remain the coordinate system for
coverage and retirement, but they should not be the only budget. This avoids treating a wide string
range and a narrow integer range as equivalent just because they contain the same number of rows.

Consumers benefit when information coverage arrives before they admit their own morsels. The
scheduler should therefore maintain a target lead for good information producers:

- increase lead when a producer is cheap, selective, high-confidence, or repeatedly causes skips;
- decrease lead when the producer is expensive, unselective, low-confidence, or fills buffers;
- increase consumer speculation when information is late and the consumer is cheap or on the
  retirement critical path;
- decrease consumer speculation when its outputs are wide, likely dead, or backpressured.

This turns "sideways information" into an explicit scheduling signal. A filter does not need to be
physically above a projection to help it. If both morsels share a domain and the filter's resource
can cover the projection's range, the scheduler can run the filter further ahead and let projection
morsels observe the refined resource before they fetch or force values. For SIP, the same mechanism
crosses operator boundaries in the opposite direction of data flow: a join build morsel can publish
a keyed resource, scans can lower it if possible, and scan morsels whose domains overlap that
coverage gain or lose priority accordingly.

Retirement remains separate:

- ordered edges retire by `order_key` when data or empty proofs are complete;
- unordered edges may emit immediately but still consume memory budgets;
- sorted edges retire through k-way merge over stream heads;
- information resources may advance independently of data retirement, but cannot violate
  correctness because unknown information is treated as live/possible.

The design should bias toward operator fusion where it removes overhead, but not hide information
flow. A fused scan/filter/projection pipeline still registers separate resource-producing and
resource-consuming morsel roles internally. Fusion should reduce function calls and buffers; it
should not force symmetric polling of predicates and projections when one side is clearly more
valuable to run ahead.

Deadlock and starvation rules:

- every ordered edge has a retirement cursor and a bounded reorder buffer;
- every source on the oldest blocked retirement path gets a minimum progress credit;
- if no high-value morsel can run, run the cheapest morsel that can prove the oldest blocked range
  live, dead, or waiting on external I/O;
- information producers can receive lead, but cannot consume all memory or CPU credits forever;
- cancellation drops in-flight morsels whose output is no longer needed and releases their budgets.

This gives a query-engine-level generalization of scan-local late materialization. Scan-local
`RowDemand` is one information resource over an ordinal domain. SIP filters are information
resources over keyed or sorted domains. Zone maps and sortedness pruning are resource refiners.
Conjunct ordering, projection skipping, join runtime filters, and ordered stream readahead all
become instances of the same policy: prioritize morsels that produce high-value information early
enough for neighboring or back-propagated consumers to use it, while retiring required outputs in
the required order.

Prototype status in `vortex-layout/src/v2/dataflow.rs`:

- `PartitionScheduler` models one partition-local scheduler, intended to map to one DataFusion
  partition initially and later to a thread/core-style driver when the embedding allows it;
- the scheduler owns a bounded `BinaryHeap` priority queue;
- `SchedulerEvent` can be a data `SchedulerMorsel`, a `SchedulerIoRequest`, or a
  work-stealing/balancing `SchedulerControlEvent`;
- `make_progress` advances exactly one event, and data morsels move through one pipeline stage at a
  time before being requeued or completed;
- priorities encode `info_gain / cost + lead_credit + age_credit - pressure`, scaled as integers so
  the heap remains deterministic;
- information producers and retirement-critical work receive distinct event classes, so row offset
  is not the scheduler's primary ordering rule;
- memory and event-count budgets bound the local queue;
- `stealable_morsels` intentionally excludes information producers and retirement-critical sink
  morsels from early stealing.

### 3. Row-Domain Scan Lowering

Inside the scan/layout subtree, use `RowDemand` aggressively and exactly:

- every leaf or expensive node checks demand before scheduling I/O or forced execution;
- every such node checks again after await points;
- fully dead row-domain ranges return cardinality-preserving placeholders when the parent still
  expects row alignment;
- cardinality-changing compaction remains at explicit filter boundaries.

This gives scan-local late materialization without pretending that row ordinals are a general query
identity.

For non-row-domain SIP, add translation points:

- sorted/statistics/index-aware scan nodes lower key-domain filters into row ranges or row masks;
- unordered or untranslatable filters stay as normal predicates;
- operators outside the scan subtree consume/produce domain-typed resources, not `RowDemand`
  directly.

### Concrete V2 Prototype

Start with the filtered scan shape because it is where Q21/Q22 expose the issue.

1. Add `DemandCoverage` to row demand resources.
   - `Unknown`, `Partial`, `Complete`.
   - Preserve current `mask_for` correctness by treating unknown rows as demanded.
   - Add instrumentation for coverage wait time, coverage hits, and all-false skips.

2. Replace `FilterPlan` lockstep alignment with a mask-publisher path.
   - Start mask execution once for the whole `row_range`.
   - As mask batches arrive, publish them into `PublishedMaskDemand`.
   - Execute the value plan once over the same `row_range` with `demand.with_source(published)`.
   - The value stream should not be re-entered per mask batch.
   - The final filter boundary still compacts rows using the same mask batches.

3. Add demand-aware permits to the two-child filter alignment.
   - Mask child gets enough budget to stay ahead.
   - Value child gets budget only for ranges whose demand coverage is known or whose estimated wait
     cost is worse than proceeding.
   - This should emulate "mask future flows downward" while staying stream-based.

4. Extend the same mechanism to `ConjunctPlan`.
   - Keep dynamic conjunct ordering, but do not first drive all children symmetrically to an aligned
     batch.
   - Give larger windows to cheap/selective conjuncts and smaller windows to expensive or
     low-value conjuncts.
   - Publish partial conjunct results as row-domain demand for later conjuncts and projections.
   - Adapt windows using observed `ns/input_row`, selectivity, all-false rate, and downstream skip
     counts.

5. Make `AlignedArrayStream` scheduling pluggable.
   - Default policy preserves current behavior for low-risk plans.
   - Filter/conjunct policies use demand-aware permits.
   - Logs should report permits granted, rows produced, wait reason, demand coverage, and skipped
     work.

### Expected Query Classes

- Multi-conjunct selective string predicates should benefit most. Demand-producing string masks get
  enough read-ahead to collapse row demand, while projected values stay near the known-demand
  frontier.
- Single-conjunct filters need value-side demand checks and cheap predicate execution, not conjunct
  scheduling.
- Cheap numeric/date filters should not pay overhead for elaborate scheduling. The policy should
  converge to small windows or current-like lockstep.
- No-filter aggregation/grouping queries should stay on the default stream policy.

### Validation Plan

For each prototype, report V1 as the benchmark, not just V2-before/V2-after:

- focused Q21/Q22/Q23/Q26/Q38 comparisons with milliseconds, ratios, and percentages;
- all ClickBench sanity pass after promising changes;
- TPCH Q6/Q13 smoke pass;
- profile worker occupancy before/after for the focused query;
- counters for demand coverage, producer permits, all-false skips, bytes requested, mask execution
  rows, and projected value rows actually forced downstream.

The key success criterion is not merely fewer batches. It is: the plan should spend CPU/I/O on rows
whose demand is either known true or rationally worth speculating on, and it should converge to V1
or better without hard-coding a single query's chunk size or conjunct order.

### Prototype Status

The isolated model lives in `vortex-layout/src/v2/dataflow.rs`. It models:

- `Domain::{Ordinal, Keyed, Sorted}` with sorted domains able to advertise an exact ordinal
  lowering;
- vectorized batches over a domain range;
- an `OrdinalDemand` resource that separates correctness from scheduling by treating unknown rows
  as demanded while tracking explicit coverage/frontier state;
- a coarse `PermitPolicy` that can drive a demand producer, wait for demand, proceed over known
  live demand, skip all-false covered ranges, or grant small speculative value work.

The tests exercise the key feedback loop:

- unknown demand is correctness-true but can still make the scheduler wait;
- demand producers advance to the first uncovered frontier;
- all-false covered prefixes advance coordinates without polling value work;
- known-live prefixes permit value work;
- cheap/low-value unknown ranges receive small speculative permits;
- sorted key domains can declare an ordinal lowering path.

The first executable integration is gated by `VORTEX_V2_FILTER_DATAFLOW=1` in `FilterPlan`. It
drives the mask stream, publishes mask batches into an exact row-demand resource, uses the
`PermitPolicy` to wait/skip/proceed, and only constructs value streams for covered windows. It is a
functioning prototype, but not a performance win yet.

Focused ClickBench Q22, `datafusion:vortex`, 3 iterations, bench-profile binary:

- V1: `218.986ms`.
- Current V2: `469.021ms`, `2.14x` slower than V1, `+114.2%`.
- Existing full-mask domain-demand experiment with range demand pulls:
  `569.076ms`, `2.60x` slower than V1, `+159.9%`.
- Dataflow filter, default `64k` min value rows:
  `569.800ms`, `2.60x` slower than V1, `+160.2%`.
- Dataflow filter, `256k` min value rows:
  `644.364ms`, `2.94x` slower than V1, `+194.2%`.
- Dataflow filter, `1m` min value rows:
  `900.949ms`, `4.11x` slower than V1, `+311.4%`.

Flow trace for the `64k` prototype:

- mask batches published: `8,217`, true rows `7,128`;
- value windows emitted: `665`, true/output rows `7,128`;
- value-span rows: `52,452,765`, with span p50 `70,949`, p90 `102,693`, p99 `205,851`,
  max `438,271`.

Interpretation:

- The model does reduce the number of value-side windows versus per-mask lockstep and cuts the
  value-side row-span substantially, so the frontier/permit abstraction is expressing the intended
  control point.
- It is still slower because it waits for mask coverage and repeatedly constructs value-plan
  executions/read_all windows. Larger windows reduce re-entry pressure but lose too much overlap,
  which pushes the prototype toward the same behavior as full-mask domain demand.
- The next prototype should keep the value plan as one long-lived execution and gate producer
  polling with permits, rather than constructing a new value stream per covered window. That likely
  means moving permits into `AlignedArrayStream`/producer tasks or adding a lazy-start/permit-aware
  child-stream wrapper.

## Current Baseline

- ClickBench Q22 final shape measured V2 at `474.1ms` median versus V1 at `210.4ms`, `2.25x` slower / `+125.3%`.
- A dict mask-pushdown prototype removed the fallback `FilterPlan`, but regressed Q22 to `520.2ms`, `2.47x` slower / `+147.2%`, so it is not the right shape as-is.
- Regular row-preserving `FlatPlan` now checks `RowDemand` and can return placeholder arrays for fully dead ranges.
- Remaining evidence points at mask/string CPU, scheduling skew, and allocation churn more than duplicated projected-column I/O.

## Investigation Loop

1. Rebaseline before changing behavior.
   - Run focused ClickBench Q22 V1 versus V2 comparisons.
   - Run a broader ClickBench pass to identify the largest remaining regressions.
   - Run a small TPCH sanity pass after meaningful changes.
   - Report absolute milliseconds, ratios, and percentages immediately after each run.

2. Separate CPU, I/O, and scheduling effects.
   - Count bytes requested/completed, duplicate byte ranges, and cancelled segment futures.
   - Count per-plan input rows, output rows, demand cardinality before execution, and demand cardinality after awaited work.
   - Summarize per-conjunct compute rows and mask rows by predicate and coordinate range.
   - Use Samply worker occupancy summaries to quantify idle gaps and stragglers.

3. Inspect fallback filtering and alignment.
   - Check whether `FilterPlan` forces projected values to be produced before masks prove rows dead.
   - Check whether `AlignedArrayStream` polls values and masks symmetrically when the mask side should run ahead.
   - Evaluate whether `FilterPlan` should become demand-aware or whether this belongs in a renamed `ConjunctPlan`.

4. Improve conjunct scheduling in a V2-native way.
   - Keep stream-based execution, but dynamically vary read-ahead/window sizes.
   - Drive cheap/selective predicates further ahead so they can publish useful `RowDemand`.
   - Use an initial cost model where zone/sortedness-prunable comparisons are cheap, equality is a bit more expensive, and string contains/LIKE predicates are expensive.
   - Recompute selectivity as chunks arrive and adjust windows.

5. Attack string predicate CPU.
   - Compare V1 and V2 row-coordinate coverage for each conjunct.
   - Confirm `str != ""` lowers to a cheap emptiness/length check where possible.
   - Check whether FSST LIKE/contains paths decompress or scan more than V1.
   - Look for avoidable bool-to-mask materialization.

6. Reduce allocation churn.
   - Track mask, bit-buffer, filter/take, and string-decode scratch allocations.
   - Consider per-execution reusable buffers or exact-size reservations where chunk sizes are known.

7. Validate broadly.
   - Rerun the focused query after each scoped change.
   - Rerun the worst ClickBench regressions and TPCH Q6 before considering the change complete.

## Findings From 2026-05-16 Follow-Up

### Command Shape

- The persistent opener path is the correct V1/V2 comparison for this branch.
- `VORTEX_USE_SCAN_API=1` exercises the DataFusion scan API path and does not measure the
  `persistent/opener.rs` layout V1/V2 switch; it produced a false Q22 near-parity comparison.

### ClickBench Rebaseline

- Focused Q22, persistent opener, 5 iterations:
  - V1: `218.217ms` median.
  - V2: `470.110ms` median.
  - V2 is `2.15x` slower, `+115.4%`, `+251.9ms`.
- All ClickBench, persistent opener, 3 iterations:
  - Worst ratio: Q22, V1 `219.227ms`, V2 `517.538ms`, `2.36x`, `+136.1%`, `+298.3ms`.
  - Next meaningful ratio regressions: Q38 `1.91x` / `+91.0%` / `+8.2ms`, Q26 `1.73x` /
    `+72.9%` / `+14.1ms`, Q36 `1.71x` / `+70.8%` / `+21.8ms`, Q21 `1.59x` / `+58.6%` /
    `+113.4ms`.
  - Largest absolute regressions after Q22: Q23 `+176.3ms` / `+44.6%`, Q28 `+154.4ms` /
    `+3.7%`, Q33 `+133.0ms` / `+17.5%`, Q27 `+130.5ms` / `+47.8%`.
  - Biggest wins: Q07 `0.33x` / `-67.0%` / `-18.3ms`, Q01 `0.34x` / `-66.4%` /
    `-12.7ms`, Q02 `0.45x` / `-55.5%` / `-21.8ms`, Q20 `0.91x` / `-9.0%` / `-27.3ms`.

### Q22 Row/Mask Diagnostics

- V2 top-level fallback filtering:
  - `8,684` projected filter batches.
  - `6,378` all-false batches, `73.4%`.
  - `99,997,497` input rows to `7,128` output rows, `0.007128%` density.
  - Largest all-false batches include `524,288`, `524,288`, `475,712`, `470,169`,
    `377,757`, and `374,961` input rows.
  - Duplicate coordinate masks: none.
- V2 conjunct compute:
  - `Title LIKE '%Google%'`: `8,206` events, `79,789,844` input rows, `32,417` output rows.
  - `URL NOT LIKE '%.google.%'`: `2,262` events, `7,185` input rows, `7,185` output rows.
  - `SearchPhrase != ''`: `5,641` events, `21,433,362` input rows, `1,200,420` output rows.
- V1 filter-stage conjunct compute:
  - `Title`: `8,665` events, `92,047,472` input rows, `34,903` output rows.
  - `URL`: `2,712` events, `2,121,597` input rows, `2,121,574` output rows.
  - `SearchPhrase`: `5,396` events, `8,334,767` input rows, `949,694` output rows.
- Interpretation:
  - V2 is not simply evaluating more predicate rows overall.
  - The strongest current signal is the fallback `FilterPlan` shape: projected values are aligned
    with the mask and produced for very large ranges that the mask later proves all-false.
  - Q22 still does not have duplicate coordinate masks in V2.

### Samply Comparison

- Fresh profiles:
  - V2: `/private/tmp/clickbench-q22-v2-followup.profile.json.gz`
  - V1: `/private/tmp/clickbench-q22-v1-followup.profile.json.gz`
- Profiled timings:
  - V1 median `206.550ms`.
  - V2 median `478.509ms`.
  - V2 is `2.32x` slower, `+131.7%`, `+272.0ms`.
- Worker occupancy:
  - V1: mean active workers/bin `64.36`, median `66`, p10 `48`, p90 `81`; only startup has
    active workers <= 4.
  - V2: mean active workers/bin `40.31`, median `46`, p10 `4`, p90 `78`; repeated low-activity
    ranges around `490..550ms`, `950..1010ms`, `1430..1500ms`, `1910..1970ms`, and
    `2380..2450ms`.
- Hot stacks:
  - V2 hot workers are dominated by `vortex_array::mask::<Mask>::execute` with FSST LIKE beneath
    it: `vortex_fsst::dfa_scan_to_bitbuf`, FSST decompression, and `memmem`.
  - V1 is more evenly filled and centered around `MaskFuture`, FSST decode, take/filter work, and
    I/O scheduling; it does not show the same repeated low-activity gaps.

### TPCH Sanity

- One iteration only, so use as smoke signal rather than stable ranking.
- Largest TPCH ratio regression: Q13, V1 `13.086ms`, V2 `21.842ms`, `1.67x`, `+66.9%`, `+8.8ms`.
- Q6: V1 `6.183ms`, V2 `7.723ms`, `1.25x`, `+24.9%`, `+1.5ms`.
- Largest win: Q11, V1 `7.675ms`, V2 `6.068ms`, `0.79x`, `-20.9%`, `-1.6ms`.

## Next Ideas

1. Prototype a mask-first fallback `FilterPlan` path for row-preserving value plans.
   - Today `FilterPlan` constructs both streams and `AlignedArrayStream` lets both producers run.
   - For projection values that cannot absorb the mask, first drive the mask for the next aligned
     range, publish/use row demand, and only then poll values for non-empty ranges.
   - Preserve V2 streaming design; do not just copy V1's whole split loop.

2. Make `AlignedArrayStream` support per-child producer permits or dynamic buffer depths.
   - Current fixed `CHILD_BUFFER_DEPTH = 4` lets less useful streams run ahead.
   - A `ConjunctPlan` scheduler should be able to give more window budget to cheap/selective
     predicates and hold back expensive/projection streams.

3. Revisit dict projection pushdown only with a streaming mask shape.
   - The earlier dict pushdown removed fallback `FilterPlan` but regressed Q22 to `520.2ms`.
   - That suggests materializing/shared-mask shape and lost overlap outweighed skipped projection.
   - A useful retry must preserve streaming overlap and avoid full-mask materialization.

4. Optimize string predicate execution separately.
   - Q22 remains FSST LIKE heavy.
   - `SearchPhrase != ''` should be lowered to a cheap non-empty/length check where possible.
   - Check whether `%Google%` can avoid full decompression or reduce bit-buffer allocation churn.

5. Investigate Q21/Q23 after Q22.
   - Q21 is the next large ratio regression with meaningful absolute cost.
   - Q23 is the next largest absolute regression.

## Feature Matrix: ClickBench Conjunct Windows and Demand

### Setup

- All runs used the persistent opener path, direct `datafusion-bench`, `RUST_LOG=error`, and
  `--display-format gh-json`.
- V2 original batching is reproducible with `VORTEX_LAYOUT_PLAN_V2=1
  VORTEX_V2_CONJUNCT_MIN_ROWS=1`.
- Temporary experiment toggles used:
  - `VORTEX_V2_CONJUNCT_MIN_ROWS=<rows>`
  - `VORTEX_V2_STATIC_CONJUNCT_ORDER=1`
  - `VORTEX_V2_DISABLE_CONJUNCT_DEMAND=1`
  - `VORTEX_V2_DISABLE_ROW_DEMAND=1`
  - `VORTEX_V2_FILTER_MASK_FIRST=1`
  - `VORTEX_V2_ALIGNED_BUFFER_DEPTH=<n>`

### All ClickBench Window Sweep

- 43 queries, 3 iterations:
  - V1 total median sum: `12,169.392ms`.
  - V2 original (`min1`): `14,708.567ms`, `+20.9%` versus V1.
  - Fixed `16k`: `14,378.331ms`, `-2.2%` versus V2 original.
  - Fixed `32k`: `14,446.873ms`, `-1.8%` versus V2 original.
  - Fixed `64k`: `15,055.632ms`, `+2.4%` versus V2 original.
  - Fixed `128k`: `15,796.507ms`, `+7.4%` versus V2 original.
  - Fixed `256k`: `16,153.088ms`, `+9.8%` versus V2 original.
  - Per-query oracle across tested windows: `14,014.830ms`, `-4.7%` versus V2 original,
    still `+15.2%` versus V1.
- Best fixed window count by query:
  - `min1`: 17 queries.
  - `16k`: 8 queries.
  - `32k`: 3 queries.
  - `64k`: 3 queries.
  - `128k`: 5 queries.
  - `256k`: 7 queries.
- Strongest per-query window wins:
  - Q22: `556.867ms -> 360.654ms` with `64k`, `-196.214ms`, `-35.2%`.
  - Q21: `337.217ms -> 274.773ms` with `64k`, `-62.444ms`, `-18.5%`.
  - Q13: `302.853ms -> 262.473ms` with `128k`, `-40.380ms`, `-13.3%`.
  - Q33/Q34 showed apparent 3-iteration window wins, but these are no-filter group-by queries;
    treat them as noise or indirect cache/scheduler effects, not evidence for conjunct batching.

### Filtered Query Subset

- Queries: Q21-Q27 and Q37-Q42, 10 iterations.
- Totals:
  - V1: `1,404.747ms`.
  - V2 original: `2,000.771ms`, `+42.4%` versus V1.
  - `16k`: `1,913.612ms`, `-4.4%` versus V2 original.
  - `32k`: `1,898.973ms`, `-5.1%`.
  - `64k`: `1,790.548ms`, `-10.5%`, still `+27.5%` versus V1.
  - `128k`: `1,888.430ms`, `-5.6%`.
- Feature toggles at original batching:
  - Static order: `2,150.366ms`, `+7.5%` versus V2 original.
  - Disable conjunct demand: `2,150.402ms`, `+7.5%`.
  - Disable flat row demand: `2,144.581ms`, `+7.2%`.
  - Naive mask-first fallback: `2,315.130ms`, `+15.7%`.
- Interactions at `64k`:
  - `64k` alone: `1,790.548ms`, `-10.5%`.
  - `64k + static`: `1,788.434ms`, `-10.6%`.
  - `64k + no conjunct demand`: `1,791.793ms`, `-10.4%`.
  - `64k + static + no conjunct demand`: `1,851.828ms`, `-7.4%`.
  - `64k + no row demand`: `1,818.372ms`, `-9.1%`.
  - `64k + mask-first`: `2,075.898ms`, `+3.8%` versus V2 original.
- Per-query filtered subset highlights:
  - Q22: `482.060ms -> 308.375ms` with `64k`, `-173.685ms`, `-36.0%`;
    V1 is `258.603ms`, so this leaves `+19.2%`.
  - Q21: `285.537ms -> 227.698ms` with `64k + no conjunct demand`, `-57.839ms`,
    `-20.3%`; V1 is `196.013ms`, leaving `+16.2%`.
  - Q23: almost insensitive to windows; best measured `581.485ms` versus `581.702ms`,
    effectively flat, and still `+35.9%` versus V1.
  - Q26: insensitive to conjunct features because it has a single filter conjunct; best remains
    V2 original at `32.075ms`, still `+41.9%` versus V1.
  - Q38: insensitive to windowing; best measured `17.500ms` with no conjunct demand,
    still `+74.1%` versus V1.

### Representative Debug Counts

- Q22 `min1`:
  - Filter projection: `8,684` batches, `6,378` all-false (`73.4%`),
    `99,997,497` input rows to `7,128` output rows.
  - Conjunct events: `16,169`.
  - Logged conjunct elapsed: Title LIKE `2,000.755ms`, URL NOT LIKE `637.146ms`,
    SearchPhrase non-empty `278.980ms`.
- Q22 `64k`:
  - Filter projection is unchanged: `8,684` batches and `99,997,497` input rows.
  - Conjunct events drop to `3,120`, `-80.7%`.
  - Logged conjunct elapsed drops to Title LIKE `1,460.682ms`, URL NOT LIKE `708.700ms`,
    SearchPhrase non-empty `312.947ms`; total logged conjunct elapsed drops about `14.9%`.
  - Interpretation: the Q22 win is mostly lower per-batch/string-expression overhead and better
    scheduling shape, not reduced final projected input rows.
- Q38:
  - Filter projection: `168` batches, `115` all-false (`68.5%`), `3,000,000` input rows to
    `47,740` output rows.
  - Conjunct events are only `66`, and logged conjunct elapsed is about `4ms` total.
  - `64k` does not materially change row counts or event counts.
  - Interpretation: Q38's remaining gap is not conjunct scheduling; investigate projected string
    decode/grouping and DataFusion aggregation work.
- Q26:
  - Single conjunct, so no `ConjunctPlan`.
  - V2 filter projection: `418` batches, `29,779,853` input rows to `3,743,321` output rows.
  - V1 filter stage: `476` batches, `33,648,500` input rows to `4,185,864` output rows.
  - Interpretation: V2 is not slower because it filters more rows. The gap is likely downstream
    CPU/materialization/sort behavior, plus the need to lower `SearchPhrase <> ''` to a cheap
    emptiness check where possible.

### Classification

- Multi-conjunct LIKE/string filters (Q21, Q22) benefit the most from larger conjunct windows.
  Their bottleneck is many small string-mask evaluations and scheduler skew. Tested best is around
  `64k`, not the largest window.
- Single-conjunct filters (Q23-Q27) do not benefit from conjunct scheduling. Their remaining gaps
  need projection/filter/sort/string materialization work.
- Multi-conjunct numeric/date filters (Q37-Q42) mostly do not benefit from larger windows because
  predicate evaluation is already cheap and event counts are small; remaining gaps are projected
  column decode and downstream grouping/aggregation.
- No-filter group/aggregation queries are outside the conjunct-window mechanism. Apparent wins and
  losses in 3-iteration all-query sweeps should be treated as noise unless reproduced with focused
  profiles.

### Adaptive Direction

1. Do not use one fixed global conjunct window. The tested per-query oracle beats original V2 by
   `4.7%`, but the best fixed all-query setting only wins `2.2%`, and large fixed windows regress
   the suite.
2. Make `ConjunctPlan` choose a window dynamically:
   - start at natural chunks or a small floor such as `16k`;
   - measure events, selectivity, elapsed per input row, all-false rate, and output density;
   - increase toward `32k/64k` when predicates are expensive strings and per-batch overhead is
     dominating;
   - avoid growing when predicates are cheap numeric/date comparisons, when there is a single
     conjunct, or when downstream row-demand responsiveness matters.
3. Keep dynamic conjunct ordering and row-demand enabled at small windows. They are clearly helpful
   there. At larger windows they are closer to neutral, but disabling both together regresses.
4. Replace the naive mask-first fallback with a V2-native mask/demand resource:
   - the naive version re-enters value execution per mask chunk and is slower;
   - the desired shape is a single value stream whose source plans can observe mask-produced
     row-demand before polling expensive segments;
   - this should preserve overlap and avoid repeated plan execution.
5. Add per-query/runtime counters to drive adaptation instead of guessing:
   - rows evaluated and output by each conjunct;
   - elapsed and allocations per conjunct event;
   - downstream projected rows skipped by row-demand;
   - all-false window count;
   - active worker occupancy or at least outstanding producer queue depth.

## Engine Snapshot Import

Copied the local `vortex-engine` working tree into
`vortex-layout/prototypes/vortex-engine-snapshot` so the pipeline/scheduler ideas can be mined
without depending on that external checkout staying around.

- Source branch: `ngates/pipeline-scheduler`.
- Source HEAD: `0a8f95e`.
- Source worktree status: dirty `Cargo.lock`.
- Snapshot index: `vortex-layout/prototypes/vortex-engine-snapshot/SNAPSHOT.md`.

Important pieces to read while iterating:

- `src/physical_plan/abi.rs`: poll-based `SourceNode` / `TransformNode` / `SinkNode` ABI and
  `Batch { array, span }`.
- `src/physical_plan/lowering.rs`: continuation-style lowering with `PipelineTail` and
  `PipelineBuilder`.
- `src/physical_plan/runtime.rs`: current pipeline driver. It still pulls one source batch, pushes
  it through transforms, drains transform output, and uses barriers between pipelines.
- `src/physical_plan/spawn.rs`: `SpawnRuntime`, `WorkHandle`, `Priority`, and `IoCost`. Priority
  and I/O cost exist but are only hints today.
- `src/operator/mod.rs`: older/larger operator model with `propagate_requirements`, `update`, and
  scheduler-selected `run(WorkKey)`, which is closer to the information-prioritized design.
- `src/domain/requirement.rs` and `src/resources/pruning.rs`: row-demand/side-information
  vocabulary to compare against V2 `RowDemand`.

Initial read: copying the physical-plan runtime alone is not enough. Its own comments explicitly
say there is no row demand or cancellation, and the driver still has a rigid source-to-sink pull
loop inside each pipeline. The better prototype path is to combine:

- physical-plan lowering and poll-based local operators;
- operator/resource requirement propagation;
- a scheduler that admits morsels and I/O by value-of-information, bounded memory, and ordered
  retirement;
- runtime-visible spans/domains so demand resources can be ordinal now and later grow into keyed or
  sorted domains.

## Single-Scheduler Lowering Prototype

Added a detached lowering surface on `LayoutPlan`:

- `LayoutPlan::lower_to_scheduler(row_range, &mut LayoutLoweringCtx)`.
- `lower_to_single_scheduler(plan, row_range)` helper for tests/local experiments.
- `LayoutLoweringCtx` records lowered nodes, records initial leaf work, owns one
  `PartitionScheduler`, and can `drive_to_completion()` by repeatedly selecting the highest
  priority scheduler event.

The default lowering assumes children share the parent's row coordinate space. Overrides cover the
first places where that assumption is wrong or incomplete:

- `ChunkedPlan`: maps parent ranges to child-local ranges while preserving root-global order ranges
  for scheduler priority.
- `MaskSlicePlan`: forwards absolute mask ranges while preserving the caller's global output range.
- `FilterPlan`: lowers both hidden children, mask before values.
- `FilteredFlatPlan`: lowers its mask child and also registers its own value leaf work.
- `DictDecodePlan`: lowers the hidden values plan plus the codes plan.
- `LetPlan`: lowers both source and body.

Current semantics are intentionally abstract: leaf work is a scheduler morsel, not a real Vortex
array execution yet. Bool leaves become information producers; other leaves become value producers.
This lets the scheduler prototype exercise value-of-information priority without integrating with
DataFusion or replacing `execute -> Stream`.

### Runnable Scheduler Bridge

Added the first end-to-end scheduler-backed stream path behind
`VORTEX_V2_SCHEDULER_EXECUTE=1`.

- `PipelineId` is now an opaque scheduler-local `usize` index into `PartitionScheduler` pipeline
  state.
- `LayoutLoweringCtx::close_pipeline_with_source(...)` and
  `close_pipeline_with_segment_source(...)` allocate pipeline state and return the opaque id.
- `SchedulerTask` remains an enum: `Work`, `Segment`, `Control`.
- Segment tasks carry `pipeline: PipelineId`, so completion can be routed into the owning pipeline
  state.
- `ScanPlan::execute` can now spawn a scheduler driver task through the partition's
  `VortexSession` runtime handle, and return an `ArrayStreamAdapter` that pops arrays from a
  bounded sink queue.
- The first runnable pipeline source is a compatibility `ExecutePlan` source held in pipeline
  state, not in the task. It delegates to the existing `LayoutPlan::execute` and forwards chunks to
  the sink queue. This validates the driver/sink shape before replacing every node with native
  pipeline operators.

Passing checks:

- `cargo test -p vortex-layout --lib v2::dataflow`
- `VORTEX_V2_SCHEDULER_EXECUTE=1 cargo test -p vortex-layout --lib \
  v2::scan::tests::diff_v1_v2_filtered_chunked_struct_single_conjunct -- --exact`
- `VORTEX_V2_SCHEDULER_EXECUTE=1 cargo test -p vortex-layout --lib \
  v2::scan::tests::diff_v1_v2_filtered_chunked_struct_two_conjuncts -- --exact`
- `cargo test -p vortex-layout --lib \
  v2::scan::tests::scheduler_execute_filtered_chunked_struct_single_conjunct -- --exact`

Next implementation gap: when a real `Segment` task completes, store the returned segment bytes in
the owning pipeline's local state and enqueue the follow-up pipeline `Work` that decodes/applies the
source and pushes arrays to the sink. At that point `FlatPlan` can stop relying on the
compatibility `ExecutePlan` source.
