# Self-Paced Execution Implementation Plan

This document is a proposed implementation and rollout plan for
[self-paced plan execution](self-paced.md). It is intentionally more detailed than a migration
outline so that API boundaries, sequencing, and stop conditions can be reviewed before production
code depends on them.

No phase assumes that the old executor is removed. The first usable path is an adapter that gathers
self-paced prefixes into the exact ArrayRef result expected by the current PlanVTable::execute
contract. The new root scan path is enabled only after semantic parity, resource bounds, and
performance are measured.

## Objective

Implement this end-to-end flow:

~~~text
optimized PlanRef
  -> open scan
     -> domains and edge maps
     -> stable ReadCatalog spine in ScanState
  -> prepare fixed morsel
     -> catalog view
     -> mutable ExecGraph
  -> refine and seal DemandLedger windows
  -> run-to-quiescence drive
  -> scheduler-owned I/O and CPU tickets
  -> child-sized prefix batches
  -> parent alignment by capping
  -> root rebatching
  -> ordered or unordered ArrayStream
~~~

The implementation is complete only when:

- all plan operators used by supported scans have an execution-node implementation;
- every coordinate translation is a declared DomainMap rather than per-operator arithmetic;
- exact outputs and observable errors match the compatibility oracle;
- projection planning may use immutable open demand for candidate I/O, while projection CPU on open
  demand requires an explicit speculation-safety classification;
- read and task registration is idempotent under duplicate wakes;
- compressed, decoded, task, and output memory are bounded, and no credit class can deadlock;
- multiple fields can expose independent work in one drive;
- batch count for a wide struct is driven by boundaries, not by field count;
- fixed morsels provide outer concurrency while internal prefixes remain variable;
- morsel boundaries are derived generically rather than from a central operator switch;
- root batching, ordering, limit, cancellation, and error behavior are defined;
- benchmark evidence supports enabling the path by default; and
- the old exact recursive path remains available until the rollback window closes.

## Current implementation boundary

The present source already provides a clean starting point:

- vortex-layout/src/plan/vtable.rs defines exact PlanVTable::execute over one row range and
  MaskFuture.
- vortex-layout/src/plan/plans contains generic SegmentScan, Concat, Pack, Eval, Take, ListPack,
  Zoned, and row-index operators.
- vortex-layout/src/plan/execution.rs contains the current segment source and session execution
  context.
- vortex-scan-v2/src/splits.rs discovers plan boundaries and subdivides large spans around a
  100,000-row ideal.
- vortex-scan-v2/src/tasks.rs coordinates pruning, filtering, early projection-read registration,
  and one exact projection result per split.
- vortex-scan-v2/src/filter.rs implements parallel or adaptively ordered filter evaluation.

The new executor should initially be parallel to these APIs. This keeps the existing path as an
oracle and avoids forcing scheduler experiments into the public PlanVTable contract prematurely.

## Proposed source ownership

Names are provisional, but dependency direction should be preserved:

| Area | Proposed home | Reason |
| --- | --- | --- |
| ExecOp, DriveResult, BatchRequest, ExecBatch, tickets, and BatchCursor | vortex-layout/src/plan/exec | Physical operator implementations already live in vortex-layout, and scan-v2 depends on it |
| DomainId, DomainMap, and ScanState | vortex-layout/src/plan/exec | Every edge map is a property of the plan, and five operators already store its inputs |
| ReadCatalog facts and plan preparation hooks | vortex-layout/src/plan/exec | Plans know segment identities, row domains, and child mappings |
| Segment, Concat, Pack, Eval, Take, ListPack, Zoned, and row-index executors | Beside their plan implementations or under plan/exec/operators | Keeps immutable plan data and corresponding open logic reviewable together |
| DemandLedger and block summaries | vortex-scan-v2/src/self_paced | Filter order and final projection demand are root scan policy |
| Morsel driver, concrete read/CPU scheduler, credits, and wake queue | vortex-scan-v2/src/self_paced | These coordinate several plan roots and multiple morsels |
| Root rebatching, ordering, limit, and compatibility selection | vortex-scan-v2/src/self_paced | These are stream-level rather than physical-operator concerns |

Do not introduce a new crate initially. A crate boundary would stabilize APIs before the contracts
have survived a vertical slice. If another consumer later needs the scheduler, move proven
interfaces after dependency and profiling evidence exists.

## Compatibility strategy

Development proceeds through three increasingly broad entry points:

~~~text
1. deterministic simulator
   fake plan nodes + fake tickets

2. exact compatibility adapter
   current row range + resolved mask
     -> self-paced prefixes
     -> gather
     -> one ArrayRef

3. self-paced scan root
   fixed morsels + DemandLedger + scheduler
     -> prefix stream
     -> RebatchExec
     -> ArrayStream
~~~

The compatibility adapter is important even though it hides streaming benefits. It proves operator
semantics and parent alignment without simultaneously changing filters, stream ordering, and
scheduler behavior.

Use a private execution-mode switch during development:

~~~text
ExactRecursive
SelfPacedExactAdapter
SelfPacedRoot
~~~

Unsupported operators fall back at the whole-morsel boundary. Do not mix old and new mutable
execution recursively unless the adapter has an explicit ownership and cardinality contract.

## Phase dependency

~~~text
Phase 0: baseline, invariants, widening question, RebatchExec
    |
Phase 1: domains, edge maps, and pure execution primitives
    |\
    | +--> Phase 2: DemandLedger and summaries
    |
    +----> Phase 3: per-scan ReadCatalog and mock scheduler
                 |
Phase 4: minimal ticket driver and resource credits
                 |
Phase 5: Segment + Concat + Pack vertical slice
                 |
Phase 6: unfiltered self-paced morsel root
                 |
Phase 7: pruning, filters, and sealed projection
                 |
Phase 8: CPU concurrency and wavefront backpressure
                 |
Phase 9: gated and coordinate-changing operators
                 |
Phase 10: root rebatching, ordering, limit, and cancellation
                 |
Phase 11: performance qualification and rollout
~~~

Phases 2 and 3 can be developed independently after the core identifiers and invariants in Phase 1
are stable. The production vertical slice should not begin until both have deterministic tests.

## Phase 0: Establish the baseline

### Rationale

V1 and current plan v2 contain behavior that is easy to lose while changing control flow:
all-false masks still cover dense rows, filtering controls fallible projection, nested coordinate
domains differ, and ordered streams constrain errors and limits. A baseline makes those semantics
an explicit oracle rather than an assumption.

### Work

1. Inventory every optimized plan operator and the scan features that construct it.
2. Identify representative existing tests for flat, chunked, struct, dictionary, list, zoned,
   row-index, pruning, filters, selections, limits, and ordering.
3. Add a differential harness interface that can execute the same prepared scan with a selected
   execution mode.
4. Record current output batches, compact row counts, errors, segment requests, and ordering for
   the representative corpus.
5. Add baseline metrics needed for later comparisons:

   - time to first batch and total time;
   - logical and physical read counts and bytes;
   - duplicate segment requests;
   - peak retained compressed and decoded bytes;
   - output batch count and size distribution; and
   - filter input and output cardinalities.

6. Decide which current behavior is contractual and which is merely an implementation artifact.
   In particular, record fallible expression and ordered-error behavior.
7. Answer the widening question: is there any supported or planned API through which demand can
   widen after a scan opens? Selection is fixed at construction, pruning and predicates intersect,
   and the only dynamic predicate is applied as file pruning before the scan opens. If nothing
   widens, DemandEpoch is deleted from the design in Phase 2 and replaced by one debug assertion.
8. Build RebatchExec against the current executor. It depends on none of the new machinery, already
   decouples the public batch size from the 100,000-row split unit, and gives the batch-size
   distribution metric a stable reference point before anything else changes.

Record read overlap between filter and projection explicitly in the baseline. Plan v2 gets it from
constructing projection futures before the filter mask resolves, and it is the property most easily
lost without anyone noticing.

### Validation

- The same input can run under two execution modes and compare arrays and errors.
- The corpus exercises every operator that must be supported before default rollout.
- Baseline metrics are obtainable without enabling the new executor.
- RebatchExec preserves output, ordering, limits, and errors on the current path.

### Exit criterion

A checked-in compatibility matrix names the oracle and expected semantics for every supported
feature. The widening question has an answer. Performance acceptance thresholds are recorded before
new-path measurements are viewed.

## Phase 1: Implement domains and pure execution primitives

### Rationale

Prefix coverage, compact-mask slicing, ticket idempotence, and multi-child fairness are easier to
prove without real I/O, layouts, or async runtime behavior. These primitives are the highest fan-out
API in the design, so mistakes should be found before operator ports begin.

Domains belong here rather than with the operators that need them. DomainMap is a refactor of state
five operators already hold — `ConcatData::row_offsets`, `RowIdxData::row_offset`, Zoned's
`zone_len`, ListPack's offsets arithmetic, and Take's codes/values split — so it can be written and
tested against `collect_plan_splits` before any execution node exists. Deferring it to Phase 9
means retrofitting a domain parameter through eight phases of row-space assumptions.

### Work

1. Add crate-private types for:

   - MorselRange and, pending Phase 0's answer, DemandEpoch;
   - DomainId and DomainMap, with `map_range`, `map_demand`, `unmap_frontier`,
     `prefix_preserving`, and `is_static`;
   - SealedDemand with its domain, mask offset, and `derive`;
   - BatchRequest with `max_rows`;
   - ExecBatch and BatchCursor;
   - ExecOp and DriveResult, with Yield carrying progress evidence;
   - ReadUseId, ReadTicket, CpuTaskKey, CpuTicket, and CreditTicket;
   - WaitSet; and
   - a bounded DriveBudget.

2. Implement dense-prefix validation and compact-array split calculations using demand rank.
3. Implement a deterministic DriveContext with fake ticket tables.
4. Implement a driver that:

   - calls one node with serialized mutable ownership;
   - loops through cheap transitions;
   - stops at Batch, Blocked, Done, or Yield;
   - validates that Blocked has a viable wait condition and Yield made progress; and
   - tolerates duplicate and reordered wake-ups.

5. Implement mock leaf, Concat-like, and Pack-like nodes whose natural boundaries are configurable.
   At least one mock must sit behind a non-Identity map so the simulator cannot bake in row-space
   assumptions that Phase 9 then has to unpick.
6. Implement parent capping: round one goes wide, later rounds cap at the agreed length.
7. Add debug-only frontier, derived-demand completeness, and credit ownership assertions.

### Validation

- Property tests generate mismatched child boundaries and prove gap-free, overlap-free output.
- Every possible dense split position of a sparse mask produces correct compact slices.
- An all-false prefix advances dense progress with zero values.
- A mock Pack registers every missing child's work before blocking.
- A K-child Pack whose children share a boundary emits one batch per boundary, not K.
- A child that can serve past the cap returns exactly the cap and retains its own surplus.
- Duplicate read or CPU registration returns the same ticket.
- Random completion order produces the same batches and final result.
- A perpetually ready node yields after its transition budget, and two Yields with no frontier or
  ticket change trip an assertion.
- Derivation across each map is complete; a Coarsen derivation used for fallible work is rejected.
- `unmap_frontier` round-trips against `map_range` for every prefix-preserving map.

### Exit criterion

The simulator cannot produce a batch that violates the prefix, mask, cardinality, monotonic
frontier, or derivation invariants; parent capping keeps batch count independent of child count for
aligned children; and no test requires an event inbox for correctness.

## Phase 2: Implement DemandLedger and summaries

### Rationale

Mask ownership determines error semantics and drive frequency. It must be settled before projection
execution is connected to live filters. The scheduler summary is built alongside the ledger so the
optimization cannot become a second source of truth.

### Work

1. Divide each morsel into configurable demand blocks, initially 1,024 rows.
2. Store the exact candidate mask, remaining predicate set, revision, and Open or Sealed state per
   block.
3. Implement monotone exact intersections and independent block sealing.
4. Track the contiguous sealed frontier from the projection commit point.
5. Restrict SealedDemand construction to the ledger, and derivation to the operator owning the
   edge's DomainMap.
6. If Phase 0 found a widening case, implement a new-epoch operation for it. If it did not, delete
   DemandEpoch and keep one debug assertion that intersections never widen.
7. Maintain two authoritative facts, one derived cache, and one generation:

   - exact candidate upper counts per block;
   - block state;
   - a maybe-nonempty bit set rebuilt from the counts, kept only so the scheduler can scan many
     blocks in one bitwise pass, never written independently; and
   - one monotone summary generation.

   Do not add a separate tri-state summary or sealed-nonempty set: both are derivable, and the
   rationale for this phase is that the optimization must not become a second source of truth.
   Estimated remaining counts are scheduling-only and may be omitted from the first implementation;
   `FilterExpr::report_selectivity` records one rate per conjunct globally, so applying it
   uniformly carries no per-block information.

8. Expose changes as coarse notifications:

   - SealedFrontierAdvanced;
   - CoverageEliminated;
   - EpochReplaced; and
   - SummaryGenerationChanged.

9. Benchmark exact intersection and population count against block-summary maintenance for
   100,000-row masks at several densities.

### Validation

- Predicate results complete in every order and yield the same sealed mask.
- Projection cannot obtain SealedDemand for an open block.
- A sealed block rejects further intersection.
- Widening within an epoch is rejected; a new epoch invalidates uncommitted capabilities.
- maybe-nonempty false is always a safe elimination proof.
- Expected counts never participate in a correctness branch.
- Count-only intersection bounds contain the exact intersection for randomized masks.
- Summary generation changes are coalesced across a configurable update interval.

### Decision gate

Confirm or revise the 1,024-row block default using measured mask overhead, time-to-first-sealed
prefix, and read-coverage precision. The exact bit mask remains mandatory regardless of block size.

### Exit criterion

Projection wake decisions can be derived from the sealed frontier, and read scoring can use summary
generations without visiting projection nodes.

## Phase 3: Implement plan preparation and a per-scan ReadCatalog

### Rationale

Repeatedly asking every execution node to “offer” future reads would turn drive into a mask-update
polling loop. A stable catalog makes all statically visible I/O schedulable once and provides one
deduplication identity for speculative and required use.

The catalog spine belongs to the scan, not the morsel. Segment identity and row coverage are
morsel-independent facts, and only necessity and lifecycle vary per morsel. Phase 1's DomainMap
also supplies coverage mapping, so the catalog does not need its own notion of when a nested or
lookup operator must fall back to conservative coverage.

### Work

1. Add ReadCatalogBuilder and immutable catalog entries with:

   - stable logical use and physical read keys;
   - owning plan or execution node;
   - estimated bytes;
   - the entry's own DomainId and coverage within it;
   - scan phase;
   - cancellation group;
   - optional dependency gate; and
   - initial necessity.

2. Represent necessity and data lifecycle as independent state axes.
3. Build the catalog spine **once per scan**, in ScanState, with a cheap per-morsel view. Segment
   identity and row coverage are morsel-independent facts; only necessity and lifecycle are
   per-morsel. Rebuilding per morsel costs `columns × segments-per-morsel` entries every morsel.
4. Add preparation support for the initial row-equivalent operators:

   - SegmentScan describes its segment;
   - Concat maps morsel ranges through its Shift maps;
   - Pack visits every projected field and validity child;
   - Eval delegates physical reads to its child; and
   - shared physical keys across filter and projection uses are retained once.

5. Add GateId and one-shot gate expansion for the non-static maps, initially exercised by fakes.
6. Implement coverage-to-demand-block mapping as DomainMap composition from the owning node up to
   the ledger domain: all-static and prefix-preserving gives exact coverage, a Coarsen on the path
   gives coarsened coverage, and a gated map gives group coverage until the gate expands. This
   replaces per-operator judgement about when to fall back to "a conservative group".
7. Implement generation-stamped lazy read scoring.
7. Prototype preparation as a crate-private hook. Do not finalize a new public PlanVTable method
   until the vertical slice shows whether describe and open should share one traversal.

### Validation

- Preparing the same plan twice produces stable keys.
- A projection and predicate using one segment produce one physical request with two logical uses.
- Flat, Concat, and Pack preparation covers every segment intersecting the morsel and no segment
  wholly outside it.
- A morsel view over the shared spine allocates no per-morsel entries.
- Catalog entry count for a wide struct scales with the scan, not with morsel count.
- A required promotion preserves the original physical key.
- A gate expands once even under duplicate wakes.
- An entry covering only zero-count blocks becomes Eliminated.
- Coverage composed through Identity, Shift, and Fence edges is exact; through a gated edge it is a
  group until expansion.
- Lazy rescoring observes the newest generation before admission without eagerly visiting every
  catalog entry on a mask update.

### Decision gate

Choose the long-term plan hook:

- defaulted methods on PlanVTable;
- a companion internal execution vtable; or
- one combined prepare-and-open hook that still exposes separate semantic products.

Prefer the smallest public API. Reject an approach that requires runtime downcast chains in the
steady-state executor.

### Exit criterion

The complete statically visible read set for a row-equivalent plan can be prepared once per scan,
shared uses deduplicate by key, coverage is computed by map composition rather than per-operator
judgement, and open-mask changes require no plan-tree traversal.

## Phase 4: Build the minimal ticket scheduler

### Rationale

Operator state machines need durable, idempotent work handles before real decode logic is added.
Starting with a deterministic scheduler separates ticket semantics and credits from thread-pool
tuning.

### Work

1. Implement a read store keyed by ReadKey and wrap the current SegmentSource request future.
2. Implement candidate admission and required promotion under:

   - a global compressed-byte budget;
   - a per-morsel compressed-byte budget; and
   - a reserved progress allowance for blocking reads.

3. Implement CPU tickets with an initially deterministic or inline executor.
4. Implement separate decoded, task-result, retained, and output credits even if the first CPU
   backend runs inline. Reserve each class per morsel at admission, and never deny the oldest
   in-flight morsel: a reserve for blocking *reads* does not prevent hold-and-wait on *decoded*
   credit, where several morsels each retain partial results and none can advance.
5. Implement a runnable-morsel queue and WaitSet subscriptions.
6. Treat completions as wake hints; poll durable ticket state after every wake.
7. Implement cancellation, result release, and oversized-unit credit.
8. Add trace and metric points for every lifecycle transition.

### Validation

- Candidate-to-required promotion never duplicates the physical request.
- Required work can make progress when speculative credit is exhausted.
- With every credit class saturated by retained decoded state across several morsels, the oldest
  morsel still advances to completion and releases.
- Cancellation releases queued credit and eventually releases completed buffers.
- Duplicate completion wakes do not repeat state transitions.
- A result that completes before WaitSet registration is still observed.
- One oversized indivisible unit can run without allowing multiple oversized units to exceed the
  isolation rule.

### Exit criterion

Fake nodes can read, compute, block, wake, yield, cancel, and release resources using only tickets
and durable state.

## Phase 5: Deliver a row-equivalent vertical slice

### Rationale

SegmentScan, Concat, and Pack exercise physical reads, sequential row routing, sibling concurrency,
prefix alignment, and retained tails without introducing lookup or nested coordinate domains. They
are the smallest slice that tests the central architectural claim.

### Work

1. Implement SegmentScanExec:

   - use the prepared segment read;
   - preserve current whole-segment decode initially;
   - submit decode through a CPU ticket;
   - slice decoded output into sealed prefixes; and
   - release segment and decoded state behind the frontier.

2. Implement ConcatExec with child-local translation through its Shift maps and one BatchCursor.
3. Implement PackExec with one cursor per field and validity child, propagating demand across
   Identity maps.
4. Track committed, ready, and CPU-scheduled frontiers plus retained bytes per Pack child.
5. Drive every missing Pack child before returning Blocked.
5a. Implement capping: round one goes wide to every child and sets the agreed length, later rounds
   cap all children at it. A child that decoded past the cap keeps the surplus in node-local state
   charged to its own decoded credit; parent-owned retention exists only where a child cannot
   re-slice its own output.
6. Implement EvalExec for the projection operations needed by the initial test corpus, while
   requiring sealed demand for fallible expressions.
7. Implement the exact compatibility adapter:

   - prepare the morsel and catalog before resolving MaskFuture when safe;
   - await the exact mask and wrap it as one sealed demand region;
   - gather every returned prefix; and
   - produce the exact ArrayRef cardinality expected by PlanVTable::execute.

8. Add a private execution-mode selection for supported plan trees.

### Validation

- Differential tests compare exact arrays and errors with current plan v2.
- Struct fields with every pair of adversarial chunk boundaries align correctly.
- The 64,000-row cheap field and 8,000-row wide field scenario runs both first reads and decodes
  independently, emits 8,000 rows, and charges the cheap field's decoded surplus to that field.
- Subsequent rounds cap both fields at the agreed length; the cheap field serves them from its own
  decoded segment and Pack retains nothing.
- A K-field struct whose fields share a boundary emits one batch per boundary, not K.
- A leading child stops materializing when its retained-byte credit is full.
- Sparse and all-false masks preserve dense progress.
- Segment requests are stable and deduplicated across shared uses.
- Unsupported plans select the old whole-morsel path before execution begins.

### Decision gate

Review the real API after three operators use it:

- Is Box<dyn ExecOp> sufficient?
- Does plan preparation need a separate traversal?
- Is DriveResult expressive enough without an event payload?
- Does capping hold up, or do real encodings overshoot often enough to need parent retention?
- Are retained-byte ownership and release unambiguous?
- Does the exact adapter expose any semantic mismatch?

Do not move to an arena or publish the API unless profiling or external use justifies it.

### Exit criterion

A flat, chunked, or struct plan can execute entirely through the new graph and exact adapter with
semantic parity and bounded retained tails.

## Phase 6: Add the unfiltered self-paced morsel root

### Rationale

The next step should expose natural prefix streaming without adding live predicate refinement.
For an unfiltered scan, the initial selection is immediately sealed, which isolates morsel driving,
read-ahead, and output pacing.

### Work

1. Add MorselExec in scan-v2 with:

   - fixed row range;
   - projection commit frontier;
   - read, materialize, and emit horizons;
   - root ExecOp;
   - demand ledger with initially sealed selection; and
   - cancellation and output credits.

2. Replace `collect_plan_splits` with generic boundary derivation: walk edges whose DomainMap is
   static and prefix-preserving, translating boundaries through the map, and stop at gated maps.
   That switch already computes this by hand — its `child.row_count() == plan.row_count()` test is
   an Identity check, its `row_offset + chunk_offset` is a Shift, and taking only Take's codes child
   is skipping a GatherGated edge. Land it in one change that proves the derived boundaries match
   today's, and keep the old function until they do.
3. Take a morsel view over the scan catalog and allow candidate reads to run ahead under
   compressed credits.
4. Drive projection on sealed demand, yielding prefix batches as downstream capacity permits.
5. Initially gather or expose an internal test stream without changing public rebatching.
6. Confirm that no projection drive is needed merely to keep static read-ahead active.

### Validation

- Derived morsel boundaries match `collect_plan_splits` on the whole Phase 0 corpus, including
  zoned, dictionary, and list plans.
- One 100,000-row morsel can emit several child-sized prefixes.
- Candidate reads beyond the current 8,000-row output prefix can be in flight while decoded lead
  remains bounded.
- Drive occurs only for initial sealed demand, waited ticket completion, capacity, cancellation,
  or Yield.
- A parked morsel consumes no worker thread.
- Multiple morsels can make independent progress.

### Exit criterion

Natural internal batching and whole-morsel read discovery work end to end for unfiltered scans
without increasing drive frequency with read-ahead distance, and morsel boundaries no longer depend
on a central operator switch.

## Phase 7: Integrate pruning, filters, and sealed projection

### Rationale

This is the semantic center of the design. It replaces the current MaskFuture coupling while
preserving early projection I/O. Read scheduling may speculate from conservative demand, but
projection computation must observe an immutable final mask for each emitted window.

### Work

1. Initialize DemandLedger from Selection for each morsel.
2. Translate pruning and evidence results into monotone block intersections.
3. Run parallel or adaptive predicates over immutable stage masks.
4. Track remaining predicates per block and seal blocks independently.
5. Wake exact projection value execution only when the contiguous sealed frontier advances.
6. Keep projection planning and catalog read scheduling active for open blocks using immutable
   snapshots and conservative summaries.
7. Promote the exact reads required by each sealed projection prefix.
8. Preserve current selectivity feedback and make expected block counts scheduling-only.
9. Classify computation:

   - safe metadata or evidence;
   - explicitly safe infallible speculation; and
   - demand-sensitive or fallible work requiring sealed demand.

10. Define a new epoch or restart behavior for any API that can widen selection.

### Validation

- Predicate completions in different orders produce identical final output.
- Many open-mask revisions cause zero exact projection value drives until a prefix seals; candidate
  read rescoring remains lazy.
- A projection read can start before sealing and is promoted without duplication later.
- A fallible projection is never evaluated on rows removed before sealing.
- All-false blocks advance the sealed frontier without projection values.
- A fully eliminated read coverage is cancelled or left only according to explicit scheduler
  policy.
- Adaptive filter ordering retains its reported selectivity behavior.
- Epoch replacement cannot reuse stale SealedDemand or commit from the previous epoch.

### Decision gate

Lock the public error and dynamic-selection semantics. If the current behavior is ambiguous,
resolve it with a dedicated semantic test and review rather than letting scheduler order define it.

### Exit criterion

Filtered scans overlap conservative projection I/O with predicate work while exact projection
computation is driven only by sealed demand.

## Phase 8: Add CPU concurrency and wavefront backpressure

### Rationale

The deterministic CPU backend proves state transitions but not intra-morsel concurrency. A real
backend is useful only after ticket ownership and mask semantics are stable; adding it earlier
would make races obscure contract bugs.

### Work

1. Submit expensive decode, expression, and array-construction work to the session runtime or a
   dedicated CPU pool.
2. Require tasks to own inputs and return owned results.
3. Add a measured task-cost threshold; keep cheap coordination inline.
4. Allow one drive to register CPU tasks and reads for different children.
5. Enforce separate CPU input, output, and decoded-tail credits.
6. Compute a row-equivalent parent's materialization wavefront from:

   - the minimum child ready frontier;
   - each child's scheduled frontier;
   - estimated output bytes per row; and
   - currently retained bytes.

7. Reserve progress capacity for the child blocking the parent frontier.
8. Tune the transition budget and runnable-morsel fairness.

### Validation

- Independent struct fields decode concurrently when credits allow.
- A wide lagging field receives progress credit before a cheap leading field extends its tail.
- Compressed reads may reach the morsel end while decoded memory stays near the materialize
  horizon.
- CPU task completion order does not alter output or error semantics.
- Cancellation cannot let an old task mutate or commit execution-node state.
- Metrics expose drive calls, transitions per call, blocked duration, task queueing, child
  frontiers, and retained bytes.

### Performance gate

Run the mismatched-struct, wide-struct, and flat-segment microbenchmarks. Confirm that task-launch
overhead does not dominate small batches and that decoded memory remains bounded by credits.
Adjust thresholds from evidence, not operator-specific guesses.

### Exit criterion

The execution graph obtains useful intra-morsel I/O and CPU concurrency without concurrent mutable
access to nodes or unbounded leading-child materialization.

## Phase 9: Port gated and coordinate-changing operators

### Rationale

These operators cross domains. With DomainMap in place from Phase 1, they split into two groups
rather than one hard class: ListPack, Zoned, and row-index are prefix-preserving and differ from the
core only in needing a gate or a coarsening, while Take's values child is the single GatherGated
edge in the system and is the only operator that genuinely breaks prefix composition. Do them in
that order — the easy group first validates the gate machinery before the sub-root model is added.

### Work

1. Implement ListPackExec, which is prefix-preserving throughout:

   - derive offsets demand as `d | (d << 1)` over the Fence map, rather than requesting all offsets;
   - expand the element gate from decoded offsets, resolving the MonotoneGated map;
   - implement `unmap_frontier` as a search for the largest `k` with `offsets[k] <= element_end`;
   - buffer element-domain prefixes; and
   - enforce indivisible list output and oversized permits.

2. Implement ZonedExec evidence and data coordination through DemandLedger, with zone statistics
   behind a Coarsen map. Assert that Coarsen-derived demand never drives fallible work.
3. Implement row-index and row-index-partition execution as Shift composition to the file domain.
4. Implement TakeExec, the one non-prefix-preserving case:

   - drive codes in the outer domain across an Identity map;
   - expand the value gate from decoded codes;
   - drive the values child as a **sub-root** with its own cursor and its own sealed demand over the
     value domain, rather than inside Take's prefix cursor;
   - default to sparse per-prefix gather, falling back to full materialization below a byte
     threshold; and
   - cache value results in ScanState under explicit credits, which is what makes the incremental
     form work without any widening machinery.

5. Complete Eval variants and any optimizer-produced operator combinations.
6. Define gate cancellation and cache reuse across repeated references.

### Validation

- A sparse outer filter reads materially fewer offsets than a whole-range request.
- List prefixes compose end to end: an outer prefix yields an element prefix, and `unmap_frontier`
  agrees with the emitted outer rows.
- Dictionary domains larger and smaller than the morsel match the oracle.
- Repeated codes across successive outer prefixes hit the ScanState value cache rather than
  re-reading, and no epoch is created.
- The values sub-root can itself be a Concat and still make prefix progress.
- Empty, null, and oversized lists crossing element batches match current semantics.
- Offset and code completion expands each gate exactly once.
- Zoned evidence eliminates all, none, and partial demand blocks correctly, and a Coarsen-derived
  demand used for fallible work is rejected.
- Row indices remain absolute through sparse masks, prefix slicing, and rebatching.
- Differential coverage includes every optimized plan shape inventoried in Phase 0.

### Decision gate

Decide whether GatherGated is common enough to justify a specialized catalog coverage type or
scheduler queue. Avoid generalizing the prefix-preserving fast path before workload evidence.

### Exit criterion

Every supported plan-v2 operator can run in the new graph, every domain translation is a declared
DomainMap rather than inferred from ArrayRef length, and the one non-prefix-preserving edge is
isolated to Take's values sub-root.

## Phase 10: Complete root stream semantics

### Rationale

Internal prefix correctness does not automatically provide a stable public stream. Rebatching,
ordered merging, limits, cancellation, and errors span morsels and therefore belong at the root.

### Work

1. Implement RebatchExec to concatenate small prefixes and slice large prefixes toward the
   consumer target.
2. Merge morsels in row order for ordered scans and completion order for explicitly unordered
   scans.
3. Apply limits after final filter cardinality is known:

   - stop creating later demand;
   - trim the final sealed prefix exactly;
   - cancel later morsels and speculative reads; and
   - release buffered tails.

4. Define first-error and ordered-error behavior to match the Phase 0 contract.
5. Propagate cancellation to read uses, CPU tasks, gates, and output buffers.
6. Add backpressure from ArrayStream to morsel output credits.
7. Ensure a zero-value dense prefix does not create a spurious empty consumer batch.
8. Preserve mapper and schema behavior from the current TaskContext path.

### Validation

- Consumer batches meet the target except at natural flush boundaries.
- Ordered and unordered scans match their documented row and error behavior.
- Limits cutting through sparse masks and internal prefixes return exactly the requested count.
- Cancellation at every read, CPU, retained-tail, and rebatch state releases resources.
- Empty selected output produces no public batch but still completes all dense frontiers.
- Multiple morsels cannot exceed global output or decoded-memory credits.

### Exit criterion

SelfPacedRoot is a complete alternative scan execution mode with no dependency on exact
whole-morsel result gathering.

## Phase 11: Qualify, roll out, and retire

### Rationale

A scheduler can improve overlap while regressing small scans through coordination overhead, or hide
memory growth behind throughput. Rollout requires both semantic and resource evidence across local
and remote storage.

### Benchmark matrix

Include:

- flat arrays with one segment per morsel and several segments per morsel;
- chunked arrays with aligned and adversarial boundaries;
- the 64,000-row cheap field versus 8,000-row wide field struct;
- narrow and very wide structs;
- dense, sparse, and all-false selections;
- no filter, high-selectivity filter, and low-selectivity filter;
- parallel and adaptive conjuncts;
- dictionary and list gates;
- zoned pruning that eliminates all, some, or no reads;
- local memory, NVMe, and object-store-style latency where available;
- ordered and unordered multi-morsel scans; and
- small scans where scheduler overhead is most visible.

Measure:

| Category | Metrics |
| --- | --- |
| Latency | Time to first prefix, first public batch, and completion |
| I/O | Candidate, promoted, eliminated, cancelled, duplicate, and physical reads; bytes read and wasted |
| CPU | Task count, launch overhead, queue delay, occupancy, and decode/eval time |
| Memory | Peak compressed, decoded, task-result, retained-tail, and output bytes |
| Coordination | Drive calls, transitions per drive, Yield count, wakes, and no-progress wakes |
| Demand | Revisions, blocks sealed, time to sealed frontier, and scheduler rescoring |
| Batching | Internal prefix and public batch size distributions |

### Rollout steps

1. Run differential tests in CI with the new mode non-default.
2. Add opt-in benchmarks and tracing for the new path.
3. Enable the exact adapter for supported row-equivalent plans in development builds.
4. Enable SelfPacedRoot behind an explicit option for the full supported plan set.
5. Run shadow or A/B comparisons where the environment supports them.
6. Make the new root default only after agreed semantic, memory, and performance gates pass.
7. Retain ExactRecursive as a rollback mode for at least one release or an agreed stabilization
   interval.
8. Remove the old path only after:

   - no required operator falls back;
   - differential testing is clean;
   - production metrics show bounded memory;
   - small-scan overhead is acceptable;
   - object-store read amplification is acceptable; and
   - maintainers approve the public API and deletion.

### Exit criterion

The self-paced root is the default, the rollback interval has completed without unresolved parity
or resource regressions, and obsolete exact-recursive code can be removed in a separate reviewable
change.

## Suggested pull-request sequence

Keep changes small enough that each review proves one claim:

1. Baseline differential harness and execution-mode plumbing.
2. RebatchExec against the current executor.
3. DomainId, DomainMap, and the edge declarations for every existing operator.
4. Pure prefix, cursor, drive, ticket, capping, and simulator types.
5. DemandLedger, block summaries, and mask microbenchmarks.
6. ReadCatalog spine in ScanState, stable keys, coverage by map composition, and fake gates.
7. Minimal read store, per-morsel credit reservation, WaitSet wake queue, and scheduler tests.
8. SegmentScanExec plus exact adapter.
9. ConcatExec and PackExec with mismatched-boundary and capping tests.
10. Generic morsel-boundary derivation replacing `collect_plan_splits`.
11. Unfiltered MorselExec and whole-morsel read-ahead.
12. Pruning, adaptive filters, and sealed projection.
13. Real CPU tasks, wavefront credits, and struct concurrency benchmarks.
14. ListPack, Zoned, row-index, and Eval completion.
15. Take with a values sub-root and a bounded ScanState value cache.
16. Ordering, limits, cancellation, and full stream integration.
17. Qualification, default switch, and later old-path removal.

Item 3 is separable and worth landing early even if nothing consumes it yet: it is a refactor of
state five operators already hold, it is testable against `collect_plan_splits`, and every later
item depends on its shape.

Split a numbered item further when its test surface becomes difficult to review. Do not combine old
path removal with the default switch.

## Cross-phase test matrix

Every phase should preserve these invariants as soon as the relevant feature exists:

| Area | Required cases |
| --- | --- |
| Prefixes | Natural boundary before, at, and after target; cap honoured; minimum prefix; all-false dense progress |
| Masks | Dense, sparse, empty, block edges, out-of-order predicate completion, epoch replacement if reachable |
| Pack | Two and many children, mismatched chunks, shared boundaries emitting one batch, nullable validity, retained-byte pressure |
| Reads | Candidate admission, promotion, elimination, sharing, rejection, retry, cancellation |
| CPU | Inline and scheduled paths, completion reordering, task error, task cancellation |
| Domains | Every DomainMap variant: derivation completeness, minimality where required, `unmap_frontier` round-trip, gate expansion, Take sub-root |
| Stream | Rebatch slices, ordering, limits, empty result, mapper error, consumer backpressure |
| Resources | Every credit class, per-morsel reservation, oldest-morsel progress, oversized unit, release after error |
| Liveness | No Blocked without a live condition, no Yield without progress, bounded drives per committed row |

Use V1 where it is the behavior oracle and current plan v2 where it already defines the intended
generic-plan behavior. A test that passes through both paths without exercising a different
boundary is not sufficient evidence for the new contract.

## Risk register

| Risk | Consequence | Mitigation and proving phase |
| --- | --- | --- |
| Fallible or exact projection value work uses open demand | New observable errors or wasted fallible work | Restricted SealedDemand construction and Phase 2/7 tests; open snapshots authorize only candidate I/O and explicitly safe speculation |
| Drive becomes an event-processing loop | Poll storms and order-dependent bugs | Durable tickets, run-to-quiescence simulator in Phase 1 |
| Catalog updates cost more than exact mask work | Filter-heavy regression | Lazy generation scoring and Phase 2/3 microbenchmarks |
| Leading struct fields decode too far ahead | Unbounded retained state | Capping in Phase 5, byte-based wavefront credits in Phase 5/8 |
| Ragged child boundaries fragment wide structs | Batch count scales with field count | Capping and the shared-boundary test in Phase 1/5 |
| Speculation starves blocking work | Morsel deadlock or high latency | Required promotion and progress reserve in Phase 4 |
| Retained decoded state deadlocks across morsels | Global stall that per-morsel Blocked checks cannot see | Per-morsel reservation and never-deny-oldest in Phase 4 |
| Static catalog is too large | Preparation latency and memory growth | Per-scan spine with morsel views, plus an entry-count exit criterion, in Phase 3 |
| Row-space assumptions harden before Phase 9 | Domain parameters retrofitted through eight phases of API | DomainMap in Phase 1 and a non-Identity mock in the simulator |
| Epoch machinery built for an unreachable case | Cost and complexity with no consumer | Phase 0 widening question gates Phase 2 item 6 |
| Dynamic gates hide useful I/O | Low concurrency for nested or lookup layouts | Explicit one-shot gate expansion in Phase 3/9 |
| Whole-segment decode defeats small prefixes | High decoded memory and latency | Preserve semantics first, then add page granularity as separate work |
| Public plan API stabilizes too early | Long-term compatibility burden | Crate-private hooks through the Phase 5 review |
| Root semantics drift | Wrong limits, order, or errors | Phase 0 oracle and Phase 10 differential tests |

## Decisions for the next design review

The implementation should not silently decide these:

1. **Can demand widen after a scan opens?** This gates whether DemandEpoch exists at all. Phase 0.
2. Should declare_domains, describe_reads, and open_exec be separate vtable methods or products of
   one preparation traversal?
3. Is a 1,024-row DemandLedger block the right initial balance?
4. Should the exact compatibility adapter prepare projection reads before MaskFuture resolves?
5. Which computation classes are safe to speculate, and who owns that classification?
6. Does the existing runtime provide the CPU-task and cancellation behavior required, or is a
   scan-local pool needed?
7. Does capping remove parent retention in practice, or do real encodings overshoot often enough
   that parents must buffer anyway?
8. Which current ordered-error behavior is contractual?
9. Are morsels always row-count ranges, or should physical boundaries align or cap them? Catalog
   coverage and DomainMap composition supply the data to answer this.
10. How long must the old executor remain as a rollback mode?
11. Which benchmark thresholds block default rollout?

Question 6 of the previous list — what memory is charged where after a batch is sliced — is now
answered by rule: the node that can release it is charged for it, and a child that overshoots a cap
charges itself.

The recommended first review scope is Phases 0 through 3. Those phases settle semantics, the domain
model, the drive contract, demand ownership, and read discovery without committing to production
scheduling or a public API.
