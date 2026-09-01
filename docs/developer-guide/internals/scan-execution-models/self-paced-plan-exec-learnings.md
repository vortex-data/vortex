# Self-Paced Plan Execution Experimental Learning Ledger

This is a deliberately comprehensive list of things learned while building and tuning the
restricted self-paced executor. Entries mix measurements, code observations, and hypotheses. They
may be incomplete or wrong and are not design commitments. Confidence describes the evidence in
this experiment, not how broadly the statement applies to Vortex.

## Benchmark and layout

1. **V1 must run on natural splits.** Giving V1 self-paced morsels or silently falling back to a
   fixed split changes the established executor being measured. **Confidence: high.** The fair
   harness now calls V1 with the reopened file's real natural boundaries.

2. **Equal rows are more important than equal task shapes.** This is a scan comparison between two
   execution models, not a requirement that both receive identical scheduling units. **Confidence:
   high.** Row count and ordered output hash are checked before timing.

3. **Core count needs OS affinity as well as a logical limit.** Worker settings alone do not prove
   that both paths use the same CPUs. **Confidence: high.** Final runs use concurrency 16 and
   `taskset -c 0-15`.

4. **The real FineWeb fixture is not a small slice.** All 15 local files contain 14,868,862 rows,
   represented by 157 ingestion chunks and 1,823 natural splits. **Confidence: high.** These counts
   are printed while building and reopening the serialized fixture.

5. **Morsels can and should cross chunk boundaries in this experiment.** Merge-16 produces 116
   outer morsels from 1,823 real splits. Internal fragments recover chunk-aligned progress without
   changing the outer output unit. **Confidence: high.** Tests exercise cross-boundary morsels.

6. **A morsel is never smaller than a constituent natural split under merge-16.** It is the union
   of up to 16 consecutive splits, with a shorter final rollup. **Confidence: high.** This follows
   from the range construction and is asserted by the harness.

7. **A fixed split-count rollup is only a starting point.** Sixteen merged splits supplied enough
   Q06 morsels for 16 cores; 32 can reduce parallelism, while variable split byte sizes make either
   count a poor proxy for work. **Confidence: medium.** A byte/work-aware target still needs tests.

8. **Exact serialization matters.** Both executors must reopen the same bytes and query rather than
   compare separately constructed in-memory layouts. **Confidence: high.** The fixture reports its
   byte length and stable hash.

9. **The current result applies only to `Struct(Chunked(Flat))`.** Other layouts, compression,
   nullability, and general expressions can change both I/O and CPU behavior. **Confidence: high.**
   The experimental layout strategy rejects unsupported plans.

## Demand and predicates

10. **Whole-morsel mask sealing prevents the intended pipeline.** A later predicate cannot start
    for an early segment if it waits for a complete outer mask. **Confidence: high.** Fragment state
    was required to expose the next conjunct before sibling chunks completed.

11. **The next predicate can run once its fragment's preceding demand is known.** It need not wait
    for unrelated fragments, and sibling fragments can run in parallel. **Confidence: high.** The
    fragment-streaming test observes this ordering.

12. **This dependency is plan-execution state, not scheduler semantics.** Plan execution knows row
    masks, cache coverage, and expression dependencies; the scheduler knows resource budgets and
    readiness. **Confidence: medium-high.** Other scheduler organizations could move the boundary,
    but teaching a global scheduler mask algebra would couple it tightly to operators.

13. **Reduced demand can eliminate enormous predicate work.** FineWeb Q06 recorded 3,638 sparse
    later predicates evaluating 24,957 demanded rows while skipping 29,689,351 row applications.
    **Confidence: high for Q06.** It does not imply the same selectivity for other queries.

14. **Less aggregate predicate CPU does not guarantee lower wall time.** Sparse evaluation reduced
    measured predicate CPU from about 18.8 ms to 10.9 ms, yet the final executor remained 2.190x
    slower than V1. **Confidence: high.** Dependency publication and coordinator latency sit on the
    critical path.

15. **Predicate order and predicate parallelism are separate choices.** Historical selectivity and
    cost can choose the first predicate; expected savings versus dependency wait should decide
    whether another predicate waits or runs concurrently. **Confidence: medium.** The adaptive cost
    model is not yet implemented.

16. **Feedback becomes useful only after observations exist.** The first fragment or first scan
    needs static estimates or query order; later fragments can use observed true counts and elapsed
    predicate time. **Confidence: medium.** Cross-scan persistence has not been evaluated.

17. **Empty demand should stop remaining predicates and projection reads.** It is both a correctness
    simplification and an important selective-query optimization. **Confidence: high.** Tests cover
    empty fragment demand, including resources shared across morsels.

## Tasks and orchestration

18. **A CPU task per segment predicate is too fine-grained here.** The first fragment implementation
    was about 2.299x V1 because task allocation, dispatch, completion, and graph transitions cost
    more than the small predicate kernels. **Confidence: high for this flat in-memory fixture.**
    Expensive predicates may reverse the tradeoff.

19. **Fusing predicate evaluation with read/decode removes tasks but mixes resource classes.** It
    improved Q06, although a future scheduler may want distinct I/O and CPU admission. **Confidence:
    high on performance, medium on architecture.** A coarse continuation could preserve both.

20. **Polling ready read/decode work on the coordinator is harmful.** Decode is synchronous after
    the async read resolves, so inline polling serialized worker work and worsened the ratio.
    **Confidence: high.** Ready does not mean cheap to complete.

21. **Completion-side adoption removes a real transition class.** Direct adoption reduced the
    fragment path from roughly 33,242 transitions to 22,320. **Confidence: high.** Its wall-time
    effect varies with noise and remaining costs.

22. **The final fragment-mask merge is not the main problem.** Concatenating bit buffers outside
    the fragments took roughly 0.57 ms in the detailed Q06 trace. **Confidence: high for Q06.**

23. **One coordinator makes graph mutation easy and the critical path serial.** The same loop
    drains completions, advances nodes, selects and claims tasks, adopts masks, and yields output.
    **Confidence: high.** Whether sharding beats synchronization overhead remains unproven.

24. **Sharding by morsel or small morsel groups is the next structural experiment.** Most fragment
    state is locally owned, while shared resources need an explicit registry or owner. **Confidence:
    low-medium.** This is a design hypothesis, not a measured solution.

25. **Wave-by-wave scheduling loses pipeline overlap.** Ready fragment work should be admitted as
    completions arrive rather than waiting for a global predicate phase. **Confidence: high.** The
    current graph is event-driven, although its coordinator remains serialized.

## Masks and cache coverage

26. **A materialized all-true `BoolArray` per morsel or fragment is avoidable work.** Symbolic
    all-true state can survive until an evaluator needs physical bits. **Confidence: medium-high.**
    The experiment removes some no-op materialization but not all of it.

27. **Per-bit demand assembly on the coordinator is disastrous.** The first sparse fused path
    regressed to about 2.922x V1. Appending/copying bit-buffer ranges recovered much of that loss.
    **Confidence: high.** Bitmap construction belongs in bulk operations.

28. **Unresolved demand should probably remain a bit buffer plus version.** Intersecting and
    publishing a few fragments at once may avoid repeated array wrappers and graph transitions.
    **Confidence: medium.** The representation and batching threshold need profiles.

29. **No-op adoption should not allocate a new mask.** In the all-row fused trace, 2,796 of 5,461
    adoptions did not reduce demand. Skipping those allocations improved the best sample.
    **Confidence: high on avoided work, medium on timing magnitude.**

30. **A partial predicate cache requires explicit coverage.** Values computed only for demanded
    rows cannot be reused for newly requested rows merely because the `SegmentId` and conjunct
    match. **Confidence: high.** Cached predicates now carry an evaluated-row bitmap.

31. **Completion waiters and later consumers have different reuse guarantees.** Waiters captured
    when a task is offered may trust its captured demand; later consumers must prove their demand
    is a subset of cached coverage or evaluate against the already decoded array. **Confidence:
    high.** This distinction fixed shared-resource cases without discarding decode reuse.

## I/O and sharing

32. **Q06's gap is not caused by substantially more physical I/O.** V1 and self-paced both issue
    about 10.9k requests and read about 714.6 MB. **Confidence: high for the measured fixture.**

33. **Q06 cannot demonstrate filter/projection sharing.** Its filter and projected fields are
    disjoint, so shared-resource and shared-byte metrics correctly report zero. **Confidence:
    high.** Overlap queries and tests do demonstrate reuse.

34. **Filter/projection byte sharing exists when both use the same `SegmentId`.** One resource node
    owns the read/decode result and projection can consume the array first decoded for a predicate.
    **Confidence: high within the restricted layout.**

35. **`SegmentId` alone is an intentionally incomplete resource key.** It is acceptable for this
    isolated source but a production cache needs segment source, layout/file identity, and possibly
    decode parameters. **Confidence: high.** Do not extend the experiment's assumption silently.

36. **Speculative I/O needs an explicit policy for unknown bytes and unknown demand.** Candidate
    controls include whether speculation is enabled, an estimated byte charge, a global byte cap,
    and a minimum expected surviving-row count. **Confidence: medium.** Metrics must distinguish
    useful, shared, and wasted speculative bytes before tuning it.

37. **Segment granularity can expose filter results earlier than a split-wide V1 future.** That is
    a real opportunity when selectivity avoids downstream reads or expensive CPU. **Confidence:
    high in mechanism, workload-dependent in benefit.** Q06 shows the work reduction but not a wall
    time win.

## Interpreting performance

38. **Self-paced is strongest when avoided work exceeds its machinery.** Earlier full-suite results
    favored selective queries with meaningful downstream work and some reuse; broad scans and tiny
    queries exposed fixed overhead. **Confidence: medium-high.** Results changed with the fair
    natural-split contract, so query-level numbers matter more than one overall mean.

39. **Select-all or almost-select-all paths need a reduced-machinery mode.** Progressive masks add
    little information when almost every row survives. A symbolic all-true path or early switch to
    parallel/full-row evaluation is likely necessary. **Confidence: medium.** The switching rule is
    not implemented.

40. **Few projected columns can magnify control overhead.** When useful decode/selection work is
    small, graph and task costs occupy a larger fraction of runtime. **Confidence: medium.** This
    should be reported alongside query selectivity and bytes.

41. **Many or expensive projected columns create more opportunity for early pruning.** Avoiding
    downstream reads and decodes can amortize mask machinery. **Confidence: medium.** Compression
    and object-store latency may change the crossover substantially.

42. **Aggregate operation nanoseconds are not wall-clock attribution.** Worker predicate times can
    overlap; coordinator time and dependency stalls may not appear in them. **Confidence: high.**
    Phase-level coordinator wall and CPU timers are still required.

43. **Tracing is diagnostic, not a benchmark mode.** Large trace strings and event vectors perturb
    short scans. Use trace counts to explain a separate non-traced alternating measurement.
    **Confidence: high.**

44. **`BTreeMap`, `BTreeSet`, metrics, and trace payloads are visible costs in this execution
    object.** Some provide valuable experiment observability, but production hot state should keep
    only what it needs and compile or configure detailed tracing out of normal runs. **Confidence:
    medium-high.** A coordinator phase profile is needed before removing specific structures.

45. **A favorable aggregate result can hide severe query regressions.** Always publish per-query
    tables, wins, geometric mean, rows, bytes, split/morsel counts, and concurrency availability.
    **Confidence: high.** FineWeb Q06 was the clearest example.

## Sharded coordinators (2026-08-23)

46. **The coordinator, not the workers, was the Q06 bottleneck, now measured directly.** Phase
    timing put the single coordinator at ~89% busy over the whole run (advance ~34%, completion
    handling ~28%, dispatch ~24%, idle ~11%), with finished worker results waiting ~17 us on
    average to be adopted. **Confidence: high.** This converts learning 23 from inference to
    measurement.

47. **Work reduction on a serialized critical path buys little; parallelizing the path buys a
    lot.** Allocation-free adoption counts, batched joins, batched fragment transitions, and a
    scheduler-pass skip combined recovered ~8%; four coordinator shards recovered ~44% (2.50x ->
    1.40x) and flipped TPC-H and most ClickBench shapes to wins. **Confidence: high on this
    fixture.**

48. **Four shards beat both two and eight on Q06.** Two shards leave each coordinator too busy;
    eight cut per-shard admission to two workers and lose latency hiding. The right shard count
    is workload- and core-dependent; a shared admission budget or stealing is still missing.
    **Confidence: medium-high.**

49. **Aligned shard boundaries make resource duplication a non-issue here.** Morsel groups end on
    natural splits, so the sharded run read exactly the same 10,918 segments and 714,536,112
    bytes as the single coordinator, every segment once. General layouts with segments spanning
    shard boundaries still need an explicit cross-shard resource owner. **Confidence: high for
    this fixture, by construction elsewhere.**

50. **Per-shard `Execution` init is now a visible cost.** Each shard builds the full plan-wide
    resource table and pays the morsel-overlap scan inside the timed region; with 4 shards the
    per-shard coordinator loop accounted for ~30 ms of a ~38 ms run, the rest being init and
    output plumbing. **Confidence: medium-high.**

51. **The serialized fixture hash is not portable across hosts.** A regenerated FineWeb fixture
    reproduced the byte length exactly (1,669,473,052) and all split counts, but a different
    stable hash. Use row counts, split counts, and byte length as cross-host identity, and the
    hash only within one host. **Confidence: high on observation, unknown cause.**

52. **Catalog regeneration is only faithful when the audit writes what production wrote.** Raw
    string columns reproduced the FineWeb catalog exactly (1,823/2,527) and the i64-converted
    lineitem reproduced TPC-H's 458 spans, but a 21-column i64 ClickBench audit produced 1,424
    all-field splits versus 19,599 from the 105-column production files. An internally fair
    contract survives; comparability with earlier tables does not. **Confidence: high.**

53. **Memory bounds the fair harness before CPU does on small hosts.** The 100-file ClickBench
    fixture holds the raw i64 arrays plus one restricted-edition (uncompressed) serialized copy
    plus a per-workload rechunked copy, exceeding 22 GB; a 30 GB host OOMs above ~20-30 files.
    **Confidence: high.**

54. **Owned coordination beats both the central coordinator and pooled shards.** Sixteen threads
    that each coordinate and evaluate their own morsel group inline turned Q06 from 1.40x
    (4 pooled shards) to 0.79x and flipped 25 of 28 workloads to wins. Per-morsel state needs no
    cross-thread communication when morsel groups end on natural splits; the scheduling problem
    collapses to "which thread owns which morsels". **Confidence: high for the in-memory
    restricted fixture; object-store latency will need read-ahead or async I/O per thread.**

55. **V1's physical request count is not deterministic.** Run-to-run it varies by a few duplicate
    concurrent segment reads, and a dropped duplicate in-flight future counts its request but
    never its bytes (~0.01% byte undercount observed). Cold-scan invariants must therefore be
    floors with a small counting allowance for V1, while self-paced required reads can be held to
    exact equality. **Confidence: high.**

56. **Enforced invariants beat one-off audits.** Making every timed iteration prove it re-read
    the warmup's unique-segment floor converts "we checked there is no caching" into a property
    the harness cannot silently lose, and it surfaced the V1 counting artifact immediately.
    **Confidence: high.**

## What remains genuinely unknown

- Whether sharded plan execution makes progressive demand faster than V1 without losing resource
  sharing or deterministic cancellation.
- The crossover model for waiting on a selective predicate versus running independent predicates
  immediately in parallel.
- The right morsel target when natural splits differ greatly in bytes, decode cost, and expected
  selectivity.
- Whether compressed arrays, nullability, nested layouts, remote I/O, or expensive expressions
  make segment-level avoided work dominate the current control overhead.
- How much detailed metrics and tracing cost when disabled, minimally enabled, and fully enabled.
- Whether the best production design is one executor with an adaptive fast path or two execution
  modes selected from plan and runtime observations.
