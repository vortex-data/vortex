# Self-Paced Plan Execution Findings

This report records what was learned while implementing and optimizing the restricted
[self-paced plan execution experiment](self-paced-plan-exec-experiment.md). It includes the
original 100-iteration comparison on 2026-08-21, a capacity-saturating FineWeb follow-up on
2026-08-22, and the coordinator-sharding follow-up on 2026-08-23. It is evidence about this
experiment, not a claim about a production executor.

## Sharded coordinators (2026-08-23)

Coordinator phase timing (`VORTEX_SELF_PACED_PHASE_TIMING=1`, new `coordinator_*` metrics)
answered the previous handover's P0 question directly. On full FineWeb Q06 the single coordinator
loop accounted for the entire ~58 ms self-paced run and was busy ~89% of it: advance ~19.5 ms
(fragment rescans and transitions), completion handling ~16 ms (including ~6.8 ms fragment mask
adoption), dispatch ~14 ms (claim, operation clones, spawn), and only ~6.3 ms waiting for
workers. Completed worker results waited on average ~17 us in the completion queue (~170 ms of
cumulative dwell against a 58 ms run), so workers were starved behind the coordinator, not the
reverse.

Work-reduction micro-optimizations (allocation-free adoption counts via a fused and-count,
batched resource joins, batching all available fragment progress into one transition, skipping
the speculative necessity pass when speculation is disabled, an all-true `SelectStruct` early
exit) recovered only 2.53x -> 2.32x on this host. The structural fix was sharding:
`VORTEX_SELF_PACED_SHARDS=N` runs N coordinator threads, each owning a contiguous group of
morsels with its own `Execution`, sharing one 16-thread worker pool with a static per-shard
admission budget of `concurrency / N`. Morsel groups align with natural splits, so no segment
straddles a shard boundary in these fixtures: the sharded run issued the same 10,918 unique
segment requests and 714,536,112 bytes as the single-coordinator run, with every segment read
exactly once. Output row counts and ordered hashes are still validated before timing.

Five-iteration medians on a 16-core, 30 GB host pinned to CPUs 0-15 (note: a smaller host than
the earlier reports; V1 medians here are correspondingly slower than the 2026-08-22 numbers):

| Shards | FineWeb Q06 self-paced ms | Ratio vs V1 |
| ---: | ---: | ---: |
| 1 | 67.8 | 2.501 |
| 2 | 45.4 | 1.642 |
| 4 | 38.4 | 1.397 |
| 8 | 39.2 | 1.436 |

With 4 shards across the suites (same fair merge-16 contract, 5 alternating iterations):

- FineWeb Q00-Q08: ratios 1.115, 0.568, 0.728, 1.219, 0.995, 1.022, 1.415, 1.071, 1.127 —
  geometric mean ~0.98, self-paced wins 3 of 9 with three near-ties.
- TPC-H SF10 lineitem (fresh duckdb dbgen Parquet, real natural splits from a regenerated
  catalog with 458 spans, matching the earlier audit): Q6 0.683, Q1 0.881, V1-friendly 0.619 —
  self-paced wins all three.
- ClickBench (20 files, 20,000,000 rows; the 30 GB host cannot hold the 100-file fixture, and
  the regenerated 21-column catalog is coarser than the 105-column production files audited
  earlier, so these are internally fair but not comparable to the 2026-08-22 table): self-paced
  wins 15 of 16 shapes, geometric mean ~0.74; only the dashboard shape loses at 1.176.

Two caveats keep this honest. First, sharding adds N coordinator threads on top of the 16-worker
pool, so the process briefly runs more runnable threads than V1's 16-worker runtime; admission is
still capped at 16 evaluation tasks. Second, per-shard `Execution` construction still builds the
full plan-wide resource table, and the remaining Q06 gap sits in per-shard dispatch (two
`Operation` clones and a `cached_predicates` Vec clone per claim) and that per-run init;
phase-sum evidence: with 4 shards the summed advance/dispatch/complete phases were ~29/22/18 ms
across shards while per-shard wall time was ~30 ms.

### Owned coordination: no central coordinator at all

`VORTEX_SELF_PACED_SHARD_MODE=owned` removes the coordinator/worker split entirely. Each of 16
threads owns a contiguous morsel group and runs the single-threaded loop over it: the thread
coordinates its own fragments and evaluates every read, decode, predicate, and selection inline.
There is no worker pool, no completion channel, no dispatch, and no queue dwell; cross-thread
communication disappears because resources are deduplicated within the owning thread and morsel
groups end on natural splits. The thread total (16) now matches V1's worker count exactly, which
also resolves the pooled-mode thread-fairness caveat.

This mode wins 25 of 28 workloads on the measurement host (five alternating iterations,
`taskset -c 0-15`):

- FineWeb Q00-Q08: 0.639, 0.524, 0.552, 0.694, 0.621, 0.629, 0.792, 0.612, 0.674 — all nine
  are wins, geometric mean ~0.63. Q06, the historical worst case, runs 21.0 ms against V1's
  26.8 ms.
- TPC-H SF10: Q6 0.562, Q1 0.686, V1-friendly 0.616.
- ClickBench (20-file fixture): 13 of 16 wins, geometric mean ~0.75; the losses are dashboard
  1.069, Q40 1.069, and Q41 1.234, with Q42 at parity (0.998).

The FineWeb Q06 progression on this host: 2.53x single coordinator, 2.32x after
work-reduction micro-optimizations, 1.40x with four pooled shards, 0.76x owned.

Five additional FineWeb scan shapes (query ids 9-13) close the P1 coverage gaps, and owned mode
wins every one:

| Shape | Ratio | I/O evidence |
| --- | ---: | --- |
| Q09 wide select-all (7 fields, all rows survive) | 0.596 | identical 20,216 requests / 953.9 MB on both engines — the win is pure executor efficiency with zero avoidable work |
| Q10 shared filter/projection field | 0.489 | self-paced 3,645 requests / 238.3 MB vs V1 5,469 / 357.5 MB — one decode serves filter and projection |
| Q11 five-conjunct chain | 0.951 | near-equal I/O; the dependency chain serializes predicate rounds, the smallest win |
| Q12 empty result | 0.433 | self-paced 1,823 requests (first predicate column only) vs V1 7,292, equal bytes |
| Q13 narrow highly selective (1 projected column) | 0.652 | near-equal bytes |

Q09 is the attribution cornerstone: with byte-identical physical work and nothing to avoid,
owned self-paced still runs 41% faster than V1, so the remaining advantage is scheduling-unit
cost — 116-168 merged morsels with inline per-thread coordination against V1's 1,823-2,527
per-split scan futures on a shared runtime.

### The pipeline executor: extensible nodes, pluggable demand, arbitrary child boundaries

`vortex-layout/src/plan/exec/pipeline.rs` (`VORTEX_SELF_PACED_SHARD_MODE=pipeline`) rebuilds the
executor around two seams. The scheduler knows exactly one trait, `MorselPipeline` (morsel range
in, `ExecBatch` out), so arbitrary execution nodes are added without scheduler changes; and the
shared per-morsel demand mask that gates every struct child is computed by a pluggable
`DemandPolicy` (`VORTEX_SELF_PACED_DEMAND=cascade|eager`). Alignment stopped being a
precondition: each field exposes chunks at its native boundaries and consumers cut them to root
row ranges (`overlapping_chunks`), so mutually unaligned children work — covered by a unit test
zipping fields chunked `[0,3,10)` against `[0,6,10)` byte-identically to an aligned reference.
Per-thread decoded-chunk caching preserves filter/projection decode sharing. The reactor's slot,
offer/claim, and fragment machinery does not exist in this mode.

It is also the fastest executor measured (five iterations, same fair contract, cold-scan I/O
invariant enforced; physical I/O identical to the reactor, e.g. Q06 at 10,918 requests /
714,536,112 bytes):

- FineWeb Q00-Q13: every shape wins, geometric mean ~0.32 — Q01 0.131, Q12 0.216, Q02 0.246,
  Q09 0.271, Q06 0.414 (11.4 ms vs V1's 27.6 ms; the owned reactor measured 21.0 ms).
- TPC-H: Q1 0.537, V1-friendly 0.572, Q6 0.938 (29 two-million-row morsels leave a thread-tail
  imbalance the reactor's finer pipelining hides; work stealing is the fix).
- ClickBench (20 files): 13 of 16 wins; the same three uneven-morsel losses (dashboard 1.06,
  Q40 1.22, Q41 1.31).

Cascade and eager demand differ by within-noise amounts on these shapes (Q06 0.414 vs 0.422,
Q09 0.271 vs 0.245, Q01 identical): the cascade's chunk-skipping pays off on empty or highly
selective shapes, eager avoids gating arithmetic on select-all shapes, and swapping them touches
nothing but the policy object. Executor totals: single coordinator 2.53x -> pooled shards 1.40x
-> owned 0.79x -> pipeline 0.41x on FineWeb Q06.

The pipeline's row-domain handling was then formalized as an executor vtable rather than inline
arithmetic: every node relationship is a **down demand transform** plus **up mask/array
transforms** (`FieldDomain::push_demand` / `pull_mask` / `pull_array`), each modeled on the
layout's native metadata — `ConcatDomain` uses the chunk-offset prefix sums (binary search down,
ordered append up), the struct node's identity relationship shares one demand mask by refcount
and packs zero-copy, and a list node would implement the same two methods over its offsets
buffer. Demand policies and the projection gather now speak only to the vtable. Re-measured after
the refactor, the seam is effectively free because dispatch is per chunk, never per row: FineWeb
geometric mean ~0.34 versus ~0.32 (Q06 0.420 vs 0.414) and TPC-H unchanged (0.93/0.53/0.58).

Three further changes completed the sweep. The suites were widened to 18 FineWeb shapes (adding
a score-band range predicate, a rare-flag filter projecting all fourteen fields, a two-range
conjunction, and a shared-and-deep shape as Q14-Q17) and 21 ClickBench shapes (adding wide
select-all, shared filter/projection, a five-conjunct chain, empty-result, and narrow-selective
as Q43-Q47). The pipeline scheduler switched from static contiguous morsel groups to threads
self-scheduling morsels off one shared atomic cursor (order restored by index), which eliminated
every few-morsel tail loss: ClickBench dashboard 1.06 -> 0.82, Q40 1.22 -> 0.67, Q41 1.31 ->
0.64, and FineWeb Q01 improved to 0.092. And the predicate kernel's dense-but-partial regime
(demand between one fifth and all rows) was switched from a per-row demand-consulting `map_cmp`
to two vectorized passes — full evaluation then AND — which is what made the five-conjunct
chains competitive under the cascade (ClickBench Q45 1.06 -> 0.95) without needing the eager
policy. After all three: every measured workload beats V1 — 18/18 FineWeb (geometric mean
~0.33), 3/3 TPC-H, 21/21 ClickBench (geometric mean ~0.6) — with the one structural remainder
being TPC-H Q6 at ~0.94, whose 29 two-million-row morsels bound the makespan at two serial
morsels per thread regardless of scheduling; finer intra-morsel parallelism would need the
merge-16 contract revisited.

### Adaptive demand, wider suites, and the statpopgen small-data regime (2026-08-23)

`AdaptiveDemand` (now the default; `VORTEX_SELF_PACED_DEMAND=cascade|eager` selects the
deterministic policies) orders conjuncts by observed survival, most selective first, learning
across morsels within a run. Output is unchanged — conjunction commutes and every mask is adopted
as a subset of the demand it was evaluated under — but the effect on the former weak spots is
direct: ClickBench dashboard 0.84 -> 0.71-0.77 and the five-conjunct chain Q45 0.89 -> 0.76,
with FineWeb unchanged at ~0.33 geometric mean. Because adaptive ordering legitimately skips
different chunks as its statistics improve, the harness's byte-exact/floor invariant is enforced
through the deterministic policies, which share the same read path.

ClickBench gained Q48-Q51 (equality+flag, pure time window, three geometry ranges, region band +
flag): 0.48-0.59, all wins; the suite now spans 25 shapes, all won, geometric mean ~0.56.

A statpopgen suite was added end to end: gnomAD chr21 VCF converted through vortex-bench's
data-gen (100k and 1M row variants), ten scalar columns (POS, QUAL milli-units, hashed ID/REF,
AN populations) as the i64 fixture, an audit mode producing its split catalog, and six
genomics-flavored shapes (region interval, quality threshold, well-genotyped region, wide
select-all, empty, shared population field). It exposed two real findings. First, the fixed
merge-16 roll-up collapses compact data (1M rows compress to 8 natural splits) into one morsel
and concurrency 1; the harness now targets ~2x the worker count
(`merge = clamp(splits/32, 1, 16)`), which leaves every large suite at merge 16 and slightly
improves ClickBench (per-file merge 4-5). Second, with parallelism restored, statpopgen's
sub-millisecond scans are the first genuine self-paced losses (three shapes at 2.2-2.7x, the
16-way shapes near parity): the pipeline carries ~100us of per-morsel fixed work (demand and
included-mask buffers, lazy filter setup, selection, pack) that a 0.5 ms scan cannot amortize —
confirmed thread-count-independent (1, 2, 4, and 8 threads within noise) after the scheduler
switched to a reused worker pool (per-run thread spawns were the first suspect and are now
eliminated). Reducing per-morsel constant cost is the open item for the tiny-scan regime; at
merge 16 (both engines at concurrency 1) the same shapes won at 0.12-0.34, so the executor's
fixed cost per *scan* remains far below V1's.

A first round of per-morsel constant cuts followed: full-demand predicate evaluation no longer
allocates a per-morsel all-true mask (`evaluate_predicate_full`), `pull_mask` returns the single
part zero-copy when one segment tiles the morsel, and `pull_array` prices its coverage check from
the segments' already-computed demanded counts (no bit scan) and reuses the single segment's
demand slice as the filter mask. statpopgen 1M moved from three 2.2-3.4x losses to Q01 1.00,
Q04 0.77, Q03 0.91, with the remaining gather-heavy shapes at 1.7-1.8x — the residual is one
`Mask`/filter construction per field per morsel, which a struct-level single filter (valid when
all fields gather identical row sets, i.e. aligned chunks) would remove. The cuts also set new
bests on the large suites: FineWeb Q06 0.385 (10.8 ms), Q12 0.184, Q01 0.086.

### Tiny-scan round 2: measurement discipline, shared masks, density-aware adaptive

Three more findings on the statpopgen sub-millisecond regime. First, five-iteration medians are
unreliable at this scale: shapes swung 0.8-2.5x between identical runs; 100-iteration medians are
now the standard for sub-millisecond workloads, and under them most of the apparent losses shrank
(Q00 1.7 -> ~1.0-1.4 before any code change). Second, the per-morsel selection `Mask` is now
built once and shared by every projection field that gathers the whole range
(`FieldDomain::pull_array` takes the parent's shared mask), removing per-field
`Mask::from_buffer` scans. Third, the eager-policy experiment showed dense demand makes gating
cost more than it avoids (Q02, 87.7% survival: cascade 2.38 vs eager 1.31), so `AdaptiveDemand`
now switches per conjunct to full-evaluate-and-intersect when current demand density is >= 50%.
At 100 iterations the statpopgen suite stands at 0.96 / 1.00 / 2.62 / 0.52 / 0.85 / 1.02 —
five of six at or better than parity. Q02 remains open: it responds to the eager *policy* (1.31)
but not to the equivalent in-policy dense switch (2.62), an unexplained delta; a samply profile
of the workload is captured for the follow-up. Self-paced also reads half of V1's requests on
these shapes (Q00: 12 vs 24) via chunk skipping.

### External oracle validation (2026-08-23)

The harness's own correctness gates (V1-vs-self-paced hash equality, `run_eager` parity) share
the parquet ingestion and query construction, so a consistent bug there would be invisible to
them. Seventeen workloads were therefore checked against DuckDB running equivalent SQL over the
original source Parquet, replicating each fixture derivation (substring-contains flags, date
digit-folding, score truncation to ppm/milli units, byte lengths, decimal/date-to-i64
conversion, hash-equality predicates by their defining strings): all seventeen output row counts
matched exactly — statpopgen 6/6 (84,350 / 0 / 877,404 / 1,000,000 / 0 / 959,526), TPC-H Q6
1,139,264 and Q1 59,142,609, FineWeb Q01/Q02/Q04/Q05/Q06/Q07/Q10 (including hash-based language
equality at 11,898), ClickBench Q47 19,491 and Q01 331,750. The timed loop additionally asserts
every iteration's row count against its warmup for both engines. Remaining validation gaps:
oracle coverage is row counts rather than full values (value-level agreement rests on the
cross-engine ordered hash), and zero-row workloads compare only row counts.

### I/O read patterns (2026-08-23)

A per-request order dump (`VORTEX_SELF_PACED_IO_DUMP` in trace mode) recorded every segment
request's identity and size for both engines across all 42 workloads. Findings:

- **V1 re-reads shared segments; the pipeline never does.** V1's worst cases read the same
  segments twice or more: tpch_v1_friendly 916 requests / 960 MB versus the pipeline's 458 /
  480 MB (the filtered column is the projected column and V1 reads it once per role), TPC-H Q6
  3,206 requests / 3.36 GB versus 1,832 / 1.92 GB, ClickBench Q00/Q01/Q07 exactly 2x. The
  pipeline's per-thread decoded-chunk cache reads every segment at most once per thread, with at
  most a handful of cross-thread boundary duplicates (Q45: one).
- **Demand skipping shows directly in bytes**: FineWeb Q16 297 MB vs V1's 505 MB, Q17 358 vs
  596 MB, Q12 (empty) 1,823 requests vs 7,292.
- **One byte regression**: ClickBench dashboard reads 637 MB vs V1's 558 MB despite fewer
  requests — query-order predicate evaluation reads a wide early column where V1's plan prunes
  with a cheaper one first. Predicate ordering by observed cost/selectivity (the old adaptive
  policy, as a `DemandPolicy`) is the fix.
- **Request sizes are layout-bound**: FineWeb segments average 47-64 KB per request — far below
  object-store sweet spots — while TPC-H/ClickBench run 0.5-1 MB. The serialized layout
  interleaves fields by chunk, so a narrow projection's reads are strided and cannot coalesce;
  wide scans are highly coalescible: merging file-adjacent requests would cut FineWeb Q15 from
  32,882 requests to 384 ranges (~86x) and averages a ~25% request reduction across workloads.
- **Arrival order is non-sequential in both engines** under 16-way parallelism (V1 up to 33%
  adjacent arrivals in single-field phases, pipeline ~0% due to work stealing); fine for
  concurrent object stores, worth a per-thread field-major sort if targeting spinning media.

None of this changes wall time on the in-memory source (requests are refcount clones), so the
actionable items are recorded for the real-I/O phase: a ranged/multi-get `SegmentSource` API for
run coalescing, writer-side chunk sizing for FineWeb-like data, adaptive predicate ordering as a
demand policy, and a cross-thread once-cell for boundary chunks.

Plan-time materialization of the cutting was built, measured, and removed. Eagerly compiling
per-morsel segment lists for every plan field regressed FineWeb from ~0.34 to ~0.39, and even
trimmed to query-touched fields it stayed ~0.36: that "compilation" was compute relocated onto a
serial pre-thread path, while runtime cutting costs ~100ns per segment distributed across all 16
threads, and per-scan planning amortizes nothing. The lesson, kept as the module's design rule:
planning does no compute — building the pipeline wires topology (domains, touched fields, output
names), the scan computes its morsel splits once, and the struct node shares one refcounted
demand handle with every child; all remaining work happens at execution on the owning threads.
The final state re-measured at Q06 0.404 (11.6 ms), Q01 0.135, Q12 0.262.

The harness now also enforces the no-caching contract per iteration instead of asserting it once:
every timed run's `CountingSource` totals are compared against its cold warmup. Self-paced totals
must match the warmup exactly (its required reads are deterministic; Q06 re-issued 10,918
requests / 714,536,112 bytes on every iteration), and both engines must stay above the warmup's
unique-segment floor (one read of every distinct segment). V1 gets a 1% counting allowance below
the floor because it sometimes drops a duplicate in-flight segment future whose request was
counted but whose bytes never resolved — the observed undershoot is ~0.01%, while any real
cross-run caching would remove a large fraction of the floor and fail the run. On selective
queries the invariant also documents the honest I/O difference: on Q01 self-paced issues 5,101
requests / 247.8 MB against V1's 37,905 requests / 257.7 MB with identical validated output.

## Original headline result

In the original comparison at 131,072 rows per self-paced morsel and concurrency 16, self-paced
execution won 15 of 28 scan workloads. The unweighted geometric mean of self-paced/V1 median time
ratios was `0.891`, or 10.9% faster overall.

| Suite | Workloads | Wins | Geometric mean self-paced/V1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| ClickBench scan shapes | 16 | 7 | 0.918 | Selective/reuse wins offset a 2.5-3.3% tax on broad scans |
| TPC-H scan shapes | 3 | 3 | 0.764 | Q6 benefits strongly from progressive filtering; the V1-friendly case is near parity |
| FineWeb scan analogues | 9 | 5 | 0.889 | Mixed; sub-millisecond cases expose fixed control costs |
| Combined | 28 | 15 | 0.891 | Promising for a restricted experiment, with workload-dependent wins |

The ratio is the geometric mean of per-workload median ratios. Summing all wall times gives a
different and less useful answer because long broad scans dominate that calculation.

### Fair complete-data merge-16 follow-up

The final comparison replaces fixed 128K self-paced morsels with ranges formed by merging 16 real
natural splits. V1 receives those natural boundaries directly and never receives morsels. Both
paths reopen the same query-specific serialized file, whose edition permits only
`Struct<Chunked<Flat<i64>>>`, and receive clones of the same materialized query object. The process
is pinned to CPUs 0-15 and both paths use concurrency 16 (or the morsel count when smaller).

These are medians of ten alternating iterations over every locally available benchmark row. The
unweighted geometric mean of the 28 self-paced/V1 ratios is `1.176`; self-paced wins 6 of 28.

| Workload | V1 ms | Self-paced ms | Ratio |
| --- | ---: | ---: | ---: |
| ClickBench selective | 6.717 | 7.253 | 1.080 |
| ClickBench dashboard | 13.568 | 19.858 | 1.464 |
| ClickBench Q00 | 5.618 | 5.407 | 0.963 |
| ClickBench Q01 | 3.946 | 4.939 | 1.252 |
| ClickBench Q02 | 7.452 | 11.285 | 1.514 |
| ClickBench Q03 | 6.304 | 8.134 | 1.290 |
| ClickBench Q04 | 6.228 | 8.312 | 1.334 |
| ClickBench Q05 | 5.847 | 7.656 | 1.309 |
| ClickBench Q06 | 5.924 | 7.794 | 1.316 |
| ClickBench Q07 | 3.949 | 4.985 | 1.263 |
| ClickBench Q08 | 8.079 | 13.299 | 1.646 |
| ClickBench Q09 | 11.955 | 18.982 | 1.588 |
| ClickBench Q39 | 16.349 | 12.434 | 0.761 |
| ClickBench Q40 | 9.889 | 11.271 | 1.140 |
| ClickBench Q41 | 8.643 | 9.719 | 1.124 |
| ClickBench Q42 | 6.144 | 8.085 | 1.316 |
| TPC-H Q6 | 15.312 | 12.558 | 0.820 |
| TPC-H Q1 | 6.796 | 8.570 | 1.261 |
| TPC-H V1-friendly | 3.356 | 3.128 | 0.932 |
| FineWeb Q00 | 8.562 | 10.949 | 1.279 |
| FineWeb Q01 | 82.335 | 42.325 | 0.514 |
| FineWeb Q02 | 15.364 | 11.273 | 0.734 |
| FineWeb Q03 | 20.081 | 30.828 | 1.535 |
| FineWeb Q04 | 18.542 | 22.920 | 1.236 |
| FineWeb Q05 | 18.503 | 22.523 | 1.217 |
| FineWeb Q06 | 21.508 | 39.935 | 1.857 |
| FineWeb Q07 | 19.068 | 22.741 | 1.193 |
| FineWeb Q08 | 23.736 | 25.585 | 1.078 |

The complete inputs are all 100 ClickBench shards (99,997,497 rows), TPC-H SF10 lineitem
(59,986,052 rows), and all 15 local FineWeb shards (14,868,862 rows). This remains a comparison of
restricted scan analogues rather than full SQL query runtimes.

FineWeb Q06 explains the largest remaining regression. V1 and self-paced issue about 10.9k reads
and return about 714.7 MB each, but self-paced performs 11,386 scheduled operations, 22,903 state
transitions, and 23,827 node inspections. Polling fused `ReadDecodeFlat` work on the coordinator
made the ratio worse (`2.082`): a ready request also performs synchronous decode, so the attempted
fast path serialized work that needs to remain parallel. The next useful optimization boundary is
coarser multi-segment read/decode submission, not inline polling of the fused task.

## Comparison contract

The comparison is a scan comparison, not a like-for-like execution-model comparison:

- V1 runs through `ScanBuilder::with_natural_splits` using the file's real natural layout
  boundaries. It never receives a morsel size and never falls back to automatic layout splitting.
- Self-paced ranges merge 16 consecutive natural splits. A morsel can cross chunk boundaries and
  is never smaller than a constituent natural split.
- Both paths use at most 16 workers, the same serialized Vortex layout, in-memory `SegmentSource`,
  filter, projection, input rows, and warm fixture state.
- They do not use identical worker executors: V1 is driven by the 16-worker Tokio runtime, while
  self-paced non-inline tasks use a shared futures thread pool behind the same concurrency cap.
- Each path gets a warm-up. Ten measured iterations alternate which executor runs first, and the
  reported time is the median.
- Every warm-up compares output row count and a stable ordered hash before timings are accepted.
- Timed runs consume every output. Fixture construction and Parquet ingestion are outside timing.

The original data sets were:

- the first ten real ClickBench Parquet shards, converted to the experiment's supported `i64`
  fields and totaling 10,000,000 rows;
- a deterministic 2,097,152-row TPC-H lineitem-shaped fixture; and
- one 1,046,615-row FineWeb Parquet sample converted to `i64` scan features.

FineWeb ingestion can now scale beyond the default sample. `VORTEX_FINEWEB_PARQUET` accepts either
a Parquet file or a directory; directory inputs use every `.parquet` file in sorted order.
`VORTEX_FINEWEB_MAX_FILES` optionally caps that list for repeatable size sweeps. The runner prints
the resulting file, chunk, and row counts before execution.

The complete-data runner also accepts `VORTEX_CLICKBENCH_MAX_FILES`; setting it to 100 consumes
every local ClickBench shard. `VORTEX_TPCH_LINEITEM_PARQUET` switches from the synthetic fixture
to a real lineitem Parquet table, converting decimal quantities and Arrow dates into the restricted
executor's `i64` domain. `clickbench_all`, `tpch_all`, and `fineweb_all` load only their selected
fixture and execute every scan shape in that suite.

All nine FineWeb analogues can be selected without loading the unrelated fixtures:

- `VORTEX_SELF_PACED_COMPARE_WORKLOAD=fineweb_q00` through `fineweb_q08`, or `fineweb_all`;
- `VORTEX_SELF_PACED_TRACE=fineweb-q00-128k` through `fineweb-q08-128k`; and
- `VORTEX_SELF_PACED_PROFILE=fineweb-q00-self-128k` or `fineweb-q00-v1-128k`, with any query ID.

The ClickBench and FineWeb cases are scan-input analogues. They preserve useful filter and
projection shapes, but exclude aggregation, grouping, ordering, strings, and disjunction because
those are outside the restricted evaluator. They must not be reported as full query runtimes.

## Historical fixed-128K results

These values predate the fair natural-split contract and are retained only as optimization history.
They are median milliseconds over 100 alternating iterations. Ratios below one favor self-paced.

| Workload | V1 ms | Self-paced ms | Ratio |
| --- | ---: | ---: | ---: |
| ClickBench selective | 0.864 | 0.743 | 0.860 |
| ClickBench dashboard | 1.708 | 1.651 | 0.967 |
| ClickBench Q00 | 11.512 | 11.823 | 1.027 |
| ClickBench Q01 | 2.014 | 1.051 | 0.522 |
| ClickBench Q02 | 22.572 | 23.141 | 1.025 |
| ClickBench Q03 | 11.516 | 11.896 | 1.033 |
| ClickBench Q04 | 11.516 | 11.886 | 1.032 |
| ClickBench Q05 | 11.512 | 11.893 | 1.033 |
| ClickBench Q06 | 11.512 | 11.893 | 1.033 |
| ClickBench Q07 | 2.015 | 1.050 | 0.521 |
| ClickBench Q08 | 22.558 | 23.184 | 1.028 |
| ClickBench Q09 | 44.555 | 45.651 | 1.025 |
| ClickBench Q39 | 5.810 | 5.318 | 0.915 |
| ClickBench Q40 | 1.259 | 1.299 | 1.032 |
| ClickBench Q41 | 1.838 | 1.776 | 0.966 |
| ClickBench Q42 | 2.331 | 2.259 | 0.969 |
| TPC-H Q6 scan | 1.193 | 0.622 | 0.522 |
| TPC-H Q1 scan | 11.325 | 9.897 | 0.874 |
| TPC-H V1-friendly | 2.480 | 2.421 | 0.976 |
| FineWeb Q00 analogue | 1.303 | 1.604 | 1.231 |
| FineWeb Q01 analogue | 1.000 | 0.677 | 0.677 |
| FineWeb Q02 analogue | 0.363 | 0.293 | 0.806 |
| FineWeb Q03 analogue | 0.355 | 0.397 | 1.119 |
| FineWeb Q04 analogue | 0.704 | 0.413 | 0.587 |
| FineWeb Q05 analogue | 0.313 | 0.313 | 1.001 |
| FineWeb Q06 analogue | 0.366 | 0.448 | 1.226 |
| FineWeb Q07 analogue | 0.324 | 0.316 | 0.976 |
| FineWeb Q08 analogue | 0.194 | 0.128 | 0.659 |

## Why self-paced can be faster

The main advantage is different work, not a universally cheaper executor.

Self-paced execution evaluates predicates against the current shrinking demand. Projection reads
and selection wait until demand seals, so empty or sparse morsels can avoid projection work. A
predicate result already contains false bits outside its input demand; when it was evaluated at the
current demand version, the executor adopts that result directly instead of intersecting the same
two masks again.

Shared resources are interned by `SegmentId` and retained across possible morsel users. This can
turn repeated V1 requests into one self-paced request. The clearest measured example was
ClickBench Q42:

| Metric | V1 | Self-paced | Change |
| --- | ---: | ---: | ---: |
| Output rows | 558,105 | 558,105 | identical |
| Stable output hash | `0xe0d6122ac6c3572e` | `0xe0d6122ac6c3572e` | identical |
| Segment requests | 255 | 42 | 83.5% fewer |
| Unique segments | 36 | 42 | self-paced touched more distinct segments |
| Bytes returned | 1,296,016,200 | 336,004,200 | 74.1% fewer |

This is why Q42 became competitive despite scheduler overhead: V1 repeatedly requested some
segments, while the scan-wide self-paced graph requested each of its 42 segments once. The result
also shows why unique-segment count alone is misleading; total requests and bytes explain the wall
time better.

TPC-H Q6 is the strongest predicate-pipelining result. Five conjuncts progressively reduce demand
before the two projected fields are materialized, producing a `0.522` ratio. Q1, with one broad
predicate and four projected fields, still reaches `0.874`, while the deliberately simple
V1-friendly scan reaches `0.976`. That near-parity case is useful: it bounds the fixed experimental
tax when there is little scheduling opportunity.

## Why self-paced can be slower

When nearly every row and projected value is needed, self-paced has little work to avoid. It still
pays for execution construction, slots, offers, claims, completion messages, `advance` calls,
demand masks, selection, and Struct packing. ClickBench Q00, Q02-Q06, Q08, and Q09 expose this
cost: the final ratios cluster from `1.025` to `1.033`.

The FineWeb traces separate I/O from control cost. Q03 read exactly 55 segments and 41,870,100
bytes in both executors, yet self-paced was 11.9% slower. Q06 read exactly 66 segments and
50,244,120 bytes in both, yet self-paced was 22.6% slower. With equal logical I/O and absolute
times below half a millisecond, task dispatch, mask handling, and dependency waits dominate.

The experiment does no statistics or metadata pruning. Both engines receive the same logical
filter, but their execution paths may request different segments due to late materialization,
native V1 splitting, and self-paced scan-wide retention. The counting source measures logical
requests and returned buffer bytes, not physical NVMe or object-store traffic.

## Graph and control overhead

The Q42 trace makes the size of the experimental control plane concrete:

```text
10,000,000 input rows
50 scan-wide resource nodes
616 morsel-local slots
569 advance calls
1,308 transitions
1,800 node inspections
410 offered, claimed, and completed tasks
243 direct demand adoptions
472 adaptive waits
162 predicate reorders
```

The graph is sized by resources, morsels, fields, and conjuncts rather than by logical rows.
`advance` inspected about 3.2 nodes per call in this trace and remained bounded by the transition
budget. The cost is nevertheless material on short scans because every task still crosses offer,
claim, completion, wake-up, and slot-state machinery.

The equal-I/O FineWeb traces show two different scheduler shapes:

| Metric | FineWeb Q03 | FineWeb Q06 |
| --- | ---: | ---: |
| Advance calls | 133 | 130 |
| Transitions | 288 | 341 |
| Nodes inspected | 413 | 463 |
| Tasks completed | 166 | 193 |
| Inline demand combinations | 8 | 5 |
| Direct demand adoptions | 8 | 13 |
| No-op demand adoptions | 0 | 6 |
| Adaptive launches | 8 | 16 |
| Adaptive waits | 0 | 16 |
| Predicate reorders | 0 | 3 |

Q06 has more predicate coordination without an I/O saving, matching its larger regression. This
is stronger evidence than attributing the result to mask intersection alone: only five explicit
combinations ran, and they ran inline.

Early Samply captures did not symbolicate the benchmark binary reliably, including one report
with zero of 370 raw addresses resolved. The conclusions above therefore rely on median timings,
event traces, and operation counters rather than unresolved sampled stacks.

## Morsel size

The earlier 128K/65K sweep showed that larger morsels usually amortize per-morsel graphs, masks,
tasks, queue operations, and output batches. Fixed row counts were still the wrong final contract:
they ignored the storage layout and made it too easy to accidentally subdivide V1 work in the same
way.

The final contract merges 16 consecutive natural splits for self-paced and leaves V1 at the
unmodified natural boundaries. ClickBench produces 100-110 morsels, TPC-H 29, and FineWeb 116-168,
so every final workload has enough morsels to use all 16 allowed cores. A smaller table may produce
fewer than 16 morsels; in that case the executor caps concurrency to the morsel count rather than
manufacturing smaller work units.

Merge-16 is an experimental roll-up factor, not a production constant. Larger roll-ups reduce
control overhead but may reduce early output, increase masks and temporary arrays, or leave too few
independent morsels.

Morsels partition the root row domain independently of storage chunks. The implemented layout is
`Struct<Chunked<Flat>>`; a morsel carries ordered Flat slices and may cross aligned field-chunk
boundaries. This avoids coupling scheduling granularity to physical chunking.

## Adaptive predicate scheduling

The adaptive policy supports both demand pipelining and parallel predicate execution. It records,
per conjunct, cumulative input rows, output rows, elapsed nanoseconds, and sample count from prior
completions. It then:

1. ranks predicates by expected rows eliminated per nanosecond;
2. uses observed survival, falling back to priors of 10% for equality and 50% for inequalities;
3. computes a per-morsel supply window from global concurrency and morsel count; and
4. launches another predicate only when estimated parallel latency, including a 3 microsecond
   launch cost, is lower than waiting and evaluating it on the expected survivors.

This is adaptive across completed morsels, not clairvoyant within the first morsel. Unseen
predicates use priors, and the policy waits when it lacks observations for either the outstanding
or next predicate. Reordering and launch/wait counts are explicit metrics.

Running predicates concurrently means they may capture different demand versions. Three cases
avoid unnecessary mask work:

- a result computed from the current demand is adopted directly;
- a stale result whose true count equals its captured input count eliminated nothing and is a
  no-op against any newer subset of that input; and
- only a stale result that eliminated rows needs an explicit `CombineDemand` intersection.

`CombineDemand` runs inline because it is small, dependency-critical work. `PackStruct` also runs
inline for the current adaptive policy. These choices avoid thread-pool round trips while keeping
reads, decodes, predicates, and selections parallel.

## Optimizations that mattered

The implementation converged on several small fast paths rather than one broad special case:

- reuse the materialized all-true initial demand by morsel length instead of allocating one per
  morsel;
- use direct and no-op demand adoption to remove redundant mask intersections;
- leave candidate resource tasks dormant when their speculative I/O class is disabled;
- preserve query order for the first predicate, then adapt after a morsel observes real demand;
- execute all-true Struct and Flat selections inline while keeping sparse selection parallel;
- run dependency-critical mask combination and final Struct packing inline;
- wake only morsels recorded as waiting on a completed shared resource;
- retain and look up scan-wide resources directly by `SegmentId`;
- keep task inputs and leases in `SmallVec` storage for the common small arities; and
- retain the shared bit buffer in boolean summaries so resource-local range counts do not
  canonicalize the mask again;
- cache selected-row counts by morsel-relative range, sharing one count across aligned fields;
- omit projection reads and `SelectFlat` inputs for physical resource slices with zero selected
  rows, including when a morsel crosses chunk boundaries; and
- skip copying projected Flat values when demand is all true and one Flat slice covers the range;
  and
- stop scheduler selection when the available executor capacity is filled instead of constructing
  a full admissible frontier that the caller immediately truncates;
- traverse the adaptive ready frontier newest-first without allocating a reversed copy, allowing
  newly unblocked decode, predicate, and selection work to pipeline ahead of old reads; and
- return lazy filtered projection views for partial masks instead of copying selected values into
  eager compact buffers.

All-true selection returns slices of decoded arrays. Partial selection now wraps those same slices
in Vortex's compact logical `FilterArray`, matching V1's output materialization behavior.

The Q40-Q42 work showed that scheduler policy and fixed overhead interact. One optimization pass
reduced the 128K Q40, Q41, and Q42 times by 56.8%, 48.2%, and 18.2% respectively relative to the
preceding implementation. Subsequent fast paths brought the final ratios to `1.032`, `0.966`, and
`0.969`. The progression is evidence that the initial regression was mainly execution mechanics,
not an unavoidable cost of self-paced plans; the intermediate run does not isolate one causal
change.

FineWeb Q01 exposed the resource-local projection issue. With speculation disabled, the old
morsel-wide nonempty check completed 50 segments and 39.6 MB. Range-aware projection completed 37
segments and 29.2 MB, while V1 completed 30.8 MB. Ten-iteration self-paced time fell from about
2.46 ms to 2.21 ms. The remaining `2.11x` ratio is fixed scheduling cost on a roughly 1 ms V1
query, rather than excess projection I/O.

A subsequent control-plane pass made two costs directly visible. First, each projection field
walked the same partial demand mask. An intermediate implementation cached one immutable
selected-index buffer across fields, but complete-data Q1 showed that eagerly gathering values was
the wrong output contract regardless of index reuse. Lazy `FilterArray` views replaced that cache.
Second, the concurrent runner consumed one completion before returning to the reactor and
scheduler, even when many worker results were already queued. It now drains ready completions as a
batch before advancing morsels.

The first two changes reduced 128K FineWeb Q01 from 2.214 ms to 0.927 ms and Q06 from 0.949 ms to
0.626 ms in 20-iteration follow-ups. A trace then exposed a remaining scheduler issue: with
speculative I/O disabled, 143 candidate projection reads were retained in the external offered
queue. Q01's scheduler considered 2,807 entries to admit 118 tasks, a `23.79x` ratio. Candidate
tasks now remain dormant in reactor state unless their speculative class is enabled; promotion to
required work inserts them into the runnable queue. The same trace after the fix considered 197
entries for the same 118 admitted tasks, a `1.67x` ratio. Q01 reached 0.758 ms versus V1's 1.033 ms
(`0.734x`), and Q06 reached 0.576 ms versus V1's 0.373 ms (`1.545x`).

The executor metrics now report scheduler passes, tasks considered and admitted, completion
batches, completions drained, and maximum batch size. The repository-local
`summarize_self_paced_trace.py` tool combines those totals with per-operation task latency, wait
time, and reactor work from an execution trace. This is the routine diagnostic layer. Samply
spans remain the next step when the report points to CPU cost inside a particular operation.

The full SF10 trace exposed breadth-first launch waves despite full occupancy: the FIFO frontier
started with runs of 115 reads, 115 decodes, and 474 predicates, and emitted its first morsel at
25.9 ms of a 29.0 ms traced run. Adaptive newest-ready traversal reduced those initial runs to 16,
16, and 78 and emitted the first morsel at 2.0 ms. Every recorded wait still had 16 running tasks.
This removes the hidden wave behavior and dramatically improves time to first output, but total
throughput remains similar because all 4,584 tasks still execute.

### FineWeb Q00

Q00 applies `int_score > -1` and projects one field. On the available data, all 1,046,615 rows
survive. With 128K morsels, eight output morsels cross eleven physical chunks. The original
single-slice all-true fast path therefore missed most morsels and copied their values into compact
buffers. `SelectFlat` now returns a zero-copy `ChunkedArray` of sliced Flat inputs when all-true
ranges form a complete partition of a morsel. For a one-field projection, selection also produces
the final Struct directly, removing eight separate `PackStruct` tasks.

The comparison harness now hashes both outputs during its correctness warmup and excludes hashing
from timed iterations. This matters for zero-copy output: canonicalizing its chunked views for a
verification hash moved work outside the scan and previously obscured the executor improvement.
Over 100 alternating scan-only iterations, Q00 measured 0.164 ms in V1 and 0.207 ms self-paced, a
`1.260x` ratio and a 43 us absolute gap. The final trace has 60 tasks: 22 reads, 22 decodes, eight
predicates, and eight fused select/pack operations. It reports 60 scheduler considerations for 60
admissions and no control-plane warning. The remaining gap is fixed orchestration over only eight
morsels, not mask combination, projection copying, excess I/O, or scheduler rescanning.
An attempted all-match predicate pre-scan regressed self-paced time to 0.230 ms: the existing
bitmap collector uses multiversioned vector code, while the optimistic pre-scan was scalar. That
fast path was removed.

A 20-iteration scan-only sweep after the Q00 changes measured V1/self-paced ratios of `1.305`,
`0.838`, `1.452`, `1.517`, `1.428`, `1.338`, `1.688`, `1.365`, and `0.785` for Q00 through Q08.
The Q00 paths are gated to complete all-true partitions and single-field output, so they do not
add work to the multi-field filtered queries. Q02's trace completed only 56 tasks and spent a
95 us absolute gap mostly on fixed orchestration. Q06 completed 188 tasks: 66 reads, 66 decodes,
24 predicates, 24 selections, and eight packs. Its completion batches averaged 6.92 tasks and its
scheduler considered 1.34 tasks per admission; the remaining regression is task/reactor overhead,
not serialized completion handling or scheduler rescanning.

## Historical natural-split baseline and projection fusion

This section records intermediate results before the serialized merge-16 comparison above.

The final comparison contract gives V1 only the 115 natural SF10 lineitem chunks and gives 128K
morsels only to self-paced. `SplitBy::Layout` is not a valid substitute because it silently
subdivides wide chunks. Under the corrected contract, the pre-optimization SF10 medians were
12.906/23.668 ms for Q6, 3.183/12.718 ms for Q1, and 2.203/4.443 ms for the V1-friendly shape
(V1/self-paced).

Q6 initially created five predicate tasks for every one of 458 morsels. Two strict bounds read
shipdate and two read discount, so self-paced traversed each of those decoded fields and produced a
mask twice. Planning now intersects compatible predicates on the same field into one strict range
predicate. Predicate tasks fell from 2,290 to 1,374 and aggregate traced predicate latency fell
from about 286 ms to 129 ms. A 21-iteration full SF10 run measured 13.218 ms V1 and 13.073 ms
self-paced.

Multi-field projection previously ran `SelectFlat` once per field and then packed the results. Q1
therefore created 1,832 selection tasks plus 458 packs and applied the same almost-all-true mask
four times. `SelectStruct` now gathers aligned decoded field slices, packs one morsel-local Struct,
and applies the shared selection once. Q1's total task count fell from 3,898 to 2,066 and its full
SF10 self-paced median fell from 12.357 to 9.064 ms. It remains 2.79x slower than V1 because its
nearly non-selective single predicate and 458 morsels still pay substantially more fixed control
cost than 115 natural V1 tasks.

The same projection fusion improved the complete 15-file FineWeb set without output mismatches.
Notable self-paced changes were Q03 7.220 to 6.401 ms, Q04/Q05/Q07 about 5.84 to 5.01 ms, Q06
8.615 to 7.821 ms, and Q08 4.761 to 4.215 ms. FineWeb Q01 measured 5.640 ms V1 and 5.673 ms
self-paced. All configured ClickBench queries were also validated over all 100 shards; they remain
2.19x to 3.26x slower than natural-split V1, showing that fixed per-morsel orchestration is now the
larger cost for their mostly narrow scan shapes.

Three experiments are worth retaining as evidence. Replacing resource-completion scans with
explicit waiter lists reduced completion wake candidates from roughly 421,000 to 7,048 on Q6 but
did not measurably change wall time; it remains as a bounded reverse-dependency lookup and exposed
metric. Moving worker tasks from the separate futures pool onto the Tokio session runtime regressed
Q6 by 7.1%, and sharing offered tasks through `Arc<Task>` regressed it by about 3%; both
runtime/task-representation experiments were reverted.

Q1 also tested an adaptive dense-output lane. After eight predicate observations showed at least
90% survival, it constructed the exact lazy `FilterArray` and Struct output inline in the reactor
instead of offering `SelectStruct` to the worker pool. This preserved masks, I/O, cache behavior,
and 128K morsels, but regressed the 31-iteration SF10 Q1 median from about 9.0 ms to 10.489 ms
(roughly 16%). Slicing fields, assembling cross-resource chunks, and building the exact selection
mask are cheap enough to make worker-task overhead visible, but expensive enough to stall the
single reactor. The experiment was reverted. A viable dense path must retain parallel execution,
for example by submitting adjacent sealed dense morsels as one worker batch and distributing its
results back to the original morsel slots.

A second experiment batched adjacent `SelectStruct` operations onto one worker submission after
the same dense-demand signal. It retained exact per-morsel masks and outputs, but did not improve
Q1 (about 9.588 ms versus 9.453 ms in the paired run, and 9.459 ms alone), so it was reverted.
Changing projection speculation from adaptive to eager was likewise neutral: Q1 measured about
9.005 ms adaptive and 9.107 ms eager. Neither worker submission count nor projection-read waiting
is therefore the dominant remaining Q1 cost.

All authoritative comparisons are process-pinned with `taskset -c 0-15`, in addition to setting
execution concurrency to 16. Earlier runs without CPU affinity are retained only as diagnostics.
The original shared futures executor created 96 worker threads on this host even though admission
was capped at 16. The executor is now reused per configured concurrency and creates exactly 16
workers for these comparisons. Under 16-core affinity this reduced SF10 Q1 from 9.454 to 9.288 ms
and improved the tested TPCH cases by roughly 1-3%, but the remaining Q1 gap to natural-split V1 is
still about 2.76x.

The timed self-paced path also cloned the complete immutable `SourcePlan` on every execution,
including every chunk, serialized flat encoding context, field name, and range. Execution now
borrows the plan and copies only the resource state it must own; source-specific byte estimates are
filled into that new execution state. This does not retain decoded arrays, masks, or scan results.
In a 51-iteration, 16-core-pinned SF10 Q1 comparison, the retained old binary measured 9.511 ms
self-paced and 3.311 ms V1; the concurrency-sized pool plus borrowed plan measured 9.196 ms
self-paced and 3.287 ms V1, improving self-paced by 3.3% and the ratio from 2.873x to 2.797x.

## Real-file split audit

The restricted benchmark's earlier "natural" boundaries were the chunks of its hand-built
`Struct(Chunked(Flat))` fixture. They are not the physical splits produced by the default Vortex
writer. A raw `LayoutReader::register_splits` audit, performed before `SplitBy::Layout` can insert
its own 100K-row subdivisions, measured the actual written files. Morsels were formed by greedily
combining adjacent whole natural splits up to 131,072 rows and were never allowed to cut a split.

- TPCH SF10 lineitem has 59,986,052 rows and 7,323 all-field physical splits. The Q1, Q6, and
  single-quantity query masks each expose 458 natural spans of 86,148 to 131,072 rows, so they
  produce 458 128K-target morsels.
- All 100 ClickBench files have 99,997,497 rows and 19,599 all-field physical splits. Most audited
  query masks produce 800 morsels, eight per file, ranging from 79,993 to 131,072 rows.
- ClickBench Q01 and Q07 are exceptions. Their single `AdvEngineID` input exposes only two natural
  spans per file, ranging from 473,209 to 524,288 rows. Preserving real splits produces 200 large
  morsels, not 800 128K morsels. Fixed 128K row slicing would cut 600 physical spans across the
  dataset and must not be described as natural-split rollup.
- Only one FineWeb Vortex file is currently written: `sample.vortex`, with 1,046,615 rows. Its nine
  audited query masks produce eight morsels of 129,111 to 131,072 rows. The complete 15-file,
  14,868,862-row FineWeb results elsewhere in this document use the restricted Parquet-derived
  fixture and are not evidence about the unwritten files' physical split distribution.

Consequently, 128K is a target rather than an invariant when morsels preserve physical leaves. A
natural span wider than the target must remain one larger morsel. The previous fixed-row benchmark
still measures executor overhead, but it is not a real-layout end-to-end comparison.

### Historical in-memory split-count rollup

These results were later rejected because physical-file boundaries were applied to an unrelated
coarse in-memory layout. They remain here to document how the benchmark artifact was discovered.

The follow-up replaced the row target with file-local split-count rollups. A self-paced morsel is
the complete row range covered by 16 or 32 adjacent query-visible natural splits; the final morsel
in each file takes the remainder. V1 receives every unmerged natural split. Both engines run with
`min(16, self_paced_morsel_count)` workers and the process is pinned with `taskset -c 0-15`.
Morsels may cross physical chunks within a file but never cross source files.

The data was TPCH SF10 lineitem (59,986,052 rows), all 100 ClickBench shards (99,997,497 rows), and
all 15 FineWeb shards (14,868,862 rows). The previously missing 14 FineWeb Vortex files were
written with the default `WriteStrategyBuilder` before collecting their raw boundaries. Timings
below are median milliseconds from 11 alternating iterations for TPCH and five for ClickBench and
FineWeb. Ratios are self-paced divided by V1.

| Workload | Natural splits | Morsels 16 / 32 | V1 ms 16 / 32 | Self-paced ms 16 / 32 | Ratio 16 / 32 |
|---|---:|---:|---:|---:|---:|
| TPCH Q6 | 458 | 29 / 15 | 15.167 / 15.316 | 10.902 / 9.122 | 0.719 / 0.596 |
| TPCH Q1 | 458 | 29 / 15 | 6.098 / 6.092 | 4.101 / 4.076 | 0.672 / 0.669 |
| TPCH friendly | 458 | 29 / 15 | 3.401 / 3.264 | 2.422 / 2.394 | 0.712 / 0.733 |
| Click selective | 740 | 100 / 100 | 6.745 / 6.543 | 5.661 / 5.591 | 0.839 / 0.854 |
| Click dashboard | 908 | 100 / 100 | 12.568 / 12.923 | 9.220 / 8.339 | 0.734 / 0.645 |
| Click Q00 | 800 | 100 / 100 | 5.469 / 5.514 | 4.103 / 4.122 | 0.750 / 0.748 |
| Click Q01 | 200 | 100 / 100 | 4.120 / 4.058 | 4.243 / 4.238 | 1.030 / 1.044 |
| Click Q02 | 800 | 100 / 100 | 6.916 / 6.867 | 4.733 / 4.711 | 0.684 / 0.686 |
| Click Q03 | 908 | 100 / 100 | 5.963 / 5.972 | 4.372 / 4.345 | 0.733 / 0.728 |
| Click Q04 | 908 | 100 / 100 | 5.854 / 5.866 | 4.390 / 4.321 | 0.750 / 0.737 |
| Click Q05 | 800 | 100 / 100 | 5.575 / 5.560 | 4.528 / 4.339 | 0.812 / 0.780 |
| Click Q06 | 800 | 100 / 100 | 5.545 / 5.572 | 4.387 / 4.391 | 0.791 / 0.788 |
| Click Q07 | 200 | 100 / 100 | 3.980 / 4.038 | 4.382 / 4.351 | 1.101 / 1.077 |
| Click Q08 | 908 | 100 / 100 | 7.388 / 7.514 | 4.789 / 4.707 | 0.648 / 0.626 |
| Click Q09 | 908 | 100 / 100 | 10.504 / 10.626 | 5.272 / 5.433 | 0.502 / 0.511 |
| Click Q39 | 1,316 | 110 / 100 | 14.359 / 14.170 | 8.219 / 9.040 | 0.572 / 0.638 |
| Click Q40 | 1,316 | 110 / 100 | 8.454 / 8.088 | 6.710 / 6.755 | 0.794 / 0.835 |
| Click Q41 | 1,048 | 100 / 100 | 7.103 / 7.176 | 9.105 / 9.170 | 1.282 / 1.278 |
| Click Q42 | 800 | 100 / 100 | 5.478 / 5.475 | 9.501 / 9.705 | 1.734 / 1.773 |
| FineWeb Q00 | 1,823 | 116 / 59 | 8.196 / 7.625 | 2.116 / 1.702 | 0.258 / 0.223 |
| FineWeb Q01 | 2,527 | 168 / 86 | 66.703 / 67.131 | 6.077 / 4.559 | 0.091 / 0.068 |
| FineWeb Q02 | 1,823 | 116 / 59 | 12.996 / 14.010 | 2.253 / 1.758 | 0.173 / 0.125 |
| FineWeb Q03 | 1,823 | 116 / 59 | 17.920 / 17.960 | 4.728 / 3.733 | 0.264 / 0.208 |
| FineWeb Q04 | 1,823 | 116 / 59 | 16.736 / 16.841 | 3.608 / 2.946 | 0.216 / 0.175 |
| FineWeb Q05 | 1,823 | 116 / 59 | 16.084 / 15.954 | 3.655 / 3.003 | 0.227 / 0.188 |
| FineWeb Q06 | 1,823 | 116 / 59 | 18.912 / 18.749 | 5.673 / 4.591 | 0.300 / 0.245 |
| FineWeb Q07 | 1,823 | 116 / 59 | 16.322 / 16.307 | 3.654 / 3.055 | 0.224 / 0.187 |
| FineWeb Q08 | 2,527 | 168 / 86 | 21.124 / 19.001 | 3.847 / 2.894 | 0.182 / 0.152 |

The only case with fewer morsels than the 16-worker cap was TPCH at merge 32: 15 morsels, so both
engines used 15 workers. ClickBench is usually one morsel per physical file after either rollup;
Q39 retains 110 morsels at merge 16. FineWeb retains at least 59 morsels. Self-paced wins every
TPCH and FineWeb case and 12 of 16 ClickBench shapes. ClickBench Q01, Q07, Q41, and Q42 remain
slower; Q42 is the largest regression at 1.73-1.77x.

These timings isolate execution-grain effects using real default-writer boundary distributions,
but both engines still execute the restricted in-memory `Struct(Chunked(Flat))` fixture. They are
not compressed-file end-to-end I/O timings.

## Architectural findings

The experiment supports these decisions:

- Keep the immutable source plan separate from mutable per-scan execution state.
- Keep reusable segment and decoded-array resources scan-wide, but demand and transformed arrays
  morsel-local.
- Carry boolean length, true count, and a shared bit-buffer view with every resolved mask.
  Scheduling and sealing can inspect whole-mask scalars and cache exact resource-range counts
  without canonicalizing arrays.
- Make offers descriptive and claim them into immutable input snapshots with leases. Revocation
  remains safe, and workers never access the mutable slot store.
- Transport promotion and revocation updates in addition to offers. An external scheduler can
  retain an offer after its necessity changes.
- Track possible users, joined users, and task leases separately. They answer different lifetime
  questions and allow retirement without a scan-wide graph walk.
- Bound `advance` by cheap transitions and expose work externally. Data-plane work does not belong
  in the reactor transition loop.

The experiment also revealed costs that should not automatically move into a production object:

- `BTreeMap` and `BTreeSet` favor determinism and inspection over hot-path efficiency;
- extensive trace strings and metrics enlarge the execution object and add branches;
- a full materialized boolean demand remains necessary for the evaluator, although sharing the
  all-true instance removes repeated initialization; and
- deduplicating only by `SegmentId` assumes a single segment source. A production key must include
  source identity.

These are acceptable at this highly restricted experiment boundary. They should be measured or
replaced before treating the module as production machinery.

## Speculative I/O admission

Unsealed reads are now visible to the scheduler as candidate work. Reads needed by the next
predicate, and projection reads after demand seals nonempty, are promoted to required work. The
scheduler independently configures predicate and projection candidates as disabled, eager, or
adaptive. Adaptive admission uses the current demand row count multiplied by observed or prior
survival rates for predicates that still have to run.

Admission has a byte budget. File and in-memory segment sources report exact segment sizes when
they know them; wrappers forward the estimate. A source that cannot estimate a segment returns
`None`, and the scheduler charges the configured conservative unknown-read size. Setting that
charge to zero rejects unknown-size speculative reads. Required reads never consume the
speculative budget.

The comparison benchmark accepts these controls:

- `VORTEX_SELF_PACED_SPECULATIVE_IO=off|predicate|projection|adaptive|predicate-eager|projection-eager|eager`
- `VORTEX_SELF_PACED_SPECULATIVE_IO_MAX_BYTES`, defaulting to 64 MiB
- `VORTEX_SELF_PACED_SPECULATIVE_IO_UNKNOWN_BYTES`, defaulting to 8 MiB
- `VORTEX_SELF_PACED_SPECULATIVE_IO_MIN_ROWS`, defaulting to 1 row

Metrics report candidate offers and admissions, known estimated bytes, unknown-size offers,
completed physical bytes, and the completed bytes later proved useful or wasted. Predicate and
projection offer counts are separate; a physical read used by both is counted once in byte
metrics. Trace output records each admitted read's phase, estimate, byte charge, current demand,
and expected surviving rows.

A five-iteration FineWeb follow-up showed why admission must consider projection width and
selectivity, not merely whether expected output is nonzero. On Q06, all 37.1 MB admitted early
were eventually required. On Q01, only 9.6 MB of 19.2 MB admitted early became required;
self-paced returned 49.2 MB from the source versus V1's 30.8 MB. With the default one-row
threshold, adaptive read-ahead improved a few latency-hiding cases but regressed most of the
sub-millisecond suite. This is evidence for a cost/benefit admission score, not for enabling the
current adaptive default broadly.

## Earlier serialized natural-split rollup comparison

These measurements established the correct serialized-file contract but predate the final
executor optimizations. The fair complete-data merge-16 table near the top supersedes their
performance numbers. Merge-32 is retained here only as historical evidence for choosing 16.

A later comparison replaced fixed 128K morsels with morsels formed by merging 16 or 32 consecutive
natural splits from the real benchmark Vortex files. The source catalogs contain 99,997,497
ClickBench rows in 100 files, 59,986,052 SF10 lineitem rows, and 14,868,862 FineWeb rows in 15
files. Split boundaries are query-specific unions over only the physical fields read by that query.
Merging restarts at every file boundary.

One initially collected result was invalid. It applied boundaries from the production Vortex files
to an unrelated coarse in-memory layout. V1 then evaluated several exact splits against the same
coarse segment and repeatedly decoded it, while self-paced retained the segment scan-wide. That
artifact produced implausible 7-14x FineWeb gains. Those measurements are rejected.

The corrected harness writes one complete Vortex byte buffer with a restricted
`Struct<Chunked<Flat<i64>>>` strategy, freezes it, and reopens it through `vortex-file`. The writer
edition permits only `vortex.primitive` and `vortex.chunked`, the two physical array encodings a
Flat segment can contain after slicing these inputs. The strategy rejects nullable roots and every
field type other than non-nullable `i64`, so unsupported encodings and layout strategies cannot
silently enter the fixture. `SourcePlan::try_from_layout` independently validates the reopened
footer, including aligned field-chunk boundaries. A single-chunk field retains its `Chunked`
wrapper.

Both executors scan the same reopened layout and `SegmentSource`, and the harness prints the exact
serialized byte length and a stable byte hash. Each comparison also materializes its query bundle
once and clones that same bundle into both execution paths. V1 receives every natural interval
unchanged. Self-paced alone receives unions of 16 or 32 intervals. Both are pinned with
`taskset -c 0-15`, use concurrency `min(16, morsel count)`, have speculative I/O disabled, validate
ordered output hashes before timing, and alternate execution order for 20 measured iterations.
Fixture construction, serialization, reopening, and rechunking are outside timing.

At merge factor 16, self-paced won 3 of 28 workloads. Its unweighted geometric-mean time ratio was
`1.463`, or 46.3% slower overall:

| Suite | Workloads | Self-paced wins | Geometric mean self-paced/V1 |
| --- | ---: | ---: | ---: |
| ClickBench | 16 | 1 | 1.498 |
| TPC-H | 3 | 2 | 1.093 |
| FineWeb | 9 | 0 | 1.545 |
| Combined | 28 | 3 | 1.463 |

At merge factor 32, self-paced won 2 of 28 and the combined geometric-mean ratio worsened to
`1.706`. The suite ratios were `1.504` for ClickBench, `1.328` for TPC-H, and `2.320` for FineWeb.

TPC-H illustrates the useful tradeoff. Its 458 query-relevant natural intervals become 29 morsels
at merge 16 and 15 at merge 32. Merge 16 measured Q6 at `15.940/14.886 ms` (`0.934x`), Q1 at
`6.864/10.491 ms` (`1.528x`), and the V1-friendly scan at `3.519/3.225 ms` (`0.916x`). Merge 32
caps both engines to 15-way concurrency and regresses Q6 to `1.155x` and Q1 to `2.174x`; reducing
control units did not compensate for lost parallelism and larger cross-chunk assembly work.

ClickBench usually has fewer than 16 relevant intervals per file, so both merge factors stop at one
morsel per file. Merge 16 is effectively tied on Q00 (`0.997x`) and slower on the other 15 shapes.
The largest ratios are Q39 `1.756x`, Q40 `2.315x`, Q41 `2.395x`, and Q42 `2.113x`. For Q39 and
Q40 only, merge 32 reduces 110 morsels to 100 and makes both slower (`1.872x` and `2.449x`).

FineWeb has 1,823 or 2,527 query-relevant natural intervals. Merge 16 creates 116 or 168 morsels
and ranges from `1.036x` on Q02 to `2.460x` on Q06. Merge 32 creates 59 or 86 morsels but is slower
on every query, ranging from `1.587x` to `4.294x`. In this restricted executor, fewer larger morsels
increase the number of physical resource slices assembled by each task and reduce opportunities to
schedule independent morsels. Natural-split rollup therefore needs a byte/work-aware target; a
fixed count of 32 is not a generally better aggregation policy.

## Segment-streamed predicate demand

Adaptive execution now subdivides each outer morsel into demand fragments at the serialized
`Struct(Chunked(Flat))` chunk boundaries. The outer morsel remains the scheduling and output unit;
fragments are internal mask state and do not change the fair merge-16 contract. Each fragment
starts with an all-true demand mask and advances through its predicates independently, while
different fragments can run concurrently.

The read/decode task fuses only the predicate currently requested by a fragment and captures that
fragment's current demand. After I/O, predicate evaluation visits only demanded rows and completion
adopts the result immediately. A selective result therefore exposes the next predicate or
projection read for that segment without waiting for sibling fragments or a complete outer-morsel
mask. A partial cached predicate records exactly which rows it evaluated; later resource reuse is
allowed only when that coverage contains every newly demanded row. Otherwise the decoded array is
reused by a normal predicate task. After every fragment seals, one `MergeDemandFragments`
operation concatenates their masks in row order; normal projection selection then consumes this
single outer-morsel mask.

Resources remain keyed and deduplicated by `SegmentId`. A resource used by both filter and
projection has one read/decode slot, and the projection consumes that same decoded array. Metrics
now report predicate-only, projection-only, and shared resources and bytes, shared decode reuse,
projection reuse of predicate decodes, fragment counts and updates, early projection unblocks,
fused predicates and cache hits, and nanoseconds spent evaluating fused predicates, adopting
fragment masks, and merging the final masks.

On full 15-file FineWeb Q06, pinned to CPUs 0-15, the streamed completion path read 714,536,112
bytes in 10,918 unique segment requests, versus V1's 714,601,752 bytes in 10,931 requests. It
executed 5,461 fused segment predicates across 1,823 fragments and 116 outer morsels. Moving mask
adoption into resource completion reduced reactor transitions from about 33,242 in the first
fragment implementation to 22,320, slightly below the earlier morsel-wide executor's roughly
22,903 transitions. The first three-iteration full-data check nevertheless measured V1 at 21.708
ms and self-paced at 47.708 ms (`2.198x`). Its trace attributed 18.83 ms of aggregate worker CPU to
fused predicate evaluation, 5.23 ms to fragment-demand adoption, and only 0.57 ms to the outside
mask merge. Of 5,461 adoptions, 2,796 did not reduce demand; avoiding `BoolArray` materialization
for those no-ops improved a five-iteration rerun to 22.472 ms for V1 and 46.127 ms for self-paced
(`2.053x`). The remaining Q06 gap is therefore not explained by extra physical I/O, additional
fragment-notification transitions, or the final merge. Segment-granular predicate and mask CPU is
the next optimization target.

A subsequent demand-aware experiment passed the reduced fragment mask into each fused predicate.
On Q06, 3,638 later predicate applications received only 24,957 demanded rows and skipped
29,689,351 row applications. Aggregate predicate CPU fell from about 18.8 ms to 10.9 ms. Despite
that useful work reduction, the final five-iteration comparison measured V1 at 22.240 ms and
self-paced at 48.695 ms (`2.190x`), slower than the all-row fused version. The saved worker CPU did
not shorten the critical path enough to offset mask publication and serialized orchestration.

This is primarily a plan-execution ownership issue, not predicate semantics that belong in the
global scheduler. `Execution` represents fragment demand, resource dependencies, cache coverage,
and readiness. The scheduler should admit any ready task subject to CPU, I/O, and byte budgets.
Today `run_self_paced_concurrent` keeps the complete mutable `Execution` on one async coordinator:
that one loop drains completions, advances every morsel, chooses tasks, claims work, adopts masks,
and queues outputs. The worker pool only evaluates claimed tasks. This made resource deduplication,
leases, cancellation, and tracing deterministic without locks, but it also serializes thousands of
small state transitions. A production design should shard plan execution by morsel (or a small
group of morsels), publish resource completions to the owning shards, and retain only admission and
global byte accounting in the shared scheduler.

The [implementation handover](self-paced-plan-exec-handover.md) records the exact current state and
next work. The [experimental learning ledger](self-paced-plan-exec-learnings.md) preserves less
certain observations and hypotheses separately from the measured findings in this report.

## What remains unknown

The current results do not establish performance for compressed production encodings, unaligned
field chunks, nullable or non-`i64` arrays, arbitrary expressions, dynamic filters, object-store
latency, realistic byte-budget backpressure, stealing, or multi-source segment identity. They also
do not measure time to first batch or peak memory in the final real-data sweep.

The next useful experiment is a production-shaped scan prototype that preserves the proven
contracts while replacing deterministic maps, trace strings, and fixed experiment operations with
the real plan and scheduler interfaces. Its gate should include equal output, physical I/O,
first-batch latency, peak resident memory, and CPU occupancy, with 128K retained as one point in a
morsel-size sweep rather than a default.
