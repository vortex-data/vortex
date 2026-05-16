# V2 ClickBench Follow-Up Plan

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
