<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn design history

[Current framework](README.md). PR and issue states below are the recorded 2026-09-21 snapshot. The
[current contracts](contracts.md) take precedence over historical proposals and tracker text.

The history explains the strict function boundary, nullable execution, sink contracts, and
compiler-sensitive loops. It also records abandoned proposals and comparison migrations. The tables
retain the PR and issue evidence behind those decisions.

## Chronological table

Dates are the merge date, or the close date for PRs that were closed without merging. "Stack" means
the PR merged into another feature branch rather than into `develop`, and its content reached
`develop` through a later PR.

| PR | Date | Author | Title | Decision in one line | Status |
| --- | --- | --- | --- | --- | --- |
| [#8930](https://github.com/vortex-data/vortex/pull/8930) | 2026-07-27 | connortsui20 | Rename scalar function strictness contract | Define `is_strict` (any null input forces a null output) as the contract that later lets shared null logic be lifted out. | Merged |
| [#9135](https://github.com/vortex-data/vortex/pull/9135) | 2026-08-05 | connortsui20 | Standardize scalar function array construction | Proposed removing bespoke constructors on tensor and geo functions in favor of the shared factory API. | Closed, not merged |
| [#9136](https://github.com/vortex-data/vortex/pull/9136) | 2026-08-05 | connortsui20 | Add scalar function performance baselines | Land benchmark names on `develop` first so CodSpeed can compare later migrations. | Merged |
| [#9138](https://github.com/vortex-data/vortex/pull/9138) | 2026-08-05 | connortsui20 | Add a Normalized encoding for tensor columns | Move the norm-split representation from a scalar function into an array encoding that owns its invariant. | Merged (later removed by #9767) |
| [#9245](https://github.com/vortex-data/vortex/pull/9245) | 2026-08-06 | connortsui20 | Review the RowFn prototype, merge develop, and apply the review findings | Clean up the prototype without changing the author-facing API; gate on unchanged optimized IR. | Merged into `ct/row-fn` |
| [#9215](https://github.com/vortex-data/vortex/pull/9215) | 2026-08-07 | HarukiMoriarty | refactor(vortex-geo): generalize scalar function execution | Build a bespoke shared geo execution layer for constants, null propagation, and all-null batches (the layer RowFn later replaced). | Merged |
| [#9298](https://github.com/vortex-data/vortex/pull/9298) | 2026-08-09 | connortsui20 | [DO NOT MERGE] Focused RowFn take_filter CodSpeed ablation | Show the framework alone is neutral on `take_filter` and the numeric migration causes the 16% regression. | Closed (experiment) |
| [#9203](https://github.com/vortex-data/vortex/pull/9203) | 2026-08-10 | HarukiMoriarty | feat(vortex-spatial): add area scalar function | Add a geo function on the pre-RowFn columnar path (context for the later migration). | Merged |
| [#9319](https://github.com/vortex-data/vortex/pull/9319) | 2026-08-11 | connortsui20 | `RowFn` framework and machinery | First attempt to land the whole framework in one PR; replaced by the split stack. | Closed, not merged |
| [#9320](https://github.com/vortex-data/vortex/pull/9320) | 2026-08-11 | connortsui20 | `RowFn` binary numeric `ScalarFn` rewrite | First numeric migration; discussion established that CodSpeed simulation and local wall time disagree. | Closed, not merged |
| [#9358](https://github.com/vortex-data/vortex/pull/9358) | 2026-08-11 | connortsui20 | Harden indexed lane sources | Drop the `Copy` bound on `IndexedSource::Item` so RowFn reuses the lane kernels instead of adding a second kernel abstraction. | Merged |
| [#9386](https://github.com/vortex-data/vortex/pull/9386) | 2026-08-14 | connortsui20 | Define `RowFn` and `RowVisitor` | Land the author-facing traits, tuple adapters, planning visitor, and the `unstable_row_fns` feature boundary. | Merged |
| [#9353](https://github.com/vortex-data/vortex/pull/9353) | 2026-08-18 | connortsui20 | Implement RowFn row execution | Land owned and sink row loops, prepared and deferred execution, and the `ViewLen` contract. | Merged |
| [#9468](https://github.com/vortex-data/vortex/pull/9468) | 2026-08-19 | connortsui20 | Execute owned RowFn outputs over valid rows | First landing (on the stack) of `Default` placeholders for owned valid-row execution. | Merged into stack |
| [#9496](https://github.com/vortex-data/vortex/pull/9496) | 2026-08-20 | connortsui20 | Simplify `RowFn` execution contracts | Rename to positive `INFALLIBLE`, delete the shared `RowExecution` result type, return arrays directly. | Merged |
| [#9450](https://github.com/vortex-data/vortex/pull/9450) | 2026-08-20 | connortsui20 | Implement `RowFn` batch execution | Connect row loops to the vtable: constants, strict validity, dense execution, sink valid-row execution, output validation. | Merged |
| [#9500](https://github.com/vortex-data/vortex/pull/9500) | 2026-08-20 | connortsui20 | Execute owned RowFn outputs over valid rows | Add `Default` to `OutputElement`; skipped rows get placeholders that are masked before return. | Merged |
| [#9511](https://github.com/vortex-data/vortex/pull/9511) | 2026-08-20 | connortsui20 | Replace `ScalarFnVTable::is_fallible` with `is_infallible` | Flip polarity so the conservative answer is the default across the vtable. | Merged |
| [#9518](https://github.com/vortex-data/vortex/pull/9518) | 2026-08-20 | connortsui20 | Increase binary benchmark workloads | Make workloads at least 16,384 rows so framework overhead is not exaggerated. | Merged |
| [#9351](https://github.com/vortex-data/vortex/pull/9351) | 2026-08-20 | connortsui20 | Automate RowFn benchmark comparisons | Prototype a pinned, alternating, cross-revision benchmark runner. | Closed, not merged |
| [#9469](https://github.com/vortex-data/vortex/pull/9469) | 2026-08-21 | connortsui20 | Support nullary RowFn execution | Run nullary kernels once for the row count, bypassing validity handling. | Merged |
| [#9345](https://github.com/vortex-data/vortex/pull/9345) | 2026-08-21 | connortsui20 | Execute primitive numeric operators with RowFn | Make checked add, subtract, multiply, and integer division the first production users. | Merged into stack (landed via #9517) |
| [#9517](https://github.com/vortex-data/vortex/pull/9517) | 2026-08-21 | connortsui20 | Use dense retry for nullable row functions | Add `DenseWithRetry`: evaluate all rows, retry valid rows only when reduced failure evidence reports an error. | Merged |
| [#9534](https://github.com/vortex-data/vortex/pull/9534) | 2026-08-21 | connortsui20 | Carry partial RowFn validity with MaskValues | Resolve the validity mask once and pass `MaskValues` through valid-row executors. | Merged |
| [#9346](https://github.com/vortex-data/vortex/pull/9346) | 2026-08-21 | connortsui20 | Execute primitive comparisons with RowFn | First comparison migration; closed because comparison kernels regressed under RowFn. | Closed, not merged |
| [#9349](https://github.com/vortex-data/vortex/pull/9349) | 2026-08-21 | connortsui20 | Execute spatial distance with RowFn | First spatial distance migration; superseded by #9616. | Closed, not merged |
| [#9350](https://github.com/vortex-data/vortex/pull/9350) | 2026-08-21 | connortsui20 | Execute spatial predicates with RowFn | First predicate migration with prepared constant bounding boxes; superseded by #9623. | Closed, not merged |
| [#9255](https://github.com/vortex-data/vortex/pull/9255) | 2026-08-21 | connortsui20 | [DO NOT MERGE] Experimental `RowFn` | The whole prototype branch, kept open for benchmarking, then closed. | Closed (experiment) |
| [#9542](https://github.com/vortex-data/vortex/pull/9542) | 2026-08-22 | connortsui20 | Extend MaskValuesRef to filter execution | Thread the shared mask handle through filter execution. | Merged |
| [#9581](https://github.com/vortex-data/vortex/pull/9581) | 2026-08-25 | connortsui20 | Let a row dispatch declare a runtime output dtype | Separate the logical output dtype (`with_output_dtype`) from the physical storage dtype of the sink. | Merged |
| [#9585](https://github.com/vortex-data/vortex/pull/9585) | 2026-08-25 | connortsui20 | Add physical parameters to OutputSink | Add `OutputSink::Params` and `FixedSizeListSink` with a runtime width. | Merged |
| [#9547](https://github.com/vortex-data/vortex/pull/9547) | 2026-08-25 | connortsui20 | Use RowFn for primitive numeric comparisons | Second comparison attempt on an "associated output" kernel API with runtime AVX2/AVX-512 dispatch. | Closed, not merged |
| [#9548](https://github.com/vortex-data/vortex/pull/9548) | 2026-08-25 | connortsui20 | Add associated outputs to RowFn kernels | Alternative `RowKernel` design with associated output storage; not adopted. | Closed, not merged |
| [#9616](https://github.com/vortex-data/vortex/pull/9616) | 2026-08-25 | HarukiMoriarty | refactor(vortex-spatial): execute distance with RowFn | Land `GeometryRow` and decouple row infallibility from decoder infallibility. | Merged |
| [#9618](https://github.com/vortex-data/vortex/pull/9618) | 2026-08-26 | connortsui20 | Clarify `RowFn` infallibility documentation | State that `INFALLIBLE` covers every supported call, including decoders that can reject legal input. | Merged |
| [#9620](https://github.com/vortex-data/vortex/pull/9620) | 2026-08-26 | connortsui20 | Build packed boolean output | Add `OutputElement::build_from` so Boolean output packs bits directly instead of via `Vec<bool>`. | Merged |
| [#9521](https://github.com/vortex-data/vortex/pull/9521) | 2026-08-26 | connortsui20 | Add filter-and-scatter RowFn execution | Add the generic correctness fallback for inputs that cannot decode null payloads. | Merged (later replaced by #9645) |
| [#9628](https://github.com/vortex-data/vortex/pull/9628) | 2026-08-26 | connortsui20 | Map constant RowFn inputs directly | Read a constant or a row through one indexed source so packed output works with constants. | Merged |
| [#9623](https://github.com/vortex-data/vortex/pull/9623) | 2026-08-26 | HarukiMoriarty | refactor(vortex-spatial): execute geo kernels with RowFn | Migrate area, contains, intersects, convex hull, length; delete the bespoke geo execution layer. | Merged |
| [#9644](https://github.com/vortex-data/vortex/pull/9644) | 2026-08-26 | connortsui20 | Separate RowFn decode and semantic infallibility | Proposed a function-wide `DECODE_INFALLIBLE`; closed once filter-and-scatter made it unnecessary. | Closed, not merged |
| [#9629](https://github.com/vortex-data/vortex/pull/9629) | 2026-08-27 | connortsui20 | Use indexed sources for fallible constant rows | Remove the separate `LaneZip` branch for constants in fallible owned execution. | Merged |
| [#9663](https://github.com/vortex-data/vortex/pull/9663) | 2026-08-27 | connortsui20 | Benchmark deferred Boolean RowFn output | Add the baseline before the packed deferred Boolean path lands. | Merged |
| [#9626](https://github.com/vortex-data/vortex/pull/9626) | 2026-08-29 | connortsui20 | Pack Boolean RowFn output directly | Add `visit_bool`, `visit_deferred_bool`, `visit_prepared_deferred_bool` with a const `MULTIVERSIONED` flag. | Merged |
| [#9680](https://github.com/vortex-data/vortex/pull/9680) | 2026-08-29 | connortsui20 | Decode RowFn constants directly | Add `InputElement::Constant` and `decode_constant` so constants are not one-row decoded columns. | Merged |
| [#9696](https://github.com/vortex-data/vortex/pull/9696) | 2026-08-29 | connortsui20 | Trim constant-operand cases from scalar function benchmarks | Cut 37 noisy constant-operand benchmark cases to 10. | Merged |
| [#9587](https://github.com/vortex-data/vortex/pull/9587) | 2026-08-31 | connortsui20 | Compare primitive values with RowFn | Third comparison attempt on `visit_bool`; merged only into a stack branch, never reached `develop`. | Merged into stack (abandoned) |
| [#9694](https://github.com/vortex-data/vortex/pull/9694) | 2026-09-01 | connortsui20 | Attach RowFn validity directly to canonical output | Proposed skipping the lazy `vortex.mask` for canonical output; rejected as papering over an abstraction gap. | Closed, not merged |
| [#9715](https://github.com/vortex-data/vortex/pull/9715) | 2026-09-02 | connortsui20 | Add `Utf8` types for `RowFn` | Add `Utf8Column`, `Utf8View`, and the UTF-8 sink types. | Merged |
| [#9736](https://github.com/vortex-data/vortex/pull/9736) | 2026-09-03 | connortsui20 | Benchmark RowFn filtered fallback | Add a forced-fallback baseline to measure #9645 against. | Merged |
| [#9703](https://github.com/vortex-data/vortex/pull/9703) | 2026-09-04 | connortsui20 | Compare primitive values with RowFn | Rebase of #9587 onto `develop`; closed without merging. | Closed, not merged |
| [#9347](https://github.com/vortex-data/vortex/pull/9347) | 2026-09-04 | connortsui20 | Execute tensor L2 norm with RowFn | First tensor migration with the proposed `reduce_encoded` hook; superseded by #9768. | Closed, not merged |
| [#9348](https://github.com/vortex-data/vortex/pull/9348) | 2026-09-04 | connortsui20 | Execute tensor product functions with RowFn | First product migration with prepared constant norms; superseded by #9769. | Closed, not merged |
| [#9767](https://github.com/vortex-data/vortex/pull/9767) | 2026-09-08 | connortsui20 | Replace normalized encoding with explicit normalization | Remove the `Normalized` encoding and every encoding-specific shortcut, removing the need for `reduce_encoded`. | Merged |
| [#9645](https://github.com/vortex-data/vortex/pull/9645) | 2026-09-08 | robert3005 | Replace filter-and-scatter RowFn fallback with filtered valid-row execution | Filter inputs but write into a full-length output with `FillDefault` placeholders; no scatter. | Merged |
| [#9776](https://github.com/vortex-data/vortex/pull/9776) | 2026-09-10 | connortsui20 | Add tensor row inputs for RowFn | Add dense-safe `TensorRow` input element. | Merged |
| [#9874](https://github.com/vortex-data/vortex/pull/9874) | 2026-09-14 | robert3005 | Remove unnecessary slicing from RowFn execute_all_constant | Small cleanup of the all-constant path. | Merged |
| [#9768](https://github.com/vortex-data/vortex/pull/9768) | 2026-09-15 | connortsui20 | Execute tensor L2 norm with RowFn | Migrate `L2Norm` on `TensorRow`; `L2Normalize` stays handwritten. | Merged |
| [#9769](https://github.com/vortex-data/vortex/pull/9769) | 2026-09-15 | connortsui20 | Execute tensor product functions with RowFn | Migrate inner product and cosine with no reassociation or constant-norm shortcut. | Merged |
| [#9712](https://github.com/vortex-data/vortex/pull/9712) | open | mhk197 | Add `ListTransform` as a (higher order) scalar function | Not a RowFn, but handles the same hidden-payload-behind-null problem for list elements. | Open, draft, stale |
| [#9949](https://github.com/vortex-data/vortex/pull/9949) | open | Ecthlion | Add vortex.is_nan expression with stats-based pruning | First external-contributor RowFn, suggested in #9913. | Open |

## PR summaries

Each entry gives the stated motivation and the design decision. Numbers come from the PR body unless
a comment is named.

**[#8930](https://github.com/vortex-data/vortex/pull/8930) Rename scalar function strictness contract (merged 2026-07-27).** Renamed `is_null_sensitive` to `is_strict` and fixed `Between` and `DynamicComparison`, which were wrongly marked strict. The stated driver was dictionary pushdown soundness: rewriting `f(dict(codes, values), c)` as `dict(codes, f(values, c))` is only sound when nulling any one argument forces a null result. The body also says "a large majority of our scalar functions have strict semantics, and so in the future (hopefully soon) we can lift out a lot of the shared null logic." That sentence is the seed of RowFn.

**[#9135](https://github.com/vortex-data/vortex/pull/9135) Standardize scalar function array construction (closed 2026-08-05).** Removed bespoke `new()` and `try_new_array()` constructors from tensor and geo functions in favor of `ScalarFnFactoryExt::try_new_array`. It was split out of the RowFn work because it was a source break, and it was closed without merging. It noted that `Normalized` could not fit the generic factory, which led to #9138.

**[#9136](https://github.com/vortex-data/vortex/pull/9136) Add scalar function performance baselines (merged 2026-08-05).** Added Divan baselines for `byte_length`, tensor, and geo functions because "Codspeed can only compare a later implementation change when the same benchmark name already exists on `develop`." Each binary uses vendored `mimalloc` because output allocation is inside the timed trace. Review comments set a norm: Joseph Isaacs asked for benchmarks under 1 ms, and Robert Kruszewski described that as "an arbitrary threshold we keep to stop them from decaying over time."

**[#9138](https://github.com/vortex-data/vortex/pull/9138) Add a Normalized encoding for tensor columns (merged 2026-08-05).** Turned `L2Denorm` from a scalar function into a `Normalized` array encoding with `normalized` and `norms` children. The reasoning was that the encoding "owns the physical decomposition and validates the invariant between its children," while "scalar functions remain operations over any well-typed input." This encoding was later deleted by #9767.

**[#9245](https://github.com/vortex-data/vortex/pull/9245) Review the RowFn prototype (merged 2026-08-06 into `ct/row-fn`).** A cleanup pass on the prototype branch. It reports that "the author-facing API is unchanged: every proposal that would have altered it was backed out," and that "the optimized IR of every `visit_prepared_into` monomorph is unchanged." It also notes that "runtime benchmarks were not usable as a gate on this host, where repeated pinned runs of the same binary disagreed by up to 4x," an early sign of the measurement problems that shaped later PRs.

**[#9215](https://github.com/vortex-data/vortex/pull/9215) refactor(vortex-geo): generalize scalar function execution (merged 2026-08-07).** Built a shared unary and binary execution layer inside `vortex-geo` for constant operands, strict null propagation, and all-null short-circuiting. Its motivation ("keeping those responsibilities in each function makes new unary and binary kernels harder to implement correctly") is the same problem statement as the RowFn epic, solved locally for one crate. #9623 deleted this layer once RowFn could carry the spatial functions.

**[#9298](https://github.com/vortex-data/vortex/pull/9298) Focused RowFn take_filter CodSpeed ablation (closed 2026-08-09).** A throwaway PR that ran only the `take_filter` benchmark. The framework-only revision measured 232.542 µs against 233.737 µs for `develop`, while the child commit that moved primitive numeric execution onto RowFn measured 279.491 µs, 16.37% slower. This isolated the regression to the migration rather than the framework.

**[#9319](https://github.com/vortex-data/vortex/pull/9319) and [#9320](https://github.com/vortex-data/vortex/pull/9320) (closed 2026-08-11).** The first attempt to land the framework (4,139 added lines) and the numeric rewrite as two PRs. Both were closed in favor of a finer stack. The #9320 discussion recorded two lasting positions. Connor reported that local wall time did not reproduce the CodSpeed regressions (for example `eq_i64_constant` improved to 0.817x locally while CodSpeed reported 1.221x). Joseph suggested a mode that gives kernels constant-size blocks; Connor replied that batch execution is already inside the machinery and that blocks "would make it harder to define constant handling and null propagation semantics," attributing the regressions to initialization costs (the reason `UninitElementSink` exists) and to cache-line effects from generics.

**[#9358](https://github.com/vortex-data/vortex/pull/9358) Harden indexed lane sources (merged 2026-08-11).** Removed the `Copy` bound from `IndexedSource::Item` in `vortex-compute`. The reason: "RowFn needs this for typed row inputs without adding a second kernel abstraction." RowFn reuses the existing lane-kernel sources rather than owning its own loop primitives.

**[#9386](https://github.com/vortex-data/vortex/pull/9386) Define `RowFn` and `RowVisitor` (merged 2026-08-14).** Landed the traits, tuple adapters, planning visitor, and the blanket `ScalarFnVTable` scaffold behind the `unstable_row_fns` feature, with `execute_rows` returning an error until #9353. Review threads settled several points: default values for the safety flags were removed ("true should be the scary thing"); the planning visitor was renamed `BatchPlanner` after Joseph objected to "plan" and Connor argued "it plans how to execute everything (do we use dense, do we prefilter, how do we handle errors, etc)"; the prepared visit was justified by cases the compiler cannot hoist itself ("a spatial `contains` query where we build an R-tree / index over a constant RHS"); and the `reduce_encoded` hook was pulled out for the tensor PRs.

**[#9353](https://github.com/vortex-data/vortex/pull/9353) Implement RowFn row execution (merged 2026-08-18).** Added the owned-output and sink-writing loops, prepared and deferred execution, `ViewLen`, and decode validation, without yet wiring them to `execute_rows`. Review made `FailureEvidence` a trait with a blanket impl for `Copy + Default + BitOrAssign`, flipped `DECODE_FALLIBLE` to `DECODE_INFALLIBLE` so `false` is the safe default, and moved filter-and-scatter out ("We need to see a benchmark showing that this is faster"). Connor also reported that decomposing a helper caused a regression until a `vortex_ensure!` was split into an `if` with a cold function, an early example of the codegen sensitivity theme.

**[#9496](https://github.com/vortex-data/vortex/pull/9496) Simplify `RowFn` execution contracts (merged 2026-08-20).** Renamed `FALLIBLE` constants to positive `INFALLIBLE`, deleted the shared three-outcome `RowExecution` result type and the first `DenseWithRetry`, and made executors return `VortexResult<ArrayRef>` directly. Connor's own review comment on #9450 explains why: "I am backtracking on the idea that it was good to separate the `RowExecution` type to be distinct from `VortexResult`. Does it really give us anything?" The tuple guard now checks every decoded view length before unchecked access.

**[#9450](https://github.com/vortex-data/vortex/pull/9450) Implement `RowFn` batch execution (merged 2026-08-20).** Connected the loops to the scalar-function adapter with constant handling, strict validity propagation, dense execution, direct valid-row execution for supporting sinks, and output validation. At this layer, "a partially valid signature that cannot execute directly on valid rows panics." Filter-and-scatter, owned valid-row execution, nullary execution, and encoding-aware reductions were each deferred to their own PRs.

**[#9500](https://github.com/vortex-data/vortex/pull/9500) (and [#9468](https://github.com/vortex-data/vortex/pull/9468) on the stack) Execute owned RowFn outputs over valid rows (merged 2026-08-20).** Added `Default` to `OutputElement`, used only to initialize skipped positions, which batch execution masks before returning. This gave owned outputs the same skip-invalid path that sinks already had.

**[#9511](https://github.com/vortex-data/vortex/pull/9511) Replace `ScalarFnVTable::is_fallible` with `is_infallible` (merged 2026-08-20).** Flipped the polarity of the vtable hook so "the conservative answer is the default" and a function opts in to infallibility. It aligned the vtable with the RowFn `INFALLIBLE` constants and renamed the `label_is_fallible` analysis to `label_infallible`.

**[#9518](https://github.com/vortex-data/vortex/pull/9518) Increase binary benchmark workloads (merged 2026-08-20).** "A lot of these benchmarks are way too short, and it meant that small changes in framework overhead looked bigger than they really were." Primitive cases now process at least 16,384 rows and 96 KiB per varying input.

**[#9351](https://github.com/vortex-data/vortex/pull/9351) Automate RowFn benchmark comparisons (closed 2026-08-20).** A `scripts/benchmark-rowfn.sh` prototype with separate build and measure phases, one codegen unit, fat LTO, `target-cpu=native`, alternating process order, and paired medians over seven runs. It was not merged, but its protocol is what the tracking issue later calls the "migration benchmark gate."

**[#9469](https://github.com/vortex-data/vortex/pull/9469) Support nullary RowFn execution (merged 2026-08-21).** Runs a nullary kernel once for the requested row count and bypasses validity handling because there is no input validity to propagate.

**[#9345](https://github.com/vortex-data/vortex/pull/9345) Execute primitive numeric operators with RowFn (merged 2026-08-21 via #9517).** Made primitive arithmetic the first production user. Checked add, subtract, and multiply "reduce compact failure evidence outside their vector loops," integer division "stops at the first failure and writes directly into uninitialized output," and decimal arithmetic stays columnar. It notes that the earlier 4.6x to 7.5x mixed-constant regressions were measured before an output-iterator fix in #9353 and no longer apply.

**[#9517](https://github.com/vortex-data/vortex/pull/9517) Use dense retry for nullable row functions (merged 2026-08-21).** Introduced the `DenseWithRetry` policy because "a strict row function must not expose errors caused only by payloads behind null rows." Deferred, dense-safe, decode-infallible owned kernels execute all rows once; if reduced evidence reports a possible error, a partial mask retries only valid rows. AVX2 IR keeps 32-, 16-, 8-, and 4-lane loops. On the repaired CodSpeed fixtures `add_i64_nullable` moved from a 37.53% regression to a 2% improvement and `mul_i32_nullable` from a 36.17% regression to a 1% improvement.

**[#9534](https://github.com/vortex-data/vortex/pull/9534) and [#9542](https://github.com/vortex-data/vortex/pull/9542) (merged 2026-08-21 and 2026-08-22).** Resolve the materialized `Mask` once and pass `MaskValues` (then `MaskValuesRef`) through the valid-row executors, while keeping array-backed validity lazy on optimistic paths "because resolving it can execute and linearly scan the full mask for the uncommon all-valid and all-null cases."

**[#9346](https://github.com/vortex-data/vortex/pull/9346) Execute primitive comparisons with RowFn (closed 2026-08-21).** Kept the fused x86 path for selected 64-bit comparisons, which stayed about 38% faster on `compare_u64_constant`. A comment with local Rust 1.97.1 fat-LTO numbers showed `u8` and `f32` improving by 23.52% and 15.85%, but `i32`, `f32` equality, and `u64` regressing by 21.92% to 25.88%, and LLVM 22 emitting scalar loops for mixed-constant `i32` and `u8` (8.32x to 8.53x slower). Closing comment: "comparison kernels regress quite a lot under the RowFn machinery so we can just keep the handwritten kernels." Robert asked whether `element_tuple::decode` was the overhead; Connor pointed at 64-bit SIMD trouble plus framework overhead.

**[#9349](https://github.com/vortex-data/vortex/pull/9349) and [#9350](https://github.com/vortex-data/vortex/pull/9350) (closed 2026-08-21).** Connor's first spatial migrations. Distance stayed within 2.7% of `develop` with a 13.9% gain on nullable column-by-constant. Predicates were mixed: most within 8%, two constant-left `contains` cases 16% slower, sparse-null cases 14% to 20% faster. Both were superseded by HarukiMoriarty's #9616 and #9623 once #9521 landed.

**[#9581](https://github.com/vortex-data/vortex/pull/9581) Let a row dispatch declare a runtime output dtype (merged 2026-08-25).** The problem: a function of type `T -> T` over several extension types could not build an output whose dtype depends on its inputs, and the sink had to know both the logical and physical type. The decision: sinks become "physical only" (`storage_dtype()` is static, `OutputSink` loses its `Options` parameter) and `RowVisitor::with_output_dtype` declares the logical dtype, which `finalize_output` relabels after masking nulls. Stated limitations: no runtime sink parameters yet (fixed in #9585) and no validation of storage values for constrained extension types.

**[#9585](https://github.com/vortex-data/vortex/pull/9585) Add physical parameters to OutputSink (merged 2026-08-25).** Added `OutputSink::Params` flowing through sink visits, validation, and allocation, and a `FixedSizeListSink` whose width comes from options at runtime.

**[#9547](https://github.com/vortex-data/vortex/pull/9547) and [#9548](https://github.com/vortex-data/vortex/pull/9548) (closed 2026-08-25).** A second comparison attempt on an alternative `RowKernel` API with associated output storage and runtime AVX2 or AVX-512 dispatch writing masks straight into packed Boolean storage. Same-artifact comparisons ranged from 7.75% faster to 12.27% slower than the columnar path under AVX2, and CodSpeed reported a 7.59% aggregate regression. Both PRs were closed; the packed Boolean idea returned as `build_from` and `visit_bool` (#9620, #9626) without the separate kernel API.

**[#9616](https://github.com/vortex-data/vortex/pull/9616) refactor(vortex-spatial): execute distance with RowFn (merged 2026-08-25).** Added `GeometryRow` and moved distance onto RowFn while "keep[ing] row-operation fallibility independent from input-decoder fallibility." It removed a contract check that had coupled `RowFn::INFALLIBLE` to `InputElement::DECODE_INFALLIBLE`.

**[#9618](https://github.com/vortex-data/vortex/pull/9618) Clarify `RowFn` infallibility documentation (merged 2026-08-26).** Documented that `INFALLIBLE` "covers every supported call, including selected decoders that can reject legal input data."

**[#9620](https://github.com/vortex-data/vortex/pull/9620) Build packed boolean output (merged 2026-08-26).** Added `OutputElement::build_from` with a Boolean override that collects into a packed bitmap through `BitBuffer::collect_bool`. With rustc 1.98.0 and LLVM 22.1.8 on AVX2, "the varying-input loop now emits vector compares, `vpmovmskb`, and a direct packed-word store," where `develop` wrote unpacked bytes first.

**[#9521](https://github.com/vortex-data/vortex/pull/9521) Add filter-and-scatter RowFn execution (merged 2026-08-26).** The generic correctness fallback: filter every input to jointly valid rows, run the dense kernel, scatter back. The body rejects two alternatives. Calling the general scalar API per valid row "would repeat array execution and scalar construction in the hot loop." A prepared selected-row view prototype won for sparse validity (7.67 µs vs 12.12 µs on 90% null polygon contains) but was "45% slower than filter/scatter on the dense polygon control and 10% slower on the ordinary nullable polygon case," so it stays an optional fast path. Connor stressed in comments that "we literally cannot support spatial with rowfn without this."

**[#9628](https://github.com/vortex-data/vortex/pull/9628), [#9629](https://github.com/vortex-data/vortex/pull/9629), [#9680](https://github.com/vortex-data/vortex/pull/9680) (merged 2026-08-26 to 2026-08-29).** The constant-handling trilogy. #9628 added one indexed source that reads either row `index` or the single decoded constant, so packed Boolean output works with a constant operand. #9629 extended it to fallible owned execution, removing the separate `LaneZip` branch and shrinking optimized IR by 27% (deferred Boolean) and 19% (deferred `i64`). #9680 added `InputElement::Constant` and `decode_constant` so a constant is a native value rather than a one-row decoded column.

**[#9623](https://github.com/vortex-data/vortex/pull/9623) refactor(vortex-spatial): execute geo kernels with RowFn (merged 2026-08-26).** Migrated area, contains, intersects, convex hull, and length, deleting 865 lines of bespoke geometry execution. `GeometryRow` "does not manufacture placeholder geometries for null rows or expand the shared `InputElement` API"; a partially valid batch declines direct execution and uses #9521.

**[#9644](https://github.com/vortex-data/vortex/pull/9644) Separate RowFn decode and semantic infallibility (closed 2026-08-26).** Proposed a required `RowFn::DECODE_INFALLIBLE`. The body argues `ScalarFnVTable::is_infallible` "is overloaded": speculation needs both totality over valid inputs and the guarantee that "evaluating data not referenced by the logical expression cannot introduce an execution error." Closed the same day: "we dont actually need this now that we have the filter/scatter execute from #9521, but I still think there is something wrong here (not specific to RowFn, but to ScalarFnVTable in general)."

**[#9626](https://github.com/vortex-data/vortex/pull/9626) Pack Boolean RowFn output directly (merged 2026-08-29).** Added `visit_bool`, `visit_deferred_bool`, and `visit_prepared_deferred_bool`, each with a const `MULTIVERSIONED` flag: `false` inlines the collector, `true` selects one for the current CPU at runtime. It cites LLVM issue 219235 for a remaining AVX-512 mask conversion.

**[#9663](https://github.com/vortex-data/vortex/pull/9663) and [#9696](https://github.com/vortex-data/vortex/pull/9696) (merged 2026-08-27 and 2026-08-29).** Benchmark hygiene: a deferred Boolean baseline before #9626, and trimming 37 constant-operand cases to 10 because they "measure the same path again and crowd out the per-row against per-row shape the kernels are tuned for."

**[#9587](https://github.com/vortex-data/vortex/pull/9587), [#9694](https://github.com/vortex-data/vortex/pull/9694), [#9703](https://github.com/vortex-data/vortex/pull/9703) (2026-08-31 to 2026-09-04).** The third comparison attempt, on multiversioned `visit_bool`, with #9694 attaching validity directly to canonical output to skip a lazy `vortex.mask`. Robert: "generally not a fan of optimisations like these." Connor: "I think it shows a gap in our abstraction, ideally this is not required for this optimization?" #9694 was closed, #9587 only ever merged into a stack branch, and its rebase #9703 was closed. The local code today has no `PrimitiveCompare` and no `visit_bool` users in `vortex-array/src/scalar_fn/fns`; comparisons remain on the handwritten path.

**[#9715](https://github.com/vortex-data/vortex/pull/9715) Add `Utf8` types for `RowFn` (merged 2026-09-02).** Added `Utf8Column` and `Utf8View` as input elements plus the UTF-8 sink types, upstreamed from a SpiralDB repository. It leaves open "if we want to ship a string library with Vortex."

**[#9736](https://github.com/vortex-data/vortex/pull/9736) Benchmark RowFn filtered fallback (merged 2026-09-03).** A `FilteredI64` element with `DENSE_SAFE = false` that declines null-tolerant decoding, so partially valid inputs are forced onto the fallback. Connor asked to merge this before #9645 "to see what the actual difference is."

**[#9767](https://github.com/vortex-data/vortex/pull/9767) Replace normalized encoding with explicit normalization (merged 2026-09-08).** Removed the `Normalized` encoding and "encoding-specific L2 norm, inner-product, and cosine shortcuts so these operations always derive results from decoded coordinates," adding an `L2Normalize` scalar function. This removed the motivating case for the `reduce_encoded` hook, which never landed.

**[#9645](https://github.com/vortex-data/vortex/pull/9645) Replace filter-and-scatter with filtered valid-row execution (merged 2026-09-08).** Robert's alternative: filter inputs to valid rows, but write each result at its original index into a full-length output, so no scatter is needed. The debate is in the review thread. Connor objected that requiring a skipped-row initializer breaks sinks that cannot build a placeholder ("the spatial types are a counterexample"). Robert answered "take still MUST initialise memory to something" and added `FillDefault` on the sink's `Rows` type. CodSpeed showed the forced fallback on `filtered_owned_i64[NineNullsInTen]` going from 36.9 µs to 13.1 µs.

**[#9776](https://github.com/vortex-data/vortex/pull/9776), [#9768](https://github.com/vortex-data/vortex/pull/9768), [#9769](https://github.com/vortex-data/vortex/pull/9769) (merged 2026-09-10 to 2026-09-15).** The second tensor migration. `TensorRow` is dense-safe "because tensor float kernels can consume the stored payload of null rows while RowFn owns output validity." `L2Norm`, inner product, and cosine moved over with "no encoded reduction, reassociation, or constant-norm shortcut" and tests for bitwise agreement, a deliberate retreat from the prepared constant-norm optimization in #9348.

**[#9874](https://github.com/vortex-data/vortex/pull/9874) (merged 2026-09-14)** removed unneeded slicing in `execute_all_constant`. **[#9949](https://github.com/vortex-data/vortex/pull/9949) (open)** adds `vortex.is_nan` as a strict RowFn from an external contributor, following Robert's note in #9913 that "IS NAN is a trivial RowFn." **[#9712](https://github.com/vortex-data/vortex/pull/9712) (open, stale)** is not a RowFn, but its `list_transform` design filters hidden physical elements behind null lists before evaluating the lambda, the same hazard RowFn handles for rows.

## Design themes

### Theme 1: Strictness as the boundary of the framework

The framework only supports strict functions. The epic
([#9128](https://github.com/vortex-data/vortex/issues/9128)) says: "we will focus solely on strict
functions, as the semantics around non-strict functions are complicated enough that it's probably
not worth extending this already-somewhat-complicated API further." It also rules out "columnar or
zero-copy kernels (not, list_length), kernels with state shared across rows (`like`), or
heterogeneous variadic kernels."

This rests on [#8930](https://github.com/vortex-data/vortex/pull/8930), which defined `is_strict`
for dictionary pushdown soundness and observed that most functions are strict. The consequence
inside RowFn is that output validity is always the conjunction of input validities, and, in the
words of the module docs, "a `RowFn` cannot produce null from valid inputs." That excludes
`list_sum` and `variant_get`. The tracking issue
[#9129](https://github.com/vortex-data/vortex/issues/9129) considered `Option<T>` outputs and
decided against them for now: "while it can fit into the current abstraction by returning an
`Option<T>`, this is terrible for performance."

### Theme 2: Dense execution over null rows versus valid-only execution

This is the largest theme. The question is what to do with a partially valid batch. Evaluating every
row (dense) keeps the loop branch-free and vectorizable but reads payloads behind nulls and can trip
errors that a correct strict function must not expose. Evaluating only valid rows (valid-only) is
always safe but needs per-row indexing and a way to fill skipped output slots.

The framework answers with a planning-time policy, `RowPolicy`, with three variants in today's
`visitor/plan.rs`: `Dense`, `DenseWithRetry`, and `ValidOnly`. Selection is a `const fn` over the
argument tuple: `Dense` (or `DenseWithRetry` for deferred kernels) requires `Args::DENSE_SAFE &&
Args::DECODE_INFALLIBLE`; everything else is `ValidOnly`.

How it evolved:

- [#9450](https://github.com/vortex-data/vortex/pull/9450) landed dense execution and direct valid-row execution for sinks that can initialize skipped rows. Anything else panicked.
- [#9500](https://github.com/vortex-data/vortex/pull/9500) gave owned outputs a `Default` placeholder so they could also skip invalid rows.
- [#9517](https://github.com/vortex-data/vortex/pull/9517) added `DenseWithRetry` (see Theme 3). The justification: "a strict row function must not expose errors caused only by payloads behind null rows," but checking validity inside the loop costs vector width, so run dense first and retry valid rows only if the failure evidence says so.
- [#9521](https://github.com/vortex-data/vortex/pull/9521) added filter-and-scatter as the generic correctness fallback, "not treated as an optimization," for inputs like `GeometryRow` that "cannot create a safe view over unspecified payloads behind nulls." A prepared selected-row view was prototyped, won on sparse nulls, and lost on dense data, so it was kept only as a possible optional fast path.
- [#9645](https://github.com/vortex-data/vortex/pull/9645) replaced the scatter with filtered valid-row execution: inputs are filtered, but the kernel writes to the original index in a full-length output whose skipped rows are filled by `FillDefault`. The review thread records the disagreement over whether every sink can produce a placeholder; the resolution was to require `FillDefault` on the sink's `Rows` type rather than a hand-written initializer. Today's `batch/execute/filtered.rs` documents that "batch execution tries direct skip-invalid execution first and filters only when a required input capability is unavailable."

The ordering rule from [#9130](https://github.com/vortex-data/vortex/issues/9130) still holds: try
direct valid-row execution on the original inputs, and only then filter. The issue also says there
is "no survivor threshold" for choosing between them.

### Theme 3: Deferred failure evidence versus immediate errors

Checked arithmetic on every row cannot construct a `VortexError` inside a vector loop without losing
vectorization. The framework's answer is `visit_deferred`: the kernel returns `(value,
FailureEvidence)` per row, the executor OR-reduces the evidence in a loop-local accumulator, and one
rich error is built after the loop.

Two rules were stated in [#9130](https://github.com/vortex-data/vortex/issues/9130): "the concrete
failure evidence must not be wider than the output value" because "a wider reduction lowers the
vector width and can make checked arithmetic much slower," and "the accumulator must stay local to
the loop because sink storage adds a loop-carried memory dependency." That second rule is why
deferred evidence exists only for owned outputs and not for sinks.

Immediate errors remain for cases where stopping early is cheaper.
[#9345](https://github.com/vortex-data/vortex/pull/9345) put integer division on
`VortexResult<InitializedElement>` with `UninitElementSink`: "Division is already scalar and
expensive. An immediate check can stop at the first failure. Uninitialized dense output avoids
filling every slot before the row loop."

Deferred evidence and dense execution combine into `DenseWithRetry`
([#9517](https://github.com/vortex-data/vortex/pull/9517)). Today's `execute/retry.rs` returns a
`DenseAttempt` that is either accepted `Values` or a `DeferredError` that "batch execution must
resolve input validity before deciding whether this error is observable." Decode, validation, and
allocation errors are never deferred.

### Theme 4: Infallibility contracts

Three flags describe fallibility, and their meaning was argued over several PRs.

- `RowFn::INFALLIBLE`: the semantic row operation cannot fail for any valid, well-typed input across every dispatch. [#9496](https://github.com/vortex-data/vortex/pull/9496) made the constant positive (`INFALLIBLE` rather than `FALLIBLE`) so the unsafe answer is the one you must write down. Review on [#9386](https://github.com/vortex-data/vortex/pull/9386) had already established "true should be the scary thing."
- `InputElement::DECODE_INFALLIBLE`: the decoder cannot reject legal input. Review on [#9353](https://github.com/vortex-data/vortex/pull/9353) flipped it from `DECODE_FALLIBLE` for the same reason.
- `ScalarFnVTable::is_infallible`: [#9511](https://github.com/vortex-data/vortex/pull/9511) flipped the vtable hook so functions opt in to infallibility. Consumers include dictionary pushdown and `ExpressionTakeRule`.

The unresolved part is whether the vtable flag conflates two ideas.
[#9644](https://github.com/vortex-data/vortex/pull/9644) argued it does: speculation requires
totality and also that "evaluating data not referenced by the logical expression cannot introduce an
execution error, including unused dictionary values and arbitrary payloads behind nulls." It
proposed a function-wide `RowFn::DECODE_INFALLIBLE` and was closed once filter-and-scatter removed
the immediate need, with the note that "there is something wrong here ... [with] ScalarFnVTable in
general."

The historical wording changed during this work. At the pinned source revision,
`RowFn::INFALLIBLE` covers the semantic row operation, while `InputElement::DECODE_INFALLIBLE`
covers decoding independently. The [current contracts](contracts.md) document that boundary.
The API tracker still listed a function-wide infallibility decision in the recorded snapshot.
That tracker item does not mean the two existing flags are currently coupled.

### Theme 5: The planning pass and `ensure_reproduced_by`

`RowFn::dispatch` is called twice with different visitors. The first call, with `BatchPlanner`,
validates dtypes and records a `BatchPlan`: the storage dtype the output capability builds, the
declared logical output dtype, and the `RowPolicy`. The second call, with an executing visitor, must
reproduce the same plan. `BatchPlan::ensure_reproduced_by` in today's `visitor/plan.rs` checks
policy, storage dtype, and output label with `vortex_ensure_eq!`, and is called from every executing
visit method in `visitor/execute.rs` and `visitor/retry.rs`.

The design was debated in the [#9386](https://github.com/vortex-data/vortex/pull/9386) review.
Joseph asked for a name other than "plan" ("Its more like a Execution Receiver"). Connor first
described it as "almost like a type checking phase, but it also verifies the flags," then argued it
"does do more than verification, it plans how to execute everything (do we use dense, do we
prefilter, how do we handle errors, etc)," and renamed it `BatchPlanner`. The planner also hosts the
compile-time contract asserts (`assert_owned_visit_contract` and friends) that check a dispatch's
flags against the function's declared `INFALLIBLE`.

The reason for a separate planning visitor, as far as the sources state it, is that `return_dtype`
and the nullable-row policy must be known before any input is decoded, and there is "no argument or
return witness" on the trait ([#9129](https://github.com/vortex-data/vortex/issues/9129)); the
concrete tuple chosen by `dispatch` is the only source of that information. This is a reading of the
tracking issue, not a quoted rationale.

### Theme 6: Sinks versus owned outputs, and logical versus physical output types

Two output forms coexist. `OutputElement` returns one owned value per row (`visit`,
`visit_prepared`, `visit_deferred`). `OutputSink` receives a row handle and writes in place
(`visit_into`, `visit_prepared_into`). [#9353](https://github.com/vortex-data/vortex/pull/9353)
landed both. A review exchange records why sinks exist for strings: "dont want to be allocating a
`String` on every single row."

The sink's `SinkResult` can be `()`, `InitializedElement`, `VortexResult<()>`, or
`VortexResult<InitializedElement>`, and its `WriteToken` must match the sink's.
[#9130](https://github.com/vortex-data/vortex/issues/9130) explains this "keeps the visitor methods
safe" by placing "the per-row unsafe operation inside the uninitialized-output closure."
`OutputSink` itself is an `unsafe trait` because the executor trusts its row handles and `finish`.

[#9581](https://github.com/vortex-data/vortex/pull/9581) then split the output type in two. Sinks
describe physical storage only; the function declares the logical dtype through `with_output_dtype`,
which the batch executor applies as a label after masking. Today's `validate_output_label` restricts
the label to the storage dtype itself or an extension dtype over exactly that storage, "which
restricts it to wrapping" and "trusts the row kernel to produce values that satisfy" any storage
constraint. [#9585](https://github.com/vortex-data/vortex/pull/9585) added `OutputSink::Params` for
runtime physical values such as a fixed-size-list width.

An alternative, a `RowKernel` trait with associated output storage
([#9548](https://github.com/vortex-data/vortex/pull/9548)), was tried for comparisons and closed;
the packed Boolean benefit it sought was delivered instead through `OutputElement::build_from` and
the `visit_bool` family (Theme 8).

### Theme 7: Constants and prepared visits

Constant operands are a first-class shape. The epic's pipeline lists "compute anything that is
constant/static for the batch" as its own step, and the `visit_prepared*` methods take a prepare
closure that receives each constant as `Some(value)` and each varying column as `None`. Connor's
justification in the [#9386](https://github.com/vortex-data/vortex/pull/9386) review is that the
compiler can hoist a scalar multiply by a constant on its own, but not "a tensor inner product where
RHS is constant, or ... a spatial `contains` query where we build an R-tree / index over a constant
RHS."

Three later PRs removed constant overhead from the loop.
[#9628](https://github.com/vortex-data/vortex/pull/9628) made "the row-or-constant choice ...
visible to the optimizer before traversal." [#9629](https://github.com/vortex-data/vortex/pull/9629)
removed the separate `LaneZip` branch for constants in fallible owned execution.
[#9680](https://github.com/vortex-data/vortex/pull/9680) added `decode_constant` so constants are
native values, not one-row columns.

One prepared optimization was later given up.
[#9348](https://github.com/vortex-data/vortex/pull/9348) computed a batch-constant norm once for
cosine similarity; the version that merged,
[#9769](https://github.com/vortex-data/vortex/pull/9769), has "no encoded reduction, reassociation,
or constant-norm shortcut" and tests bitwise agreement with materialized rows. The PR does not state
the reason in so many words; the emphasis on bitwise agreement suggests numeric reproducibility was
preferred over the constant shortcut. That is an inference.

### Theme 8: Packed Boolean output and `MULTIVERSIONED`

Boolean output was a special case from the start because a canonical `BoolArray` is bit-packed.
[#9620](https://github.com/vortex-data/vortex/pull/9620) added `OutputElement::build_from` with a
Boolean override that packs through `BitBuffer::collect_bool`, and showed the loop emitting "vector
compares, `vpmovmskb`, and a direct packed-word store."
[#9626](https://github.com/vortex-data/vortex/pull/9626) added dedicated `visit_bool`,
`visit_deferred_bool`, and `visit_prepared_deferred_bool` methods with a const `MULTIVERSIONED`
flag. Today's `execute/packed_bool.rs` selects `BitBuffer::collect_bool_multiversioned` when the
flag is `true` and the inlined `collect_bool` otherwise. The PR body notes LLVM normally removes the
chunk-local `[bool; 64]` and cites LLVM issue 219235 for a remaining AVX-512 mask conversion.

The irony of this theme is that the main intended consumer, primitive comparisons, never landed
(Theme 10). The packed paths are exercised today by the framework's own benchmarks and by the
spatial predicates.

### Theme 9: Monomorphization, generated code, and compile time

The authors treated optimized IR and assembly as first-class evidence.
[#9245](https://github.com/vortex-data/vortex/pull/9245) gated on "the optimized IR of every
`visit_prepared_into` monomorph is unchanged."
[#9345](https://github.com/vortex-data/vortex/pull/9345),
[#9517](https://github.com/vortex-data/vortex/pull/9517),
[#9620](https://github.com/vortex-data/vortex/pull/9620), and
[#9629](https://github.com/vortex-data/vortex/pull/9629) each report what the vector loop looks like
under a named toolchain and the "16-CGU, no-LTO benchmark profile."
[#9130](https://github.com/vortex-data/vortex/issues/9130) records this as policy: "keep
generated-code checks for deferred arithmetic alongside wall-clock benchmarks."

Codegen fragility shows up repeatedly in review. On
[#9353](https://github.com/vortex-data/vortex/pull/9353), decomposing a helper caused a regression
until a `vortex_ensure!` was split into an `if` with a cold function; a length check was kept
because it "helps LLVM know that the loop bound is correct"; and on
[#9386](https://github.com/vortex-data/vortex/pull/9386) a `row_count` hint was defended because
removing it caused a regression on Rust 1.91. Today's `retry.rs` still carries the comment "Keep
this dense row loop separate from `execute_owned`. Factoring their shared state into a helper
changes LLVM's optimized dense kernel even when the helper is inlined." In the
[#9320](https://github.com/vortex-data/vortex/pull/9320) discussion Connor attributed remaining
regressions to "cache line boundaries (something fit in 64-byte cache line but not anymore with all
these generics) which are unfixable."

Compile time as such is not discussed in any PR or issue found by this search. The monomorphization
cost appears only indirectly: `RowFn::dispatch` is generic over the visitor, every `dispatch` runs
for both the planner and each executor, and the `unstable_row_fns` feature gate keeps the module out
of default builds. Whether compile time was a consideration is therefore unknown from the record.

### Theme 10: Benchmark evidence requirements and the comparison retreat

The project set an unusually explicit evidence bar for migrations, and it was applied against its
own proposals.

- Baselines first ([#9136](https://github.com/vortex-data/vortex/pull/9136)), because CodSpeed compares by benchmark name. Reviewers held benchmarks to under about 1 ms.
- Workloads large enough that framework overhead is not exaggerated ([#9518](https://github.com/vortex-data/vortex/pull/9518)), and constant-operand cases trimmed as noisy ([#9696](https://github.com/vortex-data/vortex/pull/9696)).
- Ablation to attribute a regression to the right layer ([#9298](https://github.com/vortex-data/vortex/pull/9298)).
- Native measurement over simulation. [#9130](https://github.com/vortex-data/vortex/issues/9130): "CodSpeed CPU simulation measures a different cost model and does not replace native evidence." The protocol was "pinned, alternating local x86 measurements and generated-code inspection," prototyped in [#9351](https://github.com/vortex-data/vortex/pull/9351).
- A forced-fallback baseline before judging a fallback change ([#9736](https://github.com/vortex-data/vortex/pull/9736) before [#9645](https://github.com/vortex-data/vortex/pull/9645)).
- "Keep a columnar fallback when RowFn produces slower native code, as the primitive comparison path does" ([#9130](https://github.com/vortex-data/vortex/issues/9130)).

That last rule was applied three times to primitive comparisons
([#9346](https://github.com/vortex-data/vortex/pull/9346),
[#9547](https://github.com/vortex-data/vortex/pull/9547),
[#9587](https://github.com/vortex-data/vortex/pull/9587)/[#9703](https://github.com/vortex-data/vortex/pull/9703)).
Each attempt produced vectorized IR and mixed native numbers (from about 8% faster to about 12%
slower than the fused handwritten kernels), and each was closed. Comparisons stay on the handwritten
path today. The epic's prior-art note says the same about Velox: "Both frameworks keep columnar
fallbacks where the row form loses, notably primitive comparisons."

### Theme 11: Extension points, downstream crates, and encoding-aware shortcuts

The epic's goal was that "crates such as `vortex-tensor` and `vortex-spatial` [can] add row
representations without changing `vortex-array`." The decided extension points are `InputElement`,
`OutputElement`, and `OutputSink`; `ElementTuple` and `SinkResult` are sealed
([#9129](https://github.com/vortex-data/vortex/issues/9129)). `InputElement` is an `unsafe trait`
because "the shared indexed loop performs one length check before unchecked row reads." Downstream
elements now exist: `GeometryRow` and `PolygonSink` in `vortex-spatial`
([#9616](https://github.com/vortex-data/vortex/pull/9616),
[#9623](https://github.com/vortex-data/vortex/pull/9623)), `TensorRow` in `vortex-tensor`
([#9776](https://github.com/vortex-data/vortex/pull/9776)), and `Utf8Column`/`Utf8View` in
`vortex-array` ([#9715](https://github.com/vortex-data/vortex/pull/9715)). Both downstream crates
enable `unstable_row_fns` in their `Cargo.toml`.

Encoding-aware shortcuts had a shorter life. The epic wanted to "preserve encoding-aware shortcuts
for functions that have a better answer for a specific encoding," and
[#9347](https://github.com/vortex-data/vortex/pull/9347) proposed a `reduce_encoded` hook called
before decoding. It was pulled out of [#9386](https://github.com/vortex-data/vortex/pull/9386) at
review ("Did anyone actually use this?"), and once
[#9767](https://github.com/vortex-data/vortex/pull/9767) deleted the `Normalized` encoding and its
shortcuts, the hook had no user. There is no `reduce_encoded` in the codebase today, and
[#9129](https://github.com/vortex-data/vortex/issues/9129) still lists "add and stabilize
`reduce_encoded`" as an open step. Robert's comment on
[#7881](https://github.com/vortex-data/vortex/issues/7881) points at a different mechanism, the
optimizer kernel registry in `vortex-array/src/optimizer/kernels.rs`, which runs "BEFORE vtable
functions are executed."

## Open questions and future-direction issues

| Issue | State | What it asks |
| --- | --- | --- |
| [#9128](https://github.com/vortex-data/vortex/issues/9128) Epic: Row-oriented scalar functions | Open | Unresolved: "Are nullable outputs from otherwise valid inputs required before the first version is complete?" (needed for `list_sum`, `variant_get`). A comment floats GPU execution as a possible future target for the same row abstraction. |
| [#9129](https://github.com/vortex-data/vortex/issues/9129) Tracking: Define the `RowFn` API | Open | Remaining steps: stabilize the traits with representative users, conformance tests for extension types, decide the function-wide semantic vs decode infallibility contract, add `reduce_encoded`, decide on nullable row outputs, preserve serialized metadata compatibility, document when to use RowFn versus a custom vtable, stabilize the public API. |
| [#9130](https://github.com/vortex-data/vortex/issues/9130) Tracking: Execute `RowFn` | Closed 2026-09-01 | Follow-ups listed at close: measure skip-invalid against filtered execution "for each new element with substantial decode work," and keep generated-code checks alongside wall-clock benchmarks. |
| [#7881](https://github.com/vortex-data/vortex/issues/7881) Tracking: Pluggable Scalar Functions | Open (May 2026) | Predates RowFn. Asks how a third-party extension type (TurboQuant) can plug in specialized execution and a different return dtype; comments note the optimizer kernel registry and that `return_dtype` would need a session-aware overload. RowFn does not answer this. |
| [#9913](https://github.com/vortex-data/vortex/issues/9913) Expressions missing that could improve pushdown | Open | Suggests `IS NAN` as "a trivial RowFn that mostly needs pruning rewrite rules"; #9949 implements it. First sign of RowFn being the default recommendation for new simple functions. |
| [#2272](https://github.com/vortex-data/vortex/issues/2272) Add `Expr::return_nullability` | Open (2025) | Short-circuit result nullability of an expression. Related to strictness propagation but not linked to RowFn in the record. |
| [#1440](https://github.com/vortex-data/vortex/issues/1440) Shortcircuit compare operations using stats | Open (2024) | Stats-based short-circuiting of compares. Not linked to RowFn in the record. |
| [#3454](https://github.com/vortex-data/vortex/issues/3454) Move canonical impl of compute fn alongside the fn definition | Closed 2026-04 | Code organization for compute functions; only loosely related. |

Design questions raised in PRs but not tracked in an issue:

- Whether `ScalarFnVTable::is_infallible` should be split into totality and speculation safety ([#9644](https://github.com/vortex-data/vortex/pull/9644) closing comment).
- Whether the lazy `vortex.mask` step after canonical output is an abstraction gap ([#9694](https://github.com/vortex-data/vortex/pull/9694) discussion).
- Whether Vortex should ship a string library for `Utf8` kernels ([#9715](https://github.com/vortex-data/vortex/pull/9715)).
- Whether a prepared selected-row view should be added as an optional fast path for sparse validity ([#9521](https://github.com/vortex-data/vortex/pull/9521), [#9623](https://github.com/vortex-data/vortex/pull/9623)).
- Whether constant-size blocks should be offered to kernels in addition to rows (Joseph's suggestion on [#9320](https://github.com/vortex-data/vortex/pull/9320), declined at the time).

No issue or PR found by this search discusses selection vectors, definedness, demand masks, `CASE
WHEN` integration with RowFn, or moving RowFn into a standalone crate. Those topics are absent from
the GitHub record as of 2026-09-21, at least under those names.

## Search coverage

All searches were scoped to `vortex-data/vortex`. PR searches used `repo:vortex-data/vortex <terms>`
sorted by creation date; issue searches used the semantic issue search scoped to the repository, so
exact-keyword counts may differ from the GitHub web UI.

| Kind | Query | Results | Notes |
| --- | --- | --- | --- |
| PR | `RowFn` | 51 | All 6 pages read; the core list. |
| PR | `"row function"` | 28 | 3 pages read; most 2024 to 2025 hits unrelated. Added #8930, #9712. |
| PR | `RowVisitor` | 5 | All in the core list. |
| PR | `OutputSink` | 4 | Added #9581, #9585. |
| PR | `visit_deferred` | 1 | #9663. |
| PR | `DenseWithRetry` | 2 | #9496, #9548. |
| PR | `unstable_row_fns` | 3 | #9353, #9386, #9450. |
| PR | `InputElement` | 8 | All in the core list. |
| PR | `visit_prepared` | 0 | |
| PR | `ElementTuple` | 0 | |
| PR | `Utf8Sink` | 0 | `Utf8` types found via the `RowFn` query (#9715). |
| PR | `FixedSizeListSink` | 2 | #9581, #9585. |
| PR | `with_output_dtype` | 1 | #9581. |
| PR | `"row kernel"` | 46 | First page was entirely `vortex-row` byte-encoder PRs (unrelated); not paged further. |
| PR | `is_infallible` | 3 | #9496, #9511, #9644. |
| PR | `strict scalar function` | 44 | Read newest page only; relevant hits already in the core list. |
| PR | `spatial row function` | 19 | 2 pages read. Added context PRs #9203, #9215. |
| PR | `tensor row function` | 22 | Read newest page. Added #9138, #9767. |
| PR | `numeric row` | 40 | Read newest page; relevant hits already in the core list. |
| PR | `reduce_encoded` | 0 | Confirms the hook never landed as its own PR. |
| PR | `RowKernel` | 2 | #9547, #9548. |
| Issue | `RowFn` | 2 | #9129, #9130. |
| Issue | `row function scalar function per-row kernel` | 1 | #9128. |
| Issue | `selection vector` | 0 | |
| Issue | `definedness` | 0 | |
| Issue | `demand mask` | 1 | #439 (2024, unrelated compute `mask` function). |
| Issue | `short circuit short-circuit evaluation` | 2 | #1440, #2272. |
| Issue | `case when expression` | 1 | #1782 (closed 2025, projection expressions; unrelated). |
| Issue | `standalone separate crate scalar functions kernels` | 3 | #9128, #7881, #3454. |
| Issue | `generic type system dtype dispatch` | 5 | Old `DType` issues from 2024 to 2025; unrelated. |
| Issue | `null propagation strict function nullable inputs valid rows` | 2 | Unrelated. |
| Issue | `scalar function framework future design` | 1 | #7881 (first attempt hit a rate limit and was retried). |
| Issue | `nullable output from valid inputs list_sum variant_get optional row output` | 0 | |

In addition, PR bodies were read for 60 PRs, issue bodies for 8 issues, issue comments for #9128 and
#7881, PR comments for #9320, #9517, #9511, #9587, #9255, #9645, #9644, #9694, #9521, #9346, and
#9136 (bot comments skipped), and review threads for #9386, #9353, #9450, #9581, #9626, and #9645.
PRs #9542, #9696, #9767, #9138, and issue #9913 were found through cross-references in the tracking
issues rather than search. Local files read: `vortex-array/src/scalar_fn/unstable/row/mod.rs`,
`visitor/plan.rs`, `execute/retry.rs`, `execute/packed_bool.rs`, `batch/execute/filtered.rs`,
`types/fill.rs`, and a grep for `impl RowFn for` across the workspace.
