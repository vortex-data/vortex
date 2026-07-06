# `list_sum` — Per-List Element Summation as a Vortex Expression

*Goal: support `list_sum(l)` — the sum of each row's list elements, one output row per input
row — as a first-class Vortex scalar function, with DuckDB and DataFusion pushdown.*

---

## 1. Motivation

`list_sum` is the simplest member of the list-aggregate family (`list_min`, `list_max`,
`list_avg`, `list_count`, …) and the natural second list scalar function after `list_length`:

1. **It is the canonical "reduce each list" primitive.** Every mainstream engine ships it
   (DuckDB `list_sum`, DataFusion `array_sum` — new June 2026, ClickHouse `arraySum`, Polars
   `list.sum()`), and it is the shape user queries actually take over nested telemetry/feature
   columns: `SELECT id, list_sum(scores) FROM t`.
2. **Vortex already has the hard part.** The grouped-aggregate machinery
   (`GroupedAccumulator`, `aggregate_fn/accumulator_grouped.rs`) computes one aggregate per
   list using encoding-specialized kernels — a `ListArray` *is* a `GroupedArray*` where each
   list is a group. `list_sum` is mostly a thin scalar-fn adapter over machinery that exists
   and is tested today, plus one semantic fix-up (§4.2).
3. **It opens the pushdown door for the whole family.** The vtable shape, typing rule,
   semantics decisions, and converter arms designed here transfer directly to
   `list_min`/`list_max`/`list_avg` (§8) — deciding the semantics questions once, carefully,
   is most of the value of this doc.

## 2. Survey: how other engines implement it

Verified against engine sources (and DuckDB v1.5.3 empirically); full citations in the appendix.

|  | DuckDB `list_sum` | DataFusion `array_sum` | ClickHouse `arraySum` | Polars `list.sum()` |
| --- | --- | --- | --- | --- |
| NULL list row | NULL | NULL | n/a (non-Nullable arrays) | NULL |
| Empty list | **NULL** (SQL `SUM` of zero rows) | **NULL** | **0** (zero-init accumulator) | **0** |
| NULL elements | skipped | skipped; all-NULL list → NULL | unsupported (`ILLEGAL_COLUMN`; users pre-map with lambdas) | skipped |
| `INT32` list result type | `HUGEINT` (int128) | `Float64` (always) | `Int64` | `Int32` (no promotion) |
| NaN in a float list | poisons to NaN (IEEE) | poisons to NaN (IEEE) | poisons to NaN | poisons to NaN |
| Strategy | per-row **aggregate state** + batched scattered update; reuses any aggregate | per-row loop over offsets into a flat child pre-cast to `Float64` | pure segmented reduction, one linear pass | segmented reduction fast path (no nulls) / per-row amortized `Series::sum` |

### DuckDB — generic "list × any aggregate"

`list_sum` is not C++ at all: it is a SQL macro `(l) AS list_aggr(l, 'sum')`
(`src/catalog/default/default_functions.cpp`), like `list_min`/`list_max`/`list_avg`/….
`list_aggregate` (`extension/core_functions/scalar/list/list_aggregates.cpp`) binds the named
aggregate from the catalog at bind time, stashes the `BoundAggregateExpression` in bind data,
and at execution allocates one aggregate state per row, batching element→state updates through
a selection vector 2048 at a time — grouped hash-aggregation mechanics reused verbatim. All
semantics fall out of SQL `SUM`: empty list → state never updated → NULL; null elements
skipped; every integer width sums into `HUGEINT` so overflow is practically unreachable (and
`sum(HUGEINT)` itself throws on wrap). Design goal was "any aggregate over a list" (the same
file binds histogram aggregates for `list_distinct`), not a fast sum.

### DataFusion — specialized kernel, brand new

`array_sum` (alias `list_sum`) landed in `datafusion-functions-nested` in **June 2026**
(PR [#22542], merged 2026-06-08, closing [#7214] from 2023). Signature coercion casts *every*
element type to `Float64` up front and `return_type` is unconditionally `Float64` — much
cruder than DuckDB. The kernel is a per-row loop over `value_offsets()` into the flat
`Float64` child with a per-element validity branch; a row with no valid elements (empty or
all-NULL) appends null. The code comment pins the semantics to "PostgreSQL array_sum, DuckDB
list_sum, Spark aggregate": SQL `SUM` over the unnested elements.

**Version gate (verified):** Vortex pins DataFusion **54** (workspace `Cargo.toml:140`), and
the `datafusion-functions-nested-54.0.0` crate Vortex builds against has **no `array_sum`**
module. The DataFusion converter arm (§6.2) is therefore designed here but lands only when
Vortex bumps past the release containing #22542.

### ClickHouse — the branchless extreme

`arrayAggregation.cpp` implements `arrayMin/Max/Sum/Average/Product` as one segmented
reduction template: a single linear pass over the flat data column between offsets,
`Int64`/`UInt64`/`Float64` widening, empty array → value-initialized accumulator → **0**.
Nullable children are rejected outright (`ILLEGAL_COLUMN`) — the price of a branchless,
vectorizable inner loop.

### Polars — dispatch on child nullability

`sum_mean.rs`: when the inner values have no nulls, a textbook segmented reduction
(`offsets.windows(2).map(|w| sum_slice(&values[w[0]..w[1]]))`) with the *list* validity cloned
onto the output; otherwise a per-row amortized `Series::sum` slow path. Empty list → 0.
Minimal promotion (`Int32` sums into `Int32`, wrap-on-overflow risk).

### Takeaways

1. **Two viable architectures**: DuckDB's generic aggregate-reuse vs a specialized segmented
   reduction (ClickHouse/Polars/DataFusion). Vortex's `GroupedAccumulator` is exactly the
   DuckDB architecture — already built, already encoding-specialized (§3).
2. **The semantics are not settled across engines**, and the fork that matters for Vortex is
   empty/all-null lists: NULL (SQL camp: DuckDB, DataFusion) vs 0 (kernel camp: ClickHouse,
   Polars). *Both engines Vortex integrates with are in the SQL camp* — pushdown must not
   change query results, which decides §4.2.
3. **NaN**: every surveyed engine lets NaN poison the sum (IEEE float addition). Vortex's own
   default is NaN-*skipping* (§3) — a second, quieter divergence the options design must carry.

## 3. Where Vortex is today

- **Scalar functions** live in `vortex-array/src/scalar_fn/`; an expression node is
  `Expression { scalar_fn: ScalarFnRef, children }` (`expr/expression.rs:27`) over a
  `ScalarFnVTable` impl (`scalar_fn/vtable.rs:39`). `list_length`
  (`scalar_fn/fns/list_length.rs`) is the closest template: unary, `EmptyOptions`,
  `execute_until::<AnyList>` (`:176`) to reach `List`/`ListView`/`FixedSizeList`, answer from
  structure, validity passthrough (`:101`), registered in `scalar_fn/session.rs:72`, builder
  in `expr/exprs.rs:765`.
- **`Sum` is an aggregate function** (`aggregate_fn/fns/sum/mod.rs`), not a compute function.
  Its typing rule (`Sum::return_dtype`, `sum/mod.rs:101`) is the widening `list_sum` should
  inherit verbatim: `Bool → U64?`, unsigned → `U64?`, signed → `I64?`, floats → `F64?`,
  decimal → `min(precision + 10, MAX_PRECISION)?` (the Spark/DataFusion heuristic, cited in
  the source) — **always `Nullable`, because overflow yields a null sum value**, not an error.
- **The grouped path is the implementation lever.** `GroupedArray`
  (`aggregate_fn/accumulator_grouped.rs:45`) is `ListView | FixedSizeList` with
  `elements()`, `group_ranges(ctx)` (`(offset, size)` per group, `:74`), `group_validity(ctx)`
  (`:82`). `GroupedAccumulator::<Sum>::try_new(Sum, opts, elem_dtype)` +
  `DynGroupedAccumulator::accumulate_list(&list_array, ctx)` + `finish()` produces **one sum
  per list**: `accumulate_list` canonicalizes the input list internally and dispatches through
  the session's grouped kernel registry
  (`aggregate_fn/session.rs:198`, `register_grouped_encoding_kernel` at `:210`), where
  `PrimitiveGroupedSumEncodingKernel` (`sum/grouped.rs:26`) → `try_grouped_sum` (`:48`) →
  `grouped_sum` (`:70`) reuses the scalar primitive-sum reductions so per-group overflow/NaN
  semantics match scalar `sum` exactly. The element validity mask is materialized **once**
  (`execute_mask`) and sliced per group via its contiguous valid runs (`sum_masked_group`,
  `:131`) — no per-element `is_valid` calls, per the repo's hot-loop guidance.
- **Current grouped-sum semantics** (verified in `collect_sums`, `sum/grouped.rs:106`):
  null group → null; overflow → null; **empty group → `Some(0)`** (accumulator default, never
  updated); **all-null-elements group → `Some(0)`** (`AllOr::None` ⇒ no update). So Vortex
  today sits in the ClickHouse/Polars camp, opposite its two integration targets.
- **NaN handling is an option, not a constant**: `NumericalAggregateOpts { skip_nans }`
  (`aggregate_fn/vtable.rs:183`), default `skip_nans()` (`:193`, NaNs treated as missing);
  `include_nans()` (`:201`) gives the engines' poison-to-NaN behavior. It already has proto
  serialization (`:206`).
- Nothing outside `aggregate_fn/` consumes `GroupedArray` yet — `list_sum` would be the first
  cross-module consumer. Everything needed is already `pub` and re-exported
  (`aggregate_fn/mod.rs:18`), and `scalar_fn` is the same crate regardless.

## 4. Design

### 4.1 Vtable shape

```rust
pub struct ListSum;

impl ScalarFnVTable for ListSum {
    type Options = NumericalAggregateOpts;   // { skip_nans: bool }
    // id: "vortex.list.sum"
    // arity: Exact(1), child 0 = "input"
    ...
}
```

**Options = `NumericalAggregateOpts`, not `EmptyOptions`.** Rejected alternative: a stateless
fn hard-coding one NaN behavior. Carrying the option costs nothing (it is a bool with existing
proto serde, satisfying every `Options` bound) and is load-bearing for §6: Vortex-native use
defaults to `skip_nans()` — **identical to Vortex's own `sum()`**, the least-surprise anchor
for Vortex users — while the DuckDB/DataFusion converter arms bind `include_nans()` so pushed
expressions reproduce engine float semantics exactly. Hard-coding either behavior forces a
wrong answer on the other consumer.

### 4.2 Semantics — the decision table

The one real decision in this design. Proposed semantics, per input row:

| input row | output | who agrees | who disagrees |
| --- | --- | --- | --- |
| `null` list | `null` | everyone | — |
| `[]` | **`null`** | DuckDB, DataFusion, SQL `SUM` | ClickHouse, Polars, **Vortex grouped sum today** |
| `[null, null]` | **`null`** | DuckDB, DataFusion | Vortex grouped sum today (`0`) |
| `[1, null, 3]` | `4` (nulls skipped) | everyone | — |
| overflow (`i64`/`u64`/decimal) | `null` value | Vortex `sum()` | DuckDB (int128 + throw) — see §7.2 |
| `[1.0, NaN]` | option: NaN (`include_nans`) / `1.0` (`skip_nans`, default) | engines ↔ `include_nans`; Vortex `sum()` ↔ `skip_nans` | — |

**Recommendation: SQL semantics (`NULL` on empty and all-null), baked into the function.**
The argument is pushdown correctness, which is non-negotiable: both engines Vortex converts
expressions for return `NULL`, and a pushed-down `list_sum` that returns `0` silently changes
query results. The alternatives are worse:

- *Match Vortex grouped-sum today (`0`) and fix up in the converters* — the fix-up expression
  ("null iff no valid element") is not expressible without `list_count`-style helpers that
  don't exist, and every future converter must remember it. Rejected.
- *Make it an option like NaN* — an option whose two values disagree on whether `[]` is `0` or
  `NULL` invites silent misuse, and no consumer was identified that wants `0`. Rejected; can
  be added compatibly later if one appears.

Note this decision is **about `list_sum` only** — it does not change `GroupedAccumulator` or
grouped-sum kernels (used by group-by machinery with their own contract). The fix-up lives in
`list_sum::execute` (§4.4). See §7.1 for an internal inconsistency this surfaces.

### 4.3 Typing

```text
return_dtype(options, args):
    match args[0]:
        List(elem, _) | FixedSizeList(elem, _, _)
            => Sum::return_dtype(elem)   // aggregate_fn/fns/sum/mod.rs:101 widening
               (bail if None: unsupported element type)
        other => bail "list_sum expects a list input"
```

Delegating to `Sum::return_dtype` is deliberate: one widening rule in one place, and
`list_sum`'s per-list results are bit-identical to `sum()` over each list slice. The result is
**always `Nullable`** regardless of list/element nullability — forced independently by three
paths: null lists, overflow-to-null, and `NULL`-on-empty (§4.2). Supported element types are
exactly `Sum`'s: bool, primitives, decimal; anything else (strings, nested lists, structs)
bails at typing, matching DuckDB (binder error for `SUM(VARCHAR)`).

### 4.4 Execution

```rust
fn execute(&self, options, args, ctx) -> VortexResult<ArrayRef> {
    let input = args.get(0)?;

    // 1. Constant input: sum the scalar's one list once, wrap in ConstantArray
    //    (same shape as list_length.rs:84).
    if let Some(scalar) = input.as_constant() { ... }

    // 2. One grouped-sum pass: one sum per list, via the encoding-specialized
    //    grouped kernels. accumulate_list canonicalizes to ListView/FSL internally.
    let mut acc = GroupedAccumulator::try_new(Sum, (*options).clone(), elem_dtype)?;
    acc.accumulate_list(&input, ctx)?;
    let sums = acc.finish()?;          // null groups / overflow already null here

    // 3. SQL-semantics fix-up (§4.2): null out groups with zero *valid* elements.
    //    - element mask all-valid ⇒ empty ⇔ size == 0, read from group_ranges;
    //    - otherwise one materialized element Mask, count_range per (offset, size)
    //      gap (SIMD popcount) — never per-element is_valid in the loop.
    //    Final validity = sums.validity ∧ (valid_element_count > 0).
}
```

Step 3 is the only genuinely new code. It follows the same bulk-materialization discipline as
`grouped_sum` itself: `execute_mask` once, `Mask`/`BitBuffer::count_range` per group range
(works unchanged for `ListView`'s overlapping/reordered ranges), and the all-valid fast path
reduces it to a `size == 0` scan over `group_ranges`. An alternative — threading an
`empty_as_null` flag *into* the grouped kernels to avoid the second pass — touches the
aggregate layer's contract for every kernel and future aggregate; deferred until profiling
shows the fix-up pass matters (§8.3).

**Rejected execution alternative: a bespoke segmented reduction in `list_sum`**
(ClickHouse-style loop over offsets + flat elements). It would duplicate overflow/NaN/decimal
semantics that `grouped_sum` already gets right by construction, forfeit the per-encoding
grouped-kernel registry (a future dictionary-elements kernel accelerates `list_sum` for
free), and past experience (per-element validity in hot loops) says the hand-rolled version
starts slower. If profiling later shows the accumulator path lagging a fused loop, that is a
grouped-kernel improvement, not a `list_sum` rewrite.

**Vtable flags:**

- `validity` → `Ok(None)` (computed): unlike `list_length`, output validity is **not** the
  child's — a valid empty list yields a null sum. This is the one structural difference from
  the `list_length` template and deserves a comment at the impl site.
- `is_null_sensitive` → `false`: masking list rows commutes with per-list summation, same
  argument as `list_length`.
- `is_fallible` → `false`: overflow is a null *value* (Sum's contract), unsupported dtypes are
  rejected at typing, and execution has no other failure mode.
- Display/`fmt_sql`: `vortex.list.sum($)`, following `vortex.list.length($)`; suffix the
  non-default option when set (e.g. `vortex.list.sum($, include_nans)`) so plans distinguish
  the two behaviors.

### 4.5 Algebraic rewrites

Kept deliberately minimal for v1:

1. **Constant folding** — the constant fast path in `execute` plus the normal deferred
   machinery; no `simplify_untyped` rule needed.
2. `list_sum(list_transform(l, f))` **does not fuse** in v1; noted as future work (§8.4)
   alongside the `list_transform` design (sibling worktree `list-transform`, its §4.4), whose
   fusion rules this would compose with.

There is no `list_length`-style "structure only" shortcut: `list_sum` inherently needs the
elements. That asymmetry is what makes the layout story (§5) different from `list_length`'s.

### 4.6 Public API

```rust
/// Sum the elements of each list in `input`, one result per list.
///
/// Follows SQL `SUM` semantics per list: null and empty lists, and lists whose
/// elements are all null, yield null; null elements are skipped; integer and
/// decimal overflow yields null. The result dtype follows `sum`'s widening and
/// is always nullable. By default NaN values are skipped, matching `sum`;
/// see [`NumericalAggregateOpts`] for the NaN-including variant.
pub fn list_sum(input: Expression) -> Expression
```

in `expr/exprs.rs` next to `list_length` (`:765`), defaulting to
`NumericalAggregateOpts::skip_nans()`; a `list_sum_opts(input, opts)` variant (or builder
method) exposes `include_nans` for the converters. Vtable registered in
`scalar_fn/session.rs` (after `:72`) and declared in `scalar_fn/fns/mod.rs`. Python bindings
(`vortex-python/src/expr/mod.rs`) as a follow-up, matching however `list_length` is exposed.

## 5. Layout pushdown

Day one: nothing required — layout readers `.apply()` the expression against materialized
lists, which works immediately.

With the shredded `ListLayout` (elements / offsets / validity sub-layouts), `list_sum(col)`
reads all three children — unlike `list_length`, which reads offsets only — so the near-term
win is *not* IO elision on the list itself but **composition**: `list_sum(list_transform(l,
x -> x.a))` should route element reading to the `a` sub-layout only (the `list_transform`
design's §5 does the routing; `list_sum` just consumes the result). Speculative and
explicitly future: pruning via element zone maps (bounding a list's sum by
`[min·len, max·len]` per chunk) — noted in §8.5, not designed here.

## 6. Engine integration

### 6.1 DuckDB

`vortex-duckdb/src/convert/expr.rs` matches bound functions **by name string**
(`try_from_bound_function`, `:201`; cf. `"array_length"/"len"/"length"` → `build_list_length`
at `:79`). Two wrinkles are specific to `list_sum`:

1. **The name is never `list_sum`.** DuckDB's `list_sum` is a catalog macro expanding to
   `list_aggr(l, 'sum')` *before binding*, so the converter sees the `list_aggregate` family
   (canonical name plus aliases `list_aggr`, `array_aggregate`, `array_aggr`, `aggregate`)
   with the aggregate selected by a constant argument / bind data. The arm must match the
   family and extract the aggregate name — verify at implementation time whether
   vortex-duckdb's expression view exposes the constant `'sum'` child, the bind-data
   `BoundAggregateExpression`, or both. Only `'sum'` converts in v1; any other aggregate
   falls through (DuckDB executes it).
2. **Result type: `HUGEINT` blocks integer pushdown.** DuckDB sums every integer width into
   int128; Vortex has no `i128` primitive (`PType` stops at 64 bits), so there is no honest
   `cast(list_sum(x), HUGEINT-equivalent)`, and the overflow contracts differ anyway (throw
   vs null — §7.2). **v1 pushes down `FLOAT[]`/`DOUBLE[]` only**, where DuckDB's result is
   `DOUBLE` ↔ Vortex `f64?` exactly, overflow is moot, and binding `include_nans()` reproduces
   IEEE NaN propagation. Integer and decimal lists are rejected in `can_push_expression`
   (`:342`) and evaluated by DuckDB — correctness first, coverage later (§8.2).

With §4.2's SQL semantics, empty/all-null/null-element behavior matches with no shims. Add an
slt suite following the existing list_length pushdown tests.

### 6.2 DataFusion (version-gated)

Blocked on a DataFusion upgrade: Vortex pins DF 54 and `array_sum` is absent from
`datafusion-functions-nested` 54.0.0 (verified; upstream PR #22542 merged 2026-06-08).
Recipe when the bump lands, following the `ArrayLength` arm end to end:

- downcast arm for the `ArraySum` UDF in `try_convert_scalar_function`
  (`vortex-datafusion/src/convert/exprs.rs:188`, cf. `try_convert_array_length` `:166`),
  emitting `cast(list_sum_opts(input, include_nans()), Float64)` — DF's return type is
  unconditionally `Float64`, and the cast from Vortex's widened `u64?/i64?/f64?` is lossy in
  exactly the way DF itself is (it pre-casts elements to `Float64`), so results match;
- whitelist entries in **both** gates: `can_scalar_fn_be_pushed_down` (`:612`) and the mirror
  inside `is_convertible_expr` (`:546-550`);
- an slt suite mirroring the list_length pushdown one.

Semantics line up with no further work: NULL on empty/all-null matches §4.2, nulls skipped
matches, NaN matches via `include_nans`. Track the DF upgrade as the enabling dependency.

## 7. Risks and open questions

1. **Vortex-internal inconsistency surfaced by §4.2.** Scalar `sum()`'s own docs say all-null
   float sums are null (`sum/mod.rs:116` comment), while the grouped path provably yields `0`
   for empty/all-null groups (`collect_sums`). `list_sum` sidesteps it with its fix-up pass,
   but the discrepancy between scalar and grouped `Sum` deserves its own issue — pin both
   behaviors with tests during implementation and reconcile there, not here.
2. **Overflow contract divergence.** Vortex nulls on `i64`/`u64`/decimal overflow; DuckDB
   effectively cannot overflow (int128) and throws if it does. Pushing down integer-list sums
   would turn an engine error (or a correct huge value) into a silent null. v1's answer is
   "don't push down integers" (§6.1); revisit only with `i128` support or a checked/widened
   accumulator story.
3. **`list_aggregate` bind-data visibility in vortex-duckdb** (§6.1.1): the converter design
   assumes the aggregate name is recoverable from the bound expression. If it is only in
   opaque bind data, the arm may need a small vortex-duckdb FFI addition — scope that before
   committing to the DuckDB milestone.
4. **`GroupedAccumulator` from `scalar_fn` is a new dependency direction** (first consumer of
   the grouped machinery outside `aggregate_fn`). Same crate, no cycle, but if it feels like
   layering violation during review, the alternative is promoting a
   `grouped_sum(list: &ArrayRef, opts, ctx) -> VortexResult<ArrayRef>` helper into
   `aggregate_fn`'s public surface and keeping `list_sum` ignorant of accumulator internals.
5. **Decimal pushdown** is deferred everywhere: DuckDB widens to precision 38, Vortex to
   `precision + 10` — reconciling casts and overflow-null vs throw needs its own look.

## 8. Future work (in likely order)

1. **`list_min` / `list_max`** — same adapter over `Min`/`Max` grouped kernels; no widening,
   element dtype out, and the `NULL`-on-empty question recurs identically (min/max of nothing
   is NULL in SQL). The §4.2 decision is precedent.
2. **Integer/decimal DuckDB pushdown** — gated on the overflow story (§7.2).
3. **`empty_as_null` inside grouped kernels** — fold §4.4's fix-up pass into the single
   grouped-sum pass if profiling justifies touching the aggregate layer's contract.
4. **Fusion with `list_transform`** — `list_sum(list_transform(l, f))` evaluates `f` over the
   elements child then sums; with the transform's deferred-elements design the composition is
   already lazy, so the remaining win is layout routing (§5), not an expression rewrite.
5. **Zone-map pruning** — `list_sum(x) > c` bounded by per-chunk element min/max × lengths.
   Speculative; note only.
6. **DataFusion arm** when the DF upgrade lands (§6.2) — mechanical by then.

## 9. Implementation checklist (v1)

- [x] `vortex-array/src/scalar_fn/fns/list_sum/mod.rs` — `ListSum` vtable:
      `id "vortex.list.sum"`, `arity Exact(1)`, `child_name "input"`,
      `Options = NumericalAggregateOpts` (+ proto serde reuse), `return_dtype` delegating to
      `Sum::return_dtype` (§4.3), `execute` via `GroupedAccumulator` + validity fix-up (§4.4),
      `validity → None`, `is_null_sensitive = false`, `is_fallible = false`, `fmt_sql`.
- [x] Declare in `scalar_fn/fns/mod.rs`; register in `scalar_fn/session.rs` (after `:72`);
      builders `list_sum` / `list_sum_opts` in `expr/exprs.rs` (near `:765`).
- [x] Tests in `fns/list_sum/mod.rs` mirroring `list_length.rs`'s suite (`rstest`,
      `VortexResult<()>`, `assert_arrays_eq!`): offset widths; null list → null; **empty list
      → null**; **all-null elements → null**; mixed nulls skipped; `i64` overflow →
      null; unsigned widening; NaN under both option values; bool elements; `ListView`
      incl. overlapping views; `FixedSizeList`; sliced + taken inputs; constant input;
      non-list and non-numeric-element inputs bail; `test_display`
      (`"vortex.list.sum($)"`); proto round-trip with both option values. (Decimal elements
      ride the generic grouped fallback, same path the bool test covers.)
- [ ] Also pin current scalar-vs-grouped `Sum` all-null behavior with tests (§7.1) and file
      the reconciliation issue.
- [ ] Bench in `vortex-array/benches/` (fix-up-pass cost: all-valid fast path vs masked
      `count_range` path vs a no-fix-up baseline), per the repo's benchmark-backed-loops rule.
- [ ] DuckDB converter arm + `can_push_expression` gate (float lists only, `include_nans`)
      + slt suite (§6.1).
- [ ] Tracking note/issue for the DataFusion arm gated on the DF ≥ 55 upgrade (§6.2).

---

## Appendix: sources

- DuckDB: `extension/core_functions/scalar/list/list_aggregates.cpp`;
  `src/catalog/default/default_functions.cpp` (`list_sum` macro);
  `extension/core_functions/scalar/list/functions.json` (aliases); PR duckdb/duckdb#3274
  (original `list_aggregate`); semantics verified empirically on DuckDB v1.5.3
  (`list_sum([]::INT[]) → NULL`, `[1,NULL,3] → 4`, `[NULL,NULL] → NULL`,
  `INT[] → HUGEINT`, `FLOAT[] → DOUBLE`, `DECIMAL(4,2)[] → DECIMAL(38,2)`).
- DataFusion: `datafusion/functions-nested/src/array_sum.rs` @ main; PR apache/datafusion#22542
  (merged 2026-06-08); issue apache/datafusion#7214; absence from the pinned crate verified
  against `datafusion-functions-nested-54.0.0` sources.
- ClickHouse: `src/Functions/array/arrayAggregation.cpp`.
- Polars: `crates/polars-ops/src/chunked_array/list/sum_mean.rs`.
- Vortex: `vortex-array/src/scalar_fn/{vtable.rs,fns/list_length.rs}`,
  `vortex-array/src/expr/{expression.rs,exprs.rs,proto.rs}`,
  `vortex-array/src/aggregate_fn/{vtable.rs,accumulator_grouped.rs,session.rs,fns/sum/{mod.rs,grouped.rs,primitive.rs}}`,
  `vortex-duckdb/src/convert/expr.rs`, `vortex-datafusion/src/convert/exprs.rs`;
  sibling design: `.agents/worktrees/list-transform/.../list_transform/design.md`.
