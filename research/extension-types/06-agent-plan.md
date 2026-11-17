<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Extension Type Agent Plan

> **For agentic workers:** You implement one task of this plan. Read "Global Constraints", "Agent
> protocol", "Review Focus" and your own task. Ignore the other tasks, except their Interfaces
> blocks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the open PRs of [`05-implementation.md`](05-implementation.md): claims, typing in
each function, statistics, and the independent fixes U1, SF, SN3 and NU.

**Architecture:** Each extension type lists its claims (`Eq`, `Ord`) on `ExtVTable`. Each function
checks the claim that it needs in its own `return_dtype`, and runs on storage only for calls that
typing accepted. Code that builds no node checks `DType::holds` directly (X36 to X38).

**Tech Stack:** Rust workspace (`vortex-array`, `vortex-layout`, `vortex-expr`, `vortex-file`,
`vortex-row`, `vortex-arrow`, `vortex-duckdb`, `vortex-datafusion`), cargo-nextest, rstest. Codex
CLI agents in Rex tabs.

**Spec:** [`03-design.md`](03-design.md), [`../decisions.md`](../decisions.md) (X13 to X16, X20 to
X22, X30, X36 to X38), [`05-implementation.md`](05-implementation.md), and for detail
[`agents/design-notes.md`](agents/design-notes.md) (sections 3, 4, 7 and 12) and
[`agents/findings.md`](agents/findings.md).

All paths below are relative to the repository root, at `develop` `3ebf68d239`. Line numbers are
hints. If a file moved, find the code by name.

## Global Constraints

- Branch names use the prefix `ct/`. Each task names its branch and its base.
- Every commit has `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>`. Use `git commit -s`.
  The worktrees have that email in their Git configuration.
- No co-author trailer, no agent name and no "Generated with" line in commits or reports.
- Commit subjects are short, imperative and in plain English, for example "Add extension claims".
- Never use the em dash character in code, comments, docs or commit messages.
- Follow `AGENTS.md`, `STYLE.md` and the nested `AGENTS.md` of every directory that you change.
  Also follow `/Users/connor/.config/claude/skills/rust-style/SKILL.md`, which holds the Rust
  preferences of the maintainer.
- New source files need the SPDX header of their language (`AGENTS.md`, "New and generated files").
- Tests: prefer `rstest` cases, test functions that return `VortexResult<()>`, modules named
  `tests`, and `assert_arrays_eq!` with the result under test on the left. Use
  `assert_execution_paths` (`vortex-array/src/test_harness/execution_paths/mod.rs`) to check the
  array path, the row path and constant inputs together.
- Imports go at module level, not inside functions.
- Do not change expected test output to match a behavior unless the spec says that behavior is
  intended. If a test fails for a reason outside your task, report it.
- Do not reduce the scope of a task. If a step is harder than expected, report that instead.
- Do not touch generated files under `*/flatbuffers/generated/` or `*/proto/generated/`.
- Never push, never open a PR, never comment on GitHub, and never change another worktree. Work only
  in the worktree that Codex created for your session.
- The typing error text is fixed. `DType::require_claim` returns
  `"{function} requires the {claim} claim, but {dtype} does not hold it"`.
- Run checks once, on the tip of your branch, never for each commit. The crates to check are listed
  in each task:

  ```bash
  cargo clippy -p <crate> --all-targets --all-features -- -D warnings
  cargo +nightly-2026-11-13 fmt -p <crate>
  cargo nextest run -p <crate>
  ```

  If cargo fails with exactly `sccache: error: Operation not permitted`, run the same command again
  with `RUSTC_WRAPPER=` set to empty. Run `cargo test --doc -p <crate>` only if you changed doc
  comments with code. Do not run workspace-wide checks: CI runs them.

## Agent protocol

1. Create your branch from the base that the task names: `git fetch origin` first, then
   `git switch -c <branch> <base>`.
2. Follow the steps of your task in order.
3. Before you report, make the history clean: one commit for each logical change, no fixup
   commits.
4. Write your report to `<reports>/<ID>.md`, where `<reports>` is the directory in your prompt. Use
   these sections, in plain English:
   - Status: done, or blocked.
   - Worktree path, branch and head commit.
   - What changed, in five to ten lines.
   - Choices that the plan left open, and what you chose.
   - Differences from the plan, and why.
   - Checks: each command that you ran and its result. Name the checks that you did not run.
   - Anything unexpected.
5. Then create the empty file `<reports>/<ID>.done`. It must be the last file that you write.
6. Wait. A review arrives as the message "Read the review at `<path>`". Address each point, rewrite
   the history if needed, run the checks again, rewrite the report, and create
   `<reports>/<ID>.done` again.
7. If a step needs a design decision that the plan and the spec do not make, or a fact in the plan
   is wrong, stop. Write a report with the status "blocked", the facts and the options, and create
   the `.done` file.

## Review Focus

These five inputs are the most likely to break for a user, and no single task owns all of them.
Each line names the test that pins it and the task that adds the test.

1. Nested extension values: a compare of `struct{doc: Json}` values, or `list_contains` over
   `List<Json>`, must fail when it is bound, and the same over `Timestamp` must work. Tests:
   `compare_nested_extension_needs_claim` (TY1), `list_contains_requires_eq` (TY3).
2. Mixed nullability: a compare of a nullable and a non-nullable `Timestamp(ms)` column must work,
   because claims ignore the top-level nullability. Tests: `claims_ignore_top_level_nullability`
   (CL1), `compare_accepts_mixed_nullability` (TY1).
3. Engine fallback: in DuckDB, `WHERE json_a = json_b`, `WHERE json_col = '{"a":1}'` and
   `SELECT max(json_col)` must return DuckDB's own answers, not an error. Tests:
   `json_filters_are_not_pushed_down` (TY1), `max_over_json_is_not_pushed_down` (TY2).
4. Old files: a file whose statistics hold storage-order min and max for a `Json` column must read,
   must not export those values to engines, and must log one warning. A new file must still store
   logical min and max for `Timestamp`. Tests: `file_stats_ignore_min_max_without_ord` (ST1),
   `zone_map_writes_min_max_only_for_ord` (TY2).
5. Constant and row paths: a rejected call must fail in the same way on the array path, the row
   path and with constant inputs, and an accepted call must return the same values. Tests:
   `compare_accepts_claimed_types_on_every_path` (TY1), `min_max_returns_input_dtype` (TY2).

---

## Waves

| Wave | Tasks | Starts when |
| --- | --- | --- |
| A | CL1 then CL2 (one agent), U1, NU, SF then SN3 (one agent) | The plan is approved |
| B | TY1, TY2, TY3 then TY4 (one agent), ST1 then ST2 (one agent) | CL2 passed review. The branches start from CL2, and each task starts a new agent |
| C | TY5, CL3, VC1 then VC2, LP4 | Wave B is in review |
| D | RU1 to RU3, BD1, BD3, AF, LP1 to LP3 | Each needs a decision first (see "Wave D") |

Tasks that follow "then" run in one agent, one after the other, as separate branches and PRs.

---

### Task CL1: claims

Branch `ct/ext-claims` from `origin/develop`. Check `vortex-array`.

**Files:**
- Create: `vortex-array/src/dtype/claim.rs`
- Modify: `vortex-array/src/dtype/mod.rs`, `vortex-array/src/dtype/extension/{vtable,typed,erased}.rs`
- Create: `vortex-array/src/extension/tests/opaque.rs`
- Modify: `vortex-array/src/extension/tests/{mod,divisible_int}.rs`

**Interfaces:**
- Consumes: `ExtVTable::storage_dtype(metadata)` and `ExtDType::try_new(metadata, nullability)`.
- Produces:

  ```rust
  // vortex_array::dtype
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
  #[non_exhaustive]
  pub enum Claim { Eq, Ord }
  impl Claim { pub fn implies(self, other: Claim) -> bool }   // Ord implies Eq, each implies itself
  impl fmt::Display for Claim                                 // "Eq", "Ord"
  impl DType {
      pub fn holds(&self, claim: Claim) -> bool;
      pub fn require_claim(&self, claim: Claim, function: impl fmt::Display) -> VortexResult<()>;
  }
  // ExtVTable, with this default:
  fn claims(metadata: &Self::Metadata) -> &'static [Claim] { &[] }
  // ExtDTypeRef
  pub fn claims(&self) -> &'static [Claim];
  // Test type, for later tasks: vortex_array::extension::tests::Opaque, over i64, no claims
  ```

- [ ] **Step 1: Write the tests** in a `tests` module of `claim.rs`:
  - `builtin_dtypes_hold_claims`, an `rstest` over `(dtype, claim, expected)`: `i64`, `f64`,
    `Bool`, `Utf8`, `Binary`, `Decimal` and `Null` hold `Eq` and `Ord`. `Variant` holds neither.
    `struct{a: i64}` and `List<i64>` hold `Ord`. `struct{a: Variant}` holds neither.
  - `ord_implies_eq`: `Ord.implies(Eq)`, `Eq.implies(Eq)`, and not `Eq.implies(Ord)`.
  - `extension_holds_declared_and_implied_claims`: `DivisibleInt` holds `Ord` and `Eq`. `Opaque`
    holds neither. `List<Opaque>` holds neither. `struct{a: DivisibleInt}` holds `Ord`.
  - `claims_ignore_top_level_nullability`: each extension dtype above gives the same answers when
    nullable and when not.
  - `try_new_rejects_a_claim_that_storage_lacks`: a test type with storage `List<Opaque>` that
    claims `Eq`. `ExtDType::try_new` fails, and the message names the type ID, `Eq` and the storage
    dtype.
  - `require_claim_error_names_function_dtype_and_claim`: `Opaque` with `Claim::Ord` and the
    function `"min"` gives exactly the error text of "Global Constraints".
- [ ] **Step 2: Run the tests and see them fail to compile.** Run
  `cargo nextest run -p vortex-array claim`.
- [ ] **Step 3: Implement.**
  - `holds` follows X13. For an extension dtype it returns
    `ext.claims().iter().any(|c| c.implies(claim))` and does not look into the storage, which was
    checked when the dtype was built (X15).
  - `ExtDType::try_new` checks every claim of `V::claims(&metadata)` against the declared storage
    dtype, once. Make sure every constructor of `ExtDType` and the read path
    (`metadata_from_storage`) reach this check.
  - `DivisibleInt` claims `Ord`. `ForeignExtDType` keeps the default.
  - Document `claims` on `ExtVTable`: what a claim promises (X13), that the default makes the type
    opaque, and that the claims must not depend on the top-level nullability.
- [ ] **Step 4: Run the checks** for `vortex-array`. Expected: all pass.
- [ ] **Step 5: Commit** "Add extension claims".

### Task CL2: claims of the repository types

Branch `ct/ext-claims-types` from `ct/ext-claims`. Check `vortex-array`, `vortex-uuid`,
`vortex-tensor`, `vortex-json`, `vortex-spatial`.

**Files:**
- Modify: `vortex-array/src/extension/datetime/{date,time,timestamp}.rs`,
  `encodings/uuid/src/vtable.rs`, the vtables under `vortex-tensor/src/types/vector/` and
  `vortex-tensor/src/types/fixed_shape_tensor/`
- Test: the `tests` modules next to each vtable, and one test each in `vortex-json` and
  `vortex-spatial`

**Interfaces:**
- Consumes: `Claim`, `ExtVTable::claims` (CL1).
- Produces: `Date`, `Time`, `Timestamp` and `Uuid` claim `&[Claim::Ord]`. `Vector` and
  `FixedShapeTensor` claim `&[Claim::Eq]`. Every other type keeps the default (X16).

- [ ] **Step 1: Write the tests:** `temporal_types_claim_ord` (an `rstest` over the three types,
  several units and time zones, and both nullabilities: each holds `Eq` and `Ord`),
  `uuid_claims_ord`, `vector_and_tensor_claim_eq_only` (hold `Eq`, not `Ord`), `json_claims_nothing`
  and `wkb_claims_nothing`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement** the `claims` methods. Add a one-line comment only where the reason is not
  obvious: `Uuid` orders by its bytes, which is the hex text order (X16).
- [ ] **Step 4: Run the checks** for the five crates. Expected: all pass. No other test changes,
  because nothing reads the claims yet.
- [ ] **Step 5: Commit** "Give the repository extension types their claims".

### Task TY1: compare, between and DynamicComparison check claims

Branch `ct/ext-compare-claims` from `ct/ext-claims-types`. Check `vortex-array`,
`vortex-datetime-parts`, `vortex-duckdb`, `vortex-datafusion`.

**Files:**
- Modify: `vortex-array/src/scalar_fn/fns/binary/mod.rs` (`return_dtype`, lines 142-178), the
  operator type of `Binary`, `vortex-array/src/scalar_fn/fns/between/mod.rs` (lines 302-325),
  `vortex-array/src/scalar_fn/fns/dynamic.rs` (lines 81-97)
- Modify: `vortex-duckdb/src/convert/{expr,table_filter}.rs`, `vortex-duckdb/src/table_function.rs`,
  and `vortex-duckdb/cpp/table_function.cpp` only if Step 4 needs it
- Test: `vortex-array/src/scalar_fn/fns/binary/compare/tests.rs`, the `tests` modules of `between`
  and `dynamic`, `encodings/datetime-parts/src/compute/compare.rs`,
  `vortex-duckdb/src/e2e_test/`, `vortex-datafusion/src/convert/exprs/tests.rs`

**Interfaces:**
- Consumes: `DType::require_claim`, `Claim`, `Opaque` (CL1), the claims of CL2.
- Produces: `fn required_claim(self) -> Option<Claim>` on the operator type of `Binary`: `Eq` for
  `=` and `!=`, `Ord` for `<`, `<=`, `>` and `>=`, `None` for the others.

- [ ] **Step 1: Write the tests** in `compare/tests.rs`:
  - `compare_rejects_types_without_claim`, an `rstest` over `(dtype, operator)`: `Opaque` with `=`
    and `<`, `Json` with `=`, `Vector` with `<`. Binding the expression fails with the error text of
    "Global Constraints". `Vector` with `=` binds.
  - `compare_nested_extension_needs_claim`: `=` over `struct{doc: Json}` fails when bound. `<` over
    `struct{ts: Timestamp(ms)}` binds and gives the right values.
  - `compare_accepts_claimed_types_on_every_path`: `Timestamp(ms)` `<` a `Timestamp(ms)` literal,
    and `Uuid` `=` a `Uuid` literal, through `assert_execution_paths`.
  - `compare_rejects_types_without_claim_on_every_path`: the `Opaque` case through
    `assert_execution_paths`, which expects the same error on each path.
  - `compare_accepts_mixed_nullability`: a nullable and a non-nullable `Timestamp(ms)` column.
  - `compare_rejects_variant_when_bound`: two `Variant` operands fail when bound, not when executed.
  - In `between`: `between_requires_ord` (`Vector` fails, `Timestamp` works). In `dynamic`:
    `dynamic_comparison_requires_claim` (`Opaque` fails when bound).
  - In `datetime-parts`: make sure that a test runs a `Timestamp` compare through `Binary` on a
    `DateTimeParts` array and asserts that the `DateTimeParts` compare kernel answered (X32). Add
    one if none exists.
- [ ] **Step 2: Run the `vortex-array` tests and see the new ones fail.**
- [ ] **Step 3: Implement the checks.** In `Binary::return_dtype`, after
  `validate_comparison_dtypes`, call `lhs.require_claim(claim, operator)` when
  `operator.required_claim()` is `Some`. `Between` requires `Ord`. `DynamicComparison` requires the
  claim of its operator, because it calls `Binary.execute` directly. Keep the extension compare
  kernel and the lone-extension case of `compare_arrays`: they now run only for accepted calls.
  Keep the order in which compare dispatches encoding kernels before the storage path.
- [ ] **Step 4: Keep DuckDB filters correct.** Read "Order constraints", item 7, of
  `05-implementation.md`. DuckDB drops a filter from its plan when Vortex accepts it, and a bind
  error later fails the query. Complex filters (`table_function.rs:479-498`) can decline. Table
  filters (`convert/table_filter.rs`) are mandatory once pushed. Make every filter that Vortex
  rejects stay in DuckDB's plan. If DuckDB offers no way to decline a table filter for a column
  without a design choice (for example a per-type pushdown callback in the C++ glue), stop and
  report the options.
- [ ] **Step 5: Write the engine tests.** DuckDB: `json_filters_are_not_pushed_down`, with a Vortex
  file that has two `Json` columns: `WHERE json_a = json_b` and `WHERE json_a = '{"a":1}'` return
  DuckDB's answers. DataFusion: `json_comparison_is_not_pushed_down`: the converter answers
  "unsupported" for a `Json` compare, and a `Timestamp` compare still converts.
- [ ] **Step 6: Run the checks** for the four crates. Expected: all pass. A failing test that
  compares an extension type without the claim is expected to change only if the spec rejects that
  call. Report each such test.
- [ ] **Step 7: Commit** "Check compare claims when binding" (and a second commit for the engine
  changes if they are large).

### Task TY2: aggregates check claims and run on storage

Branch `ct/ext-aggregate-claims` from `ct/ext-claims-types`. Check `vortex-array`, `vortex-uuid`,
`vortex-layout`, `vortex-duckdb`.

**Files:**
- Modify: `vortex-array/src/aggregate_fn/fns/{min,max,min_max,bounded_min,bounded_max,is_sorted,is_constant,all_non_distinct}/mod.rs`,
  `min_max/extension.rs`, `is_constant/extension.rs`
- Delete: `vortex-array/src/aggregate_fn/storage_pushdown.rs`
- Modify: `vortex-array/src/aggregate_fn/{mod,session}.rs`, `encodings/uuid/src/lib.rs`,
  `encodings/uuid/tests/arrow_properties.rs` if it names the adapter
- Modify: `vortex-layout/src/layouts/zoned/{schema,builder,writer}.rs`
- Create: `vortex-array/src/aggregate_fn/extension.rs`
- Test: the `tests` modules of each aggregate, `vortex-layout/src/layouts/zoned/writer.rs`,
  `vortex-duckdb/src/e2e_test/`

**Interfaces:**
- Consumes: `DType::holds`, `Claim`, `Opaque` (CL1), the claims of CL2.
- Produces, in `vortex_array::aggregate_fn::extension`:

  ```rust
  /// The storage dtype of an extension input that holds `claim`, or `None`.
  pub(crate) fn claimed_storage_dtype(input: &DType, claim: Claim) -> Option<&DType>;
  ```

  The public items `register_storage_ordered` and `ExtensionStoragePushdown` are removed.

- [ ] **Step 1: Write the tests:**
  - `aggregates_check_claims`, an `rstest` over `(aggregate, dtype, accepted)`: `MinMax`, `Min`,
    `Max`, `BoundedMin`, `BoundedMax` and `IsSorted` accept `Timestamp(ms)` and `Uuid`, and reject
    `Opaque`, `Json` and `Vector`. `IsConstant` and `AllNonDistinct` accept `Vector` and reject
    `Opaque` and `Json`. Check `return_dtype` and `Accumulator::try_new`.
  - `min_max_returns_input_dtype`: over `Timestamp(ms, "UTC")` as a canonical, a chunked and a
    constant array, the min and max have the input dtype and the right values. `min` and `max`
    too.
  - `is_sorted_works_for_ordered_extension_types`: sorted and unsorted `Timestamp` arrays give
    `true` and `false` (EXT-A1). The `IsSorted` statistic of a sorted `Timestamp` array is `true`.
  - `min_over_uuid_orders_by_bytes`.
  - `partial_state_has_input_dtype`: the partial of `min` over `Timestamp(ms)` leaves the
    accumulator with the input dtype, comes back through `partial_from_scalar`, and merges (X21).
  - `zone_map_writes_min_max_only_for_ord`: a zone map of a `Json` column has a null count and no
    min or max. A zone map of a `Timestamp` column stores min and max with the `Timestamp` dtype,
    so the format does not change.
  - DuckDB `max_over_json_is_not_pushed_down`: `SELECT max(json_col)` returns DuckDB's answer.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement the checks.** Each order aggregate accepts an extension input only when it
  holds `Ord` and the aggregate supports its storage dtype. `is_constant` and `all_non_distinct`
  need `Eq`. The code example in "Aggregates" of `03-design.md` shows the shape.
- [ ] **Step 4: Implement the storage paths.** Each accepted aggregate unwraps each batch and runs
  over the storage, so the storage kernels still apply. Value results get the input dtype, built
  with `ExtensionArray::new_unchecked` or the scalar equivalent. Partial states follow X21. Move
  the useful body of `min_max/extension.rs` into the shared helpers. Keep the NaN handling of
  `min_max` for types stored as floats.
- [ ] **Step 5: Remove the adapter.** Delete `storage_pushdown.rs` and its tests, the composite IDs,
  `register_storage_ordered`, the registrations in `aggregate_fn/session.rs` and the one in
  `encodings/uuid/src/lib.rs`. Rewrite the `Uuid` tests that used the composite IDs to use the
  plain aggregates.
- [ ] **Step 6: Remove the writer fallback.** The zone map writer stops using the storage fallback of
  `aggregate_state_dtype` (`vortex-layout/src/layouts/zoned/schema.rs:177-188`), so it drops any
  aggregate whose `return_dtype` rejects the column. The legacy reader keeps
  `legacy_stats_table_dtype`, so old files still read.
- [ ] **Step 7: Run the checks** for the four crates. Expected: all pass.
- [ ] **Step 8: Commit** in two or three commits: the claim checks and storage paths, the removal of
  the adapter, and the zone map writer.

### Task TY3: list_contains needs Eq

Branch `ct/ext-list-contains-claims` from `ct/ext-claims-types`. Check `vortex-array`.

**Files:**
- Modify: `vortex-array/src/scalar_fn/fns/list_contains/mod.rs` (`return_dtype`, lines 177-198)

**Interfaces:**
- Consumes: `DType::require_claim`, `Opaque` (CL1), the claims of CL2.

- [ ] **Step 1: Write the test** `list_contains_requires_eq`, an `rstest`: `List<Opaque>` and
  `List<Json>` with a needle of the element dtype fail when bound. `List<Timestamp(ms)>` and
  `List<Vector>` work, with the right values, through `assert_execution_paths`.
- [ ] **Step 2: Run it and see it fail.**
- [ ] **Step 3: Implement** `require_claim(Claim::Eq)` on the element dtype in `return_dtype`.
- [ ] **Step 4: Run the checks** for `vortex-array`. Expected: all pass.
- [ ] **Step 5: Commit** "Check the Eq claim in list_contains".

### Task TY4: the row encoding accepts ordered extension types

Branch `ct/ext-row-encoding-claims` from `ct/ext-claims-types`. The same agent runs it after TY3.
Check `vortex-row`, and `vortex-array` if Step 3 changes it.

**Files:**
- Modify: `vortex-row/src/{lib,encoder,encode,size,codec}.rs`
- Test: `vortex-row/src/tests.rs`

**Interfaces:**
- Consumes: `DType::holds`, `DType::require_claim`, `Opaque` (CL1), the claims of CL2.

- [ ] **Step 1: Write the tests:** `row_encoding_orders_timestamp_by_value` (ascending, descending,
  and with nulls first and last: the byte order of the rows equals the value order),
  `row_encoding_accepts_nested_ordered_extension` (`struct{ts: Timestamp(ms)}`), and
  `row_encoding_rejects_types_without_ord` (`Json`, `Vector` and `Opaque` fail with the error text
  of "Global Constraints").
- [ ] **Step 2: Run them and see them fail.** Today the encoder rejects every extension dtype
  (`encoder.rs:113`, `:127-150`).
- [ ] **Step 3: Implement.** The encoder accepts an extension column whose dtype holds `Ord` and
  encodes its storage. It checks `DType::holds` itself, because it runs on raw columns (X38).
  `RowEncode` and `RowSize` check the same claim in `return_dtype`. Update the comment at
  `vortex-row/src/lib.rs:24-28`. Read `vortex-array/src/scalar_fn/unstable/row/` and say in the
  report whether it needs the same change. Do not change it in this task.
- [ ] **Step 4: Run the checks.** Expected: all pass.
- [ ] **Step 5: Commit** "Row-encode extension types that claim Ord".

### Task ST1: statistics readers need Ord

Branch `ct/ext-stats-readers` from `ct/ext-claims-types`. Check `vortex-layout`, `vortex-file`,
`vortex-expr`, `vortex-duckdb`, `vortex-datafusion`.

**Files:**
- Modify: `vortex-layout/src/layouts/file_stats.rs` (`get` and `to_stats_set`, lines 205-232),
  `vortex-layout/src/layouts/zoned/zone_map.rs` (lines 219-336), `vortex-file/src/pruning.rs`,
  `vortex-expr/src/stats/rewrite/builtins.rs` (`has_nans`, lines 757-759),
  `vortex-duckdb/src/column_statistics.rs` (lines 84-112),
  `vortex-datafusion/src/convert/stats/mod.rs` (`constant_distinct_count`, lines 92-100),
  `vortex-datafusion/src/persistent/format.rs`

**Interfaces:**
- Consumes: `DType::holds`, `Claim` (CL1), the claims of CL2.

- [ ] **Step 1: Write the tests:**
  - `file_stats_ignore_min_max_without_ord`: file statistics for a `Json` column that hold min and
    max return neither, and log one warning for the file and column. Build the statistics
    directly, not through the writer, so the test still holds after TY2.
  - `zone_map_pruning_ignores_min_max_without_ord`: the zone map reader binds no min or max field of
    a `Json` column.
  - `has_nans_looks_through_extension_dtypes`: a test type over `f64` that claims `Ord`. Pruning of
    `x > lit(5)` adds the NaN guard (EXT-A5).
  - DuckDB `column_statistics_skip_types_without_ord`: no min or max for a WKB column.
  - DataFusion `constant_distinct_count_uses_eq`: a constant `Uuid` column reports one distinct
    value. A `Json` column reports none.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Every reader ignores stored min and max of a dtype that does not hold
  `Ord` (X22, X38). Log the warning at most once for each file and column. `has_nans` looks
  through extension dtypes to their storage. `constant_distinct_count` replaces its temporal list
  with `dtype.holds(Claim::Eq)`.
- [ ] **Step 4: Run the checks.** Expected: all pass.
- [ ] **Step 5: Commit** "Ignore min and max of extension types without Ord".

### Task ST2: the bloom filter needs Eq

Branch `ct/ext-bloom-eq` from `ct/ext-claims-types`. The same agent runs it after ST1. Check
`vortex-layout`.

**Files:**
- Modify: `vortex-layout/src/layouts/zoned/aggregates/bloom_filter/mod.rs` (`is_bloom_valid_dtype`,
  lines 406-411), `bloom_filter/scalar_fn.rs` (`BloomEqRewrite`, lines 233-277)

**Interfaces:**
- Consumes: `DType::holds`, `Claim`, `Opaque` (CL1), the claims of CL2.

- [ ] **Step 1: Write the tests:** `bloom_rejects_extension_without_eq` (`Json` and `Opaque` are not
  valid) and `bloom_accepts_timestamp` (the tests at `bloom_filter/canonical/extension.rs:39` and
  `:69` keep passing).
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** An extension dtype is valid when it holds `Eq` and its storage is valid
  (EXT-A4). `BloomEqRewrite` checks the same.
- [ ] **Step 4: Run the checks.** Expected: all pass.
- [ ] **Step 5: Commit** "Build bloom filters only for extension types that claim Eq".

### Task U1: one cast rule

Branch `ct/ext-no-storage-casts` from `origin/develop`. Check `vortex-array`, `vortex-arrow`.

Read the U1 section of [`04-urgent-changes.md`](04-urgent-changes.md) and EXT-C1 to EXT-C3 and
EXT-B2 in [`agents/findings.md`](agents/findings.md). The line numbers there are older. At
`3ebf68d239`, the code is at `vortex-array/src/scalar/cast.rs:61-65`,
`vortex-array/src/scalar/typed_view/extension/mod.rs:87-104` and
`vortex-array/src/arrays/extension/compute/cast.rs:16-41`.

**Files:**
- Modify: those three files, and the Arrow exporters under `vortex-arrow/src/executor/` that call
  `array.cast(target)`

**Interfaces:**
- Produces: the only cast from or to an extension dtype is a nullability change of the same dtype
  (X30). Code that wants storage calls `ext_storage`.

- [ ] **Step 1: Write the tests:** `cast_never_reinterprets_storage`, through
  `assert_execution_paths`: `Timestamp(ms)` to `i64`, `i64` to `Timestamp(ms)`, `Timestamp(ms)` to
  `Timestamp(ns)` and `Uuid` to its storage dtype fail on every path. A nullability cast of
  `Timestamp(ms)` works on every path. `same_type_cast_returns_the_storage_error` (EXT-C3). An
  Arrow export test: exporting a `Timestamp` column with an `Int64` target fails, and
  `ext_storage` gives the storage.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement** the five changes of the U1 section.
- [ ] **Step 4: Fix the callers.** Run the checks. Code that depended on the leak fails, for example
  a comparison of an extension column with a storage literal. Fix each caller with an explicit
  `ext_storage`, or report it if the fix changes behavior that users see.
- [ ] **Step 5: Commit** "Stop casts from reinterpreting extension storage".

### Task NU: new_unchecked, validate_array and the conformance harness

Branch `ct/ext-new-unchecked` from `origin/develop`. Check `vortex-array`.

Read "Phase 1" of [`05-implementation.md`](05-implementation.md) (its last three items) and section
8 of [`agents/design-notes.md`](agents/design-notes.md).

**Files:**
- Modify: `vortex-array/src/dtype/extension/vtable.rs` (`validate_array`),
  `vortex-array/src/arrays/extension/array.rs` (`new_unchecked`)
- Create: an `ExtVTable` conformance module under `vortex-array/src/compute/conformance/`

**Interfaces:**
- Produces:

  ```rust
  // ExtVTable, with a default that checks each non-null value with validate_scalar_value:
  fn validate_array(metadata: &Self::Metadata, storage: &ArrayRef) -> VortexResult<()>;
  // ExtensionArray
  pub unsafe fn new_unchecked(dtype: ExtDTypeRef, storage: ArrayRef) -> Self;
  // conformance, for later tasks (CL3 adds the claim checks):
  pub fn test_ext_vtable_conformance<V: ExtVTable>(metadata: V::Metadata, storage: ArrayRef);
  ```

  `new_unchecked` is `unsafe` with a `# Safety` section, like the other `new_unchecked`
  constructors of `vortex-array`. Match the naming of the existing conformance functions, such as
  `test_cast_conformance`. If the
  existing signatures of `ExtVTable` make a different shape clearly better, choose it and say so
  in the report.

- [ ] **Step 1: Write the tests:** `validate_array_default_checks_each_value` (a `Time` array with a
  value over one day fails, and nulls are skipped), `new_unchecked_checks_in_debug` (a debug build
  panics on an illegal value), and the conformance harness run over every extension type in
  `vortex-array`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** `new_unchecked` skips the same checks that `ExtensionArray::new` skips
  today, and a debug build checks the promise with `validate_array`. The harness checks the
  metadata round trip through the serialized bytes and the stored storage, and that a
  `validate_array` override agrees with the default.
- [ ] **Step 4: Run the checks.** Expected: all pass.
- [ ] **Step 5: Commit** "Add ExtensionArray::new_unchecked and validate_array".

### Task SF: dtype errors in lazily parsed struct fields

Branch `ct/lazy-struct-field-errors` from `origin/develop`. Check `vortex-array`, `vortex-file`.

Read issue [#8848](https://github.com/vortex-data/vortex/issues/8848) with `gh issue view 8848`.

- [ ] **Step 1: Investigate.** `StructFields::fields()` calls `vortex_expect` on a fallible
  flatbuffer decode of a lazily parsed field dtype. Find every caller that can reach a lazily
  parsed dtype from a file.
- [ ] **Step 2: Report the design and stop.** Write a "blocked" report with the options, for
  example eager validation when a file is opened, fallible accessors on the read paths, or both.
  For each, give the cost for wide schemas and the size of the API change. Wait for the review.
- [ ] **Step 3: Write the regression test** from #8848: opening a file whose footer has a struct
  field with an unknown dtype discriminant returns an error and does not panic.
- [ ] **Step 4: Implement the approved option.**
- [ ] **Step 5: Run the checks.** Expected: all pass.
- [ ] **Step 6: Commit** "Return an error for an invalid lazy struct field dtype".

### Task SN3: Arrow import drops masked nulls below every nested level

Branch `ct/arrow-import-masked-nulls` from `origin/develop`. The same agent runs it after SF.
Check `vortex-arrow`.

Read the SN3 item in [`05-implementation.md`](05-implementation.md), "Phase 0", and the merged PRs
[#11223](https://github.com/vortex-data/vortex/pull/11223) and
[#11497](https://github.com/vortex-data/vortex/pull/11497), which did the same for structs and
lists.

**Files:**
- Modify: the Arrow import code under `vortex-arrow/src/` for list views, maps, run-end arrays and
  dictionaries

- [ ] **Step 1: Write the tests:** for each of list view, map, run-end and dictionary, an Arrow array
  with a null parent row over a null child slot at a non-nullable level imports, and the child
  slot is dropped. A visible null at a non-nullable level still fails. A masked dictionary key
  gets a valid target when `new_null_array` builds keys outside the values.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement** by passing visible rows to the children, as #11223 and #11497 do.
- [ ] **Step 4: Run the checks.** Expected: all pass.
- [ ] **Step 5: Commit** "Drop masked nulls below every nested Arrow level".

---

## Wave C

These tasks start after wave B is in review. Each is specified enough to run.

- **TY5:** make the variants of `BoundExpression` private (`vortex-array/src/expr/bound_expression.rs`),
  so no code builds a node without `return_dtype` (X8). Branch `ct/bound-expression-private` from
  `origin/develop`. Test: the crate compiles without public variants, and every caller uses a
  checked constructor. Coordinate with #10762.
- **CL3:** add the claim checks to the conformance harness of NU (claims ignore the top-level
  nullability), and the `DecimalLike` and `Refined` reference types of
  [`agents/design-notes.md`](agents/design-notes.md), section 13. The differential test against the
  built-in `Decimal` waits for TY1 to TY3 and for CD8a
  ([#10701](https://github.com/vortex-data/vortex/issues/10701)).
- **VC1, then VC2:** "Phase 6" of [`05-implementation.md`](05-implementation.md) and section 8 of
  the design notes. VC1 needs NU.
- **LP4:** registration stops replacing and duplicating silently (EXT-R2).

## Wave D

Each of these needs a decision before an agent can start:

- **RU1 to RU3:** XQ10, whether the rules of a type stay on `ExtVTable` after #11091 and #11092.
- **BD1:** the shape of the converter traits for DuckDB, Python and JNI.
- **BD3:** the developer guide, which describes the final API, so it comes last.
- **AF:** the API of [#11286](https://github.com/vortex-data/vortex/issues/11286).
- **LP1 to LP3:** low priority. LP1 needs XQ1, and LP3 needs an edition.

---

## Orchestration

This section is for the coordinating session, not for the agents.

### Launch

Each agent runs in its own Rex tab. Codex creates a managed Git worktree for each session
(`--worktree`), so the `vortexN` worktrees stay untouched. `<ID>` is the task, `<base>` the base
branch and `<reports>` the report directory.

```bash
rex window new -s "$REX_SESSION" "<ID>" -c /Users/connor/spiral/vortex-data/vortex6 \
  --focus=false --keep-open --json -- \
  codex -m gpt-6.2-sol \
    -c model_reasoning_effort=high \
    -c service_tier=priority \
    --approve-for-me \
    --worktree \
    --add-dir /Users/connor/spiral/vortex-data/.bare \
    --add-dir "<reports>" \
    -C /Users/connor/spiral/vortex-data/vortex6 \
    "You implement task <ID> of the Vortex extension type plan. Read /Users/connor/spiral/vortex-data/vortex6/research/extension-types/06-agent-plan.md: 'Global Constraints', 'Agent protocol', 'Review Focus' and 'Task <ID>'. Read the spec files that it names in /Users/connor/spiral/vortex-data/vortex6/research, but do not change them. Work only in the worktree that Codex created for this session. Your base is <base>. Your report directory is <reports>."
```

- `service_tier=priority` is the Fast tier, and `model_reasoning_effort=high` is the high effort.
- `--approve-for-me` sends approval requests to automatic review in the workspace-write sandbox.
- `--worktree` starts the session in a new managed worktree. The agent then creates its branch from
  `<base>`, so the commit that the worktree starts from does not matter.
- `--add-dir` for the bare repository lets the agent commit, because the Git directory of a
  worktree is outside it. The report directory is the other writable path.
- An agent that runs two tasks gets a second prompt in the same tab after the first task passes
  review: "Task <ID2> next. Read 'Task <ID2>' in the plan. Your base is <base2>."
- Each new worktree starts with an empty `target/` directory, so the first build of each agent is
  a cold build.

### Monitoring and review

1. For each agent, a background shell waits for `<reports>/<ID>.done` and wakes the coordinating
   session.
2. The coordinator reads the report and the branch (`git log <base>..<branch>`, `git diff <base>...<branch>`),
   and reviews the diff against the task, the spec and the Review Focus. The review uses the
   `code-review` and `rust-style` skills.
3. If changes are needed, the coordinator writes `<reports>/<ID>-review-<n>.md`, deletes the
   `.done` file, and sends "Read the review at `<path>`" to the tab with `rex send` and
   `rex send-key enter`.
4. After two review rounds, the coordinator fixes small remaining issues itself.
5. The maintainer is not available while the agents run (2026-11-17). So the coordinator answers a
   "blocked" report itself, picks the option that the spec supports best, and records the choice
   in the PR body.
6. `rex capture -s "$REX_SESSION" -b <block> --trim` shows a tab that seems stuck.

### Publishing

After a branch passes review, the coordinator pushes it and opens a draft PR. The PR body follows
the `connor-voice` skill. The maintainer marks PRs ready. CL1 and CL2 form a stack. The wave B PRs
use the CL2 branch as their base until CL2 merges, and then rebase onto `develop`. The coordinator
also updates the PR plans in [#11114](https://github.com/vortex-data/vortex/issues/11114) and
[#11115](https://github.com/vortex-data/vortex/issues/11115) with the new IDs and links, after it
reads their `updatedAt` so that no other edit is lost.

The agents build on `develop`, not on the crate moves of
[#10762](https://github.com/vortex-data/vortex/pull/10762) and
[#10769](https://github.com/vortex-data/vortex/pull/10769). Whichever lands second ports the other's
changes.

### Agents

| Agent | Wave A | Wave B |
| --- | --- | --- |
| 1 | CL1, then CL2 | TY1 |
| 2 | U1 | TY2 |
| 3 | NU | TY3, then TY4 |
| 4 | SF, then SN3 | ST1, then ST2 |

Wave B starts new agents, each in a new managed worktree, as soon as CL2 has passed review. After
the work of an agent is published, the coordinator removes its managed worktree and any leftover
branch that Codex created, with the `git-workspace-cleanup` skill.
