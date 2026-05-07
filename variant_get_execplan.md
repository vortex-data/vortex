# Implement VariantArray Core Storage and VariantGet

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository follows `/Users/adamgs/.codex/PLANS.md`; this file is written to be self-contained so a future agent can continue from this document alone.

## Purpose / Big Picture

Vortex has a `Variant` dtype and a Parquet Variant encoding, but the canonical `VariantArray` still exposes one child and there is no Vortex `VariantGet` expression that extracts a typed path from a variant column. After this work, a user can read or construct Parquet Variant data, keep the raw unshredded storage row-aligned with optional shredded typed values through row-preserving operations, and evaluate expressions such as "get path `data[1].a` as UTF-8" without first materializing unrelated variant fields.

The end-to-end behavior is demonstrated by tests that build Parquet Variant arrays containing objects, lists, missing paths, top-level nulls, present variant nulls, typed shredded values, and raw fallback rows. The observable result is that `VariantGet` returns nullable typed arrays when a dtype is requested, nullable variant arrays when no dtype is requested, and preserves row alignment after slice, filter, and take. When an encoding-specific array such as `ParquetVariant` is canonicalized, any logical shredded child owned by that encoding, such as Parquet `typed_value`, is surfaced as the canonical `VariantArray::shredded` child so downstream canonical operations do not need to know Parquet-specific slot names.

## Progress

- [x] (2026-05-07T11:21Z) Read the ExecPlan guide, the two local RFCs, the Parquet Variant encoding and shredding specs, the local `parquet-variant-compute` `variant_get` implementation, current Vortex `VariantArray`, `GetItem`, and Parquet Variant files, plus the `adamg/variant-array` reference branch for inspiration.
- [x] (2026-05-07T12:08Z) Added a detailed implementation inventory covering the current single-child canonical `VariantArray`, the RFC 0058 target shape, existing Parquet Variant hooks, the scalar function patterns used by `GetItem`, and the canonicalization contract for moving encoding-specific shredded children into canonical `VariantArray::shredded`.
- [x] (2026-05-07T12:42Z) Added and identified baseline tests for current variant behavior: explicit outer null versus present `variantnull`, Arrow Parquet Variant storage roundtrips including a present Variant null with a separate outer null row, and slice/filter/take over typed-only shredded Parquet Variant arrays.
- [x] (2026-05-07T13:24Z) Implemented the canonical `VariantArray` shape change with `core_storage` and optional `shredded` slots, checked constructors, serde metadata for shredded dtype, validation, accessors, scalar/validity delegation to core storage, Parquet canonicalization that exposes `typed_value` as canonical shredded data, and focused tests.
- [x] (2026-05-07T14:06Z) Updated row-preserving canonical Variant transformations so slice, filter, take, mask validity execution, canonical-validity execution, and recursive canonicalization transform `core_storage` and optional `shredded` together; added regression tests that assert both children follow the same row selection.
- [x] (2026-05-07T12:15Z) Added and registered the `VariantGet` scalar function skeleton with strict path options, optional dtype options, expression helper, nullable return dtype inference, SQL/display formatting, protobuf serialization, generated proto bindings, public API locks, and construction/type-error/serde tests.
- [ ] Implement the unshredded Parquet Variant fallback.
- [ ] Implement shredded fast-path extraction and partial-shredding merge behavior for direct `ParquetVariant`, canonical variants with a canonical `shredded` child, and canonical variants whose `core_storage` child still exposes encoding-specific shredded data.
- [ ] Wire the narrowest useful higher-level integration after core behavior is covered.
- [ ] Complete final review, validation, and signed-off commits.

## Surprises & Discoveries

- Observation: The current `vortex-parquet-variant` crate already has a four-slot Parquet storage array with `validity`, `metadata`, optional `value`, and optional `typed_value`, plus slice, filter, and take kernels that transform those children together.
  Evidence: `encodings/parquet-variant/src/array.rs` defines the slots and validates `value` or `typed_value`; `encodings/parquet-variant/src/kernel.rs` applies row selections to each present child.

- Observation: The canonical `vortex-array` `VariantArray` is still a one-child wrapper whose child dtype must equal the outer `DType::Variant`; its scalar and validity behavior simply delegate to that child.
  Evidence: `vortex-array/src/arrays/variant/mod.rs` defines `SLOT_NAMES = ["child"]` and `VariantArrayExt::child()`, while `vortex-array/src/arrays/variant/vtable/operations.rs` and `validity.rs` delegate to `child()`.

- Observation: `parquet-variant-compute` already implements much of the desired path semantics, including traversal through shredded object fields, row-wise fallback to raw value bytes, and optional typed output.
  Evidence: `/Users/adamgs/code/arrow-rs/parquet-variant-compute/src/variant_get.rs` has `follow_shredded_path_element`, `shredded_get_path`, `try_perfect_shredding`, and `GetOptions`.

- Observation: The existing baseline already covered many Parquet Variant cases, but filter and take over typed-only shredded arrays were not tested next to the existing slice case.
  Evidence: Before this milestone, `encodings/parquet-variant/src/kernel.rs` had `test_slice_shredded_typed_value` but no corresponding filter or take tests for typed-only shredded storage.

- Observation: The canonical `VariantArray` serde path needs the optional shredded child's dtype in metadata because the child dtype list only contains present child dtypes, while slot 1 can be absent.
  Evidence: `vortex-array/src/arrays/variant/vtable/mod.rs` now stores `shredded_dtype` in the variant metadata and deserializes children as `core_storage` plus optional `shredded`.

- Observation: Keeping `VariantArrayExt::child()` as a compatibility shim lets existing transformation call sites continue compiling, but those call sites still rebuild from core storage only until the next milestone migrates them.
  Evidence: `VariantArrayExt::child()` now delegates to `core_storage()`, while the planned row-preserving transform work still covers `filter`, `take`, `mask`, and canonical execution paths.

- Observation: Direct slicing of canonical variants needs a `Variant` parent reduce rule, not only the generic `SliceArray` execution path.
  Evidence: `vortex-array/src/arrays/variant/compute/slice.rs` now implements `SliceReduce` for `Variant`, and `vortex-array/src/arrays/variant/compute/rules.rs` registers slice, filter, and take parent reduce adapters for `Variant`.

- Observation: `vortex-proto` has no crate-local tests, so `cargo nextest run -p vortex-proto` compiles the crate and then exits with "no tests to run".
  Evidence: The skeleton milestone reran that validation as `cargo nextest run -p vortex-proto --no-tests pass`, which passed after the exact command compiled successfully but returned nextest's no-tests exit code.

- Observation: Regenerating public API locks for the `VariantGet` public API also picked up public reduce impl entries from the preceding Variant row-preserving transform milestone.
  Evidence: `./scripts/public-api.sh` updated `vortex-array/public-api.lock` for the new `variant_get` module and helper, and also added `Variant` slice/filter/take reduce entries that were already present in Rust code.

## Decision Log

- Decision: Treat RFC 0015, RFC 0058, and the Parquet specs as normative; use `adamg/variant-array` only as implementation inspiration.
  Rationale: The user explicitly asked to prefer RFCs and specifications when unclear, while allowing that branch as a reference for tests and metadata ideas.
  Date/Author: 2026-05-07 / Codex.

- Decision: Implement the canonical array shape before `VariantGet` execution.
  Rationale: `VariantGet`, slice, filter, take, masking, serialization, and Parquet import all depend on a stable definition of `core_storage` and optional `shredded`.
  Date/Author: 2026-05-07 / Codex.

- Decision: Initial typed `VariantGet` results are nullable even when the source variant is non-nullable.
  Rationale: RFC 0015 says missing paths, unexpected types, and present `variantnull` values can occur row by row. A non-null source row does not guarantee a non-null extracted path.
  Date/Author: 2026-05-07 / Codex.

- Decision: The first Parquet Variant path implementation should prioritize correctness by adapting or delegating to `parquet-variant-compute` behavior, then improve performance with native Vortex kernels once tests lock down semantics.
  Rationale: The local Arrow crate already implements binary path traversal and shredded/raw merge semantics from the Parquet specs; reimplementing that first would increase correctness risk.
  Date/Author: 2026-05-07 / Codex.

- Decision: Canonicalization moves encoding-specific shredded data into the canonical `VariantArray::shredded` child whenever it can expose that data as a row-aligned logical child.
  Rationale: Canonical arrays are the stable boundary for generic Vortex operations. Moving Parquet `typed_value` or another encoding's shredded tree into the canonical child prevents generic code from depending on encoding-specific slot layouts, while still allowing direct encoding-specific `VariantGet` kernels before canonicalization.
  Date/Author: 2026-05-07 / Codex.

- Decision: The first shape-change implementation exposes Parquet `typed_value` as canonical `shredded` without physically stripping it from the Parquet `core_storage` child.
  Rationale: This preserves direct Parquet Variant execution behavior and avoids rewriting encoding-specific storage while still satisfying the canonical contract that generic callers find logical shredded data through `VariantArray::shredded()`.
  Date/Author: 2026-05-07 / Codex.

## Outcomes & Retrospective

The planning, baseline-test, canonical shape, row-preserving transformation, and `VariantGet` skeleton milestones are complete. The current implementation exposes a required `core_storage` child and optional `shredded` child from canonical `VariantArray`, validates row alignment, preserves the existing one-child constructor for compatibility, serializes the optional shredded dtype, canonicalizes Parquet Variant `typed_value` into the canonical shredded child, and keeps both canonical Variant children aligned through slice, filter, take, masking, canonical validity execution, and recursive canonicalization. The scalar expression layer now has a registered `vortex.variant_get` function with strict path and optional dtype options, nullable return dtype inference, SQL/display formatting, expression helper support, and protobuf roundtrips. The next milestone can implement the unshredded Parquet Variant execution fallback behind that expression.

## Context and Orientation

The Vortex workspace is a Rust monorepo. The relevant crates for this work are `vortex-array`, which owns canonical arrays, scalar functions, expressions, and dtypes; `vortex-proto`, which owns protobuf messages used to serialize expressions; and `vortex-parquet-variant`, which owns the Parquet Variant storage encoding.

A Vortex `DType::Variant(Nullability)` describes a column whose row values can be primitive scalars, lists, objects, or the special present payload value `variantnull`. A top-level nullable `Variant` row is different from `variantnull`: a top-level null means the array slot is absent, while `variantnull` means the slot is present and the payload is null.

The Parquet Variant binary format stores each value using a one-byte header plus type-specific bytes. Objects contain field ids and offsets sorted by field name. Arrays contain element offsets. The top-level and nested `metadata` bytes contain the field-name dictionary used to interpret object field ids. The shredding spec stores raw bytes in an optional `value` field and typed columnar values in an optional `typed_value` field. For a single logical value, null `value` plus non-null `typed_value` means the value is represented by the typed child; non-null `value` plus null `typed_value` means the raw variant bytes are authoritative; both present is allowed only for partially shredded objects, where `typed_value` contains shredded fields and `value` contains the unshredded object fields not represented in `typed_value`.

RFC 0058 changes the canonical Vortex `VariantArray` from a one-child wrapper to a row-aligned structure with a required `core_storage` child and an optional `shredded` child. `core_storage` owns raw variant semantics and must have the same length and outer variant dtype. `shredded`, when present, is a same-length typed tree, usually a struct tree for object fields. Paths absent from `shredded` must still be extractable from `core_storage`.

Canonicalization is the boundary where encoding-specific shredded data becomes canonical shredded data. For example, a `ParquetVariant` array stores shredded values in its optional Parquet `typed_value` slot. When that array is converted into a canonical `VariantArray`, the logical typed tree from `typed_value` should be moved or exposed as `VariantArray::shredded`, and the raw Parquet storage remains reachable through `VariantArray::core_storage`. This does not remove the need for encoding-specific execution: `VariantGet` must support running directly on `ParquetVariant` before canonicalization, running on canonical `VariantArray` when `VariantArray::shredded` is populated, and running on canonical `VariantArray` whose `core_storage` child is itself an encoding such as `ParquetVariant` that still exposes shredded data internally.

The current Vortex code to inspect first is:

- `vortex-array/src/arrays/variant/mod.rs` and `vortex-array/src/arrays/variant/vtable/*.rs` for canonical variant shape, scalar delegation, validity, and serde.
- `vortex-array/src/arrays/filter/execute/mod.rs`, `vortex-array/src/arrays/dict/execute.rs`, `vortex-array/src/arrays/masked/execute.rs`, and `vortex-array/src/canonical.rs` for places that currently rebuild `VariantArray::new(child)`.
- `vortex-array/src/scalar_fn/fns/get_item.rs`, `vortex-array/src/scalar_fn/session.rs`, `vortex-array/src/expr/exprs.rs`, and `vortex-proto/proto/expr.proto` for scalar function registration, expression helpers, SQL formatting, return dtype inference, and protobuf options.
- `encodings/parquet-variant/src/array.rs`, `encodings/parquet-variant/src/kernel.rs`, `encodings/parquet-variant/src/operations.rs`, `encodings/parquet-variant/src/vtable.rs`, and `encodings/parquet-variant/src/validity.rs` for Arrow import/export, row-preserving kernels, scalar reconstruction, serde, and validity.
- `/Users/adamgs/code/arrow-rs/parquet-variant-compute/src/variant_get.rs` for the runtime behavior reference.

## Implementation Inventory

The canonical Vortex variant array is currently a one-child wrapper. In `vortex-array/src/arrays/variant/mod.rs`, `NUM_SLOTS` is `1`, `SLOT_NAMES` is `["child"]`, and `VariantArrayExt::child()` returns slot 0. `Array<Variant>::new(child)` derives the outer dtype as `DType::Variant(child.dtype().nullability())`, copies the length and statistics from the child, and stores only that child. In `vortex-array/src/arrays/variant/vtable/mod.rs`, validation requires slot 0 to be present, requires the outer dtype to be `DType::Variant`, requires the child dtype to exactly equal the outer variant dtype, and requires the child length to equal the outer length. Deserialization similarly expects exactly one child and rebuilds the same shape. `vortex-array/src/arrays/variant/vtable/operations.rs` delegates `scalar_at` to `array.child().execute_scalar(index, ctx)`, and `vortex-array/src/arrays/variant/vtable/validity.rs` delegates validity to `array.child().validity()`. Generic canonical call sites such as `vortex-array/src/canonical.rs`, `vortex-array/src/arrays/filter/execute/mod.rs`, `vortex-array/src/arrays/dict/execute.rs`, and `vortex-array/src/arrays/masked/execute.rs` currently rebuild variants with `VariantArray::new(transformed_child)`, so they will drop a future second child unless migrated.

RFC 0058 changes that canonical shape from a single child to row-aligned raw and typed storage. The canonical array should expose a required `core_storage` child and an optional `shredded` child through `VariantArrayExt::core_storage()` and `VariantArrayExt::shredded()`. In Vortex terms, the array's own dtype remains `DType::Variant(nullability)`, `core_storage` is the raw semantic source of the variant values and must have the same length and variant dtype as the outer array, and `shredded` is a same-length typed tree that may contain only selected paths. For object shredding, the natural Vortex representation is a `StructArray` whose fields are shredded object names; nested object fields are represented by nested `StructArray` values. Paths not present in `shredded` must still be extractable from unchanged `core_storage`. Row-preserving transformations such as slice, filter, take, and masking must apply the same row operation to both children before rebuilding the canonical variant.

The existing Parquet Variant encoding already models the Parquet shredding slots, but they are encoding-specific. In `encodings/parquet-variant/src/array.rs`, `ParquetVariant` has four slots named `validity`, `metadata`, `value`, and `typed_value`. `metadata` must be non-nullable binary and same length as the array, at least one of `value` or `typed_value` must be present, `value` must be binary when present, and `typed_value` must be row-aligned when present. `ParquetVariantData::from_arrow_variant` converts Arrow's `parquet_variant_compute::VariantArray` into `ParquetVariant::try_new(...)`, then currently wraps that encoding in the one-child canonical `VariantArray::new(pv.into_array())`. `encodings/parquet-variant/src/kernel.rs` implements slice, filter, and take by transforming validity, metadata, value, and typed_value together, which is the row-alignment behavior the canonical variant will also need. `encodings/parquet-variant/src/operations.rs` reconstructs scalar values according to the Parquet shredding table: outer null returns a null variant scalar; a valid `typed_value` takes priority; a valid raw `value` is decoded from metadata and bytes; both raw and typed data for objects are merged with typed fields taking priority and unshredded object fields filled from raw value. These operations already distinguish top-level nulls from present Parquet Variant null bytes.

The Parquet Variant vtable is also the current canonicalization boundary. `encodings/parquet-variant/src/vtable.rs` can execute the encoding into a `VariantArray::new(array.as_ref().clone().into_array())`, and tests in that file inspect `VariantArray::new(inner_pv.into_array())`. After the canonical shape change, any canonicalization path that converts a `ParquetVariant` into the canonical `vortex.variant` representation must expose `typed_value` as the logical canonical `shredded` child. The `core_storage` child may still be the `ParquetVariant` array, and that encoding may physically retain its `typed_value` slot for direct encoding kernels, but generic canonical callers must find shredded data through `VariantArray::shredded()` rather than by knowing the Parquet-specific slot name. This means the public contract is logical movement into canonical `shredded`; physical removal from `core_storage` remains an unresolved implementation choice.

`GetItem` is the closest scalar function pattern for `VariantGet`. In `vortex-array/src/scalar_fn/fns/get_item.rs`, the scalar function uses `FieldName` as options, returns id `vortex.get_item`, serializes options with `vortex_proto::expr::GetItemOpts`, declares exact arity 1, names the child `input`, formats SQL as `<child>.<field>`, infers return dtype by looking up a field in an input struct dtype, and makes a non-nullable field nullable when the input struct is nullable. Execution evaluates the child as a `StructArray`, selects an unmasked field by name, and masks the result with the parent struct validity when needed. `GetItem` also implements pack-specific `reduce` and untyped simplification rules, exposes stats through a `FieldPath`, reports itself as null-sensitive, and marks itself infallible once type checking succeeds. The function is registered in `vortex-array/src/scalar_fn/session.rs`, and user-facing helpers are in `vortex-array/src/expr/exprs.rs` as `col` and `get_item`. `VariantGet` should follow the same registration, option serialization, expression helper, return dtype inference, SQL/display formatting, and construction-test patterns, while its execution dispatches on `DType::Variant` rather than `DType::Struct`.

The local `parquet-variant-compute` crate provides the runtime semantics to mirror. Its `variant_get` accepts `GetOptions` containing a `VariantPath`, an optional Arrow `Field` for typed output, and Arrow cast options. `follow_shredded_path_element` traverses shredded object fields when `typed_value` is a struct, returns missing when both raw and typed storage prove the path absent, and returns not-shredded when raw `value` must be used. `shredded_get_path` accumulates nulls while walking shredded fields, returns null arrays for known-missing paths, uses row-wise raw extraction when the path leaves the shredded tree, returns a nested `VariantArray` for untyped extraction, returns a perfectly shredded primitive typed child when `typed_value` exactly matches and raw `value` is absent or all null, and recursively extracts requested struct fields when typed output is a struct. It does not yet support array indexes through shredded storage, so Vortex should treat direct list-index shredded traversal as a later optimization and rely on raw fallback for initial list-index behavior.

The Parquet specs confirm the correctness constraints this inventory depends on. Variant raw values are self-describing bytes with object field ids sorted by field name, field names case-sensitive and unique, and arrays encoded as ordered offset lists. Shredding uses optional `value` and `typed_value` together: both null means a missing object field, raw value only means an unshredded value that may have any type including Variant null, typed value only means a value represented by the shredded type, and both present means a partially shredded object where the typed tree contains shredded fields and raw value contains unshredded object fields. The specs require readers to interpret columns by name and say typed columns take priority if a reader elects to handle invalid cases with both a field's raw and typed values present. Databricks `variant_get` is useful prior art for SQL shape and examples: `variant_get(variantExpr, path, type)` returns null for missing objects and raises `INVALID_VARIANT_CAST` for invalid casts, while `try_variant_get` nulls invalid casts. This plan keeps Vortex's final cast behavior explicit as an unresolved decision until tests choose strict or safe semantics.

## Plan of Work

First, record the implementation inventory in this plan and identify the exact baseline tests that already exist. Add tests before changing behavior where the current implementation can support them. Baseline tests should cover top-level null versus present `variantnull`, `ParquetVariantData::from_arrow_variant` roundtrips, and row-preserving slice, filter, and take on Parquet Variant arrays. The inventory must explicitly identify which encoding-specific slots represent shredded values, starting with Parquet `typed_value`, and how those slots should appear after canonicalization.

Next, change the canonical `VariantArray` shape in `vortex-array/src/arrays/variant`. Replace the `child()`-centric API with `core_storage()` and `shredded()` accessors. Add checked constructors instead of constructing through unchecked parts. Preserve a compatibility shim only if nearby call sites need a short migration step, and remove or de-emphasize it once call sites are updated. Validation must reject missing core storage, non-variant core storage, length mismatches, shredded length mismatches, and inconsistent outer nullability. Canonicalization for encodings such as `ParquetVariant` must transfer the encoding-specific shredded tree into the canonical `shredded` child so generic canonical operations see the same shape regardless of the source encoding.

Then update every row-preserving transformation that handles canonical variants. Slice, filter, take, masking, canonical validity execution, and recursive canonicalization must transform `core_storage` and inline `shredded` with the same row selection. Regression tests must prove both children are transformed identically by checking values in both children after each operation.

After the array shape is stable, add a `VariantGet` scalar function in `vortex-array`. It needs options containing a strict path and an optional output dtype. It needs expression helper functions, return dtype inference, registration in `ScalarFnSession`, display and SQL formatting, and protobuf serialization if expression serialization is expected for scalar functions in this repo. Construction tests should cover valid paths, optional dtype normalization, non-variant input type errors, serialization round trips, and nullable return dtype inference.

After the skeleton exists, implement Parquet Variant execution. Start with a correct fallback that uses `parquet-variant-compute` or the same behavior against Arrow arrays. This fallback must handle object fields, list indexes when supported by the reference crate, missing paths, top-level nulls, present `variantnull`, non-object path traversal, type mismatches, and optional dtype extraction. The fallback can be slower than a native Vortex implementation because this milestone is about semantics. It must be callable directly for a `ParquetVariant` child so direct encoding execution works before any canonicalization step.

Finally, add shredded fast paths. Exact shredded dtype matches can return the typed child with accumulated parent validity. Partially shredded objects must use typed shredded values where present and raw `core_storage` fallback where not present. The raw storage must not be rewritten by `VariantGet`; unchanged raw values remain available for later paths that were not shredded. Implement and test all three shapes: direct `VariantGet` on `ParquetVariant`, `VariantGet` on canonical `VariantArray` with `VariantArray::shredded`, and `VariantGet` on canonical `VariantArray` whose `core_storage` child still has encoding-specific shredded data.

## Concrete Steps

Run all commands from `/Users/adamgs/code/vortex`.

1. Maintain this plan and `PROMPT.md`.

   After every completed TODO, update only the TODO section of `PROMPT.md`, update this plan's `Progress` and any changed decisions, review the diff, run the appropriate validation commands, and commit only the files for that TODO.

   Inventory milestone status: completed in this document on 2026-05-07T12:08Z. Because this milestone changes only Markdown planning files, validate by inspecting the diff and running `git diff --check`; do not run broad Rust checks for this docs-only commit.

2. Baseline tests before shape changes.

   Add tests in the narrowest files that already cover the behavior:

       cargo nextest run -p vortex-parquet-variant
       cargo nextest run -p vortex-array variant

   Expected result: tests pass before the shape change and describe current behavior precisely enough that later failures identify regressions.

   Baseline milestone status: completed on 2026-05-07T12:42Z. New tests are `array::tests::test_arrow_variant_roundtrip_with_variant_null_and_outer_null`, `operations::tests::test_outer_null_and_variant_null_are_distinct`, `kernel::tests::test_filter_shredded_typed_value`, and `kernel::tests::test_take_shredded_typed_value`. Existing tests cover unshredded slice/filter/take, typed-only slice, and the other `ParquetVariantData::from_arrow_variant` roundtrips.

3. Canonical `VariantArray` shape.

   Edit `vortex-array/src/arrays/variant/mod.rs` and `vortex-array/src/arrays/variant/vtable/*.rs`. Add slot constants for `core_storage` and `shredded`, accessors, checked constructors, validation, serde updates, scalar delegation to `core_storage`, and validity delegation to `core_storage`. Update canonicalization for `ParquetVariant` so Parquet `typed_value` is exposed through the canonical `shredded` child instead of remaining visible only through Parquet-specific slots.

   Focused validation:

       cargo nextest run -p vortex-array variant

   Canonical shape milestone status: completed on 2026-05-07T13:24Z. New tests cover `VariantArray::try_new` with and without shredded storage, non-variant core storage rejection, shredded length mismatch rejection, Arrow Parquet Variant imports exposing `typed_value` as canonical `shredded`, and direct Parquet Variant canonical execution exposing `typed_value` as canonical `shredded`. Because this milestone changed public Rust APIs, `./scripts/public-api.sh` updated `vortex-array/public-api.lock`.

4. Row-preserving transforms.

   Update the canonical variant branches in filter, take, mask, and canonical execution paths. Add regression tests that construct a variant with a simple `core_storage` and an easy-to-check `shredded` child, then slice, filter, and take it.

   Focused validation:

       cargo nextest run -p vortex-array variant
       cargo nextest run -p vortex-parquet-variant

   Row-preserving transform milestone status: completed on 2026-05-07T14:06Z. `Variant` now has parent reduce rules for slice, filter, and take, and fallback canonical execution paths rebuild variants with both transformed children. New tests cover slice, filter, take, and mask over a canonical Variant whose core storage and shredded child contain distinct row values.

5. `VariantGet` skeleton.

   Add a scalar function module under `vortex-array/src/scalar_fn/fns/variant_get/`, register it in `vortex-array/src/scalar_fn/session.rs`, expose helpers from `vortex-array/src/expr/exprs.rs`, and add protobuf messages to `vortex-proto/proto/expr.proto` if serialization is required. If generated protobuf files must be refreshed, run the repo's established proto generation command if one exists; otherwise update generated files consistently with nearby messages and verify with tests.

   Focused validation:

       cargo nextest run -p vortex-array variant_get
       cargo nextest run -p vortex-proto

   VariantGet skeleton milestone status: completed on 2026-05-07T12:15Z. New code defines `VariantGet`, `VariantGetOptions`, `VariantPath`, and `VariantPathElement`; registers `vortex.variant_get`; exposes `expr::variant_get`; serializes path elements and optional dtype through `VariantGetOpts`; and tests path parsing/display, nullable dtype inference, non-variant input rejection, SQL formatting, options serde, and expression serde. `cargo nextest run -p vortex-proto` compiles but reports no tests, so the passing command for that crate is `cargo nextest run -p vortex-proto --no-tests pass`.

6. Parquet Variant unshredded fallback.

   Add an execution parent kernel or scalar function execution path in `encodings/parquet-variant` that can execute `VariantGet` over `ParquetVariant`. Prefer delegating to `parquet-variant-compute` first. Add tests that compare Vortex results to `parquet-variant-compute` for untyped and typed paths. Include direct `ParquetVariant` tests so the implementation does not require callers to canonicalize before extracting.

   Focused validation:

       cargo nextest run -p vortex-parquet-variant variant_get
       cargo nextest run -p vortex-array variant_get

7. Shredded and partially shredded extraction.

   Add tests and implementation for exact typed shredded output, missing shredded fields, partially shredded object rows, and typed priority when both raw fallback and shredded data are involved. Preserve raw `core_storage` unchanged. Cover canonical variants where the shredded tree is present as `VariantArray::shredded`, and canonical variants where the shredded tree is still discoverable through the `core_storage` encoding.

   Focused validation:

       cargo nextest run -p vortex-parquet-variant variant_get
       cargo nextest run -p vortex-array variant

8. Final validation before any Rust behavior commit that contributes to the feature.

   The two required feature test commands are:

       cargo nextest run -p vortex-array
       cargo nextest run -p vortex-parquet-variant

   Required formatting and lint commands are:

       cargo fmt --check
       cargo clippy --all-targets -- -D warnings

   When public Rust APIs or generated API locks may have changed, also run the repository-required public API check:

       ./scripts/public-api.sh

   If a command fails exactly with `sccache: error: Operation not permitted`, rerun that same command with `RUSTC_WRAPPER=`.

## Validation and Acceptance

The feature is acceptable only when all of the following behavior is covered by tests and the required validation commands pass.

Baseline acceptance:

- A nullable Variant row that is top-level null remains distinguishable from a non-null Variant row whose payload is `variantnull`.
- `ParquetVariantData::from_arrow_variant` roundtrips Arrow storage with value-only, typed-value-only, value-plus-typed-value, and top-level nulls.
- Slice, filter, and take over current Parquet Variant arrays preserve scalar results and validity.

Canonical array acceptance:

- `VariantArray::try_new(core_storage, None)` builds a variant whose dtype, length, validity, scalar values, and statistics come from `core_storage`.
- `VariantArray::try_new(core_storage, Some(shredded))` rejects length mismatches and exposes both children with slot names `core_storage` and `shredded`.
- Canonicalizing a `ParquetVariant` that has `typed_value` exposes that typed tree through `VariantArray::shredded`; generic callers do not need to inspect Parquet-specific slots to find shredded data.
- Slice, filter, take, and masking apply the same row selection to both `core_storage` and `shredded`.
- Recursive canonicalization and canonical validity execution do not drop or stale-reference the shredded child.

Expression acceptance:

- `VariantGet` rejects non-variant input during return dtype inference.
- With no requested dtype, `VariantGet` returns `DType::Variant(Nullable)`.
- With a requested dtype, `VariantGet` returns that dtype with nullable nullability.
- Expression serialization roundtrips path elements and optional dtype.
- SQL formatting is stable enough for explain output and debugging.

Runtime acceptance:

- Extracting `a` from `{"a": 1, "b": "x"}` returns `1`.
- Extracting `data[1].a` from `{"data": [4, {"a": "hello"}, "str"]}` returns `"hello"`.
- Missing paths return null.
- Top-level null rows return null.
- Present `variantnull` at the target path returns null for typed extraction and remains a present variant null for untyped extraction if the underlying representation can preserve that distinction.
- Non-object field traversal and out-of-range list indexes return null rather than reading unrelated data.
- Typed extraction handles type mismatches according to the final decision recorded in this plan.
- A fully shredded typed path returns the shredded child with parent validity applied.
- A partially shredded object path uses shredded values where valid and raw fallback rows otherwise.
- A path absent from the shredded tree is extracted from raw `core_storage`.
- Raw `core_storage` remains unchanged after `VariantGet`.
- The same path and dtype produce equivalent results for direct `VariantGet` on `ParquetVariant`, `VariantGet` on a canonical `VariantArray` with a canonical `shredded` child, and `VariantGet` on a canonical `VariantArray` whose `core_storage` still exposes encoding-specific shredded data.

## Idempotence and Recovery

All changes should be incremental and safe to rerun. Do not use destructive git commands. Keep unrelated untracked or modified files out of commits by staging only the files for the current TODO. If a test fails, record the failure in `Surprises & Discoveries`, fix the code or plan before committing, and rerun the narrowest failing command first. If a generated file update is required, rerun the generator or update the generated output in the same commit as the proto/source change that requires it.

## Artifacts and Notes

External source facts used in this plan:

- Parquet Variant encoding defines primitive, short string, object, and array basic types; object field ids are sorted by field name and field names are case-sensitive and unique.
- Parquet Variant shredding defines `value` and `typed_value` interpretation. Missing object fields have both null; present variant null is encoded in `value` as the Variant null bytes; typed values take priority for shredded reads when present.
- Databricks `variant_get` syntax is `variant_get(variantExpr, path, type)`. Databricks returns null for missing paths and raises `INVALID_VARIANT_CAST` for invalid casts, while `try_variant_get` returns null for invalid casts.
- RFC 0058 requires filter and slice to apply the same row operation to `core_storage` and `shredded`.

Reference URLs:

- https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
- https://github.com/apache/parquet-format/blob/master/VariantShredding.md
- https://docs.databricks.com/aws/en/sql/language-manual/functions/variant_get

Baseline validation commands run on 2026-05-07:

    cargo nextest run -p vortex-parquet-variant
    cargo nextest run -p vortex-array variant
    cargo nextest run -p vortex-array
    cargo +nightly fmt --check
    cargo +nightly fmt --all --check
    cargo clippy --all-targets -- -D warnings

All commands passed. Stable `cargo fmt --check` was also attempted, but this repository's rustfmt configuration uses nightly-only settings and stable rustfmt exited with configuration warnings, so the repository-required nightly fmt check was used for formatting validation.

Canonical shape validation commands run on 2026-05-07:

    cargo nextest run -p vortex-array variant
    cargo nextest run -p vortex-parquet-variant
    cargo nextest run -p vortex-array
    ./scripts/public-api.sh
    cargo clippy --all-targets -- -D warnings
    cargo +nightly fmt --all --check

All commands passed. The repository uses nightly-only rustfmt configuration, so the repository-required nightly fmt check was used for formatting validation.

Row-preserving transform validation commands run on 2026-05-07:

    cargo nextest run -p vortex-array variant
    cargo nextest run -p vortex-array
    cargo nextest run -p vortex-parquet-variant
    cargo +nightly fmt --all --check
    cargo clippy --all-targets -- -D warnings

All commands passed. No public Rust API changed in this milestone, so `./scripts/public-api.sh` was not required.

## Interfaces and Dependencies

The expected canonical variant API in `vortex-array` is:

    pub trait VariantArrayExt: TypedArrayRef<Variant> {
        fn core_storage(&self) -> &ArrayRef;
        fn shredded(&self) -> Option<&ArrayRef>;
    }

    impl Array<Variant> {
        pub fn try_new(core_storage: ArrayRef, shredded: Option<ArrayRef>) -> VortexResult<Self>;
    }

When canonicalizing an encoding-specific variant array, expose any encoding-specific shredded tree through this API. For `ParquetVariant`, the optional `typed_value` tree becomes the logical canonical `shredded` child. The implementation may move the child physically or expose it through metadata-backed accessors, but the public canonical contract is the same: callers ask the canonical `VariantArray` for `core_storage` and optional `shredded`, not for Parquet-specific child names.

The expected expression API in `vortex-array` is:

    pub struct VariantGet;

    pub struct VariantGetOptions {
        path: VariantPath,
        as_dtype: Option<DType>,
    }

    pub fn variant_get(path: impl Into<VariantPath>, child: Expression) -> Expression;

    pub fn variant_get_as(
        path: impl Into<VariantPath>,
        as_dtype: DType,
        child: Expression,
    ) -> Expression;

The initial path model is a strict sequence of object field names and zero-based list indexes. Full JSONPath syntax, wildcards, negative indexes, quoted escaping, and recursive descent are not required for the first implementation unless a test or integration point proves they are needed.

## Unresolved Decisions

- Decide whether a later cleanup should physically remove encoding-specific shredded children from `core_storage`. The current implementation allows `core_storage` to retain them for direct encoding kernels while still exposing the logical canonical `shredded` child.
- Decide final typed cast semantics. RFC 0015 leans toward null for mismatches, `parquet-variant-compute` supports safe nulling, and Databricks `variant_get` raises `INVALID_VARIANT_CAST` while `try_variant_get` nulls. The implementation must make the chosen behavior explicit in tests.
- Decide the exact path parser grammar. The minimum is field names and zero-based list indexes; escaping, quoted field names, negative indexes, and wildcards remain out of scope until explicitly added.
- Decide how much numeric coercion is allowed for typed extraction. Numeric widening is plausible; lossy casts, string parsing, timestamps, decimals, and timezone-sensitive casts need explicit tests before support.
- Decide how to validate consistency between raw `core_storage` and `shredded` when both can represent the same logical path. The Parquet spec says writers must avoid conflicts, but Vortex constructors may still need checked errors or debug assertions.
- Decide whether `VariantArrayExt::child()` should be removed after row-preserving transformations and callers are migrated to `core_storage()`.
- Decide the narrowest higher-level integration after core tests pass. DataFusion or scan/projection wiring should wait until the core expression and Parquet behavior are stable.

## Revision Notes

2026-05-07T11:21Z: Created the initial ExecPlan from the RFCs, Parquet specs, Databricks function docs, current Vortex files, `parquet-variant-compute`, and the `adamg/variant-array` reference branch. This revision establishes milestones, validation commands, acceptance cases, and unresolved decisions before source changes.

2026-05-07T11:45Z: Clarified the canonicalization boundary for shredded data. Encoding-specific shredded children such as Parquet `typed_value` should surface as the canonical `VariantArray::shredded` child, while `VariantGet` must also support direct execution on `ParquetVariant` and canonical variants whose `core_storage` still exposes encoding-specific shredded data.

2026-05-07T12:08Z: Added the first implementation inventory. This revision records the exact current one-child canonical variant shape, the RFC 0058 target shape, Parquet Variant storage and transformation hooks, `GetItem` scalar function patterns to reuse, and the logical canonicalization contract for surfacing Parquet `typed_value` as canonical `VariantArray::shredded`.

2026-05-07T12:42Z: Added baseline tests before changing canonical variant shape. This revision records the new coverage for outer null versus present `variantnull`, a roundtrip containing both null kinds, and filter/take over typed-only shredded Parquet Variant storage, plus the validation commands that passed for this Rust-only baseline milestone.

2026-05-07T13:24Z: Implemented the canonical `VariantArray` shape with `core_storage` and optional `shredded` slots, checked construction, serde support, validation, Parquet canonicalization of `typed_value`, public API lock updates, and focused tests. The compatibility `child()` shim remains until the row-preserving transform milestone migrates callers.

2026-05-07T14:06Z: Added row-preserving `Variant` slice/filter/take reduce rules and updated canonical filter, dict take, mask validity, canonical-validity, and recursive canonicalization paths to transform optional shredded storage with the same row operation as core storage. Regression tests now assert both children after slice, filter, take, and mask.
