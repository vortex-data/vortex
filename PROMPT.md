# Variant array and VariantGet

Use the two variant RFCs in the local `rfc` repo found in `~/code/rfcs` to formulate plan to implement the missing pieces described there.

Resources:
1. RFCs repo as described above
2. Parquet spec for Variant shredding: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
3. Variant encoding: https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
4. Can use the `adamg/variant-array` branch as a general reference, especially for useful tests and some of the metadata implementation for things. Only use it as an insperation, prefer RFCs and specification when things are unclear.
5. For runtime behavior and correctness, look at the `parquet-variant-compute` crate, thas is available locally in /Users/adamgs/code/arrow-rs/parquet-variant-compute.
6. The Databricks `variant_get` function: https://docs.databricks.com/aws/en/sql/language-manual/functions/variant_get


Requirements:

- Review your own diff before committing. Do that review in the context of the Variant array nad expression, and explicitly check for ownership and lifetime mistakes, missing cleanup on error paths, behavioral regressions, and missing test coverage.
- Do not commit if your review finds problems. Fix them first.
- Before committing, formatting and lint checks must pass
- Before committing, all required tests must pass tests

Definition of done:

- An end to end implementation exists, and its correct and allows for future improvement. Prioritize correctness over performance.
- Your review is complete and does not identify unresolved issues.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` pass before commit.
- Both required test commands pass before commit.

## Instructions

You are in a Ralph Wiggum loop. You are making progress on the plan defined above. Work through the first few TODOs in the `## TODO` section below.

- update PROMPT.md with an updated TODO list after each change
- never ever change any PROMPT.md text _except_ the items in the `## TODO` section
- you may update the TODO items as you see fit--remove outdated items, add new items, mark items as completed
- commit after each completed TODO
- use conventional commit syntax for commit messages
- if there are no items left in TODO, append a final line in PROMPT.md that contains only the emoji: ✅

## TODO
- [x] Bootstrap the implementation plan: read `/Users/adamgs/.codex/PLANS.md`, `/Users/adamgs/code/rfcs/rfcs/0015-variant-type.md`, `/Users/adamgs/code/rfcs/rfcs/0058-variant-get-expr.md`, the Parquet Variant encoding and shredding specs linked above, `/Users/adamgs/code/arrow-rs/parquet-variant-compute/src/variant_get.rs`, `vortex-array/src/arrays/variant/mod.rs`, `vortex-array/src/scalar_fn/fns/get_item.rs`, and `encodings/parquet-variant/src/{array,kernel,operations}.rs`; create `variant_get_execplan.md` as a self-contained ExecPlan with milestones, exact validation commands, acceptance cases, and unresolved decisions.
- [x] Take a first implementation inventory in `variant_get_execplan.md`: document the current single-child canonical `VariantArray`, the RFC 0058 target shape with `core_storage` plus optional `shredded`, the existing Parquet Variant encoding hooks, the scalar function patterns used by `GetItem`, and the canonicalization contract that encoding-specific shredded data such as ParquetVariant `typed_value` moves to the canonical `VariantArray::shredded` child when producing a canonical variant.
- [ ] Add or identify baseline tests that describe current variant behavior before changing it, including outer null versus `variantnull`, `ParquetVariantData::from_arrow_variant` roundtrips, and filter/slice/take behavior for Parquet Variant arrays.
- [ ] Implement the canonical `VariantArray` shape change in `vortex-array/src/arrays/variant`: add `core_storage()` and `shredded()` accessors, checked constructors, slot names, row-alignment/nullability validation, canonicalization that moves encoding-specific shredded children into the canonical `shredded` child, and focused tests.
- [ ] Update variant-preserving transformations so slice/filter/take keep `core_storage` and optional `shredded` row-aligned; include regression tests proving both children are transformed with the same row selection.
- [ ] Add and register a `VariantGet` scalar function skeleton in `vortex-array`, including path and optional dtype options, expression helper(s), return dtype inference, display/SQL formatting, serialization if required by existing expression infrastructure, and construction/type-error tests.
- [ ] Implement an initial unshredded `VariantGet` fallback for Parquet Variant by adapting behavior from `parquet-variant-compute`; cover object fields, list indexes if supported, missing paths, outer nulls, `variantnull`, type mismatches, and optional dtype behavior.
- [ ] Implement shredded fast-path extraction and partial-shredding merge behavior for all supported execution shapes: `VariantGet` directly on `ParquetVariant`, `VariantGet` on canonical `VariantArray` with a canonical `shredded` child, and `VariantGet` on canonical `VariantArray` whose `core_storage` child still exposes encoding-specific shredded data; prove typed shredded values take priority when present while unchanged raw storage is still used for rows or paths that are not shredded.
- [ ] Wire `VariantGet` through higher-level integration only after core array and Parquet Variant behavior is tested; add the narrowest useful DataFusion or scan/projection test for Databricks-like `variant_get` behavior if the existing APIs can express it.
- [ ] Before each commit, update `PROMPT.md` TODO state, review the diff against the active ExecPlan, run the focused crate checks for touched crates, and run the required formatting/lint/API checks from `AGENTS.md` when Rust public APIs changed.
