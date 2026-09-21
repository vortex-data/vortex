<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Experiments that resolve the remaining questions

[Research overview](README.md)

The next useful step is a small extraction experiment with two independent storage adapters. The
[compiled type proof](type-system/compiled-proof.md) establishes the Rust mechanisms with small
adapters. It does not reuse the actual RowFn executor or establish host integration.

These are proposed follow-ups. This research branch does not implement the extracted library.

## 1. Establish the semantic boundary

Use one unchanged function definition with a plain-slice adapter and a Vortex adapter. The plain
adapter must compile without a Vortex dependency. This exposes hidden requirements in type
dispatch, error handling, and input/output contracts.

The first cases need different properties:

| Function | Question it answers |
| --- | --- |
| Explicit wrapping `i64` addition | Can the binder select typed loops without Vortex types? |
| Checked integer division | Can demanded-row errors and null suppression remain exact? |
| Timestamp truncation | Can binding preserve unit and timezone metadata in empty and nonempty outputs? |
| UTF-8 transformation | Can borrowed inputs and variable-length output ownership remain portable? |
| Fixed-size vector operation | Can runtime shape parameters reach typed readers and writers? |

A successful experiment shares dispatch semantics and row computation. Adapter-specific imports in
the function definition indicate an unresolved boundary. Rejection of an unsupported type is an
expected result, not a failure of genericity.

## 2. Add Arrow and DataFusion

An arrow-rs adapter demonstrates a second real columnar representation. A DataFusion UDF adapter
then adds binding, scalar arguments, host metadata, and output registration.

The comparison needs empty batches, all-null batches, scalar/array combinations, sliced arrays,
Boolean output, and nested validity. Metadata must match even when the row closure never runs.

Acceptance requires the same supported semantic function in Vortex and DataFusion, with identical
values, nulls, and error behavior. Host-specific casts must be explicit in the comparison.

## 3. Decide the DuckDB boundary

The [DuckDB investigation](integrations/README.md) identifies two useful experiments:

- A C scalar-function adapter establishes the simpler native callback boundary and records
  flattening costs.
- A version-matched C++ adapter preserves vector representation information and uses a batch call
  into the shared Rust implementation.

Keep the row kernel identical in both cases. Measure constant, dictionary, flat, and string inputs
separately. A result that excludes flattening or output conversion does not establish total adapter
overhead.

## 4. Introduce demand through one expression path

Use `CASE WHEN denominator != 0 THEN numerator / denominator ELSE 0 END` as the first evaluator
experiment. Include zero denominators only outside the division's demanded rows.

The request must reach child evaluation, decoding, constant preparation, and the row loop. A mask
added only to the final loop cannot suppress earlier errors or earlier work.

Then exercise the harder cases:

- An empty request with an invalid constant expression.
- A row-local decoder error outside the requested rows.
- Unreferenced dictionary values that cause errors.
- A cached expression requested first on one subset and later on an overlapping subset.
- A nested input with independent parent and child validity.
- A null-producing function, which requires a wider contract than current RowFn.

The [definedness contract](definedness/README.md) specifies the expected outcomes. Begin with the
chosen error policy, then compare executions against it.

## 5. Attribute overhead before optimizing the design

Use the [measurement protocol](performance/README.md) to separate binding, batch setup, decoding,
row computation, and output construction. Include both a prepared invocation and a full invocation.

Match semantics, allocation ownership, constant handling, nulls, and output representation between
the baseline and the framework. Record absolute batch time and time per demanded row. Small-batch
percentages alone cannot identify a cause.

Each intervention needs its own comparison. Examples include removing repeated binding, borrowing
argument descriptors, or retaining an encoded input. Generated-code inspection then determines
whether the expected loop specialization survived compilation.

Native x86 and ARM measurements answer different questions. CPU simulation, compiler output, and
end-to-end query timing also need separate labels.

## Decisions to make after the experiments

| Decision | Evidence needed |
| --- | --- |
| Public type protocol | Two hosts can bind the same function without host-specific dispatch. |
| Public reader/writer protocol | Primitive, string, nested, and extension cases preserve ownership and semantics. |
| Demand in the initial API | Selected execution suppresses row-local work and errors through the full path. |
| Retained bound call | Binding amortizes without stale types, options, constants, or host context. |
| DuckDB adapter form | Measured conversion cost and the required ABI support policy. |
| Initial function coverage | Strict, null-producing, and null-aware contracts have explicit boundaries. |

Library publication follows these decisions. Naming a crate or stabilizing its API before the
experiments does not resolve them.
