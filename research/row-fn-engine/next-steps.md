<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Next steps

[Overview](README.md)

The next useful experiment shares a real RowFn operation and executor between Vortex and a second
adapter. It must preserve semantics, ownership, and generated code before the library API becomes
stable. The isolated [type proof](type-system/compiled-proof.md) covers only the Rust mechanisms.

The portable library and adapters remain proposals. The branch base already contains the
[September 23 and 24 RowFn improvements](current-system/recent-changes.md), including output
allocation, packed Boolean retry, and mask reduction. The steps below start from that implementation.

## 1. Fix the contract before the crate split

The prototype needs four explicit decisions:

- Accepted semantic types, outer nullability, nested constraints, and unsupported mappings.
- Binding errors, row-local errors, decoder errors, and permitted reevaluation.
- Input ownership, writer completion, allocation, and exact output metadata.
- Caller demand, successful completion, and export of partial results.

The recommendation is in [the design](architecture.md). Start with the types and execution machinery
that the shared executor consumes. Host glue resolves runtime types, decodes inputs, and constructs
results. The existing `OutputBuffer` boundary supplies collection storage and publication, so the
prototype can generalize it without requiring another output layer or one large `Engine` trait.

## 2. Prove extraction with representative functions

Start with a Vortex adapter and a plain-slice adapter that compiles without Vortex arrays.
Then add Arrow to prove the boundary with a second columnar representation.

| Function | Required evidence |
| --- | --- |
| Wrapping `i64` addition. | One binder and row operation, with equal values and output metadata. |
| Checked integer division. | Exact errors on active rows, with invalid null and inactive payloads suppressed. |
| Timestamp truncation. | Preserved unit, interpretation, timezone, and empty-output metadata. |
| UTF-8 transformation. | Borrowed input lifetime, selected validation, and owned output bytes. |
| Fixed-size vector operation. | Runtime shape, child nullability, and sink initialization. |

The function definition must not import a host array type. Host-specific extension functions can
retain a native dispatch path. Explicit rejection of an unsupported mapping is a successful outcome.

Keep the Vortex implementation in place during the prototype. Move crates after the second adapter
demonstrates the boundary. Buffer and lane-kernel dependencies need a deliberate decision, not a
second copy of compiler-sensitive loops.

## 3. Evaluate performance fixes independently

The [candidate list](performance/optimization-candidates.md) identifies work that does not depend
on extraction:

| Candidate | What the experiment must distinguish |
| --- | --- |
| Retain UTF-8 validation evidence. | Saved work versus required validation and null-payload sanitation. |
| Delay rich error construction after rejected deferred evidence. | Retry signaling versus an observable error, with and without backtraces. |
| Pack selected and filtered Boolean output directly. | Avoided byte storage versus sparse writes, failure evidence, and original row positions. Dense retry already packs directly. |
| Retain a bound call and classify constants once. | Saved binding work versus changing batch types, options, and owners. |
| Measure remaining validity and finalization work. | Composition and wrapper costs after eligible all-valid arrays gained direct lazy-mask attachment. |
| Improve output collection or reuse. | Wrapper cost versus collector code generation and ownership, using the existing allocator-aware buffer contract. |

Each candidate needs a separate comparison. Combining them with the crate move prevents useful
attribution. UTF-8 decoding already avoids the intermediate array, but still validates every decode.
The ARM harness and x86 sweep predate these changes. Establish new measurements before using their
numbers to prioritize current work, and preserve their original sources as historical evidence.

## 4. Prove demand through one conditional

Use `CASE WHEN denominator != 0 THEN numerator / denominator ELSE 0 END`. Put zero divisors only
outside the division branch first, then inside it. Propagate demand through child evaluation,
decoding, preparation, and traversal.

The next cases are empty demand, invalid constants, row-local parsing errors, unused dictionary
entries, nested validity, and overlapping requests for a cached result. Boolean error suppression
needs the host's own policy. `TRY` and functions that produce null need wider contracts.

The [worked cases](definedness/worked-cases.md) specify expected outcomes. Begin with selected
execution or compaction, then compare dense speculation under the same semantics.

## 5. Add real host integration

DataFusion can reuse Arrow readers and writers. Its adapter still needs signatures, scalar
arguments, coercion, output fields, and optimizer metadata. A foreign UDF route needs separate
measurements because it can materialize scalars before the call.

For DuckDB, compare the C callback with a version-matched C++ adapter. Include input flattening,
constants, dictionaries, strings, output allocation, and error conversion in the timing boundary.
The [integration notes](integrations/README.md) describe both contracts.

## Evidence required before publication

The prototype needs equivalent values, nulls, errors, and output metadata across supported hosts.
It also needs a review of unsafe ownership and initialization contracts.

Performance work needs matched semantic baselines, small and large batches, and separate ARM and
x86 results. Generated-code inspection must accompany compiler-sensitive changes. Host benchmarks
must include adapter work. Query-level results require actual query integration.

The [measurement plan](performance/measurement-plan.md) lists the remaining matrix. Checks remain
explicit follow-up work, under the repository's verification policy.
