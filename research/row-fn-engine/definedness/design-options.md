<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Demand API and output choices

[Demand overview](README.md) | [Contract](contract.md)

The recommendation is an explicit execution selection with an all-rows default. Kernels keep their
typed row interface. The evaluator and adapters own selection propagation and completed results.

The alternatives differ in where they place that contract. None is implemented in this branch.

## Where demand enters execution

| Option | Benefit | Limitation |
| --- | --- | --- |
| Mask the inputs of a strict function. | Reuses some current validity machinery. | Changes data nullability and cannot suppress child errors that already occurred. It does not support general partial reuse. |
| Add demand to execution arguments or a dedicated call object. | Makes the row domain and request explicit. Supports selected decoding and output positions. | Requires propagation through lazy child evaluation and adapters. |
| Store demand in `ExecutionCtx`. | Avoids another visible argument. | Hides call-specific row identity and complicates nested evaluation and restoration. |
| Compact inputs before invocation. | Every local row is requested and complete on success. | Requires gather and domain mapping, plus scatter or merge for conditionals. |

An explicit call object or `ExecutionArgs` extension is the best initial boundary. Compact batches
remain its all-rows special case. Query context continues to own services such as allocation and
cancellation, without becoming an implicit selection stack.

For already evaluated, pure strict inputs, these expressions agree on successful values:

```text
mask(f(a, b), demand)
f(mask(a, demand), mask(b, demand))
```

This value identity is not an evaluation-order or error-suppression proof. The first expression can
evaluate `f` before masking. Either input expression can fail before the function receives it.
The identity also fails for general non-strict functions such as `is_null`.

## How the current policies can contribute

| Policy | Useful existing mechanism | Additional requirement for demand |
| --- | --- | --- |
| `Dense`. | Direct typed traversal and lazy validity attachment. | Safe and effect-free extra work, with no visible inactive-row errors. |
| `ValidOnly`. | Set-bit traversal or filtered input with writes into original output positions. | Selection must reach row-local decoding before it can reject inactive values. |
| `DenseWithRetry`. | Compact failure evidence followed by selected replay. | Resolve demand as well as validity, preserve the host error policy, and protect prior completed output. |

Current `DENSE_SAFE` proves null-payload access under the input contract. It does not prove that an
arbitrary partial input is initialized. Nor does it make value parsing outside demand safe to expose.

## Output choices

| Representation | Useful case | Required contract |
| --- | --- | --- |
| Full-length initialized storage plus completion coverage. | Shared partial results and repeated requests. | Reads stay within completion. A null result counts as complete. |
| Null placeholders outside demand. | A host explicitly accepts nullable filled output. | Preserve demand or completion separately. Select rows before a cast to a non-nullable result. |
| Compact output with row mapping. | Filters and hosts that already compact branch inputs. | Retain the mapping to the original domain for merge or scatter. |
| Unspecified slots behind a private execution handle. | An executor owns the result and restricts all reads. | Maintain initialization and ownership proofs. Do not export an ordinary complete array. |

The null-placeholder option is useful for Vortex storage, but cannot define the semantic contract
alone. A computed null and an uncomputed row must remain distinguishable at a cache or merge boundary.

The first general design can use initialized storage behind an opaque completion wrapper.
A successful ordinary call needs no extra completion bitmap when its request already carries
coverage. Persistent partial results retain that coverage. Sparse uninitialized storage is a
separate optimization with a separate unsafe contract.

## Conditional propagation

For `CASE`, each selected branch receives its remaining requested rows. Null conditions select
neither a true branch nor an error by themselves. `COALESCE` requests its next argument only where
earlier completed results are null, subject to the host's child-error policy.

Kleene `AND` can omit right-side rows where the left side is false. `OR` can omit rows where the
left side is true. Null rows still need the other operand. Error behavior and operand reordering
remain host policies, not consequences of three-valued logic alone.

Dictionary execution maps requested logical rows into the referenced value domain. Unreferenced
error-producing values must remain inactive. Nested execution also needs a mapping from child
errors back to requested parent rows. The [worked cases](worked-cases.md) cover both.

## Representation and tuning

All, none, ranges, bitmaps, and index lists describe useful selections. Conversion belongs at the
batch boundary. No fixed density threshold follows from the contract.

The crossover depends on callback cost, run lengths, decoding, and output initialization.
A sparse row loop can still write a full-length output. Measure those costs before selecting dense
speculation or compaction. The [adoption plan](adoption.md) lists the correctness and timing cases.
