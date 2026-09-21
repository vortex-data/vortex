<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Design choices and evidence needed

The first portable API can support demand without implementing a complete expression evaluator.
The host must restrict child evaluation before invoking RowFn, or document that it cannot provide
that guarantee. Host adapters cannot suppress errors that already occurred.

## Choose the execution boundary

| Design | What it provides | Main tradeoff |
| --- | --- | --- |
| Compact batches only. | Every local row is requested. Completion is implicit on success. | The host owns selection, compaction, and scatter. Shared partial results need external coverage. |
| Explicit demand over a stable domain. | Selected execution, preserved row identity, and partial reuse. | Every decoder and output adapter needs a domain contract. |
| Demand encoded as nulls. | Reuses some strict null paths. | Changes logical data and fails for non-strict functions, nullable results, and partial reuse. |
| Fully eager execution. | Preserves the existing whole-array calling convention. | Cannot guarantee inactive row-error suppression for general conditionals. |

Compact batches are a valid semantic design. They can provide the first Arrow, DataFusion, or
DuckDB adapter when the host already supplies the relevant subset. No extra physical definedness
bitmap is necessary in that case.

Explicit demand is useful when the engine must avoid repeated compaction, retain encoded inputs,
or share partial results. Support compact batches as the all-rows special case. Keep an all-rows
representation that reaches a direct dense loop.

Encoding demand as nulls is not a general solution. It loses the difference between an uncomputed
row and a computed null. It also changes what a function such as `is_null` observes.

## Minimum portable contract

Start with a bound, pure, deterministic RowFn and a declared error policy. Give execution a row
domain, typed input access, and a selection with all, none, range, or sparse forms.

On success, completion over the requested domain is implicit. If a partial result escapes for later
reuse, store its coverage. Initialize output storage through the existing builder discipline.

The adapter chooses direct selected decoding or compact inputs. Compaction must occur before any
row-local parsing that the selection is meant to suppress. If the backend cannot meet that rule,
report the unsupported capability rather than silently executing irrelevant rows.

Keep these properties independent in binding metadata:

- Whether input nulls force an output null.
- Whether valid inputs can produce null.
- Whether the row operation can report a row-local error.
- Whether decoding can report a row-local error.
- Whether execution is pure, deterministic, and safe to repeat.
- Whether inactive payloads can be read safely.

The first version can restrict some properties instead of supporting every combination. For
example, a strict, null-preserving, pure kernel remains useful across multiple hosts. The restriction
must be explicit so that future nullable or volatile functions do not inherit invalid optimizations.

## Propagate demand in dependency order

1. Define the host conditional and error semantics. Distinguish binding, row-local, and infrastructure
   errors.
2. Establish row-domain ownership and compact-to-original mappings.
3. Restrict child-expression evaluation before a RowFn invocation.
4. Restrict row-local decoding, constant preparation, and row traversal.
5. Complete requested null rows and preserve the output storage invariants.
6. Track partial coverage only at caching or merge boundaries.
7. Enable dictionary deduplication and dense speculation only after their preconditions hold.

For a full Vortex implementation, the propagation path crosses scalar arguments, array execution,
filter fallback, validity evaluation, RowFn decoding, and output finalization. A new row-loop
parameter by itself covers only one stage.

The [DataFusion adapter](../integrations/datafusion.md) and
[DuckDB adapter](../integrations/duckdb.md) describe their actual invocation boundaries. Their host
semantics determine which parts can stay outside the portable engine.

## Proof obligations and targeted regressions

The following is a proposed validation matrix. No tests in this matrix were executed for this
research.

| Property | Necessary evidence |
| --- | --- |
| Inactive errors stay inactive. | Place a zero divisor or row-local parse error outside demand, then inside demand. |
| Strict nulls suppress unnecessary decoding. | Put a decode error in one input where another input is null. |
| Null results count as complete. | Return null from a valid input, then reuse the cached result without re-evaluation. |
| Conditions propagate demand. | Exercise multiple CASE branches, null conditions, nested CASE, and empty ELSE demand. |
| Boolean error policy is stable. | Compare operand orders under the host policy, including error/false AND and error/true OR. |
| Dictionary mapping is correct. | Include duplicates, null codes, unused error entries, and entries excluded only by caller demand. |
| Nested errors retain ownership. | Map element errors to the correct demanded parent rows. |
| Empty demand does no value work. | Count decode, prepare, and row calls separately for constants and nullary functions. |
| Partial reuse preserves earlier results. | Use overlapping and disjoint demands with successful values, nulls, and captured errors. |
| Retry is semantically safe. | Compare selected replay with a reference evaluator and forbid effectful kernels. |
| Representation safety survives abandonment. | Audit initialization and destructor proofs across every error prefix and panic path. |
| Host export is complete. | Reject unfilled partial output or require an explicit compact/fill operation. |

Property tests can compare selected execution against a simple scalar reference for each requested
row. They must compare nulls and error outcomes, not only valid values. Mask lengths, tail bits,
empty domains, repeated indices, and nested mappings need explicit coverage.

Tests cannot establish unsafe trait soundness. The review must trace ownership and initialization
proofs from safe construction through unchecked access and finalization.

## What to measure

Measure demand plumbing separately from the work it avoids. Compare dense execution, selected
execution, and compaction with the same row function and error policy.

Vary batch size, selected density, run length, null density, dictionary cardinality, and callback
cost. Report mask construction, selected decoding, allocation, output initialization, and merge
costs. Include all-rows and no-rows controls.

For retry, separate the successful dense case from inactive-only failures and demanded failures.
Measure repeated preparation and decoding explicitly. A demand bitmap that avoids expensive row
work can still add overhead to a cheap dense primitive kernel.

Count physically materialized bytes as well as invoked rows. Sparse traversal over full-length
initialized output can retain an `O(n)` write cost even when only a few rows execute. Compact output
trades that cost for later mapping or scatter.

These measurements belong in the [framework overhead analysis](../performance/README.md). No speedup or
fixed per-row mask cost follows from the contract alone.
