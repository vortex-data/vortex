<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Velox demand, errors, and partial results

Velox provides direct prior art for this execution contract. It passes selected rows through
expression evaluation and maintains separate state for errors and partial results. Its public
documentation describes recursive selection, encoding translation, null pruning, and partial
common-subexpression reuse. [Expression evaluation documentation](https://facebookincubator.github.io/velox/develop/expression-evaluation.html).

The source references below use commit `c53a86adc4c971a752c68d964b4000e436edf2c8`, fetched on
2026-09-21. The portable RowFn proposals in this tree remain independent design recommendations.

## Selection has its own row domain

`SelectivityVector` selects positions without relocating the data. It stores bits, logical size,
selected bounds, and a cached all-selected property. Its iteration separates the all-selected loop
from bitmap traversal. This establishes an example of explicit demand without a branch per row in
the dense path. It does not establish the performance of a Rust extraction.
[SelectivityVector](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/vector/SelectivityVector.h#L33-L50),
[iteration](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/vector/SelectivityVector.h#L438-L451).

`translateToInnerRows` maps selected outer positions through dictionary indices. It skips outer
nulls before reading their code. The resulting inner selection deduplicates references through
bitmap membership. This is the concrete operation behind `D_values = c(D intersect V_codes)`.
[Dictionary row translation](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/vector/SelectivityVector.cpp#L91-L104).

## Conditionals preserve earlier branch results

`SwitchExpr` evaluates each condition on remaining rows. It evaluates a branch value on matching
rows, removes those rows, and evaluates ELSE on the remainder. It passes the selection into child
evaluation. [Switch execution](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/SwitchExpr.cpp#L71-L184).

`EvalCtx::isFinalSelection` distinguishes a final request from an intermediate selection.
`moveOrCopyResult` preserves a previously populated result when required. `EvalCtx` also provides
per-row errors and mappings from nested element errors to parent rows.
[Result preservation](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/EvalCtx.h#L432-L503),
[element error mapping](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/EvalCtx.h#L357-L369).

The design lesson is that writable result ownership and completion coverage need coordination.
A branch cannot assume that it owns every position in its destination.

## Errors

`ConjunctExpr` temporarily disables immediate throwing. It keeps rows with errors active so that a
later operand can determine their outcome. Finalization removes errors from rows whose final
Boolean result makes those errors irrelevant. This supports operand reordering without exposing
otherwise suppressed errors.
[Conjunct evaluation and finalization](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/ConjunctExpr.cpp#L66-L178).

`TryExpr` handles captured row errors by setting output nulls on selected error rows. It can wrap a
constant result to add row-specific nulls. The validity update represents an explicit error-to-null
operation. It is not evidence that errors and nulls are interchangeable.
[TRY result handling](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/TryExpr.cpp#L106-L166).

The design lesson is that demand and error capture solve different problems. Selection prevents
known-unnecessary work. Error capture supports parents that cannot yet decide whether a child error
is observable.

## Cached coverage

`Expr` stores common-subexpression values together with the rows those values cover. It removes
error rows from the reusable coverage. A later request computes missing rows and merges successful
coverage. Its final-selection handling preserves previously computed values.
[Shared expression results](https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/Expr.cpp#L978-L1047).

The design lesson is that a cache must remember successful nulls as completed results. It must also
distinguish unavailable rows from initialized placeholder storage.
