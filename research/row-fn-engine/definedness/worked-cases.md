<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Worked cases

These examples specify proposed behavior. They are algebraic examples, not results from executed
Vortex tests. The [contract](contract.md) defines the symbols.

## Inactive division by zero

For integer `10 / x`, let `x = [2, 0, 5, 0, 1]` and `D = {0, 2, 4}`.

| Row | Input | Requested? | Outcome |
| --- | --- | --- | --- |
| 0 | 2 | Yes. | 5. |
| 1 | 0 | No. | Incomplete. |
| 2 | 5 | Yes. | 2. |
| 3 | 0 | No. | Incomplete. |
| 4 | 1 | Yes. | 10. |

All input rows are valid. Validity alone cannot distinguish the required work. A dense division
followed by an output mask can report an error before that mask exists.

For `CASE WHEN x != 0 THEN 10 / x ELSE 0 END`, the division demand is `{0, 2, 4}`. The ELSE demand
is `{1, 3}`. The final output is `[5, 0, 2, 0, 10]`, complete on every row.

This requires conditional demand before branch evaluation. A RowFn mask cannot undo an error that
the host already produced while evaluating a branch argument.

## Strict null propagation

Consider an infallible binary function with input validities `V_0 = {0, 2, 3}` and `V_1 = {0, 1, 3}`.
Let `D = {0, 1, 2}`. Then `A = {0}`.

The function computes row 0. Rows 1 and 2 become complete null results. Row 3 remains incomplete,
although both inputs are valid there. Therefore `C = {0, 1, 2}` and `V = {0}`.

Every row-local decoder must enforce `A` or establish that extra work is semantically safe. Decoding
input 0 on all of its valid rows can fail on row 2. Input 1 already makes that result null.
Per-input null tolerance does not establish joint-row demand.

## Valid input that produces null

Consider a function that returns null for an empty list. Its input is `[[], [4], null]`, and all rows
are requested.

| Row | Input state | Output | Completed? | Valid? |
| --- | --- | --- | --- | --- |
| 0 | Valid empty list. | Null. | Yes. | No. |
| 1 | Valid nonempty list. | 4. | Yes. | Yes. |
| 2 | Null list. | Null. | Yes. | No. |

Neither output null requires a retry. Neither is an incomplete row. This example also explains why
null-producing functions need a result-nullability contract beyond current RowFn strictness.

## Nested conditional demand

For `CASE WHEN p THEN f(x) WHEN q THEN g(x) ELSE h(x) END`, define `R_0 = D`. At branch `k`:

```text
evaluate condition_k only on R_k
T_k = rows in R_k where condition_k is true
evaluate value_k only on T_k
R_(k+1) = R_k minus T_k
```

Null conditions remain in `R_(k+1)`. The ELSE demand is the final remainder. Without ELSE, that
remainder becomes complete null results.

Suppose `D = {0, 1, 2, 3}`, `p` is true on `{0, 2}`, and `q` is true on `{1, 2}`. The demands are
`f: {0, 2}`, `g: {1}`, and `h: {3}`. `q` needs evaluation only on `{1, 3}`.

For a nested conditional inside `f`, intersect its selection with `{0, 2}`. A child cannot replace
its inherited demand with a new all-rows mask.

## AND and OR

For a left-to-right three-valued `AND`, evaluate the right operand on requested rows where the left
operand is true or null. For `OR`, evaluate it where the left operand is false or null.

These rules preserve `null AND false = false` and `null OR true = true`. A null left operand does not
end evaluation.

The error rule needs another decision. If an engine reorders operands, evaluating an error first
must not change a result that a later false `AND` operand determines. Velox uses captured row errors
for this purpose. [Velox Boolean error handling](velox.md#errors).

A demand mask alone cannot provide order-independent error suppression. Immediate-error kernels
need selected replay or a row-error capture capability. An adapter must declare which contract it
supports.

## Dictionary rows and dictionary entries

Let dictionary values be `[2, 0, 5]`, and let codes be `[0, 2, 0, 2]`. For `10 / value`, no logical
row needs entry 1. Evaluating every dictionary value introduces a division error.

Let `c` map a logical row to its dictionary entry, and let `V_codes` identify non-null codes. The
inner demand is:

```text
D_values = { c(i) | i is in D intersect V_codes }
```

This mapping uses code validity before code access. An unspecified code payload behind a null must
not become an index. Value validity then controls whether the strict row function executes that
entry.

For `D = {0, 2}`, the inner demand is `{0}`. One pure deterministic result can serve both logical
rows. A value error at entry `j` maps back to every demanded row whose code equals `j`.

An all-values-referenced flag over the full code array is insufficient after selection. For codes
`[0, 1, 2]` and `D = {0, 2}`, entry 1 is still inactive. The optimizer needs a demand-relative proof.

Deduplication changes call count. It requires purity and deterministic results, independently of
infallibility. Functions that depend on logical row identity need that identity in their key or must
retain logical-row execution.

With two dictionary operands, independent entry demands do not establish valid input pairs.
Different code mappings require logical-row evaluation or deduplication of the paired keys.

## Nested values

A list introduces an element domain. For a function that reads every element, demand includes the
ranges of demanded, non-null parent rows. Errors in element rows map back to their parent rows.

A list-length function does not need element payloads. A predicate that stops at the first match
can need only part of a list. Parent demand therefore does not specify universal child demand.

The first RowFn API can keep this distinction inside each input capability. A later recursive demand
API must describe field or element requirements explicitly. The logical list type alone cannot
provide that execution policy.

## Retry and compact failure evidence

Suppose a dense pass OR-reduces all row failures into one Boolean. It reports failure but loses the
failing row. The evaluator cannot intersect that Boolean with demand or validity.

There are three approaches:

- Evaluate only active rows, then reduce their evidence.
- Retain row or lane error locations, then restrict errors to observable rows.
- Discard the failed attempt and repeat execution only on active rows.

Current RowFn uses the third approach for eligible deferred kernels with partial input validity.
Its decoder and allocation errors remain immediate. [Current retry path](current-vortex.md#errors-and-retry).

Retry can repeat preparation and successful row calls. `Fn` does not establish purity because
interior mutation and external IO remain possible. Current RowVisitor documentation separately
forbids side effects and panics in callbacks. Extraction must retain that requirement.
[Row callback contract](https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs#L80-L85).

Any future effectful function support needs a separate contract. Such functions cannot automatically
inherit retry, constant folding, or dictionary deduplication.

An error capture mode can turn only demanded row errors into nulls for `TRY`. Infrastructure errors
remain errors. For dictionary results, error locations must return to the logical domain before
the parent handles them.

For `TRY`, repeating one whole-batch call still loses row locations. Selected replay must recover
individual outcomes through singleton requests or another row-location protocol. Its cost belongs
in the error-path measurements.

## Empty demand

For `D = {}`, every input can contain zero divisors or other row-local invalid values. No value
needs parsing or execution. The completed set can remain empty.

An invalid signature can still fail during binding. A constant expression with a value error must
not fail during runtime evaluation of empty demand. An engine that folds constants earlier needs an
explicit rule for unreachable constant errors.

## Partial-result reuse

First evaluate `f(x)` on `{0, 2}`. Cache its successful results with `C = {0, 2}`. Later request
`D = {1, 2, 3}`.

```text
missing = D minus C = {1, 3}
new_completed = C union successful(missing)
```

Row 2 needs no repeated work. A null cached at row 0 remains a complete result. Placeholder zeroes
at rows 1 and 3 do not qualify for reuse.

The cache key must include function identity, options, input identity, row mapping, and relevant
execution semantics. Timezone, collation, error mode, and volatility can affect those semantics.
Captured errors must retain their status and domain, or remain uncached.

Merging requires preservation of completed rows outside the current demand. Overwriting the whole
buffer after the second evaluation can destroy the row 0 result. Completion is therefore useful
output metadata, even when every individual call satisfies its whole request.
