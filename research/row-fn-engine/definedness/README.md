<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Caller demand and defined results

[Research overview](../README.md).

A portable RowFn engine needs caller demand separate from validity. An output completion guarantee
also becomes necessary when results survive across partial evaluations.

Demand answers which rows the caller needs. Completion answers which rows have a result. Validity
answers whether a completed result is non-null. Storage initialization answers whether memory is
safe to access. A single mask cannot answer all four questions.

These distinctions do not require a new allocated bitmap on every call. A compact batch or a
successful call over known demand can carry completion implicitly. Persistent partial results need
coverage metadata. All-rows coverage can use a constant representation.

For example, consider `CASE WHEN x != 0 THEN 10 / x ELSE 0 END`. A zero in an unselected branch
must not cause a division error under the proposed conditional contract. Marking that row null after
division is too late. Demand must reach the branch expression, its decoders, and the row kernel.

The proposed first contract has three parts:

- Pass a row domain and requested rows into execution. Keep demand outside the logical type.
- Return results with an explicit guarantee that every requested row is complete, or return an
  error. A null is a complete result.
- Keep every exported host array structurally valid. Partial evaluation does not authorize reads
  from uninitialized memory or expose placeholders as computed values.

Exact selected-row execution is the baseline. Dense speculation requires explicit guarantees about
safe access, errors, and effects. Current `DENSE_SAFE` alone does not establish those guarantees.

This requires changes above RowFn. Vortex currently passes arrays and a row count through
`ExecutionArgs`, with no caller selection. Its nullable RowFn policies select rows from input
validity. They do not express conditional demand. The
[current Vortex analysis](current-vortex.md) distinguishes implemented shortcuts from missing
contracts.

Read the detailed notes by question:

- [What do the masks mean, and what must the API guarantee?](contract.md)
- [How do conditionals, nulls, dictionaries, retries, and caching compose?](worked-cases.md)
- [Where must Vortex propagate demand?](current-vortex.md)
- [What does Velox already implement?](velox.md)
- [Which design and validation steps are necessary?](adoption.md)

Demand is independent of the [generic type system](../type-system/README.md). Supporting functions
that produce null from valid inputs is a separate expansion of the current RowFn contract.
