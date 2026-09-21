<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Velox function authoring

[Prior-art overview](../prior-art.md) | [Demand and partial results](../definedness/velox.md)

Velox is the closest authoring comparison. A Simple Function supplies typed row operations.
`SimpleFunctionAdapter` supplies vector access, null handling, selected execution, and output.
A `VectorFunction` supplies a whole-vector implementation when the row interface is insufficient.

The source references below use `c53a86adc4c971a752c68d964b4000e436edf2c8`.
The adapter's status path was inspected again during synthesis.

## Authoring and type tags

A function defines `call` in a class template. `registerFunction<F, Return, Args...>` registers each
supported signature. RowFn instead selects a concrete tuple through runtime `dispatch`.
Both arrangements lead to compiled typed row operations.

| Velox capability | RowFn comparison |
| --- | --- |
| Primitive and string type tags. | Native elements and `Utf8Column` cover similar input shapes. |
| `Array`, `Map`, and `Row` readers and writers. | Fixed-size-list output and domain-specific tensor or geometry adapters cover a narrower set. |
| `Generic<T1>` and orderable type variables. | Current dispatch selects concrete element tuples. |
| `Variadic<T>`. | Current tuples have fixed arity, up to twelve inputs. |
| `Constant<T>`. | Prepared visits expose optional constants, without a general required-constant signature. |
| `callAscii`. | No standard batch-level ASCII capability in current RowFn. |

A `void` return means the call does not produce a null result. It does not by itself prove the
absence of an exception. A Boolean return can signal a null result. Those are separate properties
from error behavior. [Authoring documentation][docs].

## Nulls and preparation

Default null handling suppresses calls on null inputs. `callNullable` exposes nulls to the function.
`callNullFree` provides a stronger path after the adapter establishes the required nested null
properties. Current RowFn is strict and cannot produce null from valid inputs.

`initialize` receives types, configuration, and available constant arguments. It retains prepared
state at the function instance. Current RowFn preparation is per batch and can repeat on retry.
This distinction matters for caches: a batch constant does not necessarily remain constant across
calls. Initialization failures also need a row-demand policy. [Authoring documentation][docs].

## Output and representation

String writers can retain input buffers through `setNoCopy` and `reuse_strings_from_arg`.
`findReusableArg` can select reusable result storage under its preconditions. Current RowFn string
sinks copy into owned buffers and do not provide this input-owner transfer.

The adapter selects flat or constant readers where possible. Generic readers handle other encodings.
It also separates all-not-null and fixed-width paths. `SelectivityVector` supplies the requested row
domain. These mechanisms are useful precedents, but do not establish performance for a Rust adapter.
[SimpleFunctionAdapter][adapter].

## Errors

Velox supports exception capture through `EvalCtx`. The inspected adapter also carries `Status`
results and records unsuccessful statuses through `EvalCtx::setStatus`. The historical
[status proposal](https://github.com/facebookincubator/velox/issues/9635) is not evidence that
status handling is absent from this source revision. [Adapter error path][status].

RowFn's deferred path reduces compact evidence across rows and can replay valid rows after a
rejection. That is distinct from retaining errors by row. Per-row capture permits `TRY` and Boolean
parents to decide which errors remain observable. Current RowFn's fail-fast API does not provide
that wider result model. [Evaluator comparison](../definedness/velox.md).

## Design lessons

- Preserve typed nested readers and writers without making every nested schema a Rust type.
- Treat required constants, nullable outputs, and custom null policies as explicit capabilities.
- Retain a batch interface for buffer reuse and encoding-aware operations.
- Give output reuse an ownership contract that retains every referenced input buffer.
- Keep demand, row-error capture, and partial-result preservation together at the evaluator boundary.
- Preserve RowFn's plan-reproduction check without claiming that other engines lack binding phases.

A Velox adapter can expose a version-matched vector function with one batch call into Rust.
It needs signatures, selected views, output ownership, and error translation. There is no implemented
adapter or measured ABI in this research.

[docs]: https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/docs/develop/scalar-functions.rst
[adapter]: https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/SimpleFunctionAdapter.h
[status]: https://github.com/facebookincubator/velox/blob/c53a86adc4c971a752c68d964b4000e436edf2c8/velox/expression/SimpleFunctionAdapter.h#L638-L705
