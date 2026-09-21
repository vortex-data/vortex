<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Relevant prior art

[Research overview](README.md)

The useful comparisons separate function authoring, type semantics, column transport, and expression
evaluation. No source reviewed here establishes a universal function ABI across the target engines.

## Velox: shared execution behind a row interface

The 2024 paper *Simple (yet Efficient) Function Authoring for Vectorized Engines* describes Velox's
simple-function interface. It shares RowFn's central approach: authors provide row operations, and
the framework supplies vector traversal and encoding handling.

The paper covers nested views and writers, generic types, variadic arguments, and several null
policies. These provide concrete reference designs for features beyond current RowFn. It also
describes the tradeoff between specialized loops and binary size.

Its limitations are equally useful. Some operations reuse entire buffers or manipulate encodings
more effectively through a vector interface. The paper gives `map_keys`, `is_null`, and
`array_sort` as examples. Its performance results apply to the reported Velox implementations and
workloads. They do not establish RowFn overhead or cross-host equivalence.

Source: [Sakka et al., PVLDB 17(12), 4187-4199, 2024](https://www.vldb.org/pvldb/vol17/p4187-pedreira.pdf),
especially sections 3-5. The preceding paragraphs summarize this paper. The recommendations below
are conclusions for this investigation.

**Design implication:** preserve both row and batch implementation paths. Extend null and nested
value contracts deliberately, rather than assuming that a generic `DType` makes them available.

Velox's selection and error machinery is a separate precedent. The
[definedness notes](definedness/README.md) inspect that machinery using current source.

## Substrait: a semantic vocabulary with explicit extensions

Substrait separates a type's class, parameters, nullability, and variation. It also requires explicit
casts instead of defining implicit coercion rules. This is useful precedent for separating type
meaning from representation. Source:
[Substrait type system](https://substrait.io/types/type_system/), accessed 2026-09-21.

The type-class specification makes the distinction concrete. A host can represent `i8` using wider
storage while preserving the specified behavior for `i8` values. The storage width alone does not
define the logical operation. Source:
[Substrait type classes](https://substrait.io/types/type_classes/), accessed 2026-09-21.

Type variations describe representation differences with common semantics. A variation also states
whether function resolution inherits the base behavior or requires separate resolution. Source:
[Substrait type variations](https://substrait.io/types/type_variations/), accessed 2026-09-21.

Extensions provide identifiers for type and function definitions. A RowFn registry needs similarly
explicit identity, even without Substrait plan serialization. Source:
[Substrait extensions](https://substrait.io/extensions/), accessed 2026-09-21.

**Design implication:** use a small semantic protocol and explicit extension contracts. Substrait is
a useful compatibility target, but adopting its schema does not supply column readers, output
writers, or executable kernels.

Choosing Substrait as the core public type model also imports its expressiveness and versioning
choices. Compare those choices against actual function requirements before adopting it. An optional
mapping layer leaves that decision independent from the first extraction experiment.

## Arrow: distinguish transport from function execution

Arrow's format and C Data Interface address data representation and ownership exchange. They do
not define every engine's function registration, coercion, or error semantics. Arrow C++ compute and
an arrow-rs library adapter are different integration targets.

The [Arrow integration page](integrations/README.md) provides the source-backed details. DataFusion
can reuse Arrow storage access while retaining a separate UDF adapter. DuckDB can use native vector
access or an Arrow transport path, with different conversion costs.

**Design implication:** implement an Arrow adapter, but do not make Arrow transport the definition
of portability. Measure each input and output conversion on the selected path.

## Current Vortex tracking issues are historical context

The [RowFn epic](https://github.com/vortex-data/vortex/issues/9128),
[author API tracker](https://github.com/vortex-data/vortex/issues/9129), and
[executor tracker](https://github.com/vortex-data/vortex/issues/9130) describe the original layering
and implementation history. They were inspected on 2026-09-21.

Some issue descriptions lag the pinned source. For example, the executor tracker still describes
an optional skipped-row initializer and a compact-output scatter fallback. Current code requires
default filling and writes filtered results into their original output positions.

The [current-system inventory](current-system/README.md) uses code at the pinned revision as its
authority. The issue bodies provide motivation, not proof of current behavior.
