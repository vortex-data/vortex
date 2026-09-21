<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Three designs and their tradeoffs

The alternatives solve different problems. A host type parameter removes a dependency. A semantic
description gives functions a shared vocabulary. A typed capability connects that vocabulary to
concrete values and buffers.

These designs are proposals. The Rust sketches are abridged and uncompiled.

## 1. Make the framework generic over a host

The smallest structural change replaces Vortex types with associated host types:

```rust,ignore
trait Host {
    type NativeType;
    type Column;
    type Context;
}

trait RowFn<H: Host> {
    fn dispatch<V: RowVisitor<H>>(
        &self,
        args: &[H::NativeType],
        visitor: V,
    ) -> Result<V::VisitResult, BindError>;
}
```

This supports a framework instantiated with Vortex, Arrow, or DuckDB objects. It does not make an
existing function implementation portable. Generic code cannot match `H::NativeType` against
`DType::Primitive`. Every semantic operation must come from another trait or a host-specific
implementation.

Adding `is_integer`, `decimal_parameters`, and `timestamp_parameters` gives generic code that
vocabulary. The vocabulary then becomes a semantic type interface, even if it is not an enum.

There is another constraint: a generic dispatch must prove that its selected inputs exist for `H`.
`InputElement<H>` on every marker requires bounds for every marker the function selects. Associated
capability families can collect those bounds without requiring every host to support every domain.

**Useful for:** a reusable executor, host-specific functions, or an early extraction seam.

**Limitation:** one execution library can still contain separate overload binders for every host.
This does not satisfy the strongest “define the function once” goal by itself.

## 2. Convert native types to a portable semantic description

Functions can bind against a shared vocabulary:

```rust,ignore
enum SemanticType {
    Integer { signed: bool, bits: u16 },
    Utf8,
    Decimal { precision: u16, scale: i16 },
    Timestamp(TimestampSemantics),
    Extension(ExtensionIdentity),
}
```

These fields describe values. Offset width, view layout, dictionary indices, and decimal storage
width remain physical adapter choices. A parameterized timestamp needs more than an integer width.
It needs unit, interpretation, and the metadata its function preserves. [Mapping examples](mappings.md).

A closed enum is understandable and useful for common domains. It becomes restrictive if every new
domain requires a release of the core library. An extension arm permits unknown identities to
survive, but it cannot supply the meaning of an unfamiliar operation.

The conversion must be partial. Converting a decimal with precision 50 to DuckDB cannot preserve its
full domain. DuckDB supports at most 38 decimal digits. Its decimal division also returns a floating
result, which differs from Vortex's decimal division contract. [DuckDB numeric types][duckdb-decimal]
and [Vortex decimal arithmetic][vortex-decimal].

**Useful for:** shared overload selection, output inference, diagnostics, and conformance cases.

**Limitation:** it creates a shared semantic contract that needs governance. A type catalog alone
does not specify overflow, rounding, collation, timezone behavior, or errors.

## 3. Bind semantic capabilities and typed witnesses

A function can ask for the input it needs, such as signed 64-bit integers or fixed-width float
vectors. An adapter checks metadata and creates the corresponding access capability. A witness is
the resulting typed evidence that the binding succeeded.

For example, a `TimestampTicks` input can expose `i64` after validating timestamp semantics. It is a
different capability from unrestricted `I64`, although both use the same Rust scalar.

A geometry predicate can accept several extension identities through one geometry capability. An
unknown extension can remain opaque until a registered plugin supplies that capability. Plugin
recognition must validate identity, metadata, and value rules. Equal storage is insufficient.

Typed witnesses have two separate lifetimes. A function binding can retain semantic parameters
across batches. A batch binding proves properties of one actual column and its borrowed view.
Using a witness for another column must not become a safe route to unchecked access.

**Useful for:** extensible domains, multiple physical layouts, and static row loops.

**Limitation:** capabilities still require precise semantics. An unchecked downcast registry is not
a type system. A registry entry must state which values and parameters the operation accepts.

## Recommended combination

Keep the native type behind a host adapter. Define small semantic descriptions for shared function
domains. Bind each accepted signature to typed input and output capabilities.

| Decision | Owner |
| --- | --- |
| What does this function compute? | Function contract. |
| Which logical inputs and parameters does it accept? | Function binder and semantic domain. |
| Can this host preserve those semantics? | Host adapter. |
| Which physical representation is available for this batch? | Batch adapter. |
| Which Rust row values and sink handles reach the kernel? | Typed capabilities. |
| Which rows need execution? | Execution demand and validity policy. |

This retains a clear escape route for host-specific functions. Such a function can inspect native
metadata without claiming portability to hosts that lack an equivalent contract.

## Rust constraints

Current `RowFn` and `RowVisitor` are not trait objects. They require `Sized` and generic visitor
methods. Input and sink traits also use generic associated types. A type parameter does not remove
these restrictions. Rust's dyn compatibility rules exclude these forms from ordinary trait-object
dispatch. [Rust Reference][rust-dyn].

A runtime registry can instead erase a concrete bound function behind a batch method. Each registry
entry calls an already compiled implementation. Its inner typed loop can remain generic and static.
This adds a potential batch dispatch cost, not a required per-row virtual call.

Separate adapter crates also face Rust's orphan rules. An adapter cannot generally implement a
foreign input trait for foreign `i64`. A local `ArrowHost` trait parameter or local input wrapper can
make the implementation legal. Uncovered parameters before the local type still matter. Public
blanket implementations can prevent later specialized implementations. [Rust Reference][rust-coherence].

The present `RowFn -> ScalarFnVTable` blanket implementation already makes the two implementations
mutually exclusive on one type. An extraction can use an explicit `VortexFunction<F>` wrapper to
keep that choice in the Vortex adapter. [Current vtable boundary][vortex-vtable].

No Rust trait layout is proposed as a plugin ABI. A separately loaded binary needs an explicit
interface, ownership rules, versioning, and precompiled function bindings. This requirement is
separate from source portability and typed row execution.

## Keep the specialization space bounded

An open semantic type system does not require one Rust type for every nested schema. Tensor rows
already demonstrate the distinction: runtime shape and width share a slice-valued row family.
General nested types can retain runtime descriptors and specialize only the operations that need
concrete scalar widths. [Tensor input adapter][tensor-input].

For `n` independent scalar choices across `k` arguments, exhaustive specialization can require
`n^k` combinations before physical layouts are considered. Equal-type binary arithmetic needs only
`n` scalar combinations. Coercion and declared same-type constraints can keep this space smaller.
This is a design count, not a measured code-size result.

A binder must not turn every dictionary key width, list depth, tensor dimension, and layout into a
Rust type parameter automatically. Those choices need separate evidence that specialization helps
the row loop. Otherwise metadata stays at runtime, outside the loop where possible.

[Proposed contract](portable-contract.md) · [Back to the overview](README.md).

[duckdb-decimal]: https://duckdb.org/docs/current/sql/data_types/numeric#fixed-point-decimals
[vortex-decimal]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/fns/binary/numeric/decimal.rs#L4
[rust-dyn]: https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility
[rust-coherence]: https://doc.rust-lang.org/reference/items/implementations.html#trait-implementation-coherence
[vortex-vtable]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/vtable.rs#L105
[tensor-input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-tensor/src/scalar_fns/row.rs#L55
