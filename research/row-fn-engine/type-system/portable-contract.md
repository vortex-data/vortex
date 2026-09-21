<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# A proposed binding contract

The proposed core executes typed row kernels. Host adapters bind semantic requests to their arrays
and construct host-native outputs. The shared contract describes what the function means and which
inputs it accepts.

This is a design proposal. Every Rust sketch is abridged and uncompiled. Error definitions,
registration, lifetime bounds, and executor details are omitted where they do not explain the
boundary. These sketches do not establish an implementable public API by themselves.

A separate [compilation experiment](compiled-proof.md) demonstrates host-independent dispatch,
borrowed views, and cross-crate adapter implementations. It does not implement this complete design.

## Separate five kinds of information

| Information | Example | Lifetime |
| --- | --- | --- |
| Semantic type. | Decimal precision and scale, tensor shape, timestamp interpretation. | Function binding. |
| Nullability envelope. | The argument or a nested child can contain null. | Function binding. |
| Physical access. | `&[i64]`, UTF-8 views, constant value, selected input rows. | Batch binding. |
| Caller demand. | Compute rows selected by the parent expression. | Execution call. |
| Output initialization. | Which sink slots contain initialized values. | Sink traversal and finish. |

The first two form the function's type contract. The last three must not become extra variants of
`DType`. In particular, an unrequested row is not a new nullable value.
[Definedness research](../definedness/README.md).

Nullability belongs at every relevant nesting level. A nullable struct with a non-nullable child
differs from a non-nullable struct with a nullable child. A valid list can contain null elements.
Outer strictness does not decide how a kernel treats those elements. [Mapping details](mappings.md).

## Function binding and batch binding are separate

Function binding consumes options, semantic argument descriptions, required metadata, and relevant
execution configuration. It selects an overload, an output semantic type, and an error policy.
This binding can persist across batches that satisfy the same contract.

Batch binding consumes actual columns and their current representations. It handles constants,
decoding, selection, lifetimes, and typed views. A semantically identical column can require another
physical binding in a later batch.

Preparation also has two categories. A timezone or output precision can be fixed at function
binding. A lookup table for a batch-constant argument belongs to batch preparation. The common API
must preserve function-owned preparation without treating every constant as immutable across calls.

Current `RowFn::dispatch` receives options and types, while `visit_prepared*` receives batch constants.
The proposal makes this distinction explicit and permits caching the first stage.
[Current contracts][row-fn], [visitor preparation][visitor].

## A semantic type use

The shape below makes nullability explicit while keeping the semantic domain separate:

```rust,ignore
struct TypeUse<T> {
    semantic: T,
    nullable: bool,
}

struct DecimalSemantics {
    precision: u16,
    scale: i16,
}

struct TimestampSemantics {
    unit: TimeUnit,
    interpretation: TimestampInterpretation,
    timezone: Option<String>,
}
```

The numeric widths are illustrative, not proposed limits. Supported ranges belong to domain
validation and adapter capabilities. `TimestampInterpretation` must distinguish an instant from a
local calendar value. A timezone name does not define every timestamp operation.

An input also retains host metadata needed for exact output preservation. In Arrow this includes
the `Field`, because its extension metadata and nullability do not live on bare `DataType`.
[Current Arrow boundary][arrow-field].

## Bind a row value family to a host

A row kind identifies the values that a kernel consumes. The adapter supplies the input binding.
This avoids making the kernel's value family depend on `ArrayRef`:

```rust,ignore
trait RowKind {
    type Value<'a>;
}

struct I64;

impl RowKind for I64 {
    type Value<'a> = i64;
}

trait Host: Sized {
    type Column;
    type NativeType;
    type Context;
}

trait BindInput<K: RowKind>: Host {
    type Input<'a>
    where
        Self: 'a;

    fn bind_input<'a>(
        &'a self,
        column: &'a Self::Column,
        request: &InputRequest<K>,
        ctx: &mut Self::Context,
    ) -> Result<Self::Input<'a>, BindError>;
}
```

`InputRequest<K>` represents a validated semantic request. `Input` must implement a row-view
contract for exactly `K::Value<'a>`. Those omitted contracts retain the current length, lifetime,
constant, and unchecked-access guarantees. A practical implementation can borrow the column or own
decoded buffers.

`BindInput<I64>` is an optional capability. A host without a supported signed 64-bit input cannot
instantiate a function that requires it. A tensor domain can define another capability without
adding tensor methods to every host.

Output binding needs the same separation. A typed output builder accepts a kernel value or sink
handle and a validated output request. It constructs the native result type promised at binding.
For timestamps, this can mean a Vortex extension wrapper or an Arrow timestamp array. The core does
not assume that semantic metadata always means an extension wrapper.

A value-preserving relabel and a value conversion must be distinct operations. Rescaling decimal
values, changing timestamp units, and changing offset widths can require work or fail. None of those
operations belongs in an unchecked metadata relabel.

Output inference must also work before any row executes. Empty batches, all-null batches, and null
constants still need a complete result type. An untyped null literal needs overload context or an
explicit coercion rule. Its payload cannot select a concrete row type.

## What a portable function author supplies

A function author defines its semantic overload rules and typed arithmetic. A host adapter supplies
the capabilities. Conceptually, an integer addition binder can have this shape:

```rust,ignore
fn bind_add<H: IntegerHost>(
    host: &H,
    args: &[TypeUse<SemanticType>],
) -> Result<H::BoundCall, BindError> {
    let integer = require_matching_integers(args)?;

    host.bind_integer_binary(integer, Overflow::Error, CheckedAdd)
}
```

`CheckedAdd` stands for implementations over the supported Rust widths. `IntegerHost` selects the
matching compiled input and output bindings. It does not invent an implementation for a width or
domain that the library never compiled.

The function contract must also decide coercion. A portable checked addition can require equal
input types. A host adapter can request casts under a declared policy before binding the function.
Those casts are host operations with their own errors and costs.

DataFusion explicitly separates signature coercion from return-type inference. Its UDF API can
return a field and use literal arguments for output inference. An adapter can retain that split
without copying DataFusion's type system into the row core. [DataFusion UDF contract][datafusion-udf].

## Required laws

These are proposed correctness conditions for each adapter:

1. Accepted inputs preserve the function's semantic contract after any declared conversion.
2. A typed input binding refers to the actual column or decoded storage whose invariants it proves.
3. All valid row accesses produce the declared row value, with stable lifetimes and lengths.
4. Dense execution occurs only under the selected adapter and kernel's null-payload guarantees.
5. The output builder returns the planned logical type, metadata, length, and values.
6. Outer validity follows the selected null policy. Child validity remains part of the row domain.
7. Binding and caching include every option or configuration value that changes semantics.
8. Unsupported mappings return an error instead of silently weakening the contract.

These laws also define conformance work. A descriptor comparison alone cannot prove them. Witness
constructors that justify unsafe access need private fields, checked construction, or an explicit
unsafe contract. The existing input and sink traits supply useful starting invariants.
[Input safety][input], [sink safety][sink].

## Runtime erasure belongs outside the row loop

A host registry can store an erased batch entry point after a concrete function is instantiated:

```rust,ignore
trait BoundCall<H: Host>: Send + Sync {
    fn execute(
        &self,
        args: &[H::Column],
        ctx: &mut H::Context,
    ) -> Result<H::Column, ExecuteError>;
}
```

This sketch omits caller demand. That belongs in the execution contract once its semantics are
settled. A concrete implementation can decode typed views and run a statically dispatched loop.
One virtual call per batch is possible. Per-row virtual dispatch is not required by this design.

An open extension registry can add compiled handlers for new domains. An opaque extension ID alone
cannot select a Rust type or implement arithmetic. A runtime plugin must provide executable code
and an agreed interface. Source extensions, runtime registration, and a binary plugin ABI are three
separate promises.

## First prototype

The first prototype can preserve current semantics while testing the extraction boundary:

1. Separate row values and traversal contracts from Vortex decoding and output construction.
2. Implement Vortex and Arrow bindings for checked `i64` addition and UTF-8 input.
3. Add one metadata-bearing function, such as timestamp truncation with explicit semantics.
4. Retain Vortex registration and serialization behind `VortexFunction<F>`.
5. Exercise the shared binder through the host integration paths described elsewhere in this report.

The timestamp function needs a stated unit, interpretation, timezone, valid range, and overflow
policy. A tensor function is an alternative that tests shape and nested child constraints. Decimal
then tests runtime output parameters and working-width selection.

The acceptance criteria are semantic equality across accepted mappings, explicit rejection of
unsupported mappings, and unchanged Vortex contracts. Performance acceptance needs a measured
comparison that includes adapter work. None of these prototype checks ran during this research.

Open design choices remain: the initial semantic domains, extension identity/versioning, literal
arguments in binding, configuration capture, and the degree of host-native output control.

[Back to the type-system overview](README.md).

[row-fn]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs#L81
[visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs#L125
[arrow-field]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-arrow/src/session.rs#L395
[datafusion-udf]: https://docs.rs/datafusion-expr/55.1.0/datafusion_expr/trait.ScalarUDFImpl.html#method.return_field_from_args
[input]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs#L16
[sink]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs#L75
