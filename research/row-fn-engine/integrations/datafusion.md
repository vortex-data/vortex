<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# DataFusion adapter

DataFusion is the closest additional host. A Rust `ScalarUDFImpl` can wrap the Arrow adapter and
register through `SessionContext::register_udf`. This shares the row implementation and Arrow
storage code. DataFusion planning and expression execution remain host responsibilities.
[Registration guide][guide].

This page describes DataFusion 55.1.0 at [`7d3835c`][source], inspected on 2026-09-21.

## Binding and evaluation

The relevant native methods are:

```rust
fn signature(&self) -> &Signature;
fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef>;
fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue>;
```

The signature governs accepted arguments and coercion. `return_field_from_args` receives input
fields and any known scalar arguments. It can preserve extension metadata and derive output
nullability. An implementation still supplies `return_type`, but DataFusion uses the field method
when overridden. [UDF contract][udf].

The adapter must make coercion an explicit host policy. `Signature::exact` expresses an overload,
but the planner can still insert supported casts. A shared function must not assume that identical
SQL text reaches every engine with identical argument types. A strict first prototype can require
explicit SQL casts and verify the bound types.
[Signature definitions][signature], [UDF coercion contract][udf].

`ScalarFunctionArgs` supplies already evaluated arguments, argument fields, row count, planned
output field, and execution configuration. The adapter maps arrays to borrowed Arrow readers and
scalars to constant readers. It returns the planned logical type and the expected row count.
DataFusion's physical expression checks array length, with a special case for scalar-only inputs
that return one element. [Physical scalar execution][physical].

`ColumnarValue::Scalar` must remain scalar through the native adapter. `into_array` and
`values_to_arrays` broadcast scalars into full arrays. That convenience is unnecessary for a
framework that already specializes constant inputs. [ColumnarValue implementation][columnar].

## Nulls, errors, and optimizer metadata

For the current RowFn subset, return `is_strict() == true` and derive output nullability from input
fields. DataFusion strictness means that null input implies null output. Current RowFn requires
the stronger reverse direction for valid inputs: a valid input tuple cannot produce null.
[DataFusion strictness][udf], [RowFn contract][rowfn].

The native UDF callback returns one batch result or a `DataFusionError`. It does not return an error
per row. A checked row operation can fail the batch. A future framework that supports row errors
must define how the adapter converts those errors, including any `TRY` behavior.
[Native callback][udf].

Declare immutable behavior only for a pure function whose explicit arguments and bound options
fully determine the result. DataFusion distinguishes immutable, query-stable, and volatile
functions. Planning can evaluate eligible functions. A timezone or locale from execution
configuration is therefore a semantic dependency, not an incidental adapter setting.
[Volatility definitions][signature].

Current RowFn is not an implementation of all scalar SQL functions. `coalesce`, `is_null`, and
functions that create nulls from valid values need a broader null contract. Side effects violate
the current visitor contract. Volatile or nondeterministic functions also need an explicit
evaluation-count contract before constant folding or retries are allowed.
[Current row-visitor requirements][visitor].

## Row demand and conditional expressions

The ordinary physical scalar expression evaluates each child over the batch before it invokes
the UDF. There is no demand mask in `ScalarFunctionArgs`. Setting `short_circuits` or
`conditional_arguments` changes optimizer information. It does not change this callback into a
lazy argument interface. [Scalar expression execution][physical], [UDF metadata][udf].

DataFusion's `CASE` implementation can filter the input batch for a branch, evaluate that branch,
then merge or scatter its result. The UDF receives a smaller batch with local row indices. It does
not receive the original batch plus its selection. [CASE implementation][case].

Consequently, the first adapter can request every row of its received batch. A richer demand mask
inside the shared engine can still help composition. It cannot recover rows or deferred child work
that this host interface does not expose. Planning-time evaluation also needs separate attention
when the function can fail.

## Source integration and binary integration

A native Rust adapter compiles against the application's DataFusion and Arrow versions. That is
a source-level integration. It is not a stable Rust plugin ABI.

DataFusion supplies `datafusion-ffi` and `FFI_ScalarUDF` for foreign libraries. Its documentation
recommends matching versions and describes a Rust-library interface built with stable FFI types.
It recommends a Rust wrapper for a library in another language.
[DataFusion FFI scope][ffi-readme].

The pinned foreign UDF wrapper has two material differences from a native wrapper:

- It converts each input with `to_array(number_rows)` before export. The receiver reconstructs
  `ColumnarValue::Array`, so scalar inputs lose their native constant form.
- Its FFI structure and `ForeignScalarUDF` implementation do not forward `is_strict`. The foreign
  wrapper keeps the default conservative strictness metadata.

The same-library shortcut can recover the original implementation and avoid the foreign path.
These findings concern an actual foreign-library call. [Foreign UDF implementation][ffi-udf].

This existing FFI is a deployment option with measurable adapter work. A portable RowFn ABI can
preserve scalar markers and semantic metadata, but it becomes a separately versioned protocol.
Arrow C Data alone does not supply those fields.

## What the existing Vortex integration proves

The current Vortex adapter translates a supported set of DataFusion expressions into Vortex scan
expressions. Its scalar-function handling recognizes specific functions. It does not register any
arbitrary RowFn as a DataFusion UDF. A general registration layer is new work.
[Current expression conversion][vortex-convert].

[source]: https://github.com/apache/datafusion/tree/7d3835c71f30cbd3c3ae4041732267f1f453097a
[guide]: https://datafusion.apache.org/library-user-guide/functions/adding-udfs.html
[udf]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/expr/src/udf.rs
[signature]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/expr-common/src/signature.rs
[columnar]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/expr-common/src/columnar_value.rs
[physical]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/physical-expr/src/scalar_function.rs#L236-L285
[case]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/physical-expr/src/expressions/case.rs#L980-L1024
[ffi-readme]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/ffi/README.md
[ffi-udf]: https://github.com/apache/datafusion/blob/7d3835c71f30cbd3c3ae4041732267f1f453097a/datafusion/ffi/src/udf/mod.rs
[rowfn]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/row_fn.rs
[visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs
[vortex-convert]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-datafusion/src/convert/exprs.rs#L231-L275
