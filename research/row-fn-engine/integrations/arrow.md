<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Arrow has several integration boundaries

Arrow supplies a columnar format, libraries in several languages, and an interchange ABI. Those
pieces do not create one universal UDF registry. An Arrow Rust callable, an Arrow C++ compute
function, and an Arrow C Data export are different integration products.

This page uses the [pinned snapshots](README.md#source-snapshots). The adapter descriptions are
proposals based on those contracts.

The Arrow format and C Data documentation links are live pages accessed on 2026-09-21. Their
displayed version was 25.0.1. The C++ and Rust source links point to fixed commits.

## Arrow Rust

A direct Rust adapter can accept `ArrayRef` values and a distinct scalar operand. Arrow Rust
already has `Datum` and `Scalar<T>` for kernels that support scalar operands. A scalar is a
one-element array with an explicit scalar marker, not an ordinary length-one array that every
caller must broadcast. [Arrow Rust scalar interface][rust-scalar].

The proposed adapter binds an input signature once, downcasts each array once per batch, and
passes typed views into the shared loop. It returns an Arrow array or scalar result. Applications
can call this interface directly. DataFusion can register it through its own UDF interface.

Do not make `ArrayRef -> Vortex ArrayRef -> Arrow ArrayRef` the required path. That route reuses
existing Vortex execution, but it preserves the dependency that extraction is intended to remove.
It is a useful comparison baseline for a later prototype.

Binding needs a field descriptor in addition to `DataType`. In Arrow Rust, `Field` carries
nullability and extension metadata. `Utf8`, `LargeUtf8`, and `Utf8View` also describe different
physical layouts for UTF-8 values. A portable string function can accept all three without
requiring a shared physical layout. [Arrow Rust fields][rust-field],
[Arrow Rust data types][rust-types].

Dictionary inputs require logical validity. A valid key can refer to a null dictionary value.
Arrow Rust's `DictionaryArray::nulls()` returns key validity, while `logical_nulls()` combines both
sources. Calling only `nulls()` is an incorrect strict-function adapter.
[Dictionary validity implementation][rust-dictionary].

## Arrow C Data interchange

The C Data interface provides stable `ArrowArray` and `ArrowSchema` structures plus release
callbacks. It can share compatible buffers across runtimes in one process. It does not define
function registration, type coercion, volatility, error policy, or lazy argument evaluation.
Those contracts belong to the library or host that surrounds it. [C Data specification][c-data].

A RowFn plugin protocol can use C Data for arguments and results, but that protocol must also
define these items:

- Function identity, version, signature, options, and error representation.
- A scalar marker and a logical batch length independent from array length.
- Input field metadata, including logical extension identity.
- Output allocation and release ownership.
- A demand mask if the caller needs partial evaluation.

Those items are proposed protocol requirements. They are not fields supplied by `ArrowArray`.
The ABI can keep all function calls at batch boundaries. Passing an element callback over FFI
inside every row iteration prevents the intended inlining and adds a distinct per-row cost.

An exported array has an offset, child arrays, optional dictionary storage, and producer-owned
buffers. Consumers must retain the producer until they release the export. A raw pointer alone
does not carry that lifetime. The C Data format permits unaligned buffers, so a Rust adapter must
check alignment before it creates typed slices. [C Data layout and ownership][c-data].

## Arrow C++ compute registration

Arrow C++ has a concrete compute registry. An adapter creates a `ScalarFunction`, adds one or more
`ScalarKernel` implementations, then calls `FunctionRegistry::AddFunction`. Exact types, type
matchers, and computed output types can connect the registry to shared binding.
[Function registry][cpp-registry], [function definition][cpp-function],
[kernel type resolution][cpp-kernel].

The callback has this shape:

```cpp
Status execute(KernelContext*, const ExecSpan&, ExecResult*);
```

`ExecSpan` is a borrowed batch view. Its values can be arrays or scalars. `KernelInit` can prepare
state from input types and function options. The Rust library can expose one batch call for this
wrapper, with a compiled typed loop behind that call. The shared row closure remains Rust code.
[Execution span][cpp-exec], [kernel callback and state][cpp-kernel].

The default scalar kernel intersects input validity and preallocates fixed-width output.
Variable-width and nested output still need allocation by the kernel. A wrapper can use the host
allocation directly, or request ownership of a returned array. It must also declare whether it
supports slices of a larger output allocation. [Scalar kernel output contract][cpp-kernel].

Arrow C++ also exposes `ScalarFunction::is_pure`, which requires the same result for the same
inputs. Current RowFn forbids side effects but has no explicit determinism or volatility metadata.
The adapter must establish that extra property before it declares purity. Purity does not imply
infallibility. A checked operation still returns an error through `Status`.
[Purity metadata][cpp-function], [current RowVisitor contract][vortex-visitor].

`ExecBatch` contains a selection-vector field, but the normal `ArrayKernelExec` callback receives
an `ExecSpan` without that field. Its constructor does not transfer a selection vector. This is
insufficient evidence for a general demand-mask callback contract. A prototype must establish the
actual caller path before it promises selective execution. [Batch and span definitions][cpp-exec].

## Scope limits

Arrow schema compatibility does not establish function compatibility. A decimal addition still
needs a precision and overflow policy. A timestamp function needs units and timezone semantics.
An extension field needs a registered semantic interpretation. Treat an unsupported interpretation
as a bind error, even when the storage bytes are familiar.

No Arrow adapter automatically registers the function with another engine. Arrow C++ registration
does not populate DataFusion's Rust registry, and C Data export does not populate either registry.
The [worked example](worked-example.md) makes those wrappers explicit.

[rust-scalar]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-array/src/scalar.rs
[rust-field]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-schema/src/field.rs
[rust-types]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-schema/src/datatype.rs
[rust-dictionary]: https://github.com/apache/arrow-rs/blob/f90e061326bd821a7af09281d9e92de6f3b603d9/arrow-array/src/array/dictionary_array.rs#L737-L768
[c-data]: https://arrow.apache.org/docs/format/CDataInterface.html
[cpp-registry]: https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/cpp/src/arrow/compute/registry.h
[cpp-function]: https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/cpp/src/arrow/compute/function.h
[cpp-kernel]: https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/cpp/src/arrow/compute/kernel.h
[cpp-exec]: https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/cpp/src/arrow/compute/exec.h
[vortex-visitor]: https://github.com/vortex-data/vortex/blob/96bd521eb0565555def2af7b8e97e96891728da6/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs
