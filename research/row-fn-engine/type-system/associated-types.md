<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Associated types and a built-in vocabulary

[Type overview](README.md) | [Recommended binding contract](portable-contract.md)

One extraction option uses a `TypeSystem` trait and a closed vocabulary for standard element kinds.
It gives the current framework a concrete migration path. It needs additional rules for partial
host support, semantic extensions, and metadata-bearing output.

This page preserves that option as a design alternative. The sketches are incomplete and uncompiled.

## Shape of the interface

The proposed type system owns a native type and a label. It maps recognized native types to a common
kind, maps supported kinds back, and validates output labels over storage.

```rust,ignore
trait TypeSystem {
    type Type;
    type Label;
    type Error;

    fn kind(ty: &Self::Type) -> Option<BuiltinKind>;
    fn from_kind(kind: &BuiltinKind) -> Option<Self::Type>;
    fn validate_label(
        storage: &Self::Type,
        label: &Self::Label,
    ) -> Result<(), Self::Error>;
}

struct ColumnType<T> {
    ty: T,
    nullable: bool,
}
```

The initial vocabulary can cover Boolean, signed and unsigned integers, floating point, UTF-8,
binary, and fixed-size lists. Recognition must establish an accepted semantic domain. It cannot
reinterpret every type with matching physical storage as an ordinary primitive.

Input decoding and output construction belong to the engine adapter. Putting those methods on
`TypeSystem` combines type meaning with array ownership and obscures the boundary.

## Two function paths

| Function path | Dispatch information | Example |
| --- | --- | --- |
| Host-specific `RowFn`. | Native type metadata. | Existing geometry or tensor extension dispatch. |
| Portable function. | A declared semantic domain with supported typed bindings. | Equal-type checked integer addition. |

Both paths can use the same executor. A mechanical adapter can map supported common kinds into a
portable binder. Spatial and tensor functions can become portable later through explicit domain
capabilities. They do not need to remain permanently tied to Vortex.

An opaque kind can preserve unknown metadata. It cannot supply a decoder or operation for that
metadata. Unsupported execution must remain an error.

## Required refinements

**Partial host support.** Requiring every standard kind simplifies blanket implementations, but
excludes hosts that lack one kind. A function over `i64` does not need a host to support `f16`.
Optional capabilities or fallible binding avoid that unnecessary constraint. An output constructor
must not panic because an unsupported `from_kind` result was assumed impossible.

**Nested nullability.** `ColumnType` removes only outer nullability. Child fields retain their own
constraints. Vortex's recursive `eq_ignore_nullability` does not become exact type equality after
this split. Every call site needs the comparison appropriate to its contract.

**Null type.** `DType::Null` cannot become non-nullable. The boundary needs a special null-literal
rule or an explicit coercion into the selected argument type.

**Output labels.** Arrow needs field metadata. DuckDB and Velox can use logical output types.
Vortex can wrap storage in an extension array. Equal layout alone does not establish a valid
semantic relabel. Value conversion remains a separate operation.

**Public evolution.** A public closed enum imposes compatibility choices when new variants appear.
A domain registry or non-exhaustive vocabulary leaves room for growth. The prototype must establish
which mechanism supports third-party domains without weakening validation.

**Rust coherence.** A local host parameter or marker type permits adapter implementations across
crates. Broad blanket implementations can still block later specializations. The
[compiled proof](compiled-proof.md) exercises a small legal arrangement, with its limits recorded.

**Code size.** Runtime tensor dimensions and nested schemas need not become Rust type parameters.
Specialize scalar widths where required. Retain shape and metadata at runtime unless measurement
justifies another specialization.

## Why Substrait remains a mapping layer

The surveyed Substrait type vocabulary lacks the unsigned, half-precision, and fixed-size-list
coverage needed by existing Vortex users. It also embeds nullability in type descriptions.
Its function declarations remain useful for identities, signatures, options, and required constants.
The [prior-art survey](../prior-art/other-systems.md#substrait-extension-functions) records that role.

The recommendation keeps the useful associated-type structure and small common vocabulary.
The [semantic capability design](alternatives.md#3-bind-semantic-capabilities-and-typed-witnesses)
supplies the extension and validation boundary that storage kinds alone cannot provide.
