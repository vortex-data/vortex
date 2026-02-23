# Vtables and Dispatch

Vortex uses custom vtable traits rather than Rust's built-in `dyn Trait` vtables. This design
gives us capabilities that are not possible with trait objects: colocating instance methods with
type-level methods (e.g. `deserialize` does not require an existing instance), providing
type-safe public APIs backed by a different internal representation, and enforcing pre- and
post-conditions at the boundary between the two.

The `ExtDTypeVTable` is the canonical example of this pattern. Other vtables (arrays, layouts,
expressions) are being migrated to follow the same design.

## The Pattern

Every vtable-backed type in Vortex has two forms:

- A **typed form** `Foo<V>` -- generic over the vtable type `V`. This is what plugin authors
  implement against. It provides compile-time type safety and direct access to the vtable's
  associated types (e.g. `V::Metadata`).

- An **erased form** `FooRef` -- a concrete, non-generic struct that hides the vtable type
  behind a trait object. This is what the rest of the system passes around. It can be stored
  in collections, serialized, and threaded through APIs without propagating generic parameters.

Both forms are internally `Arc`-wrapped, making cloning cheap. Upcasting from `Foo<V>` to
`FooRef` is free (just moving the `Arc`). Downcasting from `FooRef` to `Foo<V>` is a checked
pointer cast -- also free after the type check.

## Example: ExtDType

Extension dtypes follow this pattern exactly. A vtable implementation defines the extension's
ID, metadata type, serialization, and validation:

```rust
trait ExtDTypeVTable: Sized + Send + Sync + Clone + Debug {
    type Metadata: Send + Sync + Clone + Debug + Display + Eq + Hash;

    fn id(&self) -> ExtId;
    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;
    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;
    fn validate(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()>;
}
```

The typed form `ExtDType<V>` wraps an `Arc` containing the vtable instance, the metadata, and
the storage dtype. Users who know the concrete type get full access to the typed metadata:

```rust
let ts: ExtDType<Timestamp> = ...;
let unit: &TimeUnit = &ts.metadata().unit;    // V::Metadata is concrete
```

The erased form `ExtDTypeRef` wraps the same `Arc` behind a private object-safe trait. Code
that does not need to know the concrete type works with `ExtDTypeRef` and can pattern-match
to recover the typed form when needed:

```rust
let ext: &ExtDTypeRef = dtype.ext();
if let Some(meta) = ext.metadata_opt::<Timestamp>() {
    // meta is &TimestampMetadata -- type-safe from here
}
```

## Why Not `dyn Trait`

Rust's `dyn Trait` vtables have several limitations that make them unsuitable for Vortex's
needs:

**No type-level methods.** A `dyn Trait` requires an existing instance to call any method.
But operations like deserialization need to construct a new instance from raw bytes -- there
is no instance yet. Vortex vtables colocate instance logic (e.g. `serialize`) with type-level
logic (e.g. `deserialize`) in the same struct, since the vtable itself is a lightweight value
(often zero-sized) that can be default-constructed.

**No associated type access.** A `dyn Trait` erases all associated types. With Vortex vtables,
the typed form `Foo<V>` preserves full access to `V::Metadata` and other associated types,
enabling type-safe APIs for plugin authors while the erased form handles heterogeneous storage.

**No pre/post-condition enforcement.** Vortex vtables separate the internal vtable trait (what
plugin authors implement) from the public API (what callers use). The public API can validate
inputs, enforce invariants, and transform outputs without exposing those concerns to vtable
implementors. With `dyn Trait`, the trait surface is the public API.

## Registration and Deserialization

Vtables are registered in the session by their ID. When a serialized value is encountered
(e.g. an extension dtype in a file footer), the session's registry resolves the ID to the
vtable instance and calls `deserialize` on it to reconstruct the typed value. The result is
returned as the erased form (`ExtDTypeRef`) so it can be stored in the dtype tree without
generic parameters.

This pattern -- register by ID, deserialize via vtable, store as erased -- is consistent
across all vtable-backed types in Vortex.
