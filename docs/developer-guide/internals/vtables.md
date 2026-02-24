# Vtables and Dispatch

Vortex uses custom vtable traits rather than Rust's built-in `dyn Trait` vtables. This design
gives us capabilities that are not possible with trait objects: colocating instance methods with
type-level methods (e.g. `deserialize` does not require an existing instance), providing
type-safe public APIs backed by a different internal representation, and enforcing pre- and
post-conditions at the boundary between the two.

## The Pattern

Every vtable-backed type in Vortex follows the same structural pattern. For a concept `Foo`,
there are six components:

| Component     | Name                | Visibility | Role                                                 |
|---------------|---------------------|------------|------------------------------------------------------|
| VTable trait  | `FooVTable`         | Public     | Non-object-safe trait that plugin authors implement  |
| Typed wrapper | `Foo<V: FooVTable>` | Public     | Cheaply cloneable typed handle, passed to VTable fns |
| Erased ref    | `FooRef`            | Public     | Type-erased handle for heterogeneous storage         |
| Inner struct  | `FooInner<V>`       | Private    | Holds vtable + data, sole implementor of `DynFoo`    |
| Sealed trait  | `DynFoo`            | Private    | Object-safe trait implemented only by `FooInner`     |
| Plugin        | `FooPlugin`         | Public     | Registry trait for ID-based deserialization          |

**Typed form** `Foo<V>` is generic over the vtable type `V`. This is what plugin authors
construct and what callers use when they know the concrete type. It provides compile-time
type safety and direct access to the vtable's associated types (e.g. `V::Metadata`).

**Erased form** `FooRef` is a concrete, non-generic struct that hides the vtable type behind
a trait object. This is what the rest of the system passes around. It can be stored in
collections, serialized, and threaded through APIs without propagating generic parameters.

Both forms are internally `Arc`-wrapped, making cloning cheap. Upcasting from `Foo<V>` to
`FooRef` is free (just moving the `Arc`). Downcasting from `FooRef` to `Foo<V>` is a checked
pointer cast -- also free after the type check.

**VTable methods** receive `&Foo<V>` -- the typed wrapper. Since `Foo<V>` provides access to
the underlying data (metadata, children, buffers, etc.), there is no need to expose the
inner struct. `FooInner<V>` is a private implementation detail that holds the data and
implements the sealed `DynFoo` trait. Both `Foo<V>` and `FooRef` are thin `Arc` wrappers
around `FooInner<V>` and `dyn DynFoo` respectively.

**Plugin** `FooPlugin` is a separate trait for registry-based deserialization. It knows how to
reconstruct a `FooRef` from serialized bytes without knowing `V` at compile time. Plugins are
registered in the session by their ID.

## Example: ExtDType

Extension dtypes follow this pattern. The vtable defines the extension's ID, metadata type,
serialization, and validation:

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
let ts: ExtDType<Timestamp> =...;
let unit: & TimeUnit = & ts.metadata().unit;    // V::Metadata is concrete
```

The erased form `ExtDTypeRef` wraps the same `Arc` behind the private `DynExtDType` trait.
Code that does not need to know the concrete type works with `ExtDTypeRef` and can
pattern-match to recover the typed form when needed:

```rust
let ext: & ExtDTypeRef = dtype.ext();
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

## File Layout Convention

Each vtable-backed concept `Foo` lives in its own module directory with a consistent file
structure:

| File         | Contents                                                          |
|--------------|-------------------------------------------------------------------|
| `vtable.rs`  | `FooVTable` — the non-object-safe trait users implement           |
| `plugin.rs`  | `FooPlugin` — registry trait for deserialization + blanket impl   |
| `typed.rs`   | `Foo<V>` + `FooInner<V>` + `DynFoo` + impl (typed wrapper + guts) |
| `erased.rs`  | `FooRef` + Display/Debug/PartialEq/Hash impls (erased public API) |
| `matcher.rs` | `Matcher` trait + blanket impl for `V: FooVTable`                 |
| `mod.rs`     | Re-exports, `FooId` type alias, sealed module                     |

The private internals (`FooInner`, `DynFoo`, sealed module) are `pub(super)` within the
concept's module. Everything else is re-exported from `mod.rs`.

## Registration and Deserialization

Vtables are registered in the session by their ID. When a serialized value is encountered
(e.g. an extension dtype in a file footer), the session's registry resolves the ID to the
`FooPlugin` and calls `deserialize` to reconstruct the value. The result is returned as the
erased form (`FooRef`) so it can be stored without generic parameters.

This pattern -- register plugin by ID, deserialize via plugin, store as erased ref -- is
consistent across all vtable-backed types in Vortex.

## Migration Status

All four vtable-backed types are converging on the pattern described above.

### ExtDType -- Done

The reference implementation. All components follow the convention:

- `ExtVTable` (vtable trait, uniquely not prefixed `ExtDType` for historical reasons)
- `ExtDType<V>`, `ExtDTypeRef`, `ExtDTypeInner`, `DynExtDType`, `ExtDTypePlugin`
- `Matcher` trait with blanket impl
- File layout: `vtable.rs`, `plugin.rs`, `typed.rs`, `erased.rs`, `matcher.rs`
- `ExtDTypeMetadata` wrapper removed; methods inlined on `ExtDTypeRef`

### Expr -- Not started

Currently uses `VTable` (unqualified), `VTableAdapter`, `DynExprVTable` (sealed trait),
and `ExprVTable` (confusingly, the erased ref). Needs renaming to `ExprVTable`, `ExprInner`,
`DynExpr`, `ExprRef`. Typed wrapper `Expr<V>` does not exist yet.

### Layout -- Not started

Currently uses `VTable` (unqualified), `LayoutAdapter`, and `Layout` (sealed trait doubling
as public API). Needs renaming to `LayoutVTable`, `LayoutInner`, `DynLayout`, `LayoutRef`.
Typed wrapper `Layout<V>` does not exist yet.

### Array -- Not started

The largest migration. Currently uses `VTable` (unqualified), `ArrayAdapter`, `Array` (sealed
trait doubling as public API), `ArrayRef`, and `DynVTable`. In addition to renaming
(`ArrayVTable`, `ArrayInner`, `DynArray`, `ArrayPlugin`), this requires:

1. **Standardize data storage.** Replace per-encoding array structs with a common inner
   struct holding `(dtype, len, V::Metadata, buffers, children, stats)`. Per-encoding typed
   accessors (e.g. `DictArray::codes()`) become methods on `Array<DictVTable>`.
2. **Collapse sub-vtables.** Fold `BaseArrayVTable`, `OperationsVTable`, `ValidityVTable`, and
   `VisitorVTable` into `ArrayVTable`. Many methods become trivial or generic once data
   storage is standardized.
3. **Introduce typed wrapper.** Add `Array<V>` analogous to `ExtDType<V>`, replacing the
   current `type Array` associated type on the vtable.
