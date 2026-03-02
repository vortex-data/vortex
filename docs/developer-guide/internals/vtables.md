# Vtables and Dispatch

Vortex uses custom vtable traits rather than Rust's built-in `dyn Trait` vtables. This design
gives us capabilities that are not possible with trait objects: colocating instance methods with
type-level methods (e.g. `deserialize` does not require an existing instance), providing
type-safe public APIs backed by a different internal representation, and enforcing pre- and
post-conditions at the boundary between the two.

## The Pattern

Every vtable-backed type in Vortex follows the same structural pattern. For a concept `Foo`,
there are five components:

| Component    | Name                | Visibility | Role                                                |
|--------------|---------------------|------------|-----------------------------------------------------|
| VTable trait | `FooVTable`         | Public     | Non-object-safe trait that plugin authors implement |
| Data struct  | `Foo<V: FooVTable>` | Public     | Generic data struct, lives behind Arc               |
| Erased ref   | `FooRef`            | Public     | Type-erased handle, public API surface              |
| Sealed trait | `DynFoo`            | Private    | Object-safe, blanket impl for `Foo<V>`              |
| Plugin       | `FooPlugin`         | Public     | Registry trait for ID-based deserialization         |

Three layers of dispatch:

```
FooRef          thin Arc wrapper, delegates to DynFoo
  → DynFoo      sealed, thin blanket forwarder to Foo<V> inherent methods
    → Foo<V>    public API with pre/post-conditions, delegates to FooVTable
      → FooVTable  plugin authors implement
```

**Data struct** `Foo<V>` is generic over the vtable type `V`. It holds the vtable instance
(which may be zero-sized for native implementations or non-zero-sized for language bindings),
common fields, and a VTable-specific associated type for concept-specific data (e.g.
`V::Metadata` for ExtDType, `V::Array` for Array). `Foo<V>` is not Arc-wrapped — it lives
behind `Arc` inside `FooRef`. `Foo<V>` has inherent methods for all operations, with
pre/post-condition enforcement. These delegate to `FooVTable` methods.

`Foo<V>` implements `Deref` to the VTable's associated data type. This means encoding-specific
methods defined on the associated type are callable directly on `&Foo<V>` (and therefore on
the result of downcasting from `&FooRef`), while common methods on `Foo<V>` remain accessible
via normal method resolution.

**Sealed trait** `DynFoo` is object-safe and has a blanket `impl<V: FooVTable> DynFoo for Foo<V>`.
This blanket impl is a thin forwarder to `Foo<V>` inherent methods — no logic of its own.
Its purpose is to enable dynamic dispatch from `FooRef` through to the typed `Foo<V>`.

**Erased form** `FooRef` wraps `Arc<dyn DynFoo>`. It delegates to `DynFoo` methods, which
forward to `Foo<V>`. It can be stored in collections, serialized, and threaded through APIs
without propagating generic parameters. Cloning is cheap (Arc clone).

**Downcasting** from `FooRef` to `Foo<V>` is borrowed:

```
FooRef::as_::<V>(&self) -> &Foo<V>           borrow, free after type check
FooRef::downcast::<V>(self) -> Arc<Foo<V>>   owned, free after type check
```

**Plugin** `FooPlugin` is a separate trait for registry-based deserialization. It knows how to
reconstruct a `FooRef` from serialized bytes without knowing `V` at compile time. Plugins are
registered in the session by their ID.

## Example: ExtDType

Extension dtypes follow this pattern:

```rust
trait ExtDTypeVTable: Sized + Send + Sync + Clone + Debug {
    type Metadata: Send + Sync + Clone + Debug + Display + Eq + Hash;

    fn id(&self) -> ExtId;
    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;
    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;
    fn validate(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()>;
}

struct ExtDType<V: ExtDTypeVTable> {
    vtable: V,
    metadata: V::Metadata,
    storage_dtype: DType,
}

struct ExtDTypeRef(Arc<dyn DynExtDType>);
```

Downcasting from the erased form recovers the typed data struct:

```rust
let ext: & ExtDTypeRef = dtype.ext();
let ts: & ExtDType<Timestamp> = ext.as_::<Timestamp>();
let unit: & TimeUnit = & ts.metadata;     // V::Metadata is concrete
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
`FooPlugin` and calls `deserialize` to reconstruct the value. The result is returned as the
erased form (`FooRef`) so it can be stored without generic parameters.

This pattern -- register plugin by ID, deserialize via plugin, store as erased ref -- is
consistent across all vtable-backed types in Vortex.

## Migration Status

All four vtable-backed types are converging on the pattern described above.

### ExtDType -- Partially done

`ExtVTable`, `ExtDType<V>`, `ExtDTypeRef`, and `ExtDTypePlugin` are in place. Remaining:

- **Drop internal Arc from `ExtDType<V>`.** Currently `ExtDType<V>` wraps
  `Arc<ExtDTypeAdapter<V>>`. It should become the data struct itself, with
  `ExtDTypeRef(Arc<dyn DynExtDType>)` holding the Arc. Remove `ExtDTypeAdapter`.
- **Rename** `ExtDTypeImpl` → `DynExtDType`, `DynExtVTable` → `ExtDTypePlugin`.
- **Remove `ExtDTypeMetadata`** erased wrapper. Its methods (`serialize`, `Display`,
  `Debug`, `PartialEq`, `Hash`) should move to `ExtDTypeRef`.

### Expr -- Not started

Currently uses `VTable` (unqualified), `VTableAdapter`, `DynExprVTable` (sealed trait),
and `ExprVTable` (confusingly, the erased ref). Needs renaming to `ExprVTable`, `DynExpr`,
`ExprRef`. Introduce `Expr<V>` data struct, remove `VTableAdapter`.

### Layout -- Not started

Currently uses `VTable` (unqualified), `LayoutAdapter`, and `Layout` (sealed trait doubling
as public API). Needs renaming to `LayoutVTable`, `DynLayout`, `LayoutRef`. Introduce
`Layout<V>` data struct, remove `LayoutAdapter`.

### Array -- Not started

The largest migration. Currently uses `VTable` (unqualified), `ArrayAdapter`, `Array` (sealed
trait doubling as public API), `ArrayRef`, and `DynVTable`.

#### Target Types

```rust
pub trait ArrayVTable: 'static + Sized + Send + Sync {
    type Array: 'static + Send + Sync + Clone + Debug;
    // Methods encoding authors implement.
    // Receive &Array<Self> for typed access.
}

pub struct Array<V: ArrayVTable> {
    vtable: V,          // ZST for native encodings, non-ZST for language bindings
    dtype: DType,
    len: usize,
    array: V::Array,    // encoding-specific data (buffers, children, etc.)
    stats: ArrayStats,
}

impl<V: ArrayVTable> Deref for Array<V> {
    type Target = V::Array;
}

pub struct ArrayRef(Arc<dyn DynArray>);

trait DynArray: sealed { ... }  // object-safe, thin forwarder
```

`Array<V>` has inherent methods for all operations (slice, filter, take, etc.)
with pre/post-condition enforcement. These delegate to `ArrayVTable` methods.
`DynArray` is a thin blanket forwarder so `ArrayRef` can reach them.

`Array<V>` derefs to `V::Array`, so encoding-specific methods defined on
the associated type are callable directly on `&Array<V>`. Common methods
(`dtype()`, `len()`) are inherent on `Array<V>` and resolve first.

Children live in `V::Array` — each encoding owns its child representation.
The VTable provides `nchildren()` / `child(i)` for generic traversal.

Constructors live on the vtable ZST: `Primitive::new(...) -> ArrayRef`.

When a VTable method needs to signal "return me unchanged" (e.g. `execute`
for canonical types), it returns `None`. The `ArrayRef` public method handles
this by cloning its own Arc.

#### Phases

**Phase 0: Rename `Array` trait → `DynArray`.**
Mechanical rename. Frees the `Array` name for the generic struct.

**Phase 1: Introduce `Array<V>`, migrate encodings.**
Per encoding:

1. Rename vtable ZST (`PrimitiveVTable` → `Primitive`).
2. Current bespoke array struct becomes `V::Array` (the associated type).
3. Wrap it in the generic `Array<V>` struct with common fields hoisted out.
4. Move constructors to vtable ZST.
5. Update all call sites (clean break, no type aliases).

**Phase 2: Update sub-vtable signatures.**
`ValidityVTable`, `OperationsVTable` methods change from `&V::Array` to
`&Array<V>`.

**Phase 3: Migrate erased layer.**

1. Blanket `impl<V: ArrayVTable> DynArray for Array<V>` forwarding to `Array<V>` inherent methods.
2. `ArrayRef` becomes concrete struct wrapping `Arc<dyn DynArray>`.
3. Move `impl dyn DynArray` methods to `impl ArrayRef`.
4. Remove old `DynArray` trait, `ArrayAdapter`, `vtable!` macro.
   Introduce `ArrayPlugin` for ID-based deserialization.
