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
`DynFoo` must stay private (`pub(super)`) because it exposes internal plumbing
(`as_any`, `metadata_any`) that should not be part of the public API.

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

## File Conventions

For a concept `Foo`, the components are organized into these files:

| File         | Contains                                                                    |
|--------------|-----------------------------------------------------------------------------|
| `vtable.rs`  | `FooVTable` trait definition                                                |
| `typed.rs`   | `Foo<V>` data struct, inherent methods, `DynFoo` sealed trait, blanket impl |
| `erased.rs`  | `FooRef` struct                                                             |
| `plugin.rs`  | `FooPlugin` trait, registration                                             |
| `matcher.rs` | Downcasting helpers (`is`, `as_`, `as_opt`, pattern matching traits)        |

For Array encodings, each encoding has its own module (e.g. `arrays/primitive/`):

| File                   | Contains                                                    |
|------------------------|-------------------------------------------------------------|
| `arrays/foo/mod.rs`    | `V::Array` associated type, encoding-specific methods on it |
| `arrays/foo/vtable.rs` | `ArrayVTable` impl for this encoding                        |
| `arrays/foo/compute/`  | Compute kernel implementations                              |

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

## Method Overlap and `same_name_method`

`Foo<V>` and `DynFoo` necessarily share method names: `Foo<V>` needs inherent methods
so callers (including VTable authors who receive `&Foo<Self>`) can use them directly,
and `DynFoo` needs the same methods for object-safe dispatch from `FooRef`.

Because `Foo<V>` implements `DynFoo`, having both an inherent `id()` and a trait `id()`
on the same type triggers `clippy::same_name_method`. The correct handling is:

1. **Logic lives in `Foo<V>` inherent methods.** Pre/post-conditions, field access, and
   delegation to `FooVTable` all happen here.
2. **`DynFoo` blanket impl is a thin forwarder.** Each method body is just `self.method()`.
   Rust's method resolution picks inherent methods over trait methods, so this calls the
   inherent impl — no infinite recursion.
3. **`#[allow(clippy::same_name_method)]`** on the `Foo<V>` inherent impl block
   acknowledges the intentional shadowing.

```rust
#[allow(clippy::same_name_method)]
impl<V: FooVTable> Foo<V> {
    pub fn id(&self) -> FooId {
        self.vtable.id()            // logic lives here
    }
}

impl<V: FooVTable> DynFoo for Foo<V> {
    fn id(&self) -> FooId {
        self.id()                   // thin forwarder to inherent
    }
}
```

`DynFoo` must stay **private** (`pub(super)`) because it exposes internal plumbing
(`as_any`, `metadata_any`) that external callers should never reach. Making it public
or implementing it for `FooRef` would leak these internals. Instead, `FooRef` has its
own inherent methods that delegate to `DynFoo` — providing a clean public API without
exposing the dispatch machinery.

Methods that exist only for erased dispatch (e.g. `as_any`, `metadata_any`,
`metadata_hash`) have no inherent counterpart on `Foo<V>` and live exclusively in
`DynFoo`.

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

File split already matches conventions: `vtable.rs`, `typed.rs`, `erased.rs`,
`plugin.rs`, `matcher.rs`. Remaining:

- **Drop internal Arc from `ExtDType<V>`.** Currently wraps
  `Arc<ExtDTypeAdapter<V>>`. Should become the data struct itself, with
  `ExtDTypeRef(Arc<dyn DynExtDType>)` holding the Arc. Remove `ExtDTypeAdapter`.
- **Rename** `ExtDTypeImpl` → `DynExtDType`, `DynExtVTable` → `ExtDTypePlugin`.
- **Remove `ExtDTypeMetadata`** erased wrapper. Its methods (`serialize`, `Display`,
  `Debug`, `PartialEq`, `Hash`) should move to `ExtDTypeRef`.

### Expr -- Not started

Currently uses `VTable` (unqualified), `VTableAdapter`, `DynExprVTable` (sealed trait),
and `ExprVTable` (confusingly, the erased ref). Needs renaming to `ExprVTable`, `DynExpr`,
`ExprRef`. Introduce `Expr<V>` data struct, remove `VTableAdapter`.

### Layout -- Implemented for serialized scan layouts

The scan layout path follows this pattern in `vortex_layout::layout_v2`:

- `layout_v2::VTable` is the layout vtable implemented by layout plugins.
- `Layout<V>` is the typed layout handle with common fields hoisted: dtype, row count, segment IDs,
  and lazy child access.
- `V::LayoutData` stores only layout-specific metadata.
- `LayoutRef` is the public type-erased layout handle.
- `DynLayout` is private erased dispatch plumbing.
- `LayoutVTablePlugin` is the registry object used for ID-based footer deserialization.

The layout vtable also owns scan expansion through `new_scan_plan`. This keeps serialized layout
metadata and runtime scan behavior registered at the same plugin point: deserializing a layout
produces `Layout<V>`, and scanning it expands that typed layout into a `ScanPlan`.

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
