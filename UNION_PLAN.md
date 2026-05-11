# Plan: Adding `Union` to Vortex

Working document for [#7705](https://github.com/vortex-data/vortex/issues/7705) (epic)
and [#7882](https://github.com/vortex-data/vortex/issues/7882) (tracking).

This is a planning doc, not user-facing documentation. The intent is to lock down
the design decisions that are expensive to revisit before any code lands, then
sequence the work across multiple PRs.

## Goal

Add `Union` as a new variant of `DType` and a canonical sparse encoding to
`Canonical`, closing the last gap with Arrow's spec and unblocking GeoArrow.

## Reference points in the existing codebase

The most relevant existing patterns:

- `DType` enum (`vortex-array/src/dtype/mod.rs:53`) — `Variant` is the most
  recently added variant; it is the best template for adding `Union`.
- `StructFields` (`vortex-array/src/dtype/struct_.rs`) — `UnionFields` will
  mirror its shape: an `Arc<Inner>` carrying names, child dtypes, and a
  `OnceLock` name→index cache.
- `StructArray` (`vortex-array/src/arrays/struct_/array.rs`) — slot layout,
  validity superposition, validation patterns.
- `Canonical` enum (`vortex-array/src/canonical.rs:120`) — where
  `Canonical::Union(UnionArray)` will be added.
- `SparseArray` (`encodings/sparse/`) — a non-canonical encoding crate that is
  expected to be useful as a *child* under `UnionArray` to recover dense-like
  storage density.
- Flatbuffer schema (`vortex-flatbuffers/flatbuffers/vortex-dtype/dtype.fbs:70`)
  and proto schema (`vortex-proto/proto/dtype.proto:77`) — both already have a
  `Variant` placeholder we can use as the template for a new `Union` table.
- Arrow `to_arrow` bail site (`vortex-array/src/arrow/executor/mod.rs:164`)
  and `from_arrow` fallback (`vortex-array/src/dtype/arrow.rs:206`) — current
  explicit gaps for Union.

## Design decisions

### Sparse is canonical; dense is a separate physical encoding

The argument in #7882 holds:

- `Canonical` in Vortex is non-recursive — `canonical.rs:73-78`: *"individual
  column child arrays may still be compressed."* A sparse union with
  `SparseArray` children recovers ~dense storage density without giving up
  trivial slicing.
- Slice, take, filter are all `O(1)` dispatch + per-child operation under
  sparse. Dense requires recomputing per-child offsets from the type tag.

Therefore:

- `Canonical::Union(UnionArray)` is sparse.
- `DenseUnionArray` is a separate non-canonical encoding (likely under
  `encodings/dense-union/` or `vortex-array/src/arrays/union_dense/`). Its
  `to_canonical()` materializes the sparse form.
- The compressor learns to prefer dense when per-type fill density is low
  *and* slicing isn't on the hot path — that's a follow-up, not v1.

### Follow Arrow strictly on validity — no top-level validity buffer

Arrow Struct has top-level validity; Arrow Union does not. The asymmetry has a
reason:

- For Struct, all fields are simultaneously "live" per row. Without a
  top-level validity bit, "row is null" can't be expressed without nulling
  every field (losing the values).
- For Union, the type tag is *already* a row-level discriminator. A null row
  is expressed by pointing the tag at a `DType::Null` child, or by the
  pointed child being nullable and null at that position.

Consequences for the Vortex implementation:

- `UnionArray` has **no validity slot.** Slots are
  `[type_ids, child_0, child_1, ..., child_N]`. This mirrors Arrow byte for
  byte (modulo the dense `offsets` buffer being absent in sparse).
- The `Nullability` on `DType::Union(_, Nullability)` becomes a *typing
  constraint* enforced in `UnionArray::validate()`:
  - `Nullable`: at least one child must be `DType::Null` or nullable.
  - `NonNullable`: no `DType::Null` children and no nullable children.
- `UnionArray::validity()` (the trait method) is *derived*: it walks
  `type_ids` and the per-child validities to produce a `Validity::Array`.
  This is `O(n)` to materialize when asked, but most compute kernels never
  ask — they work with `type_ids` directly.

This costs us `O(1)` validity that `StructArray` has, but it preserves Arrow
round-tripping and avoids the "null row with valid child value" ambiguity.

### Type IDs: support non-consecutive from v1, via optional indirection

The Arrow `Schema.fbs` `Union` table says:

> *"By default ids in the type vector refer to the offsets in the children.
> Optionally `typeIds` provides an indirection between the child offset and
> the type id for each child `typeIds[offset]` is the id used in the type
> vector."*

So:

- The per-row type tag is an `int8` (range -128..127).
- By default, type tag `i` selects child at offset `i` (consecutive `0..N`).
- Optionally, a `typeIds: [int8]` array provides indirection: child at
  offset `i` is referenced in the data by tag `typeIds[i]`. This supports
  schema evolution — removing a child doesn't renumber the others.

We will support the indirection from v1. The storage cost is trivial
(`N` bytes per schema), and adding it later would require a backward-incompatible
flatbuffer/proto migration.

`UnionFields` will carry `type_ids: Option<Arc<[i8]>>`, where `None` means
"consecutive `0..N`" (the common case).

### Mode lives on the array, not on the `DType`

Arrow's `Union` flatbuffer table carries `mode: UnionMode`. We deliberately
**do not** put mode on `DType::Union`. Reasons:

- `DType` is logical; mode is physical. Sparse and dense are two physical
  encodings of the same logical type.
- This is consistent with how Vortex treats `Utf8` (one DType, six valid
  physical encodings).

On Arrow ingest, both `DataType::Union(_, Sparse)` and
`DataType::Union(_, Dense)` produce the same `DType::Union(...)`; the array
side decides which encoding to materialize. On Arrow export, the chosen
encoding is determined by the `UnionArray` variant (canonical sparse vs
`DenseUnionArray`).

## Type system shape

```rust
// vortex-array/src/dtype/mod.rs
pub enum DType {
    // existing variants...
    Union(UnionFields, Nullability),
}

// vortex-array/src/dtype/union.rs (new)
pub struct UnionFields(Arc<UnionFieldsInner>);

struct UnionFieldsInner {
    names: FieldNames,
    dtypes: Arc<[FieldDType]>,
    /// Optional indirection from child offset to type tag in the data.
    /// `None` is equivalent to `Some([0, 1, ..., N-1])` but cheaper.
    type_ids: Option<Arc<[i8]>>,
    indices: OnceLock<HashMap<FieldName, usize>>,
}
```

`UnionFields` should mirror `StructFields` as closely as is reasonable —
naming, accessors (`field`, `field_by_index`, `find`, `project`), Arc
pointer-equality fast paths, lazy `FieldDType::View` support. The one extra
piece is `type_ids` and the `tag_to_child_index(tag) -> Option<usize>`
helper that callers will need.

## Array shape

```rust
// vortex-array/src/arrays/union/array.rs (new)
pub struct UnionArray { /* ArraySlots-backed, like StructArray */ }

// Slot layout (sparse / canonical):
//   slot 0:   type_ids   (PrimitiveArray<i8>, length = N rows)
//   slot 1:   child_0    (length = N rows; fills for inactive rows can be any
//                         valid value; SparseArray is the typical child)
//   slot 2:   child_1
//   ...
//   slot K+1: child_{K-1}
```

`UnionArray::validate()` enforces:

1. Slot count == 1 + number of children declared in the DType.
2. `type_ids` is `Primitive(I8, NonNullable)` of length `N`.
3. Each child has length `N` (sparse invariant).
4. Each `type_ids[i]` is a valid tag (either in `0..N_children` if the DType
   has no indirection, or in the `type_ids` set if it does).
5. The nullability constraint described above.

`validity()` materializes a `Validity::Array(BoolArray)` by walking
`type_ids` and looking up `children[tag_to_offset(type_ids[i])].validity()[i]`.
For DTypes with no nullable/null children, this short-circuits to
`Validity::NonNullable`.

## Serialization

### Flatbuffers (`dtype.fbs`)

Add alongside the existing `Variant` entry:

```fbs
table Union {
  names:    [string];
  dtypes:   [DType];
  type_ids: [byte];   // optional; empty means 0..N consecutive
  nullable: bool;
}

union Type { ..., Variant = 11, Union = 12 }
```

We deliberately omit `mode` here — see the rationale above.

### Proto (`dtype.proto`)

```proto
message Union {
  repeated string names = 1;
  repeated DType dtypes = 2;
  repeated int32 type_ids = 3; // empty means 0..N consecutive; values must fit in int8
  bool nullable = 4;
}
```

(Protobuf doesn't have an int8 type; the int32 values are validated at
deserialization to fit in `i8`.)

## PR sequencing

Each PR is independently shippable. CI scope is narrowed per
`CLAUDE.md` — most PRs only need targeted `cargo nextest run -p <crate>` plus
`./scripts/public-api.sh` when public APIs change.

| # | Title | Scope | Notes |
|---|---|---|---|
| 1 | Add `DType::Union` variant | `UnionFields`, new DType variant, flatbuffer/proto schemas, round-trip tests, Display/Debug/Hash/PartialEq | No `UnionArray` yet. `from_arrow` for `DataType::Union` still bails. Touches every exhaustive `match` on `DType`. |
| 2 | Canonical sparse `UnionArray` | `vortex-array/src/arrays/union/` (vtable, validation, constructors, doc tests, `Canonical::Union`, `Canonical::empty` for `DType::Union`) | Slots `[type_ids, child_0..N]`. Derived `validity()`. |
| 3 | Compute kernels | `slice`, `take`, `filter`, `mask`, scalar accessor, `cast` | Each follows `StructArray`'s per-child pattern with `type_ids` operated on alongside. |
| 4 | Arrow round-trip | `to_arrow` produces Arrow `SparseUnionArray`; `from_arrow` accepts both Arrow sparse and dense → canonical sparse | Unblocks the GeoArrow read path. Fixes the bail at `arrow/executor/mod.rs:164` and the `unimplemented!()` at `dtype/arrow.rs:206`. |
| 5 | `UnionBuilder` + `UnionScalar` | Builder mirroring `StructBuilder` (one inner builder per child + `i8` builder for type_ids); scalar.proto extension; `Scalar::Union` | Needed by Python and DataFusion. |
| 6 | `DenseUnionArray` physical encoding | Separate encoding crate or `arrays/union_dense/`; implements `to_canonical()` → sparse | Likely lives in `encodings/`. |
| 7 | Python bindings | `PyUnionDType`, `dtype_union()` factory, scalar bindings | `vortex-python/src/dtype/`. |
| 8 | FFI / JNI / C++ bridges | `strip_views` recursion case, export/import functions, C++ factory | `vortex-jni/src/dtype.rs:36`, `vortex-cxx/src/dtype.rs`, `vortex-ffi/src/scalar.rs`. |
| 9 | File / layout integration | `vortex-file`, `vortex-layout` — decide whether to add `UnionLayout` or reuse table-style layout per child | Could fold into PR 6 if scope allows. |
| 10 | DataFusion / DuckDB + docs | Integration adjustments, user docs, `docs/specs/dtype-format.md` update | Closing PR. |

GeoArrow support comes "for free" after PR 4 — the geometry types just need
to be expressible as extension types over Union, and we don't need to add
GeoArrow-specific code in this stream.

## Open questions / risks

1. **Validity materialization cost.** Code paths that today call
   `array.validity()` and pass the result to a kernel will pay `O(n)` for a
   Union where they pay `O(1)` for a Struct. We should audit hot kernels to
   see if any need a fast path that consumes `type_ids` directly. This is
   not blocking but worth measuring after PR 3.

2. **Empty / zero-child unions.** Arrow allows them (degenerate). We should
   decide whether to permit `DType::Union(fields_empty, NonNullable)`. I'd
   lean toward allowing it for round-trip fidelity and require length == 0.

3. **Duplicate field names.** `StructArray` permits duplicate names. We
   should be consistent and permit them in `UnionFields` too.

4. **DataFusion mapping.** DataFusion has its own `ScalarValue::Union(_, _, _)`
   shape; the mapping is straightforward but should be confirmed during PR 5.

5. **Compressor strategy.** When/how the compressor selects
   `DenseUnionArray` is a deliberate non-goal of this stream. Keep canonical
   sparse always, ship dense as a manually-applicable encoding, and revisit.

## Out of scope

- Smart compressor heuristics for selecting dense.
- GeoArrow-specific extension types (downstream, can land independently).
- Map type (separate Arrow gap; not part of this epic).
