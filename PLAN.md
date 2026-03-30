# Phase 5: `Array<V>` as Primary Stack Type

## Goal

Make `Array<V>` the primary type users hold on the stack. `FooArray` becomes a type alias for `Array<Foo>`. Constructors stay on `FooArray` (i.e. `impl Array<Foo>`) for first-party types, and on `FooArrayExt` extension trait for third-party encoding types.

## Key Insight

`Array<V>` already has everything:
- `Deref<Target = V::Array>` → encoding-specific field access (`.bits`, `.validity`, `.ptype()`, etc.)
- Implements `DynArray` → `scalar_at()`, `slice()`, `to_canonical()`, `is_valid()`, etc.
- Has `len()`, `dtype()`, `is_empty()`, `statistics()` directly

Today's `FooArray` is the inner data struct. After this change, `FooArray = Array<Foo>` gives users both inner fields AND DynArray methods in one type. This eliminates all 350 remaining compile errors from Phase 4b plus the `.clone().into_array()` band-aids.

## Naming Convention

| Before | After |
|--------|-------|
| `BoolArray` (struct with fields) | `BoolData` (inner data struct, `pub(crate)`) |
| `Array<Bool>` (typed wrapper) | `Array<Bool>` (unchanged) |
| N/A | `pub type BoolArray = Array<Bool>` (public alias) |
| `BoolArray::new(bits, validity)` | `BoolArray::new(bits, validity)` (now on `impl Array<Bool>`) |

For encoding crates (can't add inherent impls to `Array<Foo>` due to orphan rules):

| Before | After |
|--------|-------|
| `ALPArray` (struct) | `ALPData` (inner data struct) |
| N/A | `pub type ALPArray = Array<ALP>` |
| `ALPArray::try_new(...)` | `ALPArray::try_new(...)` (via `ALPArrayExt` trait) |

## Phases

### Phase 5a: Infrastructure

1. **Add `Array::new()` safe constructor** in `vtable/typed.rs`:
   ```rust
   impl<V: VTable> Array<V> {
       pub fn new(array: V::Array) -> Self {
           let vtable = V::vtable(&array).clone();
           let dtype = V::dtype(&array).clone();
           let len = V::len(&array);
           let stats = V::stats(&array).clone();
           unsafe { Self::new_unchecked(vtable, dtype, len, array, stats) }
       }
   }
   ```

2. **Update `vtable!` macro** to:
   - Accept inner data type name explicitly: `vtable!(Bool, BoolData)`
   - Generate `pub type BoolArray = Array<Bool>;`
   - Generate `IntoArray for BoolData` (so `.into_array()` still works on inner type)
   - Generate `From<BoolData> for ArrayRef`

### Phase 5b: Migrate Built-in Array Types (~20 types)

For each array type (Bool, Primitive, Chunked, Constant, Decimal, Dict, Extension, Filter, FixedSizeList, List, ListView, Masked, Null, ScalarFn, Shared, Slice, Struct, VarBin, VarBinView, Variant):

1. **Rename inner struct**: `BoolArray` → `BoolData`
   - Update all field references within the module
   - Update `VTable for Bool`: `type Array = BoolData;`
   - Update `vtable!(Bool)` invocation

2. **Add type alias**: `pub type BoolArray = Array<Bool>;`

3. **Move constructors** from `impl BoolData` to `impl Array<Bool>`:
   - `new()`, `try_new()`, `from_indices()`, etc.
   - These construct `BoolData` internally, then wrap via `Array::new(inner)`

4. **Move consuming methods** that can't go through Deref to `impl Array<Bool>`:
   - `into_parts()`, `into_bit_buffer()`, etc. → `self.into_inner().xxx()`

5. **Move `FromIterator` impls** to target `Array<Bool>` instead of `BoolData`

6. **Update `Canonical` enum** variants:
   - `Canonical::Bool(BoolData)` → `Canonical::Bool(BoolArray)` (i.e., `Array<Bool>`)

7. **Update re-exports** in `arrays/mod.rs`:
   - Still export `BoolArray` (now a type alias) and `Bool`

### Phase 5c: Migrate Encoding Array Types (~17 types)

Same as 5b but constructors go on extension traits since encoding crates can't add inherent impls to `Array<Foo>`:

```rust
// In encodings/alp/src/alp/array.rs
pub trait ALPArrayExt {
    fn try_new(encoded: ArrayRef, exponents: Exponents, patches: Option<Patches>) -> VortexResult<ALPArray>;
}

impl ALPArrayExt for ALPArray {
    fn try_new(...) -> VortexResult<ALPArray> {
        let inner = ALPData { ... };
        Ok(Array::new(inner))
    }
}
```

Encoding crates: ALP, ALPRD, BitPacked, ByteBool, DateTimeParts, DecimalByteParts, Delta, FoR, FSST, Pco, RLE, RunEnd, Sequence, Sparse, ZigZag, Zstd, ZstdBuffers.

### Phase 5d: Fix Callers

Most callers should "just work" because:
- `BoolArray` is still the name (now an alias for `Array<Bool>`)
- `Array<Bool>: Deref<Target = BoolData>` so field access works
- `Array<Bool>` has DynArray methods, so `scalar_at()`, `slice()`, etc. work directly

Remaining fixes:
- Remove `.clone().into_array().method()` band-aids → just `.method()` directly
- Update pattern matches on `Canonical` (variants now hold `Array<V>`)
- Update any code that constructs inner data types directly

### Phase 5e: Cleanup

- Remove inherent `len()`/`dtype()`/`is_empty()`/`validity()`/`validity_mask()` from inner data types (these come from `Array<V>` directly)
- Remove now-unused `IntoArray` import band-aids
- Run `cargo clippy --all-targets --all-features`, `cargo +nightly fmt --all`, `cargo xtask public-api`

## Ordering Strategy

1. Phase 5a (infrastructure) — must be first
2. Phase 5b for **Bool only** — proof of concept, validate the pattern compiles
3. Phase 5b for remaining built-in types — can be parallelized
4. Phase 5c for encoding types — can be parallelized
5. Phase 5d — fix callers
6. Phase 5e — cleanup

## Risks & Mitigations

- **`FromIterator` for type aliases**: `impl FromIterator<bool> for Array<Bool>` works in Rust (first-party crate owns `Array`).
- **Method resolution**: `Array<Bool>` inherent methods shadow `BoolData` methods via Deref when names conflict. `Array<V>` already has `len()`, `dtype()` — these win over `BoolData`'s versions. Remove duplicates from `BoolData` in Phase 5e.
- **`Canonical` size**: `Array<V>` is larger than `V::Array` (extra vtable + dtype + len + stats fields). Acceptable tradeoff.
- **`VTable::build()`**: Returns `V::Array` (inner type). Still works — construct `BoolData`, wrap in `Array<Bool>` via `Array::new()`.
- **Duplicate dtype/len/stats**: Inner type keeps its fields (needed for `VTable::build()` deserialization path). `Array<V>` caches copies. Deduplication is a future optimization.
- **Orphan rules for encoding crates**: Can't write `impl Array<ALP> { ... }` in the ALP crate. Use `ALPArrayExt` extension trait instead.

## Scope Estimate

- ~20 built-in array types in `vortex-array`
- ~17 encoding array types in `encodings/`
- ~3-5 constructors per type = ~100-200 constructors to move
- ~17 extension traits for encoding crates
- `Canonical` enum + `match_each_canonical!` macro
- All test/bench code that constructs arrays
