# ArrayView Migration Plan

## Changes
1. `struct ArrayRef(Arc<dyn DynArray>)` — newtype with public inherent methods
2. `DynArray` — private trait, methods take `&self` + `this: &ArrayRef`
3. `ArrayView<'a, V>` — Copy view with `&'a ArrayRef` + `&'a V::ArrayData`
4. `VTable` methods take `ArrayView<'_, Self>` instead of `&Array<Self>`
5. `OperationsVTable` / `ValidityVTable` take `ArrayView`
6. All VTable implementations updated
7. All callers updated

## Files to change
- vortex-array/src/array/mod.rs — ArrayRef newtype, DynArray private, public API
- vortex-array/src/vtable/typed.rs — ArrayView already added
- vortex-array/src/vtable/mod.rs — VTable trait signatures
- vortex-array/src/vtable/dyn_.rs — DynVTable bridge
- vortex-array/src/vtable/operations.rs — OperationsVTable
- vortex-array/src/vtable/validity.rs — ValidityVTable
- ~40 VTable implementation files
- ~100+ caller files
