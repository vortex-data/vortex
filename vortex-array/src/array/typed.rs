// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed array wrappers: [`ArrayData<V>`] (heap-allocated), [`Array<V>`] (typed handle),
//! and [`ArrayView<V>`] (lightweight borrow).
//!
//! Encoding implementors normally construct arrays through [`ArrayParts`] and
//! [`Array::try_from_parts`]. Compute and serialization code should accept [`ArrayRef`] when it can
//! operate over any encoding, and downcast to [`Array<V>`] or [`ArrayView<V>`] only when it needs
//! encoding-specific state.

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::stats::StatsSet;
use crate::stats::StatsSetRef;
use crate::validity::Validity;

/// The combined allocation behind [`ArrayRef`].
///
/// Stores common metadata (len, dtype, encoding_id, slots, stats) together with the
/// encoding-specific `data` (a concrete [`ArrayData<V>`] erased to `dyn DynArrayData`).
///
/// `ArrayRef` stores `Arc<ArrayInner<dyn DynArrayData>>` — a single 16-byte fat pointer.
/// Metadata is accessed via `self.0.*` (a normal struct field read through the Arc),
/// while encoding-specific methods go through `self.0.data` (vtable dispatch).
pub(crate) struct ArrayInner<D: ?Sized> {
    pub(crate) len: usize,
    pub(crate) encoding_id: ArrayId,
    pub(crate) dtype: DType,
    pub(crate) slots: ArraySlots,
    pub(crate) stats: ArrayStats,
    pub(crate) data: D, // must be last for unsized coercion
}

/// Construction parameters for typed arrays.
pub struct ArrayParts<V: VTable> {
    /// The vtable value identifying the array encoding.
    pub vtable: V,
    /// Logical dtype of every row in the array.
    pub dtype: DType,
    /// Number of rows in the array.
    pub len: usize,
    /// Encoding-specific, non-child data.
    pub data: V::TypedArrayData,
    /// Optional child arrays owned by this encoding.
    pub slots: ArraySlots,
}

impl<V: VTable> ArrayParts<V> {
    /// Construct array parts with no child slots.
    ///
    /// The parts are not validated until they are passed to [`Array::try_from_parts`].
    pub fn new(vtable: V, dtype: DType, len: usize, data: V::TypedArrayData) -> Self {
        Self {
            vtable,
            dtype,
            len,
            data,
            slots: ArraySlots::new(),
        }
    }

    /// Attach child slots to the construction parts.
    ///
    /// Slot count, names, and meaning are encoding-specific and validated by [`VTable::validate`].
    pub fn with_slots(mut self, slots: ArraySlots) -> Self {
        self.slots = slots;
        self
    }
}

/// Shared bound for helpers that should work over both owned [`Array<V>`] and borrowed
/// [`ArrayView<V>`].
///
/// Extension traits use this to share typed array logic while still exposing the backing
/// [`ArrayRef`] and the encoding-specific [`VTable::TypedArrayData`].
pub trait TypedArrayRef<V: VTable>: AsRef<ArrayRef> + Deref<Target = V::TypedArrayData> {
    /// Returns an owned [`Array<V>`] from the reference.
    fn to_owned(&self) -> Array<V> {
        self.as_ref().clone().downcast()
    }
}

impl<V: VTable> TypedArrayRef<V> for Array<V> {}

impl<V: VTable> TypedArrayRef<V> for ArrayView<'_, V> {}
// =============================================================================
// ArrayData<V> — the concrete type stored inside Arc<dyn DynArrayData>
// =============================================================================

/// A VTable and its instance data, this can be type-erased to [`DynArrayData`](DynArrayData).
#[doc(hidden)]
pub(crate) struct ArrayData<V: VTable> {
    pub(crate) vtable: V,
    pub(crate) data: V::TypedArrayData,
}

impl<V: VTable> ArrayInner<ArrayData<V>> {
    /// Create a new validated [`ArrayInner`] from construction parameters.
    #[doc(hidden)]
    pub fn try_new(new: ArrayParts<V>) -> VortexResult<Self> {
        new.vtable
            .validate(&new.data, &new.dtype, new.len, &new.slots)?;
        Ok(ArrayInner {
            len: new.len,
            encoding_id: new.vtable.id(),
            dtype: new.dtype,
            slots: new.slots,
            stats: ArrayStats::default(),
            data: ArrayData {
                vtable: new.vtable,
                data: new.data,
            },
        })
    }

    /// Create an [`ArrayInner`] without validation.
    ///
    /// # Safety
    /// Caller must ensure dtype and len match the data.
    pub(crate) unsafe fn new_unchecked(
        vtable: V,
        len: usize,
        dtype: DType,
        data: V::TypedArrayData,
        slots: ArraySlots,
        stats: ArrayStats,
    ) -> Self {
        ArrayInner {
            len,
            encoding_id: vtable.id(),
            dtype,
            slots,
            stats,
            data: ArrayData { vtable, data },
        }
    }
}

impl<V: VTable> Deref for ArrayData<V> {
    type Target = V::TypedArrayData;
    fn deref(&self) -> &V::TypedArrayData {
        &self.data
    }
}

impl<V: VTable> DerefMut for ArrayData<V> {
    fn deref_mut(&mut self) -> &mut V::TypedArrayData {
        &mut self.data
    }
}

impl<V: VTable> Clone for ArrayData<V> {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            data: self.data.clone(),
        }
    }
}

impl<V: VTable> Debug for ArrayData<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayData")
            .field("encoding", &self.vtable.id())
            .field("inner", &self.data)
            .finish()
    }
}

// =============================================================================
// Array<V> — typed owned handle wrapping an ArrayRef
// =============================================================================

/// A typed owned handle to an array.
///
/// `Array<V>` holds an [`ArrayRef`] (shared, heap-allocated) and provides typed access
/// to the encoding-specific data via [`Deref`] to `V::TypedArrayData`.
///
/// This is the primary type for working with typed arrays. Convert to [`ArrayRef`]
/// via [`into_array()`](IntoArray::into_array) or [`AsRef<ArrayRef>`].
pub struct Array<V: VTable> {
    inner: ArrayRef,
    _phantom: PhantomData<V>,
}

impl<V: VTable> Array<V> {
    /// Create a typed array from explicit construction parameters.
    ///
    /// This is the safe construction path for encoding implementors. It calls
    /// [`VTable::validate`] before publishing the array as an [`ArrayRef`].
    pub fn try_from_parts(new: ArrayParts<V>) -> VortexResult<Self> {
        let store = ArrayInner::<ArrayData<V>>::try_new(new)?;
        let inner = ArrayRef::from_inner(Arc::new(store));
        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Create a typed array from explicit construction parameters without validation.
    ///
    /// # Safety
    /// Caller must ensure the provided parts are logically consistent.
    #[doc(hidden)]
    pub unsafe fn from_parts_unchecked(new: ArrayParts<V>) -> Self {
        let store = unsafe {
            ArrayInner::<ArrayData<V>>::new_unchecked(
                new.vtable,
                new.len,
                new.dtype,
                new.data,
                new.slots,
                ArrayStats::default(),
            )
        };
        let inner = ArrayRef::from_inner(Arc::new(store));
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Create from an existing `ArrayRef`, trusting that it contains `ArrayData<V>`.
    ///
    /// # Safety
    /// Caller must ensure the `ArrayRef` contains an `ArrayData<V>`.
    pub(crate) unsafe fn from_array_ref_unchecked(array: ArrayRef) -> Self {
        Self {
            inner: array,
            _phantom: PhantomData,
        }
    }

    /// Try to create a typed handle from an [`ArrayRef`].
    ///
    /// Returns the original [`ArrayRef`] in `Err` when the encoding id does not match `V`.
    pub fn try_from_array_ref(array: ArrayRef) -> Result<Self, ArrayRef> {
        if array.is::<V>() {
            Ok(Self {
                inner: array,
                _phantom: PhantomData,
            })
        } else {
            Err(array)
        }
    }

    /// Returns the logical dtype.
    pub fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    /// Returns the length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether this array is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Returns the encoding ID for `V`.
    pub fn encoding_id(&self) -> ArrayId {
        self.inner.encoding_id()
    }

    /// Returns this array's statistics set.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.inner.statistics()
    }

    /// Returns a reference to the encoding-specific data.
    pub fn data(&self) -> &V::TypedArrayData {
        &self.downcast_inner().data
    }

    /// Try to fetch mutable access to the encoding-specific data.
    ///
    /// Returns `None` when this handle is not the unique owner of the backing allocation.
    pub fn data_mut(&mut self) -> Option<&mut V::TypedArrayData> {
        let store = self.inner.inner_mut()?;
        let array_inner = store.data.as_any_mut().downcast_mut::<ArrayData<V>>();
        Some(&mut array_inner?.data)
    }

    /// Returns the full typed array construction parts if this handle owns the allocation.
    pub fn try_into_parts(self) -> Result<ArrayParts<V>, Self> {
        let Self { inner, _phantom } = self;
        // SAFETY: Array<V> guarantees the inner is ArrayData<V>.
        let typed_arc = unsafe { inner.downcast_inner_unchecked::<V>() };

        match Arc::try_unwrap(typed_arc) {
            Ok(store) => Ok(ArrayParts {
                vtable: store.data.vtable,
                dtype: store.dtype,
                len: store.len,
                data: store.data.data,
                slots: store.slots,
            }),
            Err(typed_arc) => Err(Self {
                inner: ArrayRef::from_inner(typed_arc),
                _phantom: PhantomData,
            }),
        }
    }

    /// Replace the array's statistics set and return the same typed handle.
    pub fn with_stats_set(self, stats: StatsSet) -> Self {
        self.statistics().replace(stats);
        self
    }

    /// Returns a clone of the inner encoding-specific data.
    pub fn into_data(self) -> V::TypedArrayData {
        self.downcast_inner().data.clone()
    }

    /// Returns the array slots.
    pub fn slots(&self) -> &[Option<ArrayRef>] {
        self.inner.slots()
    }

    /// Returns the internal [`ArrayRef`].
    #[inline(always)]
    pub fn as_array(&self) -> &ArrayRef {
        &self.inner
    }

    /// Returns an [`ArrayView`] borrowing this array's data.
    pub fn as_view(&self) -> ArrayView<'_, V> {
        let inner = self.downcast_inner();
        // SAFETY: `inner.data` is the `V::TypedArrayData` stored inside `self.inner`.
        unsafe { ArrayView::new_unchecked(&self.inner, &inner.data) }
    }

    /// Downcast the inner `ArrayRef` to `&ArrayData<V>`.
    #[inline(always)]
    fn downcast_inner(&self) -> &ArrayData<V> {
        let any = self.inner.dyn_array().as_any();
        // NOTE(ngates): use downcast_unchecked when it becomes stable
        debug_assert!(any.is::<ArrayData<V>>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(any as *const dyn Any as *const ArrayData<V>) }
    }
}

/// Public API methods that shadow `DynArrayData` / `ArrayRef` methods.
impl<V: VTable> Array<V> {
    /// Lazily or eagerly slice the array to `range`, depending on available kernels.
    pub fn slice(&self, range: std::ops::Range<usize>) -> VortexResult<ArrayRef> {
        self.inner.slice(range)
    }

    #[deprecated(
        note = "Use `execute_scalar` instead, which allows passing an execution context for more \
        efficient execution when fetching multiple scalars from the same array."
    )]
    pub fn scalar_at(&self, index: usize) -> VortexResult<crate::scalar::Scalar> {
        self.inner
            .execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())
    }

    /// Execute the array to extract a scalar at the given index.
    pub fn execute_scalar(
        &self,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<crate::scalar::Scalar> {
        self.inner.execute_scalar(index, ctx)
    }

    /// Filter the array with a selection mask.
    pub fn filter(&self, mask: vortex_mask::Mask) -> VortexResult<ArrayRef> {
        self.inner.filter(mask)
    }

    /// Gather rows from this array by index.
    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        self.inner.take(indices)
    }

    /// Returns the array's validity representation.
    pub fn validity(&self) -> VortexResult<Validity> {
        self.inner.validity()
    }

    /// Returns whether `index` is valid using the provided execution context.
    pub fn is_valid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.is_valid(index, ctx)
    }

    /// Returns whether `index` is null using the provided execution context.
    pub fn is_invalid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.is_invalid(index, ctx)
    }

    /// Returns whether every row is valid.
    pub fn all_valid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.all_valid(ctx)
    }

    /// Returns whether every row is null.
    pub fn all_invalid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.all_invalid(ctx)
    }

    #[deprecated(note = "Use Array::<V>::execute::<Canonical>() instead")]
    pub fn to_canonical(&self) -> VortexResult<crate::Canonical> {
        #[expect(deprecated)]
        let result = self.inner.to_canonical();
        result
    }

    /// Returns the estimated physical bytes owned or referenced by this array tree.
    pub fn nbytes(&self) -> u64 {
        self.inner.nbytes()
    }

    /// Returns the number of top-level buffers exposed by this encoding.
    pub fn nbuffers(&self) -> usize {
        self.inner.nbuffers()
    }

    /// Returns the scalar value when this array is known to be constant.
    pub fn as_constant(&self) -> Option<crate::scalar::Scalar> {
        self.inner.as_constant()
    }

    /// Counts valid rows, executing validity arrays when necessary.
    pub fn valid_count(&self, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        self.inner.valid_count(ctx)
    }

    /// Counts null rows, executing validity arrays when necessary.
    pub fn invalid_count(&self, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        self.inner.invalid_count(ctx)
    }

    /// Append this array's logical values to a canonical builder.
    pub fn append_to_builder(
        &self,
        builder: &mut dyn crate::builders::ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.inner.append_to_builder(builder, ctx)
    }
}

impl<V: VTable> Deref for Array<V> {
    type Target = V::TypedArrayData;

    fn deref(&self) -> &V::TypedArrayData {
        self.data()
    }
}

impl<V: VTable> Clone for Array<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<V: VTable> Debug for Array<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("encoding", &self.inner.encoding_id())
            .field("dtype", self.inner.dtype())
            .field("len", &self.inner.len())
            .finish()
    }
}

impl<V: VTable> AsRef<ArrayRef> for Array<V> {
    fn as_ref(&self) -> &ArrayRef {
        &self.inner
    }
}

impl<V: VTable> IntoArray for Array<V> {
    #[inline(always)]
    fn into_array(self) -> ArrayRef {
        self.inner
    }
}

impl<V: VTable> IntoArray for Arc<Array<V>> {
    fn into_array(self) -> ArrayRef {
        match Arc::try_unwrap(self) {
            Ok(array) => array.inner,
            Err(arc) => arc.inner.clone(),
        }
    }
}

impl<V: VTable> From<Array<V>> for ArrayRef {
    fn from(value: Array<V>) -> ArrayRef {
        value.inner
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::Array;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    #[test]
    fn typed_array_into_parts_roundtrips() {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
        let expected = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);

        let parts = array.try_into_parts().unwrap();
        let rebuilt = Array::<Primitive>::try_from_parts(parts).unwrap();

        assert_arrays_eq!(rebuilt, expected, &mut ctx);
    }

    #[test]
    fn typed_array_try_into_parts_requires_unique_owner() {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
        let alias = array.clone();

        let array = match array.try_into_parts() {
            Ok(_) => panic!("aliased arrays should not move out their backing parts"),
            Err(array) => array,
        };

        assert_arrays_eq!(array, alias, &mut ctx);
    }
}
