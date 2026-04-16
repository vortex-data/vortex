// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed array wrappers: [`ArrayInner<V>`] (heap-allocated), [`Array<V>`] (typed handle),
//! and [`ArrayView<V>`] (lightweight borrow).

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
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

/// Construction parameters for typed arrays.
pub struct ArrayParts<V: VTable> {
    pub vtable: V,
    pub dtype: DType,
    pub len: usize,
    pub data: V::ArrayData,
    pub slots: Vec<Option<ArrayRef>>,
}

impl<V: VTable> ArrayParts<V> {
    pub fn new(vtable: V, dtype: DType, len: usize, data: V::ArrayData) -> Self {
        Self {
            vtable,
            dtype,
            len,
            data,
            slots: Vec::new(),
        }
    }

    pub fn with_slots(mut self, slots: Vec<Option<ArrayRef>>) -> Self {
        self.slots = slots;
        self
    }
}

/// Shared bound for helpers that should work over both owned [`Array<V>`] and borrowed
/// [`ArrayView<V>`].
///
/// Extension traits use this to share typed array logic while still exposing the backing
/// [`ArrayRef`] and the encoding-specific [`VTable::ArrayData`].
pub trait TypedArrayRef<V: VTable>: AsRef<ArrayRef> + Deref<Target = V::ArrayData> {
    /// Returns an owned [`Array<V>`] from the reference.
    fn to_owned(&self) -> Array<V> {
        self.as_ref().clone().downcast()
    }
}

impl<V: VTable> TypedArrayRef<V> for Array<V> {}

impl<V: VTable> TypedArrayRef<V> for ArrayView<'_, V> {}
// =============================================================================
// ArrayInner<V> — the concrete type stored inside Arc<dyn DynArray>
// =============================================================================

/// The concrete array type that lives inside an `Arc` behind [`ArrayRef`].
///
/// Prefer using [`Array<V>`] (owned typed handle) for constructing arrays
/// and converting between typed and untyped representations.
/// This type is returned by reference from [`Matcher`] downcasts.
#[doc(hidden)]
pub(crate) struct ArrayInner<V: VTable> {
    pub(crate) vtable: V,
    pub(crate) dtype: DType,
    pub(crate) len: usize,
    pub(crate) data: V::ArrayData,
    pub(crate) slots: Vec<Option<ArrayRef>>,
    pub(crate) stats: ArrayStats,
}

impl<V: VTable> ArrayInner<V> {
    /// Create a new inner array from explicit construction parameters.
    #[doc(hidden)]
    pub fn try_new(new: ArrayParts<V>) -> VortexResult<Self> {
        new.vtable
            .validate(&new.data, &new.dtype, new.len, &new.slots)?;
        Ok(unsafe {
            Self::from_data_unchecked(
                new.vtable,
                new.dtype,
                new.len,
                new.data,
                new.slots,
                ArrayStats::default(),
            )
        })
    }

    /// Create without validation.
    ///
    /// # Safety
    /// Caller must ensure dtype and len match the data.
    pub(crate) unsafe fn from_data_unchecked(
        vtable: V,
        dtype: DType,
        len: usize,
        data: V::ArrayData,
        slots: Vec<Option<ArrayRef>>,
        stats: ArrayStats,
    ) -> Self {
        Self {
            vtable,
            dtype,
            len,
            data,
            slots,
            stats,
        }
    }
}

impl<V: VTable> Deref for ArrayInner<V> {
    type Target = V::ArrayData;
    fn deref(&self) -> &V::ArrayData {
        &self.data
    }
}

impl<V: VTable> DerefMut for ArrayInner<V> {
    fn deref_mut(&mut self) -> &mut V::ArrayData {
        &mut self.data
    }
}

impl<V: VTable> Clone for ArrayInner<V> {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            dtype: self.dtype.clone(),
            len: self.len,
            data: self.data.clone(),
            slots: self.slots.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl<V: VTable> Debug for ArrayInner<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayInner")
            .field("encoding", &self.vtable.id())
            .field("dtype", &self.dtype)
            .field("len", &self.len)
            .field("inner", &self.data)
            .field("slots", &self.slots)
            .finish()
    }
}

impl<V: VTable> IntoArray for ArrayInner<V> {
    fn into_array(self) -> ArrayRef {
        ArrayRef::from_inner(Arc::new(self))
    }
}

impl<V: VTable> From<ArrayInner<V>> for ArrayRef {
    fn from(value: ArrayInner<V>) -> ArrayRef {
        ArrayRef::from_inner(Arc::new(value))
    }
}

// =============================================================================
// Array<V> — typed owned handle wrapping an ArrayRef
// =============================================================================

/// A typed owned handle to an array.
///
/// `Array<V>` holds an [`ArrayRef`] (shared, heap-allocated) and provides typed access
/// to the encoding-specific data via [`Deref`] to `V::ArrayData`.
///
/// This is the primary type for working with typed arrays. Convert to [`ArrayRef`]
/// via [`into_array()`](IntoArray::into_array) or [`AsRef<ArrayRef>`].
pub struct Array<V: VTable> {
    inner: ArrayRef,
    _phantom: PhantomData<V>,
}

impl<V: VTable> Array<V> {
    /// Create a typed array from explicit construction parameters.
    pub fn try_from_parts(new: ArrayParts<V>) -> VortexResult<Self> {
        let inner = ArrayRef::from_inner(Arc::new(ArrayInner::<V>::try_new(new)?));
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
        let inner = ArrayRef::from_inner(Arc::new(unsafe {
            ArrayInner::<V>::from_data_unchecked(
                new.vtable,
                new.dtype,
                new.len,
                new.data,
                new.slots,
                ArrayStats::default(),
            )
        }));
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Create from an existing `ArrayRef`, trusting that it contains `ArrayInner<V>`.
    ///
    /// # Safety
    /// Caller must ensure the `ArrayRef` contains an `ArrayInner<V>`.
    pub(crate) unsafe fn from_array_ref_unchecked(array: ArrayRef) -> Self {
        Self {
            inner: array,
            _phantom: PhantomData,
        }
    }

    /// Try to create from an `ArrayRef`, returning `Err` if the type doesn't match.
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

    /// Returns the dtype.
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

    /// Returns the encoding ID.
    pub fn encoding_id(&self) -> ArrayId {
        self.inner.encoding_id()
    }

    /// Returns the statistics.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.inner.statistics()
    }

    /// Returns a reference to the encoding-specific data.
    pub fn data(&self) -> &V::ArrayData {
        &self.downcast_inner().data
    }

    /// Returns the full typed array construction parts if this handle owns the allocation.
    pub fn try_into_parts(self) -> Result<ArrayParts<V>, Self> {
        let Self { inner, _phantom } = self;
        let any = inner.into_inner().into_any_arc();
        let inner = Arc::downcast::<ArrayInner<V>>(any)
            .unwrap_or_else(|_| unreachable!("typed array must contain ArrayInner for its vtable"));

        match Arc::try_unwrap(inner) {
            Ok(inner) => Ok(ArrayParts {
                vtable: inner.vtable,
                dtype: inner.dtype,
                len: inner.len,
                data: inner.data,
                slots: inner.slots,
            }),
            Err(inner) => Err(Self {
                inner: ArrayRef::from_inner(inner),
                _phantom: PhantomData,
            }),
        }
    }

    pub fn with_stats_set(self, stats: StatsSet) -> Self {
        self.statistics().replace(stats);
        self
    }

    /// Returns a clone of the inner encoding-specific data.
    pub fn into_data(self) -> V::ArrayData {
        self.downcast_inner().data.clone()
    }

    /// Returns the array slots.
    pub fn slots(&self) -> &[Option<ArrayRef>] {
        &self.downcast_inner().slots
    }

    /// Returns the internal [`ArrayRef`].
    #[inline(always)]
    pub fn as_array(&self) -> &ArrayRef {
        &self.inner
    }

    /// Returns an [`ArrayView`] borrowing this array's data.
    pub fn as_view(&self) -> ArrayView<'_, V> {
        let inner = self.downcast_inner();
        // SAFETY: `inner.data` is the `V::ArrayData` stored inside `self.inner`.
        unsafe { ArrayView::new_unchecked(&self.inner, &inner.data) }
    }

    /// Downcast the inner `ArrayRef` to `&ArrayInner<V>`.
    #[inline(always)]
    fn downcast_inner(&self) -> &ArrayInner<V> {
        let any = self.inner.inner().as_any();
        // NOTE(ngates): use downcast_unchecked when it becomes stable
        debug_assert!(any.is::<ArrayInner<V>>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(any as *const dyn Any as *const ArrayInner<V>) }
    }
}

/// Public API methods that shadow `DynArray` / `ArrayRef` methods.
impl<V: VTable> Array<V> {
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

    pub fn filter(&self, mask: vortex_mask::Mask) -> VortexResult<ArrayRef> {
        self.inner.filter(mask)
    }

    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        self.inner.take(indices)
    }

    pub fn validity(&self) -> VortexResult<Validity> {
        self.inner.validity()
    }

    pub fn is_valid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.is_valid(index, ctx)
    }

    pub fn is_invalid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.is_invalid(index, ctx)
    }

    pub fn all_valid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.all_valid(ctx)
    }

    pub fn all_invalid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        self.inner.all_invalid(ctx)
    }

    #[deprecated(note = "Use Array::<V>::execute::<Canonical>() instead")]
    pub fn to_canonical(&self) -> VortexResult<crate::Canonical> {
        self.inner.to_canonical()
    }

    pub fn nbytes(&self) -> u64 {
        self.inner.nbytes()
    }

    pub fn nbuffers(&self) -> usize {
        self.inner.nbuffers()
    }

    pub fn as_constant(&self) -> Option<crate::scalar::Scalar> {
        self.inner.as_constant()
    }

    pub fn valid_count(&self, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        self.inner.valid_count(ctx)
    }

    pub fn invalid_count(&self, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        self.inner.invalid_count(ctx)
    }

    pub fn append_to_builder(
        &self,
        builder: &mut dyn crate::builders::ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.inner.append_to_builder(builder, ctx)
    }
}

impl<V: VTable> Deref for Array<V> {
    type Target = V::ArrayData;

    fn deref(&self) -> &V::ArrayData {
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
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    #[test]
    fn typed_array_into_parts_roundtrips() {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
        let expected = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);

        let parts = array.try_into_parts().unwrap();
        let rebuilt = Array::<Primitive>::try_from_parts(parts).unwrap();

        assert_arrays_eq!(rebuilt, expected);
    }

    #[test]
    fn typed_array_try_into_parts_requires_unique_owner() {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
        let alias = array.clone();

        let array = match array.try_into_parts() {
            Ok(_) => panic!("aliased arrays should not move out their backing parts"),
            Err(array) => array,
        };

        assert_arrays_eq!(array, alias);
    }
}
