// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed array wrapper [`Array<V>`] that pairs a [`VTable`] with common fields and inner data.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

/// A typed array, parameterized by a concrete [`VTable`].
///
/// This struct holds the vtable instance, common fields (dtype, len, stats), and the
/// encoding-specific data (`V::Array`). It implements [`Deref`] to `V::Array`, so
/// encoding-specific methods are callable directly on `ArrayView<'_, V>`.
///
/// Construct via encoding-specific constructors and type-erase with
/// [`into_array()`](IntoArray::into_array).
pub struct Array<V: VTable> {
    vtable: V,
    pub(crate) dtype: DType,
    pub(crate) len: usize,
    pub(crate) data: V::ArrayData,
    pub(crate) stats: ArrayStats,
}

#[allow(clippy::same_name_method)]
impl<V: VTable> Array<V> {
    /// Create a new typed array from encoding-specific data.
    ///
    /// Extracts dtype, len, vtable, and stats from the data via [`VTable`] methods.
    pub fn try_from_data(data: V::ArrayData) -> VortexResult<Self> {
        let vtable = V::vtable(&data).clone();
        let dtype = V::dtype(&data).clone();
        let len = V::len(&data);
        let stats = V::stats(&data).clone();
        // SAFETY: dtype and len are extracted from `data` via VTable methods.
        Ok(unsafe { Self::from_data_unchecked(vtable, dtype, len, data, stats) })
    }

    /// Create a new typed array without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `V::dtype(&data) == &dtype` and `V::len(&data) == len`.
    pub unsafe fn from_data_unchecked(
        vtable: V,
        dtype: DType,
        len: usize,
        data: V::ArrayData,
        stats: ArrayStats,
    ) -> Self {
        Self {
            vtable,
            dtype,
            len,
            data,
            stats,
        }
    }

    /// Returns a reference to the vtable.
    pub fn vtable(&self) -> &V {
        &self.vtable
    }

    /// Returns the dtype of this array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the encoding ID of this array.
    pub fn encoding_id(&self) -> ArrayId {
        self.vtable.id()
    }

    /// Returns the statistics for this array.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.stats.to_ref(self)
    }

    /// Returns a reference to the underlying [`ArrayStats`].
    pub fn array_stats(&self) -> &ArrayStats {
        &self.stats
    }

    /// Returns a reference to the inner encoding-specific array data.
    pub fn data(&self) -> &V::ArrayData {
        &self.data
    }

    /// Consumes this array and returns the inner encoding-specific data.
    pub fn into_data(self) -> V::ArrayData {
        self.data
    }

    /// Returns a cloned [`ArrayRef`] for this array.
    pub fn to_array_ref(&self) -> ArrayRef {
        ArrayRef::from(self.clone())
    }

    /// Calls `f` with an [`ArrayView`] backed by a temporary [`ArrayRef`].
    ///
    /// This creates a temporary `ArrayRef` via cloning, then constructs an `ArrayView` from it.
    /// The result of `f` must not borrow from the view (the view's lifetime is limited to the
    /// closure).
    pub fn with_view<R>(&self, f: impl FnOnce(ArrayView<'_, V>) -> R) -> R {
        let tmp = self.to_array_ref();
        // SAFETY: `self.data` is the `V::ArrayData` stored inside `tmp` (same clone).
        let view = unsafe { ArrayView::new(&tmp, &self.data) };
        f(view)
    }
}

impl<V: VTable> Array<V>
where
    V::ArrayData: crate::vtable::ValidityHelper,
{
    /// Returns a reference to the validity of this array.
    ///
    /// This inherent method shadows `DynArray::validity()` to provide direct access
    /// to the concrete validity without going through `VortexResult`.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> &crate::validity::Validity {
        crate::vtable::ValidityHelper::validity(&self.data)
    }
}

/// Public API methods on `Array<V>` — these shadow the `DynArray` trait methods
/// so callers don't need to import `DynArray`.
impl<V: VTable> Array<V> {
    /// Performs a constant-time slice of the array.
    #[allow(clippy::same_name_method)]
    pub fn slice(&self, range: std::ops::Range<usize>) -> VortexResult<ArrayRef> {
        <Self as crate::DynArray>::slice(self, range)
    }

    /// Fetch the scalar at the given index.
    #[allow(clippy::same_name_method)]
    pub fn scalar_at(&self, index: usize) -> VortexResult<crate::scalar::Scalar> {
        <Self as crate::DynArray>::scalar_at(self, index)
    }

    /// Wraps the array in a FilterArray such that it is logically filtered by the given mask.
    #[allow(clippy::same_name_method)]
    pub fn filter(&self, mask: vortex_mask::Mask) -> VortexResult<ArrayRef> {
        <Self as crate::DynArray>::filter(self, mask)
    }

    /// Wraps the array in a DictArray such that it is logically taken by the given indices.
    #[allow(clippy::same_name_method)]
    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        <Self as crate::DynArray>::take(self, indices)
    }

    /// Returns whether the item at `index` is valid.
    #[allow(clippy::same_name_method)]
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        <Self as crate::DynArray>::is_valid(self, index)
    }

    /// Returns whether the item at `index` is invalid.
    #[allow(clippy::same_name_method)]
    pub fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        <Self as crate::DynArray>::is_invalid(self, index)
    }

    /// Returns whether all items in the array are valid.
    #[allow(clippy::same_name_method)]
    pub fn all_valid(&self) -> VortexResult<bool> {
        <Self as crate::DynArray>::all_valid(self)
    }

    /// Returns whether all items in the array are invalid.
    #[allow(clippy::same_name_method)]
    pub fn all_invalid(&self) -> VortexResult<bool> {
        <Self as crate::DynArray>::all_invalid(self)
    }

    /// Returns the canonical representation of the array.
    #[allow(clippy::same_name_method)]
    pub fn to_canonical(&self) -> VortexResult<crate::Canonical> {
        <Self as crate::DynArray>::to_canonical(self)
    }

    /// Total size of the array in bytes, including all children and buffers.
    pub fn nbytes(&self) -> u64 {
        self.to_array_ref().nbytes()
    }

    /// Returns the number of buffers this array would serialize.
    #[allow(clippy::same_name_method)]
    pub fn nbuffers(&self) -> usize {
        self.with_view(V::nbuffers)
    }

    /// If this array is a constant, returns the scalar value.
    // TODO(ngates): remove this... we already know if we're constant or not
    pub fn as_constant(&self) -> Option<crate::scalar::Scalar> {
        self.to_array_ref().as_constant()
    }

    /// Returns the number of valid elements.
    #[allow(clippy::same_name_method)]
    pub fn valid_count(&self) -> VortexResult<usize> {
        <Self as crate::DynArray>::valid_count(self)
    }

    /// Returns the number of invalid elements.
    #[allow(clippy::same_name_method)]
    pub fn invalid_count(&self) -> VortexResult<usize> {
        <Self as crate::DynArray>::invalid_count(self)
    }

    /// Writes the array into a canonical builder.
    #[allow(clippy::same_name_method)]
    pub fn append_to_builder(
        &self,
        builder: &mut dyn crate::builders::ArrayBuilder,
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<()> {
        <Self as crate::DynArray>::append_to_builder(self, builder, ctx)
    }

    /// Returns the array as an [`ArrayRef`].
    #[allow(clippy::same_name_method)]
    #[deprecated(note = "use `.to_array_ref()` or `.into_array()` instead")]
    pub fn to_array(&self) -> ArrayRef {
        self.to_array_ref()
    }

    /// Returns the validity mask.
    #[allow(clippy::same_name_method)]
    pub fn validity_mask(&self) -> VortexResult<vortex_mask::Mask> {
        <Self as crate::DynArray>::validity_mask(self)
    }
}

impl<V: VTable> Deref for Array<V> {
    type Target = V::ArrayData;

    fn deref(&self) -> &V::ArrayData {
        &self.data
    }
}

impl<V: VTable> DerefMut for Array<V> {
    fn deref_mut(&mut self) -> &mut V::ArrayData {
        &mut self.data
    }
}

impl<V: VTable> Clone for Array<V> {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            dtype: self.dtype.clone(),
            len: self.len,
            data: self.data.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl<V: VTable> Debug for Array<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("encoding", &self.vtable.id())
            .field("dtype", &self.dtype)
            .field("len", &self.len)
            .field("inner", &self.data)
            .finish()
    }
}

impl<V: VTable> IntoArray for Array<V> {
    fn into_array(self) -> ArrayRef {
        ArrayRef::from(self)
    }
}

impl<V: VTable> IntoArray for Arc<Array<V>> {
    fn into_array(self) -> ArrayRef {
        ArrayRef::from(self)
    }
}

/// A lightweight, `Copy`-able typed view into an [`ArrayRef`].
///
/// `ArrayView` pairs a reference to the type-erased [`ArrayRef`] with a reference to the
/// encoding-specific data (`V::ArrayData`), allowing zero-cost typed access without cloning
/// or re-wrapping. It [`Deref`]s to `V::ArrayData` for direct field access.
pub struct ArrayView<'a, V: VTable> {
    array: &'a ArrayRef,
    data: &'a V::ArrayData,
}

// Manual Copy/Clone impls to avoid requiring `V: Copy/Clone`.
impl<V: VTable> Copy for ArrayView<'_, V> {}

impl<V: VTable> Clone for ArrayView<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: VTable> ArrayView<'a, V> {
    /// Create a new `ArrayView`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` is the `V::ArrayData` stored inside `array`.
    pub(crate) unsafe fn new(array: &'a ArrayRef, data: &'a V::ArrayData) -> Self {
        Self { array, data }
    }

    /// Returns the underlying [`ArrayRef`].
    pub fn array_ref(&self) -> &'a ArrayRef {
        self.array
    }

    /// Returns a reference to the encoding-specific data.
    pub fn data(&self) -> &'a V::ArrayData {
        self.data
    }

    /// Returns the dtype of this array.
    pub fn dtype(&self) -> &DType {
        self.array.dtype()
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Returns whether this array is empty.
    pub fn is_empty(&self) -> bool {
        self.array.len() == 0
    }

    /// Returns the encoding ID of this array.
    pub fn encoding_id(&self) -> ArrayId {
        self.array.encoding_id()
    }

    /// Returns the statistics for this array.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.array.statistics()
    }
}

impl<V: VTable> Deref for ArrayView<'_, V> {
    type Target = V::ArrayData;

    fn deref(&self) -> &V::ArrayData {
        self.data
    }
}

impl<V: VTable> Debug for ArrayView<'_, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayView")
            .field("encoding", &self.array.encoding_id())
            .field("dtype", self.array.dtype())
            .field("len", &self.array.len())
            .finish()
    }
}
