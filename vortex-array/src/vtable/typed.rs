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
/// encoding-specific methods are callable directly on `&Array<V>`.
///
/// Construct via encoding-specific constructors and type-erase with
/// [`into_array()`](IntoArray::into_array).
pub struct Array<V: VTable> {
    vtable: V,
    pub(crate) dtype: DType,
    pub(crate) len: usize,
    pub(crate) array: V::Array,
    pub(crate) stats: ArrayStats,
}

#[allow(clippy::same_name_method)]
impl<V: VTable> Array<V> {
    /// Create a new typed array from encoding-specific data.
    ///
    /// Extracts dtype, len, vtable, and stats from the data via [`VTable`] methods.
    pub fn try_from_data(data: V::Array) -> VortexResult<Self> {
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
        data: V::Array,
        stats: ArrayStats,
    ) -> Self {
        Self {
            vtable,
            dtype,
            len,
            array: data,
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
    pub fn inner(&self) -> &V::Array {
        &self.array
    }

    /// Consumes this array and returns the inner encoding-specific data.
    pub fn into_inner(self) -> V::Array {
        self.array
    }

    /// Returns a cloned [`ArrayRef`] for this array.
    pub fn to_array_ref(&self) -> ArrayRef {
        Arc::new(self.clone())
    }
}

impl<V: VTable> Array<V>
where
    V::Array: crate::vtable::ValidityHelper,
{
    /// Returns a reference to the validity of this array.
    ///
    /// This inherent method shadows `DynArray::validity()` to provide direct access
    /// to the concrete validity without going through `VortexResult`.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> &crate::validity::Validity {
        crate::vtable::ValidityHelper::validity(&self.array)
    }

    /// Returns the validity mask for this array.
    #[allow(clippy::same_name_method)]
    pub fn validity_mask(&self) -> VortexResult<vortex_mask::Mask> {
        Ok(self.validity().to_mask(self.len))
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

    /// Returns the number of buffers in this array.
    #[allow(clippy::same_name_method)]
    pub fn nbuffers(&self) -> usize {
        V::nbuffers(self)
    }

    /// If this array is a constant, returns the scalar value.
    pub fn as_constant(&self) -> Option<crate::scalar::Scalar> {
        self.to_array_ref().as_constant()
    }
}

impl<V: VTable> Deref for Array<V> {
    type Target = V::Array;

    fn deref(&self) -> &V::Array {
        &self.array
    }
}

impl<V: VTable> DerefMut for Array<V> {
    fn deref_mut(&mut self) -> &mut V::Array {
        &mut self.array
    }
}

impl<V: VTable> Clone for Array<V> {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            dtype: self.dtype.clone(),
            len: self.len,
            array: self.array.clone(),
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
            .field("inner", &self.array)
            .finish()
    }
}

impl<V: VTable> IntoArray for Array<V> {
    fn into_array(self) -> ArrayRef {
        Arc::new(self)
    }
}

impl<V: VTable> IntoArray for Arc<Array<V>> {
    fn into_array(self) -> ArrayRef {
        self
    }
}

impl<V: VTable> From<Array<V>> for ArrayRef {
    fn from(value: Array<V>) -> ArrayRef {
        value.into_array()
    }
}
