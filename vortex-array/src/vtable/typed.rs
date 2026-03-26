// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed array wrapper [`Array<V>`] that pairs a [`VTable`] with common fields and inner data.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

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
    /// Create a new typed array without validating that the inner array's dtype/len match.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `V::dtype(&array) == &dtype` and `V::len(&array) == len`.
    pub unsafe fn new_unchecked(
        vtable: V,
        dtype: DType,
        len: usize,
        array: V::Array,
        stats: ArrayStats,
    ) -> Self {
        Self {
            vtable,
            dtype,
            len,
            array,
            stats,
        }
    }

    /// Returns a reference to the vtable.
    pub fn typed_vtable(&self) -> &V {
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

impl<V: VTable> Deref for Array<V> {
    type Target = V::Array;

    fn deref(&self) -> &V::Array {
        &self.array
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
