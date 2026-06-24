// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::dtype::DType;
use crate::stats::StatsSetRef;
use crate::validity::Validity;

/// A lightweight, `Copy`-able typed view into an [`ArrayRef`].
///
/// Vtable methods receive `ArrayView<V>` so they can inspect common array metadata and
/// encoding-specific data without cloning or downcasting an [`ArrayRef`]. The view is only valid for
/// the lifetime of the borrowed array.
pub struct ArrayView<'a, V: VTable> {
    array: &'a ArrayRef,
    data: &'a V::TypedArrayData,
}

impl<V: VTable> Copy for ArrayView<'_, V> {}

impl<V: VTable> Clone for ArrayView<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: VTable> ArrayView<'a, V> {
    /// # Safety
    /// Caller must ensure `data` is the `V::TypedArrayData` stored inside `array`.
    pub(crate) unsafe fn new_unchecked(array: &'a ArrayRef, data: &'a V::TypedArrayData) -> Self {
        debug_assert!(array.is::<V>());
        Self { array, data }
    }

    /// Returns the erased array handle that owns this view.
    pub fn array(&self) -> &'a ArrayRef {
        self.array
    }

    /// Returns the encoding-specific data stored by this array.
    pub fn data(&self) -> &'a V::TypedArrayData {
        self.data
    }

    /// Returns this array's child slots.
    pub fn slots(&self) -> &'a [Option<ArrayRef>] {
        self.array.slots()
    }

    /// Returns the logical dtype.
    pub fn dtype(&self) -> &DType {
        self.array.dtype()
    }

    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Returns `true` when the array has no rows.
    pub fn is_empty(&self) -> bool {
        self.array.len() == 0
    }

    /// Returns the encoding ID.
    pub fn encoding_id(&self) -> ArrayId {
        self.array.encoding_id()
    }

    /// Returns the array's statistics set.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.array.statistics()
    }

    /// Returns the array's validity representation.
    pub fn validity(&self) -> VortexResult<Validity> {
        self.array.validity()
    }

    /// Clone the underlying [`ArrayRef`] and return it as an owned typed handle.
    pub fn into_owned(self) -> Array<V> {
        // SAFETY: we are ourselves type checked as 'V'
        unsafe { Array::<V>::from_array_ref_unchecked(self.array.clone()) }
    }
}

impl<V: VTable> AsRef<ArrayRef> for ArrayView<'_, V> {
    fn as_ref(&self) -> &ArrayRef {
        self.array
    }
}

impl<V: VTable> Deref for ArrayView<'_, V> {
    type Target = V::TypedArrayData;

    fn deref(&self) -> &V::TypedArrayData {
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
