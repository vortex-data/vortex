// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::VTable;
use crate::dtype::DType;
use crate::stats::StatsSetRef;

/// A lightweight, `Copy`-able typed view into an [`ArrayRef`].
pub struct ArrayView<'a, V: VTable> {
    array: &'a ArrayRef,
    data: &'a V::ArrayData,
}

impl<V: VTable> Copy for ArrayView<'_, V> {}

impl<V: VTable> Clone for ArrayView<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: VTable> ArrayView<'a, V> {
    /// # Safety
    /// Caller must ensure `data` is the `V::ArrayData` stored inside `array`.
    pub(crate) unsafe fn new_unchecked(array: &'a ArrayRef, data: &'a V::ArrayData) -> Self {
        debug_assert!(array.is::<V>());
        Self { array, data }
    }

    pub fn array(&self) -> &'a ArrayRef {
        self.array
    }

    pub fn data(&self) -> &'a V::ArrayData {
        self.data
    }

    pub fn dtype(&self) -> &DType {
        self.array.dtype()
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.len() == 0
    }

    pub fn encoding_id(&self) -> ArrayId {
        self.array.encoding_id()
    }

    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.array.statistics()
    }

    pub fn into_owned(self) -> Array<V> {
        // SAFETY: we are ourselves type checked as 'V'
        unsafe { Array::<V>::from_array_ref_unchecked(self.array.clone()) }
    }
}

impl<'a, V: VTable> ArrayView<'a, V>
where
    V::ArrayData: crate::vtable::ValidityHelper,
{
    /// Returns a reference to the validity.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> &'a crate::validity::Validity {
        crate::vtable::ValidityHelper::validity(self.data)
    }
}

impl<V: VTable> AsRef<ArrayRef> for ArrayView<'_, V> {
    fn as_ref(&self) -> &ArrayRef {
        self.array
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
