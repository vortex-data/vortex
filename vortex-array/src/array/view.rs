// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ParentMaterializer;
use crate::array::VTable;
use crate::dtype::DType;
use crate::matcher::AsParent;
use crate::stats::StatsSetRef;
use crate::validity::Validity;

/// A lightweight, `Copy`-able typed view of an array.
///
/// `ArrayView` is heap-backed: it always borrows an existing [`ArrayRef`]. Parent
/// reduction over stack-borrowed construction parts uses [`ParentView`] instead, so
/// APIs like [`Self::array`] and [`AsRef<ArrayRef>`] never hide a stack materialization.
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
    /// Construct a heap-backed view.
    ///
    /// # Safety
    /// Caller must ensure `data` is the `V::TypedArrayData` stored inside `array`.
    pub(crate) unsafe fn new_unchecked(array: &'a ArrayRef, data: &'a V::TypedArrayData) -> Self {
        debug_assert!(array.is::<V>());
        Self { array, data }
    }

    /// Returns the underlying heap-allocated [`ArrayRef`].
    #[inline]
    pub fn array(&self) -> &'a ArrayRef {
        self.array
    }

    #[inline]
    pub fn data(&self) -> &'a V::TypedArrayData {
        self.data
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn slots(&self) -> &'a [Option<ArrayRef>] {
        self.array.slots()
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn dtype(&self) -> &'a DType {
        self.array.dtype()
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        self.array.len()
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    #[inline]
    pub fn encoding_id(&self) -> ArrayId {
        self.array.encoding_id()
    }

    /// Returns the array's statistics.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.array().statistics()
    }

    /// Returns the array's validity.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> VortexResult<Validity> {
        self.array().validity()
    }

    /// Returns an owned typed handle. Forces stack-backed views to materialize.
    pub fn into_owned(self) -> Array<V> {
        // SAFETY: we are ourselves type checked as 'V'
        unsafe { Array::<V>::from_array_ref_unchecked(self.array().clone()) }
    }
}

/// A typed view over a parent array during metadata-only reduction.
///
/// `ParentView` can borrow either a heap-backed parent or stack-allocated construction
/// parts via [`ParentRef`](crate::array::ParentRef). It intentionally does not implement [`AsRef<ArrayRef>`] and
/// does not expose an `array()` method: callers that truly need an [`ArrayRef`] must call
/// [`Self::materialize_array_ref`] so the allocation boundary is visible in code review.
pub struct ParentView<'a, V: VTable> {
    data: &'a V::TypedArrayData,
    dtype: &'a DType,
    len: usize,
    slots: &'a [Option<ArrayRef>],
    encoding_id: ArrayId,
    materializer: &'a dyn ParentMaterializer,
}

impl<V: VTable> Copy for ParentView<'_, V> {}

impl<V: VTable> Clone for ParentView<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: VTable> ParentView<'a, V> {
    /// Construct a parent view borrowing any [`AsParent`].
    ///
    /// # Safety
    /// Caller must ensure `parent.is_encoding::<V>()` and that `data` is the
    /// `V::TypedArrayData` borrowed inside `parent`.
    pub(crate) unsafe fn new_unchecked<P: AsParent + ParentMaterializer>(
        parent: &'a P,
        data: &'a V::TypedArrayData,
    ) -> Self {
        debug_assert!(parent.is_encoding::<V>());
        Self {
            data,
            dtype: parent.dtype(),
            len: parent.len(),
            slots: parent.slots(),
            encoding_id: parent.encoding_id(),
            materializer: parent,
        }
    }

    /// Explicitly materialize the parent as an [`ArrayRef`].
    ///
    /// For heap-backed parents this returns the existing array. For stack-backed parents this
    /// allocates an `ArrayRef` on first call and reuses the parent's cache after that.
    #[inline]
    pub fn materialize_array_ref(&self) -> &'a ArrayRef {
        self.materializer.materialize_array_ref()
    }

    /// Explicitly materialize the parent as a heap-backed [`ArrayView`].
    ///
    /// For heap-backed parents this is free; stack-backed parents allocate an
    /// `ArrayRef` on first call and reuse the parent's cache after that.
    pub fn materialize_view(&self) -> ArrayView<'a, V> {
        self.materialize_array_ref()
            .as_typed::<V>()
            .vortex_expect("materialized parent must keep its encoding")
    }

    #[inline]
    pub fn data(&self) -> &'a V::TypedArrayData {
        self.data
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn slots(&self) -> &'a [Option<ArrayRef>] {
        self.slots
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    #[allow(clippy::same_name_method)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn encoding_id(&self) -> ArrayId {
        self.encoding_id
    }

    /// Returns an owned typed handle, explicitly materializing stack-backed parents.
    pub fn into_owned(self) -> Array<V> {
        // SAFETY: we are ourselves type checked as 'V'
        unsafe { Array::<V>::from_array_ref_unchecked(self.materialize_array_ref().clone()) }
    }
}

impl<V: VTable> Deref for ParentView<'_, V> {
    type Target = V::TypedArrayData;

    fn deref(&self) -> &V::TypedArrayData {
        self.data
    }
}

impl<V: VTable> Debug for ParentView<'_, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParentView")
            .field("encoding", &self.encoding_id())
            .field("dtype", self.dtype())
            .field("len", &self.len())
            .finish()
    }
}

impl<V: VTable> AsRef<ArrayRef> for ArrayView<'_, V> {
    fn as_ref(&self) -> &ArrayRef {
        // For heap-backed views this returns the borrowed `ArrayRef` directly. For
        // stack-backed views, materialization runs once and the cached `ArrayRef`
        // lives as long as the parent.
        self.array()
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
            .field("encoding", &self.encoding_id())
            .field("dtype", self.dtype())
            .field("len", &self.len())
            .finish()
    }
}
