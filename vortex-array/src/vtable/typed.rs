// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed array wrappers: [`ArrayInner<V>`] (heap-allocated), [`Array<V>`] (typed handle),
//! and [`ArrayView<V>`] (lightweight borrow).

use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::DynArray;
use crate::IntoArray;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

// =============================================================================
// ArrayInner<V> — the concrete type stored inside Arc<dyn DynArray>
// =============================================================================

/// The concrete array type that lives inside an `Arc` behind [`ArrayRef`].
///
/// Prefer using [`Array<V>`] (owned typed handle) for constructing arrays
/// and converting between typed and untyped representations.
/// This type is returned by reference from [`Matcher`] downcasts.
#[doc(hidden)]
pub struct ArrayInner<V: VTable> {
    pub(crate) vtable: V,
    pub(crate) dtype: DType,
    pub(crate) len: usize,
    pub(crate) data: V::ArrayData,
    pub(crate) stats: ArrayStats,
}

impl<V: VTable> ArrayInner<V> {
    /// Create a new inner array from encoding-specific data.
    #[doc(hidden)]
    pub fn try_from_data(data: V::ArrayData) -> VortexResult<Self> {
        let vtable = V::vtable(&data).clone();
        let dtype = V::dtype(&data).clone();
        let len = V::len(&data);
        let stats = V::stats(&data).clone();
        Ok(unsafe { Self::from_data_unchecked(vtable, dtype, len, data, stats) })
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
}

impl<V: VTable> ArrayInner<V> {
    /// Calls `f` with an [`ArrayView`] backed by a temporary [`ArrayRef`].
    ///
    /// This creates a clone of `self` wrapped in an `ArrayRef` so that the `ArrayView`
    /// has a valid `&ArrayRef` to reference.
    #[doc(hidden)]
    pub fn with_view<R>(&self, f: impl FnOnce(ArrayView<'_, V>) -> R) -> R {
        let array_ref = self.to_array_ref();
        // SAFETY: `self.data` is equivalent to the data inside `array_ref` (it's a clone).
        let view = unsafe { ArrayView::new(&array_ref, &self.data) };
        f(view)
    }

    /// Creates an [`ArrayRef`] by cloning self into an Arc.
    #[doc(hidden)]
    pub fn to_array_ref(&self) -> ArrayRef {
        ArrayRef::from_inner(Arc::new(self.clone()))
    }

    /// Returns a reference to the encoding-specific data.
    pub fn data(&self) -> &V::ArrayData {
        &self.data
    }

    /// Consumes this array and returns the encoding-specific data.
    pub fn into_data(self) -> V::ArrayData {
        self.data
    }

    /// Returns the dtype.
    #[allow(clippy::same_name_method)]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the length.
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array is empty.
    #[allow(clippy::same_name_method)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the encoding ID.
    #[allow(clippy::same_name_method)]
    pub fn encoding_id(&self) -> ArrayId {
        self.vtable.id()
    }

    /// Returns the statistics.
    #[allow(clippy::same_name_method)]
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.stats.to_ref(self)
    }

    /// Returns the canonical validity mask for the array.
    #[allow(clippy::same_name_method)]
    pub fn validity_mask(&self) -> VortexResult<vortex_mask::Mask> {
        DynArray::validity_mask(self)
    }

    /// Fetch the scalar at the given index.
    #[allow(clippy::same_name_method)]
    pub fn scalar_at(&self, index: usize) -> VortexResult<crate::scalar::Scalar> {
        DynArray::scalar_at(self, index)
    }

    /// Returns the constant scalar if this is a constant array.
    pub fn as_constant(&self) -> Option<crate::scalar::Scalar> {
        self.to_array_ref().as_constant()
    }

    /// Performs a constant-time slice of the array.
    #[allow(clippy::same_name_method)]
    pub fn slice(&self, range: std::ops::Range<usize>) -> VortexResult<ArrayRef> {
        DynArray::slice(self, range)
    }

    /// Returns the canonical representation of the array.
    #[allow(clippy::same_name_method)]
    pub fn to_canonical(&self) -> VortexResult<crate::Canonical> {
        DynArray::to_canonical(self)
    }

    /// Wraps the array in a filter such that it is logically filtered by the given mask.
    #[allow(clippy::same_name_method)]
    pub fn filter(&self, mask: vortex_mask::Mask) -> VortexResult<ArrayRef> {
        DynArray::filter(self, mask)
    }

    /// Wraps the array in a dict such that it is logically taken by the given indices.
    #[allow(clippy::same_name_method)]
    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        DynArray::take(self, indices)
    }

    /// Returns whether the item at `index` is valid.
    #[allow(clippy::same_name_method)]
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        DynArray::is_valid(self, index)
    }

    /// Returns whether the item at `index` is invalid.
    #[allow(clippy::same_name_method)]
    pub fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        DynArray::is_invalid(self, index)
    }

    /// Returns whether all items in the array are valid.
    #[allow(clippy::same_name_method)]
    pub fn all_valid(&self) -> VortexResult<bool> {
        DynArray::all_valid(self)
    }

    /// Returns whether the array is all invalid.
    #[allow(clippy::same_name_method)]
    pub fn all_invalid(&self) -> VortexResult<bool> {
        DynArray::all_invalid(self)
    }

    /// Returns the number of valid elements in the array.
    #[allow(clippy::same_name_method)]
    pub fn valid_count(&self) -> VortexResult<usize> {
        DynArray::valid_count(self)
    }

    /// Returns the number of invalid elements in the array.
    #[allow(clippy::same_name_method)]
    pub fn invalid_count(&self) -> VortexResult<usize> {
        DynArray::invalid_count(self)
    }

    /// Writes the array into the canonical builder.
    #[allow(clippy::same_name_method)]
    pub fn append_to_builder(
        &self,
        builder: &mut dyn crate::builders::ArrayBuilder,
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<()> {
        DynArray::append_to_builder(self, builder, ctx)
    }

    /// Total size of the array in bytes.
    pub fn nbytes(&self) -> u64 {
        self.to_array_ref().nbytes()
    }

    /// Returns the number of buffers in the array.
    #[allow(clippy::same_name_method)]
    pub fn nbuffers(&self) -> usize {
        self.with_view(V::nbuffers)
    }

    /// Returns a cloned [`ArrayRef`].
    #[allow(clippy::same_name_method)]
    pub fn to_array(&self) -> ArrayRef {
        self.to_array_ref()
    }
}

impl<V: VTable> ArrayInner<V>
where
    V::ArrayData: crate::vtable::ValidityHelper,
{
    /// Returns a reference to the validity.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> &crate::validity::Validity {
        crate::vtable::ValidityHelper::validity(&self.data)
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

impl<V: VTable> IntoArray for Arc<ArrayInner<V>> {
    fn into_array(self) -> ArrayRef {
        ArrayRef::from_inner(self)
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

#[allow(clippy::same_name_method)]
impl<V: VTable> Array<V> {
    /// Create a typed array from encoding-specific data.
    pub fn try_from_data(data: V::ArrayData) -> VortexResult<Self> {
        let inner = ArrayInner::<V>::try_from_data(data)?;
        Ok(Self::from_inner(inner))
    }

    /// Create from an `ArrayInner<V>`, wrapping it in an `ArrayRef`.
    pub(crate) fn from_inner(inner: ArrayInner<V>) -> Self {
        Self {
            inner: ArrayRef::from_inner(Arc::new(inner)),
            _phantom: PhantomData,
        }
    }

    /// Create from an existing `ArrayRef`, trusting that it contains `ArrayInner<V>`.
    ///
    /// # Safety
    /// Caller must ensure the `ArrayRef` contains an `ArrayInner<V>`.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_array_ref_unchecked(array: ArrayRef) -> Self {
        Self {
            inner: array,
            _phantom: PhantomData,
        }
    }

    /// Try to create from an `ArrayRef`, returning `Err` if the type doesn't match.
    pub fn try_from_array_ref(array: ArrayRef) -> Result<Self, ArrayRef> {
        if array.as_any().is::<ArrayInner<V>>() {
            Ok(Self {
                inner: array,
                _phantom: PhantomData,
            })
        } else {
            Err(array)
        }
    }

    /// Returns a reference to the underlying [`ArrayRef`].
    pub fn array_ref(&self) -> &ArrayRef {
        &self.inner
    }

    /// Consumes this typed array and returns the underlying [`ArrayRef`].
    pub fn into_array_ref(self) -> ArrayRef {
        self.inner
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

    /// Returns a reference to the inner `ArrayInner<V>`.
    fn inner_ref(&self) -> &ArrayInner<V> {
        // SAFETY: We only construct Array<V> when the ArrayRef contains ArrayInner<V>.
        unsafe {
            self.inner
                .as_any()
                .downcast_ref::<ArrayInner<V>>()
                .unwrap_unchecked()
        }
    }

    /// Returns a reference to the encoding-specific data.
    pub fn data(&self) -> &V::ArrayData {
        &self.inner_ref().data
    }

    /// Returns a clone of the inner encoding-specific data.
    pub fn into_data(self) -> V::ArrayData {
        self.inner_ref().data.clone()
    }

    /// Returns a cloned [`ArrayRef`].
    pub fn to_array_ref(&self) -> ArrayRef {
        self.inner.clone()
    }

    /// Calls `f` with an [`ArrayView`] backed by this array's [`ArrayRef`].
    pub fn with_view<R>(&self, f: impl FnOnce(ArrayView<'_, V>) -> R) -> R {
        // SAFETY: `self.inner_ref().data` is the data inside `self.inner`.
        let view = unsafe { ArrayView::new(&self.inner, &self.inner_ref().data) };
        f(view)
    }
}

impl<V: VTable> Array<V>
where
    V::ArrayData: crate::vtable::ValidityHelper,
{
    /// Returns a reference to the validity.
    #[allow(clippy::same_name_method)]
    pub fn validity(&self) -> &crate::validity::Validity {
        crate::vtable::ValidityHelper::validity(&self.inner_ref().data)
    }
}

/// Public API methods that shadow `DynArray` / `ArrayRef` methods.
impl<V: VTable> Array<V> {
    #[allow(clippy::same_name_method)]
    pub fn slice(&self, range: std::ops::Range<usize>) -> VortexResult<ArrayRef> {
        self.inner.slice(range)
    }

    #[allow(clippy::same_name_method)]
    pub fn scalar_at(&self, index: usize) -> VortexResult<crate::scalar::Scalar> {
        self.inner.scalar_at(index)
    }

    #[allow(clippy::same_name_method)]
    pub fn filter(&self, mask: vortex_mask::Mask) -> VortexResult<ArrayRef> {
        self.inner.filter(mask)
    }

    #[allow(clippy::same_name_method)]
    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        self.inner.take(indices)
    }

    #[allow(clippy::same_name_method)]
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.inner.is_valid(index)
    }

    #[allow(clippy::same_name_method)]
    pub fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.inner.is_invalid(index)
    }

    #[allow(clippy::same_name_method)]
    pub fn all_valid(&self) -> VortexResult<bool> {
        self.inner.all_valid()
    }

    #[allow(clippy::same_name_method)]
    pub fn all_invalid(&self) -> VortexResult<bool> {
        self.inner.all_invalid()
    }

    #[allow(clippy::same_name_method)]
    pub fn to_canonical(&self) -> VortexResult<crate::Canonical> {
        self.inner.to_canonical()
    }

    pub fn nbytes(&self) -> u64 {
        self.inner.nbytes()
    }

    #[allow(clippy::same_name_method)]
    pub fn nbuffers(&self) -> usize {
        self.with_view(V::nbuffers)
    }

    pub fn as_constant(&self) -> Option<crate::scalar::Scalar> {
        self.inner.as_constant()
    }

    #[allow(clippy::same_name_method)]
    pub fn valid_count(&self) -> VortexResult<usize> {
        self.inner.valid_count()
    }

    #[allow(clippy::same_name_method)]
    pub fn invalid_count(&self) -> VortexResult<usize> {
        self.inner.invalid_count()
    }

    #[allow(clippy::same_name_method)]
    pub fn append_to_builder(
        &self,
        builder: &mut dyn crate::builders::ArrayBuilder,
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<()> {
        self.inner.append_to_builder(builder, ctx)
    }

    #[allow(clippy::same_name_method)]
    #[deprecated(note = "use `.to_array_ref()` or `.into_array()` instead")]
    pub fn to_array(&self) -> ArrayRef {
        self.to_array_ref()
    }

    #[allow(clippy::same_name_method)]
    pub fn validity_mask(&self) -> VortexResult<vortex_mask::Mask> {
        self.inner.validity_mask()
    }
}

impl<V: VTable> Deref for Array<V> {
    type Target = V::ArrayData;

    fn deref(&self) -> &V::ArrayData {
        &self.inner_ref().data
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

// =============================================================================
// ArrayView<V> — lightweight borrow
// =============================================================================

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
    pub(crate) unsafe fn new(array: &'a ArrayRef, data: &'a V::ArrayData) -> Self {
        Self { array, data }
    }

    pub fn array_ref(&self) -> &'a ArrayRef {
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
