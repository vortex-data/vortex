// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_mask::Mask;

use crate::arrays::PrimitiveArray;
use crate::builders::{ArrayBuilder, DEFAULT_BUILDER_CAPACITY, LazyNullBufferBuilder};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// The builder for building a [`PrimitiveArray`], parametrized by the `PType`.
pub struct PrimitiveBuilder<T> {
    dtype: DType,
    values: BufferMut<T>,
    nulls: LazyNullBufferBuilder,
}

impl<T: NativePType> PrimitiveBuilder<T> {
    /// Creates a new `PrimitiveBuilder` with a capacity of [`DEFAULT_BUILDER_CAPACITY`].
    pub fn new(nullability: Nullability) -> Self {
        Self::with_capacity(nullability, DEFAULT_BUILDER_CAPACITY)
    }

    /// Creates a new `PrimitiveBuilder` with the given `capacity`.
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            values: BufferMut::with_capacity(capacity),
            nulls: LazyNullBufferBuilder::new(capacity),
            dtype: DType::Primitive(T::PTYPE, nullability),
        }
    }

    /// Appends a primitive `value` to the builder.
    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.nulls.append_non_null();
    }

    /// Appends an optional primitive value to the builder.
    ///
    /// If the value is `Some`, it appends the primitive value. If the value is `None`, it appends a
    /// null.
    ///
    /// # Panics
    ///
    /// This method will panic if the input is `None` and the builder is non-nullable.
    pub(crate) fn append_option(&mut self, value: Option<T>) {
        match value {
            Some(value) => self.append_value(value),
            None => self.append_null(),
        }
    }

    /// Returns the raw primitive values in this builder as a slice.
    pub fn values(&self) -> &[T] {
        self.values.as_ref()
    }

    /// Create a new handle to the next `len` uninitialized values in the builder.
    ///
    /// All reads/writes through the handle to the values buffer or the validity buffer will operate
    /// on indices relative to the start of the range.
    ///
    /// ## Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
    /// use vortex_dtype::Nullability;
    ///
    /// // Create a new builder.
    /// let mut builder: PrimitiveBuilder<i32> =
    ///     PrimitiveBuilder::with_capacity(Nullability::NonNullable, 5);
    ///
    /// // Populate the values in reverse order.
    /// let mut range = builder.uninit_range(5);
    /// for i in [4, 3, 2, 1, 0] {
    ///     range[i] = MaybeUninit::new(i as i32);
    /// }
    /// range.finish();
    ///
    /// let built = builder.finish_into_primitive();
    ///
    /// assert_eq!(built.as_slice::<i32>(), &[0i32, 1, 2, 3, 4]);
    /// ```
    pub fn uninit_range(&mut self, len: usize) -> UninitRange<'_, T> {
        assert_ne!(0, len, "cannot create an uninit range of length 0");

        let offset = self.values.len();
        assert!(
            offset + len <= self.values.capacity(),
            "uninit_range of len {len} exceeds builder capacity {}",
            self.values.capacity()
        );

        UninitRange {
            offset,
            len,
            builder: self,
        }
    }

    /// Finishes the builder directly into a [`PrimitiveArray`].
    pub fn finish_into_primitive(&mut self) -> PrimitiveArray {
        let validity = self
            .nulls
            .finish_with_nullability(self.dtype().nullability());

        PrimitiveArray::new(std::mem::take(&mut self.values).freeze(), validity)
    }

    /// Extends the primitive array with an iterator.
    pub fn extend_with_iterator(&mut self, iter: impl IntoIterator<Item = T>, mask: Mask) {
        self.values.extend(iter);
        self.nulls.append_validity_mask(mask);
    }
}

impl<T: NativePType> ArrayBuilder for PrimitiveBuilder<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.values.push_n(T::default(), n);
        self.nulls.append_n_non_nulls(n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.values.push_n(T::default(), n);
        self.nulls.append_n_nulls(n);
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array) {
        let array = array.to_primitive();

        // This should be checked in `extend_from_array` but we can check it again.
        debug_assert_eq!(
            array.ptype(),
            T::PTYPE,
            "Cannot extend from array with different ptype"
        );

        self.values.extend_from_slice(array.as_slice::<T>());
        self.nulls.append_validity_mask(array.validity_mask());
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if capacity > self.values.capacity() {
            self.values.reserve(capacity - self.values.len());
            self.nulls.ensure_capacity(capacity);
        }
    }

    fn set_validity(&mut self, validity: Mask) {
        self.nulls = LazyNullBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_primitive().into_array()
    }
}

/// A range of uninitialized values in the primitive builder that can be filled.
pub struct UninitRange<'a, T> {
    offset: usize,
    len: usize,
    builder: &'a mut PrimitiveBuilder<T>,
}

impl<T> UninitRange<'_, T> {
    /// Append a [`Mask`] to this builder's null buffer.
    ///
    /// # Panics
    ///
    /// Panics if the mask length is not equal to the the length of the current `UninitRange`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that they safely initialize `mask.len()` primitive values via
    ///   [`UninitRange::copy_from_slice`].
    /// - The caller must also ensure that they only call this method once.
    pub unsafe fn append_mask(&mut self, mask: Mask) {
        assert_eq!(
            mask.len(),
            self.len,
            "Tried to append a mask to an `UninitRange` that was beyond the allowed range"
        );

        // TODO(connor): Ideally, we would call this function `set_mask` and directly set all of the
        // bits (so that we can call this multiple times), but the underlying `BooleanBuffer` does
        // not have an easy way to do this correctly.

        self.builder.nulls.append_validity_mask(mask);
    }

    /// Set a validity bit at the given index. The index is relative to the start of this range
    /// of the builder.
    pub fn set_bit(&mut self, index: usize, v: bool) {
        // Note that this won't panic because we can only create an `UninitRange` within the
        // capacity of the builder (it will not automatically resize).
        self.builder.nulls.set_bit(self.offset + index, v);
    }

    /// Set values from an initialized range.
    ///
    /// Note that the input `offset` should be an offset relative to the local `UninitRange`, not
    /// the entire `PrimitiveBuilder`.
    pub fn copy_from_slice(&mut self, offset: usize, src: &[T])
    where
        T: Copy,
    {
        debug_assert!(
            offset + src.len() <= self.len,
            "tried to copy a slice into a `UninitRange` past its boundary"
        );

        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
        let uninit_src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(src) };

        let dst = &mut self[offset..][..src.len()];
        dst.copy_from_slice(uninit_src);
    }

    /// Finish building this range, marking it as initialized and advancing the length of the
    /// underlying values buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that they have safely initialized all `len` values via
    /// [`UninitRange::copy_from_slice`] as well as correctly set all of the null bits via
    /// [`set_bit`] or [`append_mask`] if the builder is nullable.
    ///
    /// [`set_bit`]: UninitRange::set_bit
    /// [`append_mask`]: UninitRange::append_mask
    pub unsafe fn finish(self) {
        // SAFETY: constructor enforces that offset + len does not exceed the capacity of the array.
        unsafe { self.builder.values.set_len(self.offset + self.len) };
    }
}

/// Note that we only implement `Deref` for parity with `DerefMut`.
impl<T> Deref for UninitRange<'_, T> {
    type Target = [MaybeUninit<T>];

    /// Returns a [`MaybeUninit`] slice of the [`PrimitiveBuilder`]'s uninitialized memory capacity
    /// (which is simply the length).
    fn deref(&self) -> &[MaybeUninit<T>] {
        // We implement this manually since there is no `spare_capacity()` method on the internal
        // `BytesMut` type.
        let base = self.builder.values.as_ptr();

        // SAFETY: `offset` is derived from the existing buffer in memory, so this won't wrap.
        let start = unsafe { base.add(self.offset) };

        unsafe {
            // SAFETY: `start + len` is checked on construction of `UninitRange` to be within the
            // total capacity of the builder.
            let dst = std::slice::from_raw_parts(start, self.len);

            // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
            let dst: &[MaybeUninit<T>] = std::mem::transmute(dst);

            dst
        }
    }
}

impl<T> DerefMut for UninitRange<'_, T> {
    fn deref_mut(&mut self) -> &mut [MaybeUninit<T>] {
        &mut self.builder.values.spare_capacity_mut()[..self.len]
    }
}
