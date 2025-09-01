// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use vortex_buffer::BufferMut;
use vortex_dtype::{DType, NativePType, Nullability};
use vortex_error::{VortexResult, vortex_bail};
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

    /// Append a [`Mask`] to this builder's null buffer.
    pub fn append_mask(&mut self, mask: Mask) {
        self.nulls.append_validity_mask(mask);
    }

    /// Appends a primitive `value` to the builder.
    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.nulls.append_non_null();
    }

    /// Appends an optional primitive (representing a nullable primitive) to the builder.
    ///
    /// # Panics
    ///
    /// This method will panic if the input is `None` and the builder is non-nullable.
    pub fn append_option(&mut self, value: Option<T>) {
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

    fn append_nulls(&mut self, n: usize) {
        self.values.push_n(T::default(), n);
        self.nulls.append_n_nulls(n);
    }

    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()> {
        if !self.dtype.eq_with_nullability_superset(array.dtype()) {
            vortex_bail!(
                "tried to extend a builder with `DType` {} with an array with `DType {}",
                self.dtype,
                array.dtype()
            );
        }

        let array = array.to_primitive()?;
        if array.ptype() != T::PTYPE {
            vortex_bail!("Cannot extend from array with different ptype");
        }

        self.values.extend_from_slice(array.as_slice::<T>());
        self.nulls.append_validity_mask(array.validity_mask());

        Ok(())
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

impl<T> Deref for UninitRange<'_, T> {
    type Target = [MaybeUninit<T>];

    fn deref(&self) -> &[MaybeUninit<T>] {
        let start = self.builder.values.as_ptr();
        unsafe {
            // SAFETY: start + len is checked on construction to be in range.
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

impl<T> UninitRange<'_, T> {
    /// Set a validity bit at the given index. The index is relative to the start of this range
    /// of the builder.
    pub fn set_bit(&mut self, index: usize, v: bool) {
        self.builder.nulls.set_bit(self.offset + index, v);
    }

    /// Set values from an initialized range.
    pub fn copy_from_init(&mut self, offset: usize, len: usize, src: &[T])
    where
        T: Copy,
    {
        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
        let uninit_src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(src) };

        let dst = &mut self[offset..][..len];
        dst.copy_from_slice(uninit_src);
    }

    /// Finish building this range, marking it as initialized and advancing the length of the
    /// underlying values buffer.
    pub fn finish(self) {
        // SAFETY: constructor enforces that offset + len does not exceed the capacity of the array.
        unsafe { self.builder.values.set_len(self.offset + self.len) };
    }
}
