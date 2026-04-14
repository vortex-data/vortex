// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::mem::MaybeUninit;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::builders::ArrayBuilder;
use crate::builders::DEFAULT_BUILDER_CAPACITY;
use crate::builders::LazyBitBufferBuilder;
use crate::canonical::Canonical;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// The builder for building a [`PrimitiveArray`], parametrized by the `PType`.
pub struct PrimitiveBuilder<T> {
    dtype: DType,
    values: BufferMut<T>,
    nulls: LazyBitBufferBuilder,
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
            nulls: LazyBitBufferBuilder::new(capacity),
            dtype: DType::Primitive(T::PTYPE, nullability),
        }
    }

    /// Appends a primitive `value` to the builder.
    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.nulls.append_non_null();
    }

    /// Appends `n` copies of `value` as non-null entries, directly writing into the buffer.
    pub fn append_n_values(&mut self, value: T, n: usize) {
        self.values.push_n(value, n);
        self.nulls.append_n_non_nulls(n);
    }

    /// Returns the raw primitive values in this builder as a slice.
    pub fn values(&self) -> &[T] {
        self.values.as_ref()
    }

    /// Returns the raw primitive values in this builder as a mutable slice.
    pub fn values_mut(&mut self) -> &mut [T] {
        self.values.as_mut()
    }

    /// Create a new handle to the next `len` uninitialized values in the builder.
    ///
    /// All reads/writes through the handle to the values buffer or the validity buffer will operate
    /// on indices relative to the start of the range.
    ///
    /// # Panics
    ///
    /// Panics if `len` is 0 or if the current length of the builder plus `len` would exceed the
    /// capacity of the builder's memory.
    ///
    /// ## Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use vortex_array::builders::{ArrayBuilder, PrimitiveBuilder};
    /// use vortex_array::dtype::Nullability;
    ///
    /// // Create a new builder.
    /// let mut builder: PrimitiveBuilder<i32> =
    ///     PrimitiveBuilder::with_capacity(Nullability::NonNullable, 5);
    ///
    /// // Populate the values.
    /// let mut uninit_range = builder.uninit_range(5);
    /// uninit_range.copy_from_slice(0, &[0, 1, 2, 3, 4]);
    ///
    /// // SAFETY: We have initialized all 5 values in the range, and since the array builder is
    /// // non-nullable, we don't need to set any null bits.
    /// unsafe { uninit_range.finish(); }
    ///
    /// let built = builder.finish_into_primitive();
    ///
    /// assert_eq!(built.as_slice::<i32>(), &[0i32, 1, 2, 3, 4]);
    /// ```
    pub fn uninit_range(&mut self, len: usize) -> UninitRange<'_, T> {
        assert_ne!(0, len, "cannot create an uninit range of length 0");

        let current_len = self.values.len();
        assert!(
            current_len + len <= self.values.capacity(),
            "uninit_range of len {len} exceeds builder with length {} and capacity {}",
            current_len,
            self.values.capacity()
        );

        UninitRange { len, builder: self }
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

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "PrimitiveBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );

        if let Some(pv) = scalar.as_primitive().pvalue() {
            self.append_value(pv.cast::<T>()?)
        } else {
            self.append_null()
        }

        Ok(())
    }

    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef) {
        let array = array.to_primitive();

        // This should be checked in `extend_from_array` but we can check it again.
        debug_assert_eq!(
            array.ptype(),
            T::PTYPE,
            "Cannot extend from array with different ptype"
        );

        self.values.extend_from_slice(array.as_slice::<T>());
        self.nulls.append_validity_mask(
            array
                .as_ref()
                .validity()
                .vortex_expect("validity_mask")
                .to_mask(
                    array.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .vortex_expect("Failed to compute validity mask"),
        );
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.nulls.reserve_exact(additional);
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        self.nulls = LazyBitBufferBuilder::new(validity.len());
        self.nulls.append_validity_mask(validity);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_primitive().into_array()
    }

    fn finish_into_canonical(&mut self) -> Canonical {
        Canonical::Primitive(self.finish_into_primitive())
    }
}

/// A range of uninitialized values in the primitive builder that can be filled.
pub struct UninitRange<'a, T> {
    /// The length of the uninitialized range.
    ///
    /// This is guaranteed to be within the memory capacity of the builder.
    len: usize,

    /// A mutable reference to the builder.
    ///
    /// Since this is a mutable reference, we can guarantee that nothing else can modify the builder
    /// while this `UninitRange` exists.
    builder: &'a mut PrimitiveBuilder<T>,
}

impl<T> UninitRange<'_, T> {
    /// Returns the length of this uninitialized range.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this range has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set a value at the given index within this range.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn set_value(&mut self, index: usize, value: T) {
        assert!(index < self.len, "index out of bounds");
        let spare = self.builder.values.spare_capacity_mut();
        spare[index] = MaybeUninit::new(value);
    }

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

    /// Set a validity bit at the given index.
    ///
    /// The index is relative to the start of this range (not relative to the values already in the
    /// builder).
    ///
    /// Note that this will have no effect if the builder is non-nullable.
    pub fn set_validity_bit(&mut self, index: usize, v: bool) {
        assert!(index < self.len, "set_bit index out of bounds");
        // Note that this won't panic because we can only create an `UninitRange` within the
        // capacity of the builder (it will not automatically resize).
        let absolute_index = self.builder.values.len() + index;
        self.builder.nulls.set_bit(absolute_index, v);
    }

    /// Set values from an initialized range.
    ///
    /// Note that the input `offset` should be an offset relative to the local `UninitRange`, not
    /// the entire `PrimitiveBuilder`.
    pub fn copy_from_slice(&mut self, local_offset: usize, src: &[T])
    where
        T: Copy,
    {
        debug_assert!(
            local_offset + src.len() <= self.len,
            "tried to copy a slice into a `UninitRange` past its boundary"
        );

        // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout.
        let uninit_src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(src) };

        // Note: spare_capacity_mut() returns the spare capacity starting from the current length,
        // so we just use local_offset directly.
        let dst =
            &mut self.builder.values.spare_capacity_mut()[local_offset..local_offset + src.len()];
        dst.copy_from_slice(uninit_src);
    }

    /// Get a mutable slice of uninitialized memory at the specified offset within this range.
    ///
    /// Note that the offsets are relative to this local range, not to the values already in the
    /// builder.
    ///
    /// # Safety
    ///
    /// The caller must ensure that they properly initialize the returned memory before calling
    /// `finish()` on this range.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` exceeds the range bounds.
    pub unsafe fn slice_uninit_mut(&mut self, offset: usize, len: usize) -> &mut [MaybeUninit<T>] {
        assert!(
            offset + len <= self.len,
            "slice_uninit_mut: offset {} + len {} exceeds range length {}",
            offset,
            len,
            self.len
        );
        &mut self.builder.values.spare_capacity_mut()[offset..offset + len]
    }

    /// Finish building this range, marking it as initialized and advancing the length of the
    /// underlying values buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that they have safely initialized all `len` values via
    /// [`copy_from_slice()`] or [`set_value()`], as well as correctly set all of the null bits via
    /// [`set_validity_bit()`] or [`append_mask()`] if the builder is nullable.
    ///
    /// [`copy_from_slice()`]: UninitRange::copy_from_slice
    /// [`set_value()`]: UninitRange::set_value
    /// [`set_validity_bit()`]: UninitRange::set_validity_bit
    /// [`append_mask()`]: UninitRange::append_mask
    pub unsafe fn finish(self) {
        // SAFETY: constructor enforces that current length + len does not exceed the capacity of the array.
        let new_len = self.builder.values.len() + self.len;
        unsafe { self.builder.values.set_len(new_len) };
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use super::*;
    use crate::assert_arrays_eq;

    /// REGRESSION TEST: This test verifies that multiple sequential ranges have correct offsets.
    ///
    /// This would have caught the `Deref` bug where it always returned from the start of the
    /// buffer.
    #[test]
    fn test_multiple_uninit_ranges_correct_offsets() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::NonNullable, 10);

        // First range.
        let mut range1 = builder.uninit_range(3);
        range1.copy_from_slice(0, &[1, 2, 3]);

        // SAFETY: We initialized all 3 values.
        unsafe {
            range1.finish();
        }

        // Verify the builder now has these values.
        assert_eq!(builder.values(), &[1, 2, 3]);

        // Second range - this would fail with the old Deref implementation.
        let mut range2 = builder.uninit_range(2);

        // Set values using copy_from_slice.
        range2.copy_from_slice(0, &[4, 5]);

        // SAFETY: We initialized both values.
        unsafe {
            range2.finish();
        }

        // Verify the builder now has all 5 values.
        assert_eq!(builder.values(), &[1, 2, 3, 4, 5]);

        let array = builder.finish_into_primitive();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]));
    }

    /// REGRESSION TEST: This test verifies that `append_mask` was correctly moved from
    /// `PrimitiveBuilder` to `UninitRange`.
    ///
    /// The old API had `append_mask` on the builder, which was confusing when used with ranges.
    /// This test ensures the new API works correctly.
    #[test]
    fn test_append_mask_on_uninit_range() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::Nullable, 5);
        let mut range = builder.uninit_range(3);

        // Create a mask for 3 values.
        let mask = Mask::from_iter([true, false, true]);

        // SAFETY: We're about to initialize the values.
        unsafe {
            range.append_mask(mask);
        }

        // Initialize the values.
        range.copy_from_slice(0, &[10, 20, 30]);

        // SAFETY: We've initialized all values and set the mask.
        unsafe {
            range.finish();
        }

        let array = builder.finish_into_primitive();
        assert_eq!(array.len(), 3);
        // Check validity using scalar_at - nulls will return is_null() = true.
        assert!(!array.scalar_at(0).unwrap().is_null());
        assert!(array.scalar_at(1).unwrap().is_null());
        assert!(!array.scalar_at(2).unwrap().is_null());
    }

    /// REGRESSION TEST: This test verifies that `append_mask` validates the mask length.
    ///
    /// This ensures that masks can only be appended if they match the range length.
    #[test]
    #[should_panic(
        expected = "Tried to append a mask to an `UninitRange` that was beyond the allowed range"
    )]
    fn test_append_mask_wrong_length_panics() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::Nullable, 10);
        let mut range = builder.uninit_range(5);

        // Try to append a mask with wrong length (3 instead of 5).
        let wrong_mask = Mask::from_iter([true, false, true]);

        // SAFETY: This is expected to panic due to length mismatch.
        unsafe {
            range.append_mask(wrong_mask);
        }
    }

    /// Test that `copy_from_slice` works correctly with different offsets.
    ///
    /// This verifies the new simplified API without the redundant `len` parameter.
    #[test]
    fn test_copy_from_slice_with_offsets() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::NonNullable, 10);
        let mut range = builder.uninit_range(6);

        // Copy to different offsets.
        range.copy_from_slice(0, &[1, 2]);
        range.copy_from_slice(2, &[3, 4]);
        range.copy_from_slice(4, &[5, 6]);

        // SAFETY: We've initialized all 6 values.
        unsafe {
            range.finish();
        }

        let array = builder.finish_into_primitive();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]));
    }

    /// Test that `set_bit` uses relative indexing within the range.
    ///
    /// Note: `set_bit` requires the null buffer to already be initialized, so we first
    /// use `append_mask` to set up the buffer, then demonstrate that `set_bit` can
    /// modify individual bits with relative indexing.
    #[test]
    fn test_set_bit_relative_indexing() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::Nullable, 10);

        // First add some values to the builder.
        builder.append_value(100);
        builder.append_value(200);

        // Create a range for new values.
        let mut range = builder.uninit_range(3);

        // Use append_mask to initialize the validity buffer for this range.
        let initial_mask = Mask::from_iter([false, false, false]);
        // SAFETY: We're about to initialize the values.
        unsafe {
            range.append_mask(initial_mask);
        }

        // Now we can use set_bit to modify individual bits with relative indexing.
        range.set_validity_bit(0, true); // Change first bit to valid
        range.set_validity_bit(2, true); // Change third bit to valid
        // Leave middle bit as false (null)

        // Initialize the values.
        range.copy_from_slice(0, &[10, 20, 30]);

        // SAFETY: We've initialized all 3 values and set their validity.
        unsafe {
            range.finish();
        }

        let array = builder.finish_into_primitive();

        // Verify the total length and values.
        assert_eq!(array.len(), 5);
        assert_eq!(array.as_slice::<i32>(), &[100, 200, 10, 20, 30]);

        // Check validity - the first two should be valid (from append_value).
        assert!(!array.scalar_at(0).unwrap().is_null()); // initial value 100
        assert!(!array.scalar_at(1).unwrap().is_null()); // initial value 200

        // Check the range items with modified validity.
        assert!(!array.scalar_at(2).unwrap().is_null()); // range index 0 - set to valid
        assert!(array.scalar_at(3).unwrap().is_null()); // range index 1 - left as null
        assert!(!array.scalar_at(4).unwrap().is_null()); // range index 2 - set to valid
    }

    /// Test that creating a zero-length uninit range panics.
    #[test]
    #[should_panic(expected = "cannot create an uninit range of length 0")]
    fn test_zero_length_uninit_range_panics() {
        let mut builder = PrimitiveBuilder::<i32>::new(Nullability::NonNullable);
        let _range = builder.uninit_range(0);
    }

    /// Test that creating an uninit range exceeding capacity panics.
    #[test]
    #[should_panic(
        expected = "uninit_range of len 10 exceeds builder with length 0 and capacity 6"
    )]
    fn test_uninit_range_exceeds_capacity_panics() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::NonNullable, 5);
        let _range = builder.uninit_range(10);
    }

    /// Test that `copy_from_slice` debug asserts on out-of-bounds access.
    ///
    /// Note: This only panics in debug mode due to `debug_assert!`.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "tried to copy a slice into a `UninitRange` past its boundary")]
    fn test_copy_from_slice_out_of_bounds() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::NonNullable, 10);
        let mut range = builder.uninit_range(3);

        // Try to copy 3 elements starting at offset 1 (would need 4 slots total).
        range.copy_from_slice(1, &[1, 2, 3]);
    }

    /// Test that the unsafe contract of `finish` is documented and works correctly.
    ///
    /// This test demonstrates proper usage of the unsafe `finish` method.
    #[test]
    fn test_finish_unsafe_contract() {
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::Nullable, 5);
        let mut range = builder.uninit_range(3);

        // Set validity mask.
        let mask = Mask::from_iter([true, true, false]);
        // SAFETY: We're about to initialize the matching number of values.
        unsafe {
            range.append_mask(mask);
        }

        // Initialize all values.
        range.copy_from_slice(0, &[10, 20, 30]);

        // SAFETY: We have initialized all 3 values and set their validity.
        unsafe {
            range.finish();
        }

        let array = builder.finish_into_primitive();
        assert_eq!(array.len(), 3);
        assert_eq!(array.as_slice::<i32>(), &[10, 20, 30]);
    }

    #[test]
    fn test_append_scalar() {
        use crate::dtype::DType;
        use crate::scalar::Scalar;

        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::Nullable, 10);

        // Test appending a valid primitive value.
        let scalar1 = Scalar::primitive(42i32, Nullability::Nullable);
        builder.append_scalar(&scalar1).unwrap();

        // Test appending another value.
        let scalar2 = Scalar::primitive(84i32, Nullability::Nullable);
        builder.append_scalar(&scalar2).unwrap();

        // Test appending null value.
        let null_scalar = Scalar::null(DType::Primitive(
            crate::dtype::PType::I32,
            Nullability::Nullable,
        ));
        builder.append_scalar(&null_scalar).unwrap();

        let array = builder.finish_into_primitive();
        assert_eq!(array.len(), 3);

        // Check actual values.
        let values = array.as_slice::<i32>();
        assert_eq!(values[0], 42);
        assert_eq!(values[1], 84);
        // values[2] might be any value since it's null.

        // Check validity - first two should be valid, third should be null.
        assert!(
            array
                .validity()
                .vortex_expect("primitive validity should be derivable")
                .is_valid(0)
                .unwrap()
        );
        assert!(
            array
                .validity()
                .vortex_expect("primitive validity should be derivable")
                .is_valid(1)
                .unwrap()
        );
        assert!(
            !array
                .validity()
                .vortex_expect("primitive validity should be derivable")
                .is_valid(2)
                .unwrap()
        );

        // Test wrong dtype error.
        let mut builder = PrimitiveBuilder::<i32>::with_capacity(Nullability::NonNullable, 10);
        let wrong_scalar = Scalar::from(true);
        assert!(builder.append_scalar(&wrong_scalar).is_err());
    }
}
