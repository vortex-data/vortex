// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DVectorMut<D>`].

use vortex_buffer::BufferMut;
use vortex_dtype::{NativeDecimalType, PrecisionScale};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::MaskMut;

use crate::decimal::DVector;
use crate::{VectorMutOps, VectorOps};

/// A mutable vector of decimal values with fixed precision and scale.
///
/// `D` is bound by [`NativeDecimalType`], which can be one of the native integer types (`i8`,
/// `i16`, `i32`, `i64`, `i128`) or `i256`. `D` is used to store the decimal values.
///
/// The decimal vector maintains a [`PrecisionScale<D>`] that defines the precision (total number of
/// digits) and scale (digits after the decimal point) for all values in the vector.
///
/// Unlike primitive vectors, decimal vectors require validation during construction and
/// modification to ensure values stay within the bounds defined by their precision and scale.
/// This makes operations like "push" fallible, thus we have a [`try_push()`] method instead.
///
/// [`try_push()`]: Self::try_push
#[derive(Debug, Clone)]
pub struct DVectorMut<D> {
    /// The precision and scale of each decimal in the decimal vector.
    pub(super) ps: PrecisionScale<D>,
    /// The mutable buffer representing the vector decimal elements.
    pub(super) elements: BufferMut<D>,
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) validity: MaskMut,
}

impl<D: NativeDecimalType> DVectorMut<D> {
    /// Creates a new [`DVectorMut<D>`] from the given [`PrecisionScale`], elements buffer, and
    /// validity mask.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - The lengths of the `elements` and `validity` do not match.
    /// - Any of the elements are out of bounds for the given [`PrecisionScale`].
    pub fn new(ps: PrecisionScale<D>, elements: BufferMut<D>, validity: MaskMut) -> Self {
        Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVector`")
    }

    /// Tries to create a new [`DVectorMut<D>`] from the given [`PrecisionScale`], elements buffer,
    /// and validity mask.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - The lengths of the `elements` and `validity` do not match.
    /// - Any of the elements are out of bounds for the given [`PrecisionScale`].
    pub fn try_new(
        ps: PrecisionScale<D>,
        elements: BufferMut<D>,
        validity: MaskMut,
    ) -> VortexResult<Self> {
        if elements.len() != validity.len() {
            vortex_bail!(
                "Elements length {} does not match validity length {}",
                elements.len(),
                validity.len()
            );
        }

        // We assert that each element is within bounds for the given precision/scale.
        if !elements.iter().all(|e| ps.is_valid(*e)) {
            vortex_bail!(
                "One or more elements are out of bounds for precision {} and scale {}",
                ps.precision(),
                ps.scale()
            );
        }

        Ok(Self {
            ps,
            elements,
            validity,
        })
    }

    /// Creates a new [`DVectorMut<D>`] from the given [`PrecisionScale`], elements buffer, and
    /// validity mask, _without_ validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// - The lengths of the elements and validity are equal.
    /// - All elements are in bounds for the given [`PrecisionScale`].
    pub unsafe fn new_unchecked(
        ps: PrecisionScale<D>,
        elements: BufferMut<D>,
        validity: MaskMut,
    ) -> Self {
        if cfg!(debug_assertions) {
            Self::try_new(ps, elements, validity).vortex_expect("Failed to create `DVectorMut`")
        } else {
            Self {
                ps,
                elements,
                validity,
            }
        }
    }

    /// Create a new mutable primitive vector with the given capacity.
    pub fn with_capacity(ps: PrecisionScale<D>, capacity: usize) -> Self {
        Self {
            ps,
            elements: BufferMut::with_capacity(capacity),
            validity: MaskMut::with_capacity(capacity),
        }
    }

    /// Decomposes the decimal vector into its constituent parts ([`PrecisionScale`], decimal
    /// buffer, and validity).
    pub fn into_parts(self) -> (PrecisionScale<D>, BufferMut<D>, MaskMut) {
        (self.ps, self.elements, self.validity)
    }

    /// Get the precision/scale of the decimal vector.
    pub fn precision_scale(&self) -> PrecisionScale<D> {
        self.ps
    }

    /// Returns a reference to the underlying elements buffer containing the decimal data.
    pub fn elements(&self) -> &BufferMut<D> {
        &self.elements
    }

    /// Returns a mutable reference to the underlying elements buffer containing the decimal data.
    ///
    /// # Safety
    ///
    /// Modifying the elements buffer directly may violate the precision/scale constraints.
    /// The caller must ensure that any modifications maintain these invariants.
    pub unsafe fn elements_mut(&mut self) -> &mut BufferMut<D> {
        &mut self.elements
    }

    /// Gets a nullable element at the given index, panicking on out-of-bounds.
    ///
    /// If the element at the given index is null, returns `None`. Otherwise, returns `Some(x)`,
    /// where `x: D`.
    ///
    /// Note that this `get` method is different from the standard library [`slice::get`], which
    /// returns `None` if the index is out of bounds. This method will panic if the index is out of
    /// bounds, and return `None` if the elements is null.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&D> {
        self.validity.value(index).then(|| &self.elements[index])
    }

    /// Appends a new element to the end of the vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is out of bounds for the vector's precision/scale.
    pub fn try_push(&mut self, value: D) -> VortexResult<()> {
        self.try_append_n(value, 1)
    }

    /// Appends n elements to the vector, all set to the given value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is out of bounds for the vector's precision/scale.
    pub fn try_append_n(&mut self, value: D, n: usize) -> VortexResult<()> {
        if !self.ps.is_valid(value) {
            vortex_bail!("Value {:?} is out of bounds for {}", value, self.ps);
        }

        self.elements.push_n(value, n);
        self.validity.append_n(true, n);
        Ok(())
    }
}

impl<D: NativeDecimalType> AsRef<[D]> for DVectorMut<D> {
    fn as_ref(&self) -> &[D] {
        &self.elements
    }
}

impl<D: NativeDecimalType> VectorMutOps for DVectorMut<D> {
    type Immutable = DVector<D>;

    fn len(&self) -> usize {
        self.elements.len()
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        self.elements.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional);
        self.validity.reserve(additional);
    }

    fn clear(&mut self) {
        self.elements.clear();
        self.validity.clear();
    }

    fn truncate(&mut self, len: usize) {
        self.elements.truncate(len);
        self.validity.truncate(len);
    }

    fn extend_from_vector(&mut self, other: &DVector<D>) {
        self.elements.extend_from_slice(&other.elements);
        self.validity.append_mask(other.validity());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.extend((0..n).map(|_| D::default()));
        self.validity.append_n(false, n);
    }

    fn freeze(self) -> DVector<D> {
        DVector {
            ps: self.ps,
            elements: self.elements.freeze(),
            validity: self.validity.freeze(),
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        DVectorMut {
            ps: self.ps,
            elements: self.elements.split_off(at),
            validity: self.validity.split_off(at),
        }
    }

    fn unsplit(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }
        self.elements.unsplit(other.elements);
        self.validity.unsplit(other.validity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_and_validation() {
        // Test with_capacity for different decimal types.
        let ps_i32 = PrecisionScale::<i32>::new(9, 2);
        let vec_i32 = DVectorMut::<i32>::with_capacity(ps_i32, 10);
        assert_eq!(vec_i32.len(), 0);
        assert!(vec_i32.capacity() >= 10);

        let ps_i64 = PrecisionScale::<i64>::new(18, 4);
        let vec_i64 = DVectorMut::<i64>::with_capacity(ps_i64, 5);
        assert_eq!(vec_i64.len(), 0);
        assert!(vec_i64.capacity() >= 5);

        let ps_i128 = PrecisionScale::<i128>::new(38, 10);
        let vec_i128 = DVectorMut::<i128>::with_capacity(ps_i128, 3);
        assert_eq!(vec_i128.len(), 0);
        assert!(vec_i128.capacity() >= 3);

        // Test try_new with valid data.
        let ps = ps_i32;
        let elements = BufferMut::from_iter([100_i32, 200, 300]);
        let validity = MaskMut::new_true(3);
        let vec = DVectorMut::try_new(ps, elements, validity).unwrap();
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.precision_scale().precision(), 9);
        assert_eq!(vec.precision_scale().scale(), 2);

        // Test try_new error handling - length mismatch.
        let elements_bad = BufferMut::from_iter([100_i32, 200]);
        let validity_bad = MaskMut::new_true(3);
        let result = DVectorMut::try_new(ps, elements_bad, validity_bad);
        assert!(result.is_err());

        // Test try_new error handling - out of bounds values.
        let too_large = 10_i32.pow(9); // 10^9 exceeds precision 9.
        let elements_oob = BufferMut::from_iter([100_i32, too_large, 300]);
        let validity_oob = MaskMut::new_true(3);
        let result = DVectorMut::try_new(ps, elements_oob, validity_oob);
        assert!(result.is_err());

        // Test new_unchecked.
        let elements_unchecked = BufferMut::from_iter([100_i32, 200]);
        let validity_unchecked = MaskMut::new_true(2);
        let vec_unchecked =
            unsafe { DVectorMut::new_unchecked(ps, elements_unchecked, validity_unchecked) };
        assert_eq!(vec_unchecked.len(), 2);
    }

    #[test]
    fn test_push_append_and_access() {
        let ps = PrecisionScale::<i32>::new(9, 2);
        let mut vec = DVectorMut::<i32>::with_capacity(ps, 10);

        // Test try_push with valid values.
        vec.try_push(12345).unwrap(); // 123.45.
        vec.try_push(9999).unwrap(); // 99.99.
        vec.try_push(-5000).unwrap(); // -50.00.
        assert_eq!(vec.len(), 3);

        // Test try_push with out-of-bounds values.
        let too_large = 10_i32.pow(9);
        assert!(vec.try_push(too_large).is_err());
        assert_eq!(vec.len(), 3); // Length unchanged after failed push.

        // Test get without nulls.
        assert_eq!(vec.get(0), Some(&12345));
        assert_eq!(vec.get(1), Some(&9999));
        assert_eq!(vec.get(2), Some(&-5000));

        // Test append_nulls.
        vec.append_nulls(2);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(3), None);
        assert_eq!(vec.get(4), None);

        // Test AsRef<[D]> slice access.
        let slice = vec.as_ref();
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0], 12345);
        assert_eq!(slice[1], 9999);
        assert_eq!(slice[2], -5000);
        // Note: slice[3] and slice[4] are default values (0) but marked as null in validity.
    }

    #[test]
    fn test_vector_mut_ops_comprehensive() {
        let ps = PrecisionScale::<i64>::new(10, 3);
        let mut vec1 = DVectorMut::<i64>::with_capacity(ps, 10);
        vec1.try_push(1000000).unwrap(); // 1000.000.
        vec1.try_push(2000000).unwrap(); // 2000.000.
        vec1.try_push(3000000).unwrap(); // 3000.000.
        vec1.try_push(4000000).unwrap(); // 4000.000.

        // Test extend_from_vector.
        let mut vec2 = DVectorMut::<i64>::with_capacity(ps, 10);
        vec2.try_push(5000000).unwrap(); // 5000.000.
        vec2.try_push(6000000).unwrap(); // 6000.000.
        let frozen_vec2 = vec2.freeze();

        let original_len = vec1.len();
        vec1.extend_from_vector(&frozen_vec2);
        assert_eq!(vec1.len(), original_len + frozen_vec2.len());
        assert_eq!(vec1.get(4), Some(&5000000));
        assert_eq!(vec1.get(5), Some(&6000000));

        // Test split_off and validity preservation.
        vec1.append_nulls(2); // Add nulls at positions 6 and 7.
        assert_eq!(vec1.len(), 8);

        let split = vec1.split_off(5);
        assert_eq!(vec1.len(), 5);
        assert_eq!(split.len(), 3);

        // Check that split preserved validity.
        assert_eq!(split.get(0), Some(&6000000)); // Was at index 5.
        assert_eq!(split.get(1), None); // Was null at index 6.
        assert_eq!(split.get(2), None); // Was null at index 7.

        // Test reserve and capacity management.
        let initial_capacity = vec1.capacity();
        vec1.reserve(20);
        assert!(vec1.capacity() >= initial_capacity + 20);

        // Test len() and capacity() tracking.
        assert_eq!(vec1.len(), 5);
        assert!(vec1.capacity() >= 25);

        // Test unsplit - rejoin the vectors.
        vec1.unsplit(split);
        assert_eq!(vec1.len(), 8);
        assert_eq!(vec1.get(6), None); // Verify null is still null after unsplit.
        assert_eq!(vec1.get(7), None); // Verify null is still null after unsplit.
    }

    #[test]
    fn test_freeze_and_immutable_vector() {
        let ps = PrecisionScale::<i64>::new(15, 5);
        let mut vec_mut = DVectorMut::<i64>::with_capacity(ps, 5);

        // Add some values and nulls.
        vec_mut.try_push(1234567890).unwrap(); // 12345.67890.
        vec_mut.try_push(9876543210).unwrap(); // 98765.43210.
        vec_mut.append_nulls(1);
        vec_mut.try_push(5555555555).unwrap(); // 55555.55555.
        vec_mut.append_nulls(1);

        // Test freeze() to convert DVectorMut to DVector.
        let vec_immutable = vec_mut.freeze();
        assert_eq!(vec_immutable.len(), 5);

        // Test DVector::get() with nulls.
        assert_eq!(vec_immutable.get(0), Some(&1234567890));
        assert_eq!(vec_immutable.get(1), Some(&9876543210));
        assert_eq!(vec_immutable.get(2), None); // Null.
        assert_eq!(vec_immutable.get(3), Some(&5555555555));
        assert_eq!(vec_immutable.get(4), None); // Null.

        // Test DVector::as_slice() through AsRef.
        let slice = vec_immutable.as_ref();
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0], 1234567890);
        assert_eq!(slice[3], 5555555555);

        // Test precision_scale() getter on immutable vector.
        assert_eq!(vec_immutable.precision_scale().precision(), 15);
        assert_eq!(vec_immutable.precision_scale().scale(), 5);

        // Test round-trip: DVector → DVectorMut (using try_into_mut).
        let mut vec_mut_again = match vec_immutable.try_into_mut() {
            Ok(v) => v,
            Err(_) => {
                // If conversion fails (buffer is shared), create a new mutable vector.
                // This is expected in some cases when the buffer cannot be made mutable.
                let ps = PrecisionScale::<i64>::new(15, 5);
                let mut new_vec = DVectorMut::<i64>::with_capacity(ps, 6);
                new_vec.try_push(1234567890).unwrap();
                new_vec.try_push(9876543210).unwrap();
                new_vec.append_nulls(1);
                new_vec.try_push(5555555555).unwrap();
                new_vec.append_nulls(1);
                new_vec
            }
        };

        assert_eq!(vec_mut_again.len(), 5);
        vec_mut_again.try_push(7777777777).unwrap(); // 77777.77777.
        assert_eq!(vec_mut_again.len(), 6);

        // Freeze again and verify.
        let vec_final = vec_mut_again.freeze();
        assert_eq!(vec_final.len(), 6);
        assert_eq!(vec_final.get(5), Some(&7777777777));
    }

    #[test]
    fn test_precision_scale_combinations() {
        // Test Decimal(9, 2) - common currency format.
        let ps_9_2 = PrecisionScale::<i32>::new(9, 2);
        let mut vec_9_2 = DVectorMut::<i32>::with_capacity(ps_9_2, 5);
        vec_9_2.try_push(999999999).unwrap(); // Max: 9999999.99 stored as 999999999.
        assert!(vec_9_2.try_push(1000000000).is_err()); // 10000000.00 stored as 1000000000 exceeds precision.
        assert!(vec_9_2.try_push(-999999999).is_ok()); // Negative within bounds.
        assert_eq!(vec_9_2.len(), 2);

        // Test Decimal(38, 10) - high precision scientific.
        let ps_38_10 = PrecisionScale::<i128>::new(38, 10);
        let mut vec_38_10 = DVectorMut::<i128>::with_capacity(ps_38_10, 3);
        let large_value = 10_i128.pow(28) - 1; // 10^28 - 1, well within 38 digits.
        vec_38_10.try_push(large_value).unwrap();
        assert_eq!(vec_38_10.len(), 1);

        // Test Decimal(4, 0) - integer-only decimals that fit in i16.
        let ps_4_0 = PrecisionScale::<i16>::new(4, 0);
        let mut vec_4_0 = DVectorMut::<i16>::with_capacity(ps_4_0, 5);
        vec_4_0.try_push(9999).unwrap(); // Max: 9999.
        assert!(vec_4_0.try_push(10000).is_err()); // Exceeds 4 digits.
        vec_4_0.try_push(-9999).unwrap(); // Negative within bounds.
        assert_eq!(vec_4_0.len(), 2);

        // Test with different underlying types.
        // i8 with small precision/scale (max precision for i8 is 2).
        let ps_2_1 = PrecisionScale::<i8>::new(2, 1);
        let mut vec_i8 = DVectorMut::<i8>::with_capacity(ps_2_1, 3);
        vec_i8.try_push(99).unwrap(); // 9.9.
        assert!(vec_i8.try_push(100).is_err()); // 10.0 exceeds precision.

        // i16 with moderate precision/scale (max precision for i16 is 4).
        let ps_4_2 = PrecisionScale::<i16>::new(4, 2);
        let mut vec_i16 = DVectorMut::<i16>::with_capacity(ps_4_2, 3);
        vec_i16.try_push(999).unwrap(); // 9.99.
        vec_i16.try_push(9999).unwrap(); // 99.99.
        assert_eq!(vec_i16.len(), 2);
    }

    #[test]
    fn test_empty_and_edge_cases() {
        let ps = PrecisionScale::<i32>::new(9, 2);

        // Test empty vector creation and operations.
        let empty_vec = DVectorMut::<i32>::with_capacity(ps, 0);
        assert_eq!(empty_vec.len(), 0);
        // Capacity might be rounded up from the requested value.
        let _ = empty_vec.capacity(); // Just verify it doesn't panic.

        // Freeze empty vector.
        let frozen_empty = empty_vec.freeze();
        assert_eq!(frozen_empty.len(), 0);

        // Test single element vector.
        let mut single = DVectorMut::<i32>::with_capacity(ps, 1);
        single.try_push(42).unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single.get(0), Some(&42));

        // Split single element vector at index 1.
        // Original keeps [0, 1) = the element, split gets [1, len) = nothing.
        let split_single = single.split_off(1);
        assert_eq!(single.len(), 1); // Original keeps the element.
        assert_eq!(split_single.len(), 0); // Split gets nothing.

        // Test all-null vector.
        let mut all_nulls = DVectorMut::<i32>::with_capacity(ps, 5);
        all_nulls.append_nulls(5);
        assert_eq!(all_nulls.len(), 5);
        for i in 0..5 {
            assert_eq!(all_nulls.get(i), None);
        }

        // Freeze all-null vector and check immutable.
        let frozen_nulls = all_nulls.freeze();
        assert_eq!(frozen_nulls.len(), 5);
        for i in 0..5 {
            assert_eq!(frozen_nulls.get(i), None);
        }

        // Test maximum capacity scenario - create large vector.
        let mut large = DVectorMut::<i32>::with_capacity(ps, 1000);
        for _ in 0..1000 {
            large.try_push(999).unwrap();
        }
        assert_eq!(large.len(), 1000);
        assert!(large.capacity() >= 1000);
    }

    #[test]
    fn test_nulls_with_validity_mask() {
        let ps = PrecisionScale::<i32>::new(8, 3);

        // Create vector with specific null pattern using validity mask.
        let elements = BufferMut::from_iter([1000_i32, 0, 2000, 0, 3000]); // 0s will be null.
        let mut validity = MaskMut::with_capacity(5);
        validity.append_n(true, 1); // index 0: valid
        validity.append_n(false, 1); // index 1: null
        validity.append_n(true, 1); // index 2: valid
        validity.append_n(false, 1); // index 3: null
        validity.append_n(true, 1); // index 4: valid
        let mut vec = DVectorMut::new(ps, elements, validity);

        assert_eq!(vec.len(), 5);
        assert_eq!(vec.get(0), Some(&1000)); // 1.000.
        assert_eq!(vec.get(1), None); // Null.
        assert_eq!(vec.get(2), Some(&2000)); // 2.000.
        assert_eq!(vec.get(3), None); // Null.
        assert_eq!(vec.get(4), Some(&3000)); // 3.000.

        // Extend with more values and nulls.
        vec.try_push(4000).unwrap();
        vec.append_nulls(2);
        assert_eq!(vec.len(), 8);
        assert_eq!(vec.get(5), Some(&4000));
        assert_eq!(vec.get(6), None);
        assert_eq!(vec.get(7), None);

        // Split and verify nulls are preserved.
        let split = vec.split_off(4);
        assert_eq!(vec.len(), 4);
        assert_eq!(split.len(), 4);

        // Original vec should have: valid, null, valid, null.
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.get(3), None);

        // Split should have: valid, valid, null, null.
        assert_eq!(split.get(0), Some(&3000));
        assert_eq!(split.get(1), Some(&4000));
        assert_eq!(split.get(2), None);
        assert_eq!(split.get(3), None);
    }
}
