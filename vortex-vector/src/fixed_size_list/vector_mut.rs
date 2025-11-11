// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVectorMut`].

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use crate::fixed_size_list::FixedSizeListVector;
use crate::{VectorMut, VectorMutOps, match_vector_pair};

/// A mutable vector of fixed-size lists.
///
/// `FixedSizeList` vectors can mostly be thought of as a wrapper around other vectors that "groups"
/// a fixed number of elements together for each list scalar.
///
/// More specifically, each list scalar in the vector has the same number of elements (fixed size),
/// with all list elements stored contiguously in a child [`VectorMut`].
///
/// Note that the validity mask tracks which lists are null, not which individual elements are null.
///
/// # Structure
///
/// For a vector of `n` lists each with size `list_size`:
/// - The `elements` vector has length `n * list_size`
/// - The `validity` mask has length `n`
/// - Each list `i` occupies `elements[i * list_size..(i+1) * list_size]
#[derive(Debug, Clone)]
pub struct FixedSizeListVectorMut {
    /// The mutable child vector of elements.
    pub(super) elements: Box<VectorMut>,

    /// The size of every list in the vector.
    pub(super) list_size: u32,

    /// The validity mask (where `true` represents a list is **not** null).
    ///
    /// Note that the `elements` vector will have its own internal validity, denoting if individual
    /// list elements are null.
    pub(super) validity: MaskMut,

    /// The length of the vector (which is the same as the length of the validity mask).
    ///
    /// This is stored here as a convenience, as the validity also tracks this information.
    pub(super) len: usize,
}

impl FixedSizeListVectorMut {
    /// Creates a new [`FixedSizeListVectorMut`] from the given `elements` vector, size of each
    /// list, and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the `validity` mask multiplied by the `list_size` is not
    /// equal to the length of the `elements` vector.
    ///
    /// Put another way, the length of the `elements` vector divided by the `list_size` must be
    /// equal to the length of the validity, or this function will panic.
    pub fn new(elements: Box<VectorMut>, list_size: u32, validity: MaskMut) -> Self {
        Self::try_new(elements, list_size, validity)
            .vortex_expect("Failed to create `FixedSizeListVectorMut`")
    }

    /// Tries to create a new [`FixedSizeListVectorMut`] from the given `elements` vector, size of
    /// each list, and validity mask.
    ///
    /// # Errors
    ///
    /// Returns and error if the length of the `validity` mask multiplied by the `list_size` is not
    /// equal to the length of the `elements` vector.
    ///
    /// Put another way, the length of the `elements` vector divided by the `list_size` must be
    /// equal to the length of the validity.
    pub fn try_new(
        elements: Box<VectorMut>,
        list_size: u32,
        validity: MaskMut,
    ) -> VortexResult<Self> {
        let len = validity.len();
        let elements_len = elements.len();

        if list_size == 0 {
            vortex_ensure!(
                elements.is_empty(),
                "A degenerate (`list_size == 0`) `FixedSizeListVectorMut` should have no underlying elements",
            );
        } else {
            vortex_ensure!(
                list_size as usize * len == elements_len,
                "Tried to create a `FixedSizeListVectorMut` of length {len} and list_size {list_size} \
                with an child vector of size {elements_len} ({list_size} * {len} != {elements_len})",
            );
        }

        Ok(Self {
            elements,
            list_size,
            validity,
            len,
        })
    }

    /// Tries to create a new [`FixedSizeListVectorMut`] from the given `elements` vector, size of
    /// each list, and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the length of the `validity` mask multiplied by the `list_size`
    /// is exactly equal to the length of the `elements` vector.
    pub unsafe fn new_unchecked(
        elements: Box<VectorMut>,
        list_size: u32,
        validity: MaskMut,
    ) -> Self {
        let len = validity.len();

        if cfg!(debug_assertions) {
            Self::new(elements, list_size, validity)
        } else {
            Self {
                elements,
                list_size,
                validity,
                len,
            }
        }
    }

    /// Creates a new [`FixedSizeListVectorMut`] with given element type, list size, and capacity.
    pub fn with_capacity(elem_dtype: &DType, list_size: u32, capacity: usize) -> Self {
        let elements = Box::new(VectorMut::with_capacity(
            elem_dtype,
            capacity * list_size as usize,
        ));

        let validity = MaskMut::with_capacity(capacity);
        let len = validity.len();

        Self {
            elements,
            list_size,
            validity,
            len,
        }
    }

    /// Decomposes the `FixedSizeListVector` into its constituent parts (child elements, list size,
    /// and validity).
    pub fn into_parts(self) -> (Box<VectorMut>, u32, MaskMut) {
        (self.elements, self.list_size, self.validity)
    }

    /// Returns the child vector of elements, which represents the contiguous fixed-size lists of
    /// the `FixedSizeListVector`.
    pub fn elements(&self) -> &VectorMut {
        &self.elements
    }

    /// Returns the size of every list in the vector.
    pub fn list_size(&self) -> u32 {
        self.list_size
    }
}

impl VectorMutOps for FixedSizeListVectorMut {
    type Immutable = FixedSizeListVector;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    /// In the case that `list_size == 0`, the capacity of the vector is infinite because it will
    /// never take up any space.
    fn capacity(&self) -> usize {
        self.elements
            .capacity()
            .checked_div(self.list_size as usize)
            .unwrap_or(usize::MAX)
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional * self.list_size as usize);
    }

    fn clear(&mut self) {
        self.elements.clear();
        self.validity.clear();
        self.len = 0;
    }

    fn truncate(&mut self, len: usize) {
        let new_len = len.min(self.len);
        self.elements.truncate(new_len * self.list_size as usize);
        self.validity.truncate(new_len);
        self.len = new_len;
    }

    fn extend_from_vector(&mut self, other: &FixedSizeListVector) {
        match_vector_pair!(
            self.elements.as_mut(),
            other.elements.as_ref(),
            |a: VectorMut, b: Vector| {
                // This will panic if `other.elements` is not the correct type of vector.
                a.extend_from_vector(b);
            }
        );

        self.validity.append_mask(&other.validity);
        self.len += other.len;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.append_nulls(n * self.list_size as usize);
        self.validity.append_n(false, n);
        self.len += n;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn freeze(self) -> FixedSizeListVector {
        FixedSizeListVector {
            elements: Arc::new(self.elements.freeze()),
            list_size: self.list_size,
            validity: self.validity.freeze(),
            len: self.len,
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        assert!(
            at <= self.capacity(),
            "split_off out of bounds: {} > {}",
            at,
            self.capacity()
        );

        let split_elements = self.elements.split_off(at * self.list_size as usize);

        let split_validity = self.validity.split_off(at);
        let split_len = self.len.saturating_sub(at);
        self.len = at;

        debug_assert_eq!(self.len, self.validity.len());

        Self {
            elements: Box::new(split_elements),
            list_size: self.list_size,
            validity: split_validity,
            len: split_len,
        }
    }

    fn unsplit(&mut self, other: Self) {
        assert_eq!(self.list_size, other.list_size);

        if self.is_empty() {
            *self = other;
            return;
        }

        self.elements.unsplit(*other.elements);
        self.validity.unsplit(other.validity);

        self.len += other.len;
        debug_assert_eq!(self.len, self.validity.len());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, PType};
    use vortex_mask::{Mask, MaskMut};

    use super::*;
    use crate::VectorOps;
    use crate::primitive::PVectorMut;

    #[test]
    fn test_core_operations() {
        // Test with_capacity constructor.
        let dtype = DType::Primitive(PType::I32, vortex_dtype::Nullability::Nullable);
        let mut vec = FixedSizeListVectorMut::with_capacity(&dtype, 3, 10);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.list_size(), 3);
        assert!(vec.capacity() >= 10);

        // Create a vector to extend from.
        let elements = Arc::new(
            PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6])
                .freeze()
                .into(),
        );
        let validity = Mask::new_true(2);
        let immutable = FixedSizeListVector::new(elements, 3, validity);

        // Test extend_from_vector.
        vec.extend_from_vector(&immutable);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.elements().len(), 6);

        // Test append_nulls.
        vec.append_nulls(3);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.elements().len(), 15); // 5 lists * 3 elements each.

        // Test freeze and accessors.
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 5);
        assert_eq!(frozen.list_size(), 3);
        assert_eq!(frozen.elements().len(), 15);
    }

    #[test]
    fn test_split_unsplit_operations() {
        // Create a vector with 6 lists, each containing 2 elements.
        let elements = PVectorMut::<i32>::from_iter([
            1, 2, // List 0
            3, 4, // List 1
            5, 6, // List 2
            7, 8, // List 3
            9, 10, // List 4
            11, 12, // List 5
        ]);
        let mut vec =
            FixedSizeListVectorMut::new(Box::new(elements.into()), 2, MaskMut::new_true(6));

        // Test split at different positions.

        // Split at position 0 (take nothing).
        let split = vec.split_off(0);
        assert_eq!(vec.len(), 0);
        assert_eq!(split.len(), 6);
        vec.unsplit(split);
        assert_eq!(vec.len(), 6);

        // Split at middle position.
        let split = vec.split_off(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(split.len(), 3);
        assert_eq!(vec.elements().len(), 6); // 3 lists * 2 elements.
        assert_eq!(split.elements().len(), 6); // 3 lists * 2 elements.

        // Verify the correct elements are in each half.
        // First half should have [1,2,3,4,5,6].
        // Second half should have [7,8,9,10,11,12].

        // Rejoin the parts.
        vec.unsplit(split);
        assert_eq!(vec.len(), 6);
        assert_eq!(vec.elements().len(), 12);

        // Split at the end (take everything).
        let split = vec.split_off(6);
        assert_eq!(vec.len(), 6);
        assert_eq!(split.len(), 0);
        vec.unsplit(split);
        assert_eq!(vec.len(), 6);
    }

    #[test]
    fn test_null_handling() {
        // Test nullable lists with non-null elements.
        let elements = PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6]);
        let validity = MaskMut::new_true(3);
        // We can't directly set individual validity, but we can create vectors with nulls.

        let mut vec = FixedSizeListVectorMut::new(Box::new(elements.into()), 2, validity);

        // Append null lists.
        vec.append_nulls(2);
        assert_eq!(vec.len(), 5);

        // After freezing, check validity is preserved.
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 5);
        assert_eq!(frozen.validity().true_count(), 3); // First 3 are valid.

        // Test non-null lists with nullable elements.
        let elements_with_nulls = PVectorMut::<i32>::from_iter([
            Some(1),
            None,
            Some(3), // First list has a null element.
            Some(4),
            Some(5),
            None, // Second list has a null element.
        ]);
        let validity = MaskMut::new_true(2); // Both lists are valid.

        let mut vec =
            FixedSizeListVectorMut::new(Box::new(elements_with_nulls.into()), 3, validity);

        assert_eq!(vec.len(), 2);
        assert_eq!(vec.elements().len(), 6);

        // Operations should preserve element nullability.
        let split = vec.split_off(1);
        assert_eq!(vec.len(), 1);
        assert_eq!(split.len(), 1);

        vec.unsplit(split);
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_edge_cases() {
        // Test empty vector (0 lists).
        let elements = PVectorMut::<i32>::from_iter(Vec::<i32>::new());
        let validity = MaskMut::new_true(0);
        let mut vec = FixedSizeListVectorMut::new(Box::new(elements.into()), 3, validity);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.list_size(), 3);
        assert_eq!(vec.elements().len(), 0);

        // Operations on empty vector.
        vec.append_nulls(1);
        assert_eq!(vec.len(), 1);

        // Test single element list.
        let elements = PVectorMut::<i32>::from_iter([42]);
        let validity = MaskMut::new_true(1);
        let vec = FixedSizeListVectorMut::new(
            Box::new(elements.into()),
            1, // List size of 1.
            validity,
        );
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.list_size(), 1);

        // Test large list size.
        let large_elements: Vec<i32> = (0..1000).collect();
        let elements = PVectorMut::<i32>::from_iter(large_elements);
        let validity = MaskMut::new_true(1); // Single list with 1000 elements.
        let vec = FixedSizeListVectorMut::new(Box::new(elements.into()), 1000, validity);
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.list_size(), 1000);
        assert_eq!(vec.elements().len(), 1000);

        // Verify operations work correctly.
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 1);
        assert_eq!(frozen.list_size(), 1000);
    }

    #[test]
    fn test_capacity_management() {
        let dtype = DType::Primitive(PType::I32, vortex_dtype::Nullability::Nullable);

        // Test initial capacity from with_capacity.
        let mut vec = FixedSizeListVectorMut::with_capacity(&dtype, 3, 10);
        assert!(vec.capacity() >= 10);
        assert!(vec.elements().capacity() >= 30); // At least 10 lists * 3 elements.

        // Test reserve works without panicking.
        // The exact capacity increase depends on the underlying allocation strategy.
        vec.reserve(100);
        // After reserving, we should be able to hold at least the current length + reserved amount.
        // Since current length is 0, capacity should be at least 100.
        assert!(vec.capacity() >= 100);

        // Test capacity calculation with different list sizes.
        let vec2 = FixedSizeListVectorMut::with_capacity(&dtype, 5, 20);
        assert!(vec2.capacity() >= 20);
        assert!(vec2.elements().capacity() >= 100); // At least 20 lists * 5 elements.

        // Edge case: capacity when list_size = 0.
        // Based on the documentation, capacity is infinite (usize::MAX) for degenerate case.
        let vec3 = FixedSizeListVectorMut::with_capacity(&dtype, 0, 10);
        assert_eq!(vec3.capacity(), usize::MAX); // Infinite capacity for degenerate case.

        // Test that capacity is preserved through operations.
        let elements = PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6]);
        vec.elements = Box::new(elements.into());
        vec.validity = MaskMut::new_true(2);
        vec.len = 2;
        vec.list_size = 3;

        vec.reserve(8); // Reserve space for 8 more lists.
        assert!(vec.capacity() >= 10);

        // Test that split_off and unsplit work without panicking.
        let split = vec.split_off(1);
        assert_eq!(vec.len(), 1);
        assert_eq!(split.len(), 1);

        vec.unsplit(split);
        assert_eq!(vec.len(), 2);
    }
}
