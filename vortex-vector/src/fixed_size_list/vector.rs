// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVector`].

use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::fixed_size_list::{FixedSizeListScalar, FixedSizeListVectorMut};
use crate::{Vector, VectorOps};

/// An immutable vector of fixed-size lists.
///
/// `FixedSizeList` vectors can mostly be thought of as a wrapper around other vectors that "groups"
/// a fixed number of elements together for each list scalar.
///
/// More specifically, each list scalar in the vector has the same number of elements (fixed size),
/// with all list elements stored contiguously in a child [`Vector`].
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
pub struct FixedSizeListVector {
    /// The child vector of elements.
    pub(super) elements: Arc<Vector>,

    /// The size of every list in the vector.
    pub(super) list_size: u32,

    /// The validity mask (where `true` represents a list is **not** null).
    ///
    /// Note that the `elements` vector will have its own internal validity, denoting if individual
    /// list elements are null.
    pub(super) validity: Mask,

    /// The length of the vector (which is the same as the length of the validity mask).
    ///
    /// This is stored here as a convenience, as the validity also tracks this information.
    pub(super) len: usize,
}

impl FixedSizeListVector {
    /// Creates a new [`FixedSizeListVector`] from the given `elements` vector, size of each list,
    /// and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the `validity` mask multiplied by the `list_size` is not
    /// equal to the length of the `elements` vector.
    ///
    /// Put another way, the length of the `elements` vector divided by the `list_size` must be
    /// equal to the length of the validity, or this function will panic.
    pub fn new(elements: Arc<Vector>, list_size: u32, validity: Mask) -> Self {
        Self::try_new(elements, list_size, validity)
            .vortex_expect("Failed to create `FixedSizeListVector`")
    }

    /// Tries to create a new [`FixedSizeListVector`] from the given `elements` vector, size of each
    /// list, and validity mask.
    ///
    /// # Errors
    ///
    /// Returns and error if the length of the `validity` mask multiplied by the `list_size` is not
    /// equal to the length of the `elements` vector.
    ///
    /// Put another way, the length of the `elements` vector divided by the `list_size` must be
    /// equal to the length of the validity.
    pub fn try_new(elements: Arc<Vector>, list_size: u32, validity: Mask) -> VortexResult<Self> {
        let len = validity.len();
        let elements_len = elements.len();

        if list_size == 0 {
            vortex_ensure!(
                elements.is_empty(),
                "A degenerate (`list_size == 0`) `FixedSizeListVector` should have no underlying elements",
            );
        } else {
            vortex_ensure!(
                list_size as usize * len == elements_len,
                "Tried to create a `FixedSizeListVector` of length {len} and list_size {list_size} \
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

    /// Tries to create a new [`FixedSizeListVector`] from the given `elements` vector, size of each
    /// list, and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the length of the `validity` mask multiplied by the `list_size`
    /// is exactly equal to the length of the `elements` vector.
    pub unsafe fn new_unchecked(elements: Arc<Vector>, list_size: u32, validity: Mask) -> Self {
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

    /// Decomposes the `FixedSizeListVector` into its constituent parts (child elements, list size,
    /// and validity).
    pub fn into_parts(self) -> (Arc<Vector>, u32, Mask) {
        (self.elements, self.list_size, self.validity)
    }

    /// Returns the element size of every list in the vector.
    pub fn element_size(&self) -> u32 {
        self.list_size
    }

    /// Returns the child vector of elements, which represents the contiguous fixed-size lists of
    /// the `FixedSizeListVector`.
    pub fn elements(&self) -> &Arc<Vector> {
        &self.elements
    }

    /// Returns the size of every list in the vector.
    pub fn list_size(&self) -> u32 {
        self.list_size
    }
}

impl VectorOps for FixedSizeListVector {
    type Mutable = FixedSizeListVectorMut;
    type Scalar = FixedSizeListScalar;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn scalar_at(&self, index: usize) -> FixedSizeListScalar {
        assert!(index < self.len());
        FixedSizeListScalar::new(self.slice(index..index + 1))
    }

    fn slice(&self, _range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        todo!()
    }

    fn clear(&mut self) {
        Arc::make_mut(&mut self.elements).clear();
        self.validity.clear();
        self.len = 0;
    }

    fn try_into_mut(self) -> Result<FixedSizeListVectorMut, Self> {
        let len = self.len;
        let list_size = self.list_size;

        // Try to unwrap the `Arc`.
        let elements = match Arc::try_unwrap(self.elements) {
            Ok(elements) => elements,
            Err(elements) => return Err(Self { elements, ..self }),
        };

        // Try to make validity mutable.
        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(Self {
                    elements: Arc::new(elements),
                    list_size,
                    validity,
                    len,
                });
            }
        };

        // Try to make the elements mutable.
        match elements.try_into_mut() {
            Ok(mutable_elements) => Ok(FixedSizeListVectorMut {
                elements: Box::new(mutable_elements),
                list_size,
                validity,
                len,
            }),
            Err(elements) => Err(Self {
                elements: Arc::new(elements),
                list_size,
                validity: validity.freeze(),
                len,
            }),
        }
    }

    fn into_mut(self) -> FixedSizeListVectorMut {
        let len = self.len;
        let list_size = self.list_size;
        let validity = self.validity.into_mut();

        // If someone else has a strong reference to the `Arc`, clone the underlying data (which is
        // just a **different** reference count increment).
        let elements = Arc::try_unwrap(self.elements).unwrap_or_else(|arc| (*arc).clone());

        FixedSizeListVectorMut {
            elements: Box::new(elements.into_mut()),
            list_size,
            validity,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_mask::Mask;

    use super::*;
    use crate::primitive::PVectorMut;
    use crate::{Vector, VectorMutOps};

    #[test]
    fn test_constructor_and_validation() {
        // Valid construction with new().
        let elements: Arc<Vector> = Arc::new(
            PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6])
                .freeze()
                .into(),
        );
        let validity = Mask::new_true(2);
        let vec = FixedSizeListVector::new(elements.clone(), 3, validity.clone());
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.list_size(), 3);

        // Valid construction with try_new().
        let result = FixedSizeListVector::try_new(elements.clone(), 3, validity);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);

        // Length mismatch error - elements length != list_size * validity length.
        let bad_validity = Mask::new_true(3); // Should be 2 for 6 elements with list_size=3.
        let result = FixedSizeListVector::try_new(elements.clone(), 3, bad_validity);
        assert!(result.is_err());

        // Degenerate case (list_size = 0) with empty elements is valid.
        let empty_elements: Arc<Vector> = Arc::new(
            PVectorMut::<i32>::from_iter(Vec::<i32>::new())
                .freeze()
                .into(),
        );
        let validity = Mask::new_true(5);
        let result = FixedSizeListVector::try_new(empty_elements, 0, validity);
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.list_size(), 0);

        // Degenerate case with non-empty elements should fail.
        let result = FixedSizeListVector::try_new(elements, 0, Mask::new_true(1));
        assert!(result.is_err());

        // Test unsafe new_unchecked in debug mode (it should still validate).
        let elements: Arc<Vector> =
            Arc::new(PVectorMut::<i32>::from_iter([1, 2, 3, 4]).freeze().into());
        let validity = Mask::new_true(2);
        let vec = unsafe { FixedSizeListVector::new_unchecked(elements, 2, validity) };
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.list_size(), 2);
    }

    #[test]
    fn test_try_into_mut_conversion() {
        // Create a vector that we solely own.
        let elements: Arc<Vector> = Arc::new(
            PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6])
                .freeze()
                .into(),
        );
        let validity = Mask::new_true(2);
        let vec = FixedSizeListVector::new(elements, 3, validity);

        // Successful conversion when solely owned.
        let result = vec.try_into_mut();
        assert!(result.is_ok());
        let mut_vec = result.unwrap();
        assert_eq!(mut_vec.len(), 2);
        assert_eq!(mut_vec.list_size(), 3);

        // Freeze and try again - roundtrip test.
        let vec = mut_vec.freeze();
        let result = vec.try_into_mut();
        assert!(result.is_ok());

        // Test failed conversion with shared ownership.
        let elements: Arc<Vector> =
            Arc::new(PVectorMut::<i32>::from_iter([1, 2, 3, 4]).freeze().into());
        let validity = Mask::new_true(2);
        let vec = FixedSizeListVector::new(elements, 2, validity);

        // Keep a clone to maintain shared ownership.
        let _shared = vec.clone();

        let result = vec.try_into_mut();
        assert!(result.is_err());

        // The error case should return the original vector.
        if let Err(returned_vec) = result {
            assert_eq!(returned_vec.len(), 2);
            assert_eq!(returned_vec.list_size(), 2);
        }
    }

    #[test]
    fn test_accessors_and_parts() {
        let elements: Arc<Vector> = Arc::new(
            PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6])
                .freeze()
                .into(),
        );
        let validity = Mask::new_true(3);
        let vec = FixedSizeListVector::new(elements, 2, validity);

        // Test accessors.
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.list_size(), 2);
        assert_eq!(vec.elements().len(), 6);
        assert_eq!(vec.validity().true_count(), 3);

        // Test into_parts.
        let (parts_elements, list_size, parts_validity) = vec.into_parts();
        assert_eq!(parts_elements.len(), 6);
        assert_eq!(list_size, 2);
        assert_eq!(parts_validity.true_count(), 3);
    }
}
