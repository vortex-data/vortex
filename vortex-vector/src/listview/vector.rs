// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVector`].

use std::fmt::Debug;
use std::ops::BitAnd;
use std::ops::RangeBounds;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use super::ListViewScalar;
use super::ListViewVectorMut;
use crate::Vector;
use crate::match_each_integer_pvector;
use crate::match_each_integer_pvector_pair;
use crate::primitive::PVector;
use crate::primitive::PrimitiveVector;
use crate::vector_ops::VectorMutOps;
use crate::vector_ops::VectorOps;

/// A vector of variable-width lists.
///
/// Each list is defined by 2 integers: an offset and a size (a "list view"), which point into a
/// child `elements` vector.
///
/// Note that the list views **do not** need to be sorted, nor do they have to be contiguous or
/// fully cover the `elements` vector. This means that multiple views can be pointing to the same
/// elements.
///
/// # Structure
///
/// - `elements`: The child vector of all list elements, stored as an [`Arc<Vector>`].
/// - `offsets`: A [`PrimitiveVector`] containing the starting offset of each list in the `elements`
///   vector.
/// - `sizes`: A [`PrimitiveVector`] containing the size (number of elements) of each list.
/// - `validity`: A [`Mask`] indicating which lists are null.
#[derive(Debug, Clone)]
pub struct ListViewVector {
    /// The child vector of elements.
    pub(super) elements: Arc<Vector>,

    /// Offsets for each list into the elements vector.
    ///
    /// Offsets are always integers, and always non-negative (even if the type is signed).
    pub(super) offsets: PrimitiveVector,

    /// Sizes (lengths) of each list.
    ///
    /// Sizes are always integers, and always non-negative (even if the type is signed).
    pub(super) sizes: PrimitiveVector,

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

impl PartialEq for ListViewVector {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        if self.validity != other.validity {
            return false;
        }
        if self.elements.len() != other.elements.len() {
            return false;
        }

        // Offsets and sizes must have matching types, then compare within the match
        match_each_integer_pvector_pair!(
            (&self.offsets, &other.offsets),
            |self_offsets, other_offsets| {
                match_each_integer_pvector_pair!(
                    (&self.sizes, &other.sizes),
                    |self_sizes, other_sizes| {
                        listview_eq_impl(
                            self.len,
                            &self.validity,
                            self.elements.as_ref(),
                            other.elements.as_ref(),
                            self_offsets,
                            other_offsets,
                            self_sizes,
                            other_sizes,
                        )
                    },
                    { false } // Size types don't match
                )
            },
            { false } // Offset types don't match
        )
    }
}

/// Helper function for ListViewVector equality comparison.
#[expect(clippy::too_many_arguments)]
fn listview_eq_impl<O, S>(
    len: usize,
    validity: &Mask,
    self_elements: &Vector,
    other_elements: &Vector,
    self_offsets: &PVector<O>,
    other_offsets: &PVector<O>,
    self_sizes: &PVector<S>,
    other_sizes: &PVector<S>,
) -> bool
where
    O: vortex_dtype::NativePType + Copy,
    S: vortex_dtype::NativePType + Copy,
    usize: TryFrom<O> + TryFrom<S>,
{
    // Fast path: if all lists are invalid, elements don't matter
    if validity.all_false() {
        return true;
    }

    // Fast path: if all lists are valid, compare elements directly
    if validity.all_true() {
        return self_elements == other_elements
            && self_offsets == other_offsets
            && self_sizes == other_sizes;
    }

    // Build element-level mask using Vec<bool> to handle overlapping slices correctly
    let elem_len = self_elements.len();
    let mut element_valid = vec![false; elem_len];
    for i in 0..len {
        if validity.value(i) {
            let offset = self_offsets
                .get_as::<usize>(i)
                .vortex_expect("offset is valid and fits in usize");
            let size = self_sizes
                .get_as::<usize>(i)
                .vortex_expect("size is valid and fits in usize");
            for j in offset..(offset + size).min(elem_len) {
                element_valid[j] = true;
            }
        }
    }
    let element_mask = Mask::from_buffer(vortex_buffer::BitBuffer::from(element_valid));

    // Clone elements and apply the element-level mask
    let mut self_elems = self_elements.clone();
    let mut other_elems = other_elements.clone();
    self_elems.mask_validity(&element_mask);
    other_elems.mask_validity(&element_mask);

    if self_elems != other_elems {
        return false;
    }

    // Compare offsets and sizes at valid positions
    (0..len).all(|i| {
        !validity.value(i)
            || (self_offsets.get_as::<usize>(i) == other_offsets.get_as::<usize>(i)
                && self_sizes.get_as::<usize>(i) == other_sizes.get_as::<usize>(i))
    })
}

impl ListViewVector {
    /// Creates a new [`ListViewVector`] from its components.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `offsets` or `sizes` contain nulls values.
    /// - `offsets`, `sizes`, and `validity` do not all have the same length
    /// - The `sizes` integer width is not less than or equal to the `offsets` integer width (this
    ///   would cause overflow)
    /// - For any `i`, `offsets[i] + sizes[i]` causes an overflow or is greater than
    ///   `elements.len()` (even if the corresponding view is defined as null by the validity
    ///   array).
    pub fn new(
        elements: Arc<Vector>,
        offsets: PrimitiveVector,
        sizes: PrimitiveVector,
        validity: Mask,
    ) -> Self {
        Self::try_new(elements, offsets, sizes, validity)
            .vortex_expect("Invalid ListViewVector construction")
    }

    /// Attempts to create a new [`ListViewVector`] from its components.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - `offsets` or `sizes` contain nulls values.
    /// - `offsets`, `sizes`, and `validity` do not all have the same length
    /// - The `sizes` integer width is not less than or equal to the `offsets` integer width (this
    ///   would cause overflow)
    /// - For any `i`, `offsets[i] + sizes[i]` causes an overflow or is greater than
    ///   `elements.len()` (even if the corresponding view is defined as null by the validity
    ///   array).
    pub fn try_new(
        elements: Arc<Vector>,
        offsets: PrimitiveVector,
        sizes: PrimitiveVector,
        validity: Mask,
    ) -> VortexResult<Self> {
        let len = validity.len();

        vortex_ensure!(
            offsets.len() == len,
            "Offsets length {} does not match validity length {len}",
            offsets.len(),
        );
        vortex_ensure!(
            sizes.len() == len,
            "Sizes length {} does not match validity length {len}",
            sizes.len(),
        );

        vortex_ensure!(
            offsets.validity().all_true(),
            "Offsets vector must not contain null values"
        );
        vortex_ensure!(
            sizes.validity().all_true(),
            "Sizes vector must not contain null values"
        );

        let offsets_width = offsets.ptype().byte_width();
        let sizes_width = sizes.ptype().byte_width();
        vortex_ensure!(
            sizes_width <= offsets_width,
            "Sizes integer width {sizes_width} must be \
                    <= offsets integer width {offsets_width} to prevent overflow",
        );

        // Check that each `offsets[i] + sizes[i] <= elements.len()`.
        validate_views_bound(elements.len(), &offsets, &sizes)?;

        Ok(Self {
            elements,
            offsets,
            sizes,
            validity,
            len,
        })
    }

    /// Creates a new [`ListViewVector`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - `offsets` and `sizes` must be non-nullable integer vectors.
    /// - `offsets`, `sizes`, and `validity` must have the same length.
    /// - Size integer width must be smaller than or equal to offset type (to prevent overflow).
    /// - For each `i`, `offsets[i] + sizes[i]` must not overflow and must be `<= elements.len()`
    ///   (even if the corresponding view is defined as null by the validity array).
    pub unsafe fn new_unchecked(
        elements: Arc<Vector>,
        offsets: PrimitiveVector,
        sizes: PrimitiveVector,
        validity: Mask,
    ) -> Self {
        let len = validity.len();

        if cfg!(debug_assertions) {
            Self::new(elements, offsets, sizes, validity)
        } else {
            Self {
                elements,
                offsets,
                sizes,
                validity,
                len,
            }
        }
    }

    /// Decomposes the [`ListViewVector`] into its constituent parts (child elements, offsets,
    /// sizes, and validity).
    pub fn into_parts(self) -> (Arc<Vector>, PrimitiveVector, PrimitiveVector, Mask) {
        (self.elements, self.offsets, self.sizes, self.validity)
    }

    /// Returns a reference to the `elements` vector.
    #[inline]
    pub fn elements(&self) -> &Arc<Vector> {
        &self.elements
    }

    /// Returns a reference to the integer `offsets` vector.
    #[inline]
    pub fn offsets(&self) -> &PrimitiveVector {
        &self.offsets
    }

    /// Returns a reference to the integer `sizes` vector.
    #[inline]
    pub fn sizes(&self) -> &PrimitiveVector {
        &self.sizes
    }
}

impl VectorOps for ListViewVector {
    type Mutable = ListViewVectorMut;
    type Scalar = ListViewScalar;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn mask_validity(&mut self, mask: &Mask) {
        self.validity = self.validity.bitand(mask);
    }

    fn scalar_at(&self, index: usize) -> ListViewScalar {
        assert!(index < self.len());
        ListViewScalar::new(self.slice(index..index + 1))
    }

    fn slice(&self, _range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        todo!()
    }

    fn clear(&mut self) {
        self.offsets.clear();
        self.sizes.clear();
        Arc::make_mut(&mut self.elements).clear();
        self.validity.clear();
        self.len = 0;
    }

    fn try_into_mut(self) -> Result<ListViewVectorMut, Self> {
        // Try to unwrap the `Arc`.
        let elements = match Arc::try_unwrap(self.elements) {
            Ok(elements) => elements,
            Err(elements) => return Err(Self { elements, ..self }),
        };

        // Try to make the validity mutable.
        let validity = match self.validity.try_into_mut() {
            Ok(v) => v,
            Err(validity) => {
                return Err(Self {
                    elements: Arc::new(elements),
                    validity,
                    ..self
                });
            }
        };

        // Try to make the offsets mutable.
        let offsets = match self.offsets.try_into_mut() {
            Ok(mutable_offsets) => mutable_offsets,
            Err(offsets) => {
                return Err(Self {
                    offsets,
                    sizes: self.sizes,
                    elements: Arc::new(elements),
                    validity: validity.freeze(),
                    len: self.len,
                });
            }
        };

        // Try to make the sizes mutable.
        let sizes = match self.sizes.try_into_mut() {
            Ok(mutable_sizes) => mutable_sizes,
            Err(sizes) => {
                return Err(Self {
                    offsets: offsets.freeze(),
                    sizes,
                    elements: Arc::new(elements),
                    validity: validity.freeze(),
                    len: self.len,
                });
            }
        };

        // Try to make the elements mutable.
        match elements.try_into_mut() {
            Ok(mut_elements) => Ok(ListViewVectorMut {
                offsets,
                sizes,
                elements: Box::new(mut_elements),
                validity,
                len: self.len,
            }),
            Err(elements) => Err(Self {
                offsets: offsets.freeze(),
                sizes: sizes.freeze(),
                elements: Arc::new(elements),
                validity: validity.freeze(),
                len: self.len,
            }),
        }
    }

    fn into_mut(self) -> ListViewVectorMut {
        let len = self.len;
        let validity = self.validity.into_mut();
        let offsets = self.offsets.into_mut();
        let sizes = self.sizes.into_mut();

        // If someone else has a strong reference to the `Arc`, clone the underlying data (which is
        // just a **different** reference count increment).
        let elements = Arc::try_unwrap(self.elements).unwrap_or_else(|arc| (*arc).clone());

        ListViewVectorMut {
            offsets,
            sizes,
            elements: Box::new(elements.into_mut()),
            validity,
            len,
        }
    }
}

// TODO(connor): It would be better to separate everything inside the macros into its own function,
// but that would require adding another macro that sets a type `$type` to be used by the caller.
/// Checks that all views are `<= elements_len`.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from nested match_each_* macros"
)]
#[allow(clippy::cast_possible_truncation)] // casts inside macro
fn validate_views_bound(
    elements_len: usize,
    offsets: &PrimitiveVector,
    sizes: &PrimitiveVector,
) -> VortexResult<()> {
    let len = offsets.len();

    match_each_integer_pvector!(&offsets, |offsets_vector| {
        match_each_integer_pvector!(&sizes, |sizes_vector| {
            let offsets_slice = offsets_vector.as_ref();
            let sizes_slice = sizes_vector.as_ref();

            for i in 0..len {
                let offset = offsets_slice[i] as usize;
                let size = sizes_slice[i] as usize;
                vortex_ensure!(offset + size <= elements_len);
            }
        });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::Buffer;
    use vortex_mask::Mask;

    use super::*;
    use crate::primitive::PVector;

    /// Helper to create a ListViewVector with i32 elements and u32 offsets/sizes
    fn make_listview(
        elements: Vec<i32>,
        offsets: Vec<u32>,
        sizes: Vec<u32>,
        validity: Mask,
    ) -> ListViewVector {
        let elem_validity = Mask::new_true(elements.len());
        let elements = PVector::new(Buffer::from(elements), elem_validity);
        let offsets_len = offsets.len();
        let sizes_len = sizes.len();
        let offsets = PVector::new(Buffer::from(offsets), Mask::new_true(offsets_len));
        let sizes = PVector::new(Buffer::from(sizes), Mask::new_true(sizes_len));
        ListViewVector::try_new(
            Arc::new(Vector::from(elements)),
            PrimitiveVector::from(offsets),
            PrimitiveVector::from(sizes),
            validity,
        )
        .unwrap()
    }

    #[test]
    fn test_listview_eq_all_valid() {
        // All lists valid - direct element comparison
        let v1 = make_listview(
            vec![1, 2, 3, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            Mask::new_true(3),
        );
        let v2 = make_listview(
            vec![1, 2, 3, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            Mask::new_true(3),
        );
        assert_eq!(v1, v2);

        // Different elements should not be equal
        let v3 = make_listview(
            vec![1, 2, 99, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            Mask::new_true(3),
        );
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_listview_eq_all_invalid() {
        // All lists invalid - elements don't matter
        let v1 = make_listview(
            vec![1, 2, 3, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            Mask::new_false(3),
        );
        let v2 = make_listview(
            vec![99, 99, 99, 99, 99],
            vec![0, 2, 3],
            vec![2, 1, 2],
            Mask::new_false(3),
        );
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_listview_eq_mixed_validity() {
        // Lists: [1,2], null, [4,5]
        // Elements at positions 2 (index of null list's elements) don't matter
        let validity = Mask::from_indices(3, vec![0, 2]);

        let v1 = make_listview(
            vec![1, 2, 3, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            validity.clone(),
        );
        let v2 = make_listview(
            vec![1, 2, 3, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            validity.clone(),
        );
        assert_eq!(v1, v2);

        // Element at position 2 is only used by the invalid list - should still be equal
        let v3 = make_listview(
            vec![1, 2, 99, 4, 5],
            vec![0, 2, 3],
            vec![2, 1, 2],
            validity.clone(),
        );
        assert_eq!(v1, v3, "Invalid list's elements should be ignored");

        // Element at position 0 is used by valid list 0 - should NOT be equal
        let v4 = make_listview(vec![99, 2, 3, 4, 5], vec![0, 2, 3], vec![2, 1, 2], validity);
        assert_ne!(v1, v4, "Valid list's elements must match");
    }

    #[test]
    fn test_listview_eq_overlapping_slices() {
        // Overlapping ranges: list0=[0..3], list1=[1..4] (overlapping at positions 1,2)
        // This tests that the Vec<bool> approach handles overlaps correctly
        let v1 = make_listview(vec![1, 2, 3, 4], vec![0, 1], vec![3, 3], Mask::new_true(2));
        let v2 = make_listview(vec![1, 2, 3, 4], vec![0, 1], vec![3, 3], Mask::new_true(2));
        assert_eq!(v1, v2);

        // Different element in overlapping region
        let v3 = make_listview(vec![1, 99, 3, 4], vec![0, 1], vec![3, 3], Mask::new_true(2));
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_listview_eq_overlapping_with_invalid() {
        // list0=[0..3] valid, list1=[1..4] invalid
        // Positions 1,2 are in overlap but list1 is invalid, so only list0's view matters
        let validity = Mask::from_indices(2, vec![0]); // only list 0 is valid

        let v1 = make_listview(vec![1, 2, 3, 4], vec![0, 1], vec![3, 3], validity.clone());

        // Element at position 3 is only used by invalid list1 - can differ
        let v2 = make_listview(vec![1, 2, 3, 99], vec![0, 1], vec![3, 3], validity.clone());
        assert_eq!(v1, v2, "Element used only by invalid list can differ");

        // Element at position 2 is used by valid list0 - must match
        let v3 = make_listview(vec![1, 2, 99, 4], vec![0, 1], vec![3, 3], validity);
        assert_ne!(v1, v3, "Element used by valid list must match");
    }

    #[test]
    fn test_listview_eq_different_offsets_sizes() {
        // Same elements but different offsets at valid positions
        let v1 = make_listview(vec![1, 2, 3, 4], vec![0, 2], vec![2, 2], Mask::new_true(2));
        let v2 = make_listview(
            vec![1, 2, 3, 4],
            vec![0, 1], // different offset for list1
            vec![2, 2],
            Mask::new_true(2),
        );
        assert_ne!(v1, v2, "Different offsets at valid positions");

        // Different sizes at valid positions
        let v3 = make_listview(
            vec![1, 2, 3, 4],
            vec![0, 2],
            vec![2, 1], // different size for list1
            Mask::new_true(2),
        );
        assert_ne!(v1, v3, "Different sizes at valid positions");
    }

    #[test]
    fn test_listview_eq_different_validity() {
        let v1 = make_listview(vec![1, 2, 3, 4], vec![0, 2], vec![2, 2], Mask::new_true(2));
        let v2 = make_listview(
            vec![1, 2, 3, 4],
            vec![0, 2],
            vec![2, 2],
            Mask::from_indices(2, vec![0]), // only first list valid
        );
        assert_ne!(v1, v2, "Different validity patterns");
    }

    #[test]
    fn test_listview_eq_empty() {
        let v1 = make_listview(vec![], vec![], vec![], Mask::new_true(0));
        let v2 = make_listview(vec![], vec![], vec![], Mask::new_true(0));
        assert_eq!(v1, v2);
    }
}
