// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVector`].

use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use super::{ListViewScalar, ListViewVectorMut};
use crate::primitive::PrimitiveVector;
use crate::vector_ops::{VectorMutOps, VectorOps};
use crate::{Scalar, Vector, match_each_integer_pvector};

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
    pub fn elements(&self) -> &Vector {
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

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn scalar_at(&self, index: usize) -> Scalar {
        assert!(index < self.len());
        ListViewScalar::new(self.slice(index..index + 1)).into()
    }

    fn slice(&self, _range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        todo!()
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
#[allow(clippy::cognitive_complexity, clippy::cast_possible_truncation)]
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
