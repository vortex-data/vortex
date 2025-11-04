// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVectorMut`].

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use super::ListViewVector;
use crate::ops::VectorMutOps;
use crate::primitive::PrimitiveVectorMut;
use crate::{VectorMut, match_each_integer_pvector_mut};

/// A mutable vector of variable-width lists.
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
/// - `elements`: The child vector of all list elements, stored as a [`Box<VectorMut>`].
/// - `offsets`: A [`PrimitiveVectorMut`] containing the starting offset of each list in the
///   `elements` vector.
/// - `sizes`: A [`PrimitiveVectorMut`] containing the size (number of elements) of each list.
/// - `validity`: A [`MaskMut`] indicating which lists are null.
#[derive(Debug, Clone)]
pub struct ListViewVectorMut {
    /// The mutable child vector of elements.
    pub(super) elements: Box<VectorMut>,

    /// Mutable offsets for each list into the elements array.
    ///
    /// Offsets are always integers, and always non-negative (even if the type is signed).
    pub(super) offsets: PrimitiveVectorMut,

    /// Mutable sizes (lengths) of each list.
    ///
    /// Sizes are always integers, and always non-negative (even if the type is signed).
    pub(super) sizes: PrimitiveVectorMut,

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

impl ListViewVectorMut {
    /// Creates a new [`ListViewVectorMut`] from its components.
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
        elements: Box<VectorMut>,
        offsets: PrimitiveVectorMut,
        sizes: PrimitiveVectorMut,
        validity: MaskMut,
    ) -> Self {
        Self::try_new(elements, offsets, sizes, validity)
            .vortex_expect("Failed to create `ListViewVectorMut`")
    }

    /// Attempts to create a new [`ListViewVectorMut`] from its components.
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
        elements: Box<VectorMut>,
        offsets: PrimitiveVectorMut,
        sizes: PrimitiveVectorMut,
        validity: MaskMut,
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

    /// Creates a new [`ListViewVectorMut`] without validation.
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
        elements: Box<VectorMut>,
        offsets: PrimitiveVectorMut,
        sizes: PrimitiveVectorMut,
        validity: MaskMut,
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

    /// Creates a new [`ListViewVectorMut`] with the specified capacity.
    ///
    /// TODO figure out how to set offsets and sizes type?
    pub fn with_capacity(_element_dtype: &DType, _capacity: usize) -> Self {
        todo!("ListViewVectorMut::with_capacity")
    }

    /// Decomposes the [`ListViewVectorMut`] into its constituent parts (child elements, offsets,
    /// sizes, and validity).
    pub fn into_parts(
        self,
    ) -> (
        Box<VectorMut>,
        PrimitiveVectorMut,
        PrimitiveVectorMut,
        MaskMut,
    ) {
        (self.elements, self.offsets, self.sizes, self.validity)
    }

    /// Returns a reference to the elements vector.
    pub fn elements(&self) -> &VectorMut {
        &self.elements
    }

    /// Returns a reference to the offsets vector.
    pub fn offsets(&self) -> &PrimitiveVectorMut {
        &self.offsets
    }

    /// Returns a reference to the sizes vector.
    pub fn sizes(&self) -> &PrimitiveVectorMut {
        &self.sizes
    }
}

impl VectorMutOps for ListViewVectorMut {
    type Immutable = ListViewVector;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        debug_assert!(
            self.offsets.capacity() <= self.sizes.capacity(),
            "the capacity of the sizes was somehow less than the offsets"
        );

        self.offsets.capacity()
    }

    fn reserve(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.sizes.reserve(additional);
        self.elements.reserve(additional * 2); // Sane default TODO
        self.validity.reserve(additional);
    }

    fn extend_from_vector(&mut self, _other: &ListViewVector) {
        todo!()
    }

    fn append_nulls(&mut self, _n: usize) {
        todo!("Need to figure out what the 'value' of nulls are for list view vectors")
    }

    fn freeze(self) -> ListViewVector {
        ListViewVector {
            offsets: self.offsets.freeze(),
            sizes: self.sizes.freeze(),
            elements: Arc::new(self.elements.freeze()),
            validity: self.validity.freeze(),
            len: self.len,
        }
    }

    fn split_off(&mut self, _at: usize) -> Self {
        todo!()
    }

    fn unsplit(&mut self, _other: Self) {
        todo!()
    }
}

// TODO(connor): It would be better to separate everything inside the macros into its own function,
// but that would require adding another macro that sets a type `$type` to be used by the caller.
/// Checks that all views are `<= elements_len`.
#[allow(clippy::cognitive_complexity, clippy::cast_possible_truncation)]
fn validate_views_bound(
    elements_len: usize,
    offsets: &PrimitiveVectorMut,
    sizes: &PrimitiveVectorMut,
) -> VortexResult<()> {
    let len = offsets.len();

    match_each_integer_pvector_mut!(&offsets, |offsets_vector| {
        match_each_integer_pvector_mut!(&sizes, |sizes_vector| {
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
