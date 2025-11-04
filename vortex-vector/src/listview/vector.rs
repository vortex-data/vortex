// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVector`].

use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use super::ListViewVectorMut;
use crate::Vector;
use crate::ops::{VectorMutOps, VectorOps};
use crate::primitive::PrimitiveVector;

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
    pub(super) offsets: PrimitiveVector,

    /// Sizes (lengths) of each list.
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
    /// TODO
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
    /// TODO
    pub fn try_new(
        _elements: Arc<Vector>,
        _offsets: PrimitiveVector,
        _sizes: PrimitiveVector,
        _validity: Mask,
    ) -> VortexResult<Self> {
        todo!()
    }

    /// Creates a new [`ListViewVector`] without validation.
    ///
    /// # Safety
    ///
    /// TODO
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
}
