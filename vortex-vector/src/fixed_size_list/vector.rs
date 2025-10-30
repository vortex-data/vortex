// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVector`].

use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::Mask;

use crate::{FixedSizeListVectorMut, Vector, VectorOps};

/// An immutable vector of fixed-size lists.
///
/// `FixedSizeListVector` can be considered a borrowed / frozen version of
/// [`FixedSizeListVectorMut`], which is created via the [`freeze`](crate::VectorMutOps::freeze)
/// method.
///
/// See the documentation for [`FixedSizeListVectorMut`] for more information.
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

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        let len = self.len;
        let list_size = self.list_size;

        let elements = match Arc::try_unwrap(self.elements) {
            Ok(elements) => elements,
            Err(elements) => return Err(FixedSizeListVector { elements, ..self }),
        };

        let validity = match self.validity.try_into_mut() {
            Ok(validity) => validity,
            Err(validity) => {
                return Err(FixedSizeListVector {
                    elements: Arc::new(elements),
                    list_size,
                    validity,
                    len,
                });
            }
        };

        match elements.try_into_mut() {
            Ok(mutable_elements) => Ok(FixedSizeListVectorMut {
                elements: Box::new(mutable_elements),
                list_size,
                validity,
                len,
            }),
            Err(elements) => Err(FixedSizeListVector {
                elements: Arc::new(elements),
                list_size,
                validity: validity.freeze(),
                len,
            }),
        }
    }
}
