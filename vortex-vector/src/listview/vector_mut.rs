// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`ListViewVectorMut`].

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::MaskMut;

use super::ListViewVector;
use crate::VectorMut;
use crate::ops::VectorMutOps;
use crate::primitive::PrimitiveVectorMut;

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
    pub(super) offsets: PrimitiveVectorMut,

    /// Mutable sizes (lengths) of each list.
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
    /// TODO
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
    /// TODO
    pub fn try_new(
        _elements: Box<VectorMut>,
        _offsets: PrimitiveVectorMut,
        _sizes: PrimitiveVectorMut,
        _validity: MaskMut,
    ) -> VortexResult<Self> {
        todo!()
    }

    /// Creates a new [`ListViewVectorMut`] without validation.
    ///
    /// # Safety
    ///
    /// TODO
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
        todo!()
    }

    fn reserve(&mut self, _additional: usize) {
        todo!()
    }

    fn extend_from_vector(&mut self, _other: &ListViewVector) {
        todo!()
    }

    fn append_nulls(&mut self, _n: usize) {
        todo!()
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
