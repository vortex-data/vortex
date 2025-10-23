// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVectorMut`].

use vortex_mask::MaskMut;

use crate::{StructVector, VectorMutOps};

/// A mutable vector of struct values (values with named fields).
///
/// TODO docs.
#[derive(Debug, Clone)]
pub struct StructVectorMut {
    /// The validity mask (where `true` represents an element is **not** null).
    pub(super) _validity: MaskMut,
}

impl StructVectorMut {
    /// Creates a new mutable boolean vector with the given `capacity`.
    pub fn with_capacity(_capacity: usize) -> Self {
        todo!()
    }
}

impl VectorMutOps for StructVectorMut {
    type Immutable = StructVector;

    fn len(&self) -> usize {
        todo!()
    }

    fn capacity(&self) -> usize {
        todo!()
    }

    fn reserve(&mut self, _additional: usize) {
        todo!()
    }

    fn extend_from_vector(&mut self, _other: &StructVector) {
        todo!()
    }

    fn append_nulls(&mut self, _n: usize) {
        todo!()
    }

    fn freeze(self) -> Self::Immutable {
        todo!()
    }

    fn split_off(&mut self, _at: usize) -> Self {
        todo!()
    }

    fn unsplit(&mut self, _other: Self) {
        todo!()
    }
}
