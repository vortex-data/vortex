// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVectorMut`].

use crate::{FixedSizeListVector, VectorMutOps};

/// TODO docs.
#[derive(Debug, Clone)]
pub struct FixedSizeListVectorMut;

impl VectorMutOps for FixedSizeListVectorMut {
    type Immutable = FixedSizeListVector;

    fn len(&self) -> usize {
        todo!()
    }

    fn capacity(&self) -> usize {
        todo!()
    }

    fn reserve(&mut self, _additional: usize) {
        todo!()
    }

    fn extend_from_vector(&mut self, _other: &Self::Immutable) {
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
