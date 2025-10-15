// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::ops::VectorMutOps;
use crate::{NullVector, VectorMut};

/// A mutable vector of null values.
pub struct NullVectorMut {
    pub(super) len: usize,
}

impl NullVectorMut {
    /// Creates a new `NullVectorMut` with the given length.
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl From<NullVectorMut> for VectorMut {
    fn from(v: NullVectorMut) -> Self {
        Self::Null(v)
    }
}

impl VectorMutOps for NullVectorMut {
    type Immutable = NullVector;

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> &DType {
        &DType::Null
    }

    fn capacity(&self) -> usize {
        usize::MAX
    }

    fn reserve(&mut self, _additional: usize) {}

    fn split_off(&mut self, at: usize) -> Self {
        let new_len = self.len - at;
        self.len = at;
        NullVectorMut { len: new_len }
    }

    fn unsplit(&mut self, other: Self) {
        self.len += other.len;
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        self.len += other.len;
    }

    fn freeze(self) -> Self::Immutable {
        NullVector::new(self.len)
    }
}
