// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability};

use super::NullVector;
use crate::VectorMutOps;

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

impl VectorMutOps for NullVectorMut {
    type Immutable = NullVector;

    fn nullability(&self) -> Nullability {
        Nullability::Nullable
    }

    fn dtype(&self) -> DType {
        DType::Null
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        usize::MAX
    }

    fn reserve(&mut self, _additional: usize) {
        // We do not allocate memory for `NullVector`, so this is a no-op.
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        self.len += other.len;
    }

    fn freeze(self) -> Self::Immutable {
        NullVector::new(self.len)
    }

    fn split_off(&mut self, at: usize) -> Self {
        // TODO(connor): This is wrong (https://docs.rs/bytes/latest/src/bytes/bytes_mut.rs.html#320-335)
        assert!(
            at <= self.capacity(),
            "split_off out of bounds: {:?} <= {:?}",
            at,
            self.capacity(),
        );

        let new_len = self.len - at;
        self.len = at;
        NullVectorMut { len: new_len }
    }

    fn unsplit(&mut self, other: Self) {
        self.len += other.len;
    }
}
