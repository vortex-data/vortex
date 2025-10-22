// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVectorMut`].

use super::NullVector;
use crate::VectorMutOps;

/// A mutable vector of null values.
///
/// Since a "null" value does not require any data storage, the nulls are stored internally with a
/// single `length` counter.
///
/// The immutable equivalent of this type is [`NullVector`].
#[derive(Debug, Clone, Copy)]
pub struct NullVectorMut {
    /// The total number of nulls.
    pub(super) len: usize,
}

impl NullVectorMut {
    /// Creates a new mutable vector of nulls with the given length.
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl VectorMutOps for NullVectorMut {
    type Immutable = NullVector;

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

    fn append_nulls(&mut self, n: usize) {
        self.len += n;
    }

    fn freeze(self) -> Self::Immutable {
        NullVector::new(self.len)
    }

    fn split_off(&mut self, at: usize) -> Self {
        assert!(
            at <= self.capacity(),
            "split_off out of bounds: {:?} <= {:?}",
            at,
            self.capacity(),
        );

        let new_len = self.len.saturating_sub(at);
        self.len = std::cmp::min(self.len, at);
        NullVectorMut { len: new_len }
    }

    fn unsplit(&mut self, other: Self) {
        self.len += other.len;
    }
}
