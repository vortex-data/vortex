// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVector`].

use vortex_mask::{Mask, MaskMut};

use crate::null::NullVector;
use crate::{Cow, VectorOps};

/// A mutable vector of null values.
///
/// Since a "null" value does not require any data storage, the nulls are stored internally with a
/// single `length` counter.
///
/// The immutable equivalent of this type is [`NullVector`].
#[derive(Debug)]
pub struct NullVector {
    /// The total number of nulls.
    pub(super) len: usize,
    /// The validity mask. We only store this in order to implement the
    /// [`validity()`](Self::validity) method.
    pub(super) validity: Cow<Mask>,
}

impl NullVector {
    /// Creates a new mutable vector of nulls with the given length.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            validity: MaskMut::new_false(len),
        }
    }
}

impl VectorOps for NullVector {
    type Immutable = NullVector;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &MaskMut {
        &self.validity
    }

    fn capacity(&self) -> usize {
        usize::MAX
    }

    fn reserve(&mut self, _additional: usize) {
        // We do not allocate memory for `NullVector`, so this is a no-op.
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    fn extend_from_vector(&mut self, other: &NullVector) {
        self.len += other.len;
    }

    fn append_nulls(&mut self, n: usize) {
        self.len += n;
    }

    fn freeze(self) -> NullVector {
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
        NullVector {
            len: new_len,
            validity: MaskMut::new_false(new_len),
        }
    }

    fn unsplit(&mut self, other: Self) {
        self.len += other.len;
    }
}
