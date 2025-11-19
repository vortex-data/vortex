// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVector`].

use std::fmt::Debug;
use std::ops::RangeBounds;

use vortex_mask::Mask;

use crate::null::{NullScalar, NullVectorMut};
use crate::{Scalar, VectorOps};

/// An immutable vector of null values.
///
/// Since a "null" value does not require any data storage, the nulls are stored internally with a
/// single `length` counter.
///
/// The mutable equivalent of this type is [`NullVectorMut`].
#[derive(Debug, Clone)]
pub struct NullVector {
    /// The total number of nulls.
    pub(super) len: usize,
    /// The validity mask. We only store this in order to implement the
    /// [`validity()`](Self::validity) method.
    pub(super) validity: Mask,
}

impl NullVector {
    /// Creates a new immutable vector of nulls with the given length.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            validity: Mask::AllFalse(len),
        }
    }
}

impl VectorOps for NullVector {
    type Mutable = NullVectorMut;

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn scalar_at(&self, index: usize) -> Scalar {
        assert!(index < self.len, "Index out of bounds in `NullVector`");
        NullScalar.into()
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        let len = crate::vector_ops::range_bounds_to_len(range, self.len());
        Self::new(len)
    }

    fn clear(&mut self) {
        self.len = 0;
        self.validity = Mask::AllFalse(0);
    }

    fn try_into_mut(self) -> Result<NullVectorMut, Self> {
        Ok(NullVectorMut::new(self.len))
    }

    fn into_mut(self) -> NullVectorMut {
        NullVectorMut::new(self.len)
    }
}
