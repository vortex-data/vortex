// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVector`].

use vortex_mask::Mask;

use crate::VectorOps;
use crate::null::NullVectorMut;

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

    fn try_into_mut(self) -> Result<NullVectorMut, Self>
    where
        Self: Sized,
    {
        Ok(NullVectorMut::new(self.len))
    }
}
