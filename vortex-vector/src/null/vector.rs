// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVector`].

use vortex_dtype::Nullability;

use crate::{NullVectorMut, VectorOps};

/// An immutable vector of null values.
///
/// Since a "null" value does not require any data storage, the nulls are stored internally with a
/// single `length` counter.
///
/// The mutable equivalent of this type is [`NullVectorMut`].
#[derive(Debug, Clone, Copy)]
pub struct NullVector {
    pub(super) len: usize,
}

impl NullVector {
    /// Creates a new immutable vector of nulls with the given length.
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl VectorOps for NullVector {
    type Mutable = NullVectorMut;

    fn nullability(&self) -> Nullability {
        Nullability::Nullable
    }

    fn len(&self) -> usize {
        self.len
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        Ok(NullVectorMut::new(self.len))
    }
}
