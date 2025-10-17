// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability};

use crate::{NullVectorMut, VectorOps};

/// An immutable vector of null values.
pub struct NullVector {
    pub(super) len: usize,
}

impl NullVector {
    /// Creates a new `NullVector` with the given length.
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl VectorOps for NullVector {
    type Mutable = NullVectorMut;

    fn nullability(&self) -> Nullability {
        Nullability::Nullable
    }

    fn dtype(&self) -> DType {
        DType::Null
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
