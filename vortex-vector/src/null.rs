// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::ops::VectorOps;
use crate::{NullVectorMut, Vector};

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

impl From<NullVector> for Vector {
    fn from(v: NullVector) -> Self {
        Self::Null(v)
    }
}

impl VectorOps for NullVector {
    type Mutable = NullVectorMut;

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> &DType {
        &DType::Null
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        Ok(NullVectorMut::new(self.len))
    }
}
