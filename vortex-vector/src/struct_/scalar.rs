// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::struct_::StructVector;
use crate::{Scalar, ScalarOps, VectorMut, VectorOps};

/// Represents a struct scalar value.
///
/// The inner value is a StructVector with length 1.
#[derive(Debug)]
pub struct StructScalar(StructVector);

impl StructScalar {
    /// Creates a new StructScalar from a length-1 StructVector.
    ///
    /// # Panics
    ///
    /// Panics if the input vector does not have length 1.
    pub fn new(vector: StructVector) -> Self {
        assert_eq!(vector.len(), 1);
        Self(vector)
    }

    /// Returns the inner length-1 vector representing the struct scalar.
    pub fn value(&self) -> &StructVector {
        &self.0
    }
}

impl ScalarOps for StructScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn repeat(&self, _n: usize) -> VectorMut {
        todo!()
    }
}

impl From<StructScalar> for Scalar {
    fn from(val: StructScalar) -> Self {
        Scalar::Struct(val)
    }
}
