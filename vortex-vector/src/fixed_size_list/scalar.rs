// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorOps;
use crate::fixed_size_list::FixedSizeListVector;

/// A scalar value for fixed-size list types.
///
/// The inner value is a length-1 fsl vector.
// NOTE(ngates): the reason we don't hold Option<Vector> representing the elements is that we
//  wouldn't be able to go back to a vector using "repeat".
#[derive(Debug)]
pub struct FixedSizeListScalar(FixedSizeListVector);

impl FixedSizeListScalar {
    /// Create a new FixedSizeListScalar from a length-1 FixedSizeListVector.
    ///
    /// # Panics
    ///
    /// Panics if the input vector does not have length 1.
    pub fn new(vector: FixedSizeListVector) -> Self {
        assert_eq!(vector.len(), 1);
        Self(vector)
    }

    /// Returns the inner length-1 vector representing the fixed-size list scalar.
    pub fn value(&self) -> &FixedSizeListVector {
        &self.0
    }
}

impl ScalarOps for FixedSizeListScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn repeat(&self, _n: usize) -> VectorMut {
        // TODO(ngates): add "repeat(n)" to the vector ops trait
        todo!()
    }
}

impl From<FixedSizeListScalar> for Scalar {
    fn from(val: FixedSizeListScalar) -> Self {
        Scalar::FixedSizeList(val)
    }
}
