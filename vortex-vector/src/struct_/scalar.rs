// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorOps;
use crate::struct_::StructVector;
use crate::struct_::StructVectorMut;
use vortex_mask::Mask;
use vortex_mask::MaskMut;

/// Represents a struct scalar value.
///
/// The inner value is a StructVector with length 1.
#[derive(Clone, Debug)]
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

    /// Returns the nth field scalar of the struct.
    pub fn field(&self, field_idx: usize) -> Scalar {
        self.0.fields()[field_idx].scalar_at(0)
    }
}

impl ScalarOps for StructScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn mask_validity(&mut self, mask: bool) {
        if !mask {
            self.0.mask_validity(&Mask::new_false(1))
        }
    }

    fn repeat(&self, n: usize) -> VectorMut {
        let fields = self
            .0
            .fields()
            .iter()
            .map(|f| f.scalar_at(0).repeat(n))
            .collect();
        let validity = MaskMut::new(n, self.is_valid());
        VectorMut::Struct(StructVectorMut::new(fields, validity))
    }
}

impl From<StructScalar> for Scalar {
    fn from(val: StructScalar) -> Self {
        Scalar::Struct(val)
    }
}
