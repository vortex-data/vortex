// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_mask::MaskMut;

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorMutOps;
use crate::VectorOps;
use crate::struct_::StructVector;
use crate::struct_::StructVectorMut;

/// Represents a struct scalar value.
///
/// The inner value is a [`StructVector`] with length 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructScalar(StructVector);

impl StructScalar {
    /// Creates a new [`StructScalar`] from a length-1 [`StructVector`].
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

    /// Returns an iterator over the field scalars of the struct.
    pub fn fields(&self) -> impl Iterator<Item = Scalar> {
        self.0.fields().iter().map(|f| f.scalar_at(0))
    }
}

impl StructScalar {
    /// Creates a zero (default fields) struct scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not a [`DType::Struct`].
    pub fn zero(dtype: &DType) -> Self {
        if !matches!(dtype, DType::Struct(..)) {
            vortex_panic!("Expected Struct dtype, got {}", dtype);
        }

        let mut vec = VectorMut::with_capacity(dtype, 1);
        vec.append_zeros(1);
        vec.freeze().scalar_at(0).into_struct()
    }

    /// Creates a null struct scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not a nullable [`DType::Struct`].
    pub fn null(dtype: &DType) -> Self {
        match dtype {
            DType::Struct(_, n) if n.is_nullable() => {}
            DType::Struct(..) => vortex_panic!("Expected nullable Struct dtype, got {}", dtype),
            _ => vortex_panic!("Expected Struct dtype, got {}", dtype),
        }

        let mut vec = VectorMut::with_capacity(dtype, 1);
        vec.append_nulls(1);
        vec.freeze().scalar_at(0).into_struct()
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
