// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::Scalar;
use crate::ScalarOps;
use crate::VectorMut;
use crate::VectorMutOps;
use crate::VectorOps;
use crate::fixed_size_list::FixedSizeListVector;

/// A scalar value for fixed-size list types.
///
/// The inner value is a length-1 [`FixedSizeListVector`].
// NOTE(ngates): the reason we don't hold Option<Vector> representing the elements is that we
//  wouldn't be able to go back to a vector using "repeat".
#[derive(Clone, Debug, PartialEq)]
pub struct FixedSizeListScalar(FixedSizeListVector);

impl FixedSizeListScalar {
    /// Create a new [`FixedSizeListScalar`] from a length-1 [`FixedSizeListVector`].
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

impl FixedSizeListScalar {
    /// Creates a zero (default elements) fixed-size list scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not a [`DType::FixedSizeList`].
    pub fn zero(dtype: &DType) -> Self {
        if !matches!(dtype, DType::FixedSizeList(..)) {
            vortex_panic!("Expected FixedSizeList dtype, got {}", dtype);
        }

        let mut vec = VectorMut::with_capacity(dtype, 1);
        vec.append_zeros(1);
        vec.freeze().scalar_at(0).into_fixed_size_list()
    }

    /// Creates a null fixed-size list scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not a nullable [`DType::FixedSizeList`].
    pub fn null(dtype: &DType) -> Self {
        match dtype {
            DType::FixedSizeList(_, _, n) if n.is_nullable() => {}
            DType::FixedSizeList(..) => {
                vortex_panic!("Expected nullable FixedSizeList dtype, got {}", dtype)
            }
            _ => vortex_panic!("Expected FixedSizeList dtype, got {}", dtype),
        }

        let mut vec = VectorMut::with_capacity(dtype, 1);
        vec.append_nulls(1);
        vec.freeze().scalar_at(0).into_fixed_size_list()
    }
}

impl ScalarOps for FixedSizeListScalar {
    fn is_valid(&self) -> bool {
        self.0.validity().value(0)
    }

    fn mask_validity(&mut self, mask: bool) {
        if !mask {
            self.0.mask_validity(&Mask::new_false(1))
        }
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
