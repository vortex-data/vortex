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
use crate::fixed_size_list::FixedSizeListVector;
use crate::fixed_size_list::FixedSizeListVectorMut;

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

    fn repeat(&self, n: usize) -> VectorMut {
        if n == 0 {
            // Return an empty vector with the correct structure
            let list_size = self.0.list_size();
            let scalar_elements = self.0.elements();
            let elements = scalar_elements.slice(0..0).into_mut();
            let validity = MaskMut::new_true(0);
            return unsafe {
                VectorMut::FixedSizeList(FixedSizeListVectorMut::new_unchecked(
                    Box::new(elements),
                    list_size,
                    validity,
                ))
            };
        }

        let list_size = self.0.list_size();

        // Get the elements from the scalar's inner length-1 vector and repeat them
        // Clone the inner Vector from the Arc
        let scalar_elements = self.0.elements();

        let mut elements = scalar_elements.as_ref().clone();
        elements.clear();
        let mut elements = elements.into_mut();
        elements.reserve((n - 1) * list_size as usize);

        if self.is_null() {
            elements.append_zeros((n - 1) * list_size as usize);
            let validity = MaskMut::new_false(n);
            return unsafe {
                VectorMut::FixedSizeList(FixedSizeListVectorMut::new_unchecked(
                    Box::new(elements),
                    list_size,
                    validity,
                ))
            };
        }

        // Repeat the elements n-1 more times (we already have 1 copy)
        for _ in 1..n {
            elements.extend_from_vector(scalar_elements.as_ref());
        }

        // SAFETY: We've repeated the elements n times, so elements.len() == n * list_size
        unsafe {
            VectorMut::FixedSizeList(FixedSizeListVectorMut::new_unchecked(
                Box::new(elements),
                list_size,
                MaskMut::new_true(n),
            ))
        }
    }
}

impl From<FixedSizeListScalar> for Scalar {
    fn from(val: FixedSizeListScalar) -> Self {
        Scalar::FixedSizeList(val)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_mask::Mask;

    use super::*;
    use crate::Vector;
    use crate::fixed_size_list::FixedSizeListVector;
    use crate::primitive::PVectorMut;

    #[test]
    fn test_repeat_valid_scalar() {
        // Create a FSL with elements [1, 2, 3] (list_size = 3)
        let elements: Vector = PVectorMut::<i32>::from_iter([1, 2, 3]).freeze().into();
        let validity = Mask::new_true(1);
        let fsl = FixedSizeListVector::new(Arc::new(elements), 3, validity);

        let scalar = FixedSizeListScalar::new(fsl);
        assert!(scalar.is_valid());

        // Repeat 4 times
        let repeated = scalar.repeat(4).freeze();
        assert_eq!(repeated.len(), 4);

        // Check validity - all should be valid
        assert_eq!(repeated.validity().true_count(), 4);

        // Freeze and check the elements
        let fsl_vec = repeated.as_fixed_size_list();
        assert_eq!(fsl_vec.len(), 4);
        assert_eq!(fsl_vec.list_size(), 3);

        // Elements should be [1,2,3, 1,2,3, 1,2,3, 1,2,3]
        let elements = fsl_vec.elements();
        assert_eq!(elements.len(), 12);
    }

    #[test]
    fn test_repeat_null_scalar() {
        // Create a null FSL scalar
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            3,
            Nullability::Nullable,
        );
        let scalar = FixedSizeListScalar::null(&dtype);
        assert!(!scalar.is_valid());

        // Repeat 3 times
        let repeated = scalar.repeat(3).freeze();
        assert_eq!(repeated.len(), 3);

        // Check validity - all should be null
        assert_eq!(repeated.validity().true_count(), 0);
    }

    #[test]
    fn test_repeat_zero() {
        // Create a valid FSL scalar
        let elements: Vector = PVectorMut::<i32>::from_iter([1, 2]).freeze().into();
        let validity = Mask::new_true(1);
        let fsl = FixedSizeListVector::new(Arc::new(elements), 2, validity);

        let scalar = FixedSizeListScalar::new(fsl);

        // Repeat 0 times - should return empty vector
        let repeated = scalar.repeat(0);
        assert_eq!(repeated.len(), 0);

        let frozen = repeated.freeze();
        let fsl_vec = frozen.as_fixed_size_list();
        assert_eq!(fsl_vec.len(), 0);
        assert_eq!(fsl_vec.list_size(), 2);
        assert_eq!(fsl_vec.elements().len(), 0);
    }

    #[test]
    fn test_repeat_one() {
        // Create a FSL with elements [10, 20]
        let elements: Vector = PVectorMut::<i32>::from_iter([10, 20]).freeze().into();
        let validity = Mask::new_true(1);
        let fsl = FixedSizeListVector::new(Arc::new(elements), 2, validity);

        let scalar = FixedSizeListScalar::new(fsl);

        // Repeat 1 time - should be same as original
        let repeated = scalar.repeat(1);
        assert_eq!(repeated.len(), 1);

        let frozen = repeated.freeze();
        let fsl_vec = frozen.as_fixed_size_list();
        assert_eq!(fsl_vec.len(), 1);
        assert_eq!(fsl_vec.elements().len(), 2);
    }
}
