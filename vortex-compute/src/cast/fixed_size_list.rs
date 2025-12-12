// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::fixed_size_list::FixedSizeListScalar;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::vector_matches_dtype;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl Cast for FixedSizeListVector {
    type Output = Vector;

    /// Casts to FixedSizeList (identity with same element dtype, size, and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same element dtype, size, and compatible nullability.
            DType::FixedSizeList(element_dtype, size, n)
                if *size == self.list_size()
                    && vector_matches_dtype(self.elements(), element_dtype)
                    && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            DType::FixedSizeList(..) => {
                vortex_bail!(
                    "Cannot cast FixedSizeListVector to {} (incompatible element type or size)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast FixedSizeListVector to {}", target_dtype);
            }
        }
    }
}

impl Cast for FixedSizeListScalar {
    type Output = Scalar;

    /// Casts to FixedSizeList (identity with same element dtype, size, and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same element dtype, size, and compatible nullability.
            // We check by verifying the scalar's underlying vector matches the target dtype.
            DType::FixedSizeList(element_dtype, size, n)
                if *size == self.value().list_size()
                    && vector_matches_dtype(self.value().elements(), element_dtype)
                    && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            DType::FixedSizeList(..) => {
                vortex_bail!(
                    "Cannot cast FixedSizeListScalar to {} (incompatible element type, size, or nullability)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast FixedSizeListScalar to {}", target_dtype);
            }
        }
    }
}
