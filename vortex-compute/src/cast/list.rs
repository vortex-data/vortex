// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::listview::ListViewScalar;
use vortex_vector::listview::ListViewVector;
use vortex_vector::vector_matches_dtype;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl Cast for ListViewVector {
    type Output = Vector;

    /// Casts to List (identity with same element dtype and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same element dtype and compatible nullability.
            DType::List(element_dtype, n)
                if vector_matches_dtype(self.elements(), element_dtype)
                    && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            DType::List(..) => {
                vortex_bail!(
                    "Cannot cast ListViewVector to {} (incompatible element type or nullability)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast ListViewVector to {}", target_dtype);
            }
        }
    }
}

impl Cast for ListViewScalar {
    type Output = Scalar;

    /// Casts to List (identity with same element dtype and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same element dtype and compatible nullability.
            DType::List(element_dtype, n)
                if vector_matches_dtype(self.value().elements(), element_dtype)
                    && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            DType::List(..) => {
                vortex_bail!(
                    "Cannot cast ListViewScalar to {} (incompatible element type or nullability)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast ListViewScalar to {}", target_dtype);
            }
        }
    }
}
