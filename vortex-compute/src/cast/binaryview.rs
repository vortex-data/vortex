// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::binaryview::BinaryViewScalar;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::binaryview::BinaryViewVector;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl<T: BinaryViewType> Cast for BinaryViewVector<T> {
    type Output = Vector;

    /// Casts to Utf8 or Binary (identity cast with compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same type with compatible nullability.
            dt if T::matches_dtype(dt) && (dt.is_nullable() || self.validity().all_true()) => {
                Ok(self.clone().into())
            }
            // Cross-cast between Utf8 and Binary is not supported.
            DType::Utf8(_) | DType::Binary(_) => {
                vortex_bail!(
                    "Cannot cast BinaryViewVector to {} (cross-cast not supported)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast BinaryViewVector to {}", target_dtype);
            }
        }
    }
}

impl<T: BinaryViewType> Cast for BinaryViewScalar<T> {
    type Output = Scalar;

    /// Casts to Utf8 or Binary (identity cast with compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same type with compatible nullability.
            dt if T::matches_dtype(dt) && (dt.is_nullable() || self.is_valid()) => {
                Ok(self.clone().into())
            }
            // Cross-cast between Utf8 and Binary is not supported.
            DType::Utf8(_) | DType::Binary(_) => {
                vortex_bail!(
                    "Cannot cast BinaryViewScalar to {} (cross-cast not supported)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast BinaryViewScalar to {}", target_dtype);
            }
        }
    }
}
