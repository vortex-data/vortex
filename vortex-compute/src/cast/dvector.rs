// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::NativeDecimalType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::decimal::DScalar;
use vortex_vector::decimal::DVector;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl<D: NativeDecimalType> Cast for DVector<D> {
    type Output = Vector;

    /// Casts to Decimal (identity with same precision/scale and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same scale, equal or larger precision, and compatible nullability.
            DType::Decimal(ddt, n)
                if ddt.precision() == self.precision()
                    && ddt.scale() == self.scale()
                    && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            // TODO(connor): cast to different precision/scale
            DType::Decimal(..) => {
                vortex_bail!(
                    "Casting DVector to {} with different precision/scale not yet implemented",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast DVector to {}", target_dtype);
            }
        }
    }
}

impl<D: NativeDecimalType> Cast for DScalar<D> {
    type Output = Scalar;

    /// Casts to Decimal (identity with same precision/scale and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same precision, scale, and compatible nullability.
            DType::Decimal(ddt, n)
                if ddt.precision() == self.precision()
                    && ddt.scale() == self.scale()
                    && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            // TODO(connor): cast to different precision/scale
            DType::Decimal(..) => {
                vortex_bail!(
                    "Casting DScalar to {} with different precision/scale not yet implemented",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast DScalar to {}", target_dtype);
            }
        }
    }
}
