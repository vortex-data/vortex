// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PScalar;
use vortex_vector::primitive::PVector;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

impl<T: NativePType> Cast for PVector<T> {
    type Output = Vector;

    /// Casts to Primitive (same PType identity).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // We're already the correct PType, and we have compatible nullability.
            DType::Primitive(target_ptype, n)
                if *target_ptype == T::PTYPE && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            // We're not the correct PType, but we do have compatible nullability.
            DType::Primitive(target_ptype, n) if n.is_nullable() || self.validity().all_true() => {
                vortex_bail!(
                    "Casting PVector from PType {} to PType {} not yet implemented",
                    T::PTYPE,
                    target_ptype
                );
            }
            _ => {
                vortex_bail!("Cannot cast PVector<{}> to {}", T::PTYPE, target_dtype);
            }
        }
    }
}

impl<T: NativePType> Cast for PScalar<T> {
    type Output = Scalar;

    /// Casts to Primitive (same PType identity).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // We're already the correct PType, and we have compatible nullability.
            DType::Primitive(target_ptype, n)
                if *target_ptype == T::PTYPE && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            // We're not the correct PType, but we do have compatible nullability.
            DType::Primitive(target_ptype, n) if n.is_nullable() || self.is_valid() => {
                vortex_bail!(
                    "Casting PScalar from PType {} to PType {} not yet implemented",
                    T::PTYPE,
                    target_ptype
                );
            }
            _ => {
                vortex_bail!("Cannot cast PScalar<{}> to {}", T::PTYPE, target_dtype);
            }
        }
    }
}
