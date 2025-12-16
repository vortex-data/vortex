// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::struct_::StructScalar;
use vortex_vector::struct_::StructVector;
use vortex_vector::vector_matches_dtype;

use crate::cast::Cast;
use crate::cast::try_cast_scalar_common;
use crate::cast::try_cast_vector_common;

/// Checks if a struct vector's fields match the given struct fields dtype.
fn struct_fields_match(vector: &StructVector, fields: &vortex_dtype::StructFields) -> bool {
    if fields.nfields() != vector.fields().len() {
        return false;
    }
    for (field_dtype, field_vector) in fields.fields().zip(vector.fields().iter()) {
        if !vector_matches_dtype(field_vector, &field_dtype) {
            return false;
        }
    }
    true
}

impl Cast for StructVector {
    type Output = Vector;

    /// Casts to Struct (identity with same fields and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if let Some(result) = try_cast_vector_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same fields and compatible nullability.
            DType::Struct(fields, n)
                if struct_fields_match(self, fields)
                    && (n.is_nullable() || self.validity().all_true()) =>
            {
                Ok(self.clone().into())
            }
            DType::Struct(..) => {
                vortex_bail!(
                    "Cannot cast StructVector to {} (incompatible fields or nullability)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast StructVector to {}", target_dtype);
            }
        }
    }
}

impl Cast for StructScalar {
    type Output = Scalar;

    /// Casts to Struct (identity with same fields and compatible nullability).
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(result) = try_cast_scalar_common(self, target_dtype)? {
            return Ok(result);
        }

        match target_dtype {
            // Identity cast: same fields and compatible nullability.
            DType::Struct(fields, n)
                if struct_fields_match(self.value(), fields)
                    && (n.is_nullable() || self.is_valid()) =>
            {
                Ok(self.clone().into())
            }
            DType::Struct(..) => {
                vortex_bail!(
                    "Cannot cast StructScalar to {} (incompatible fields or nullability)",
                    target_dtype
                );
            }
            _ => {
                vortex_bail!("Cannot cast StructScalar to {}", target_dtype);
            }
        }
    }
}
