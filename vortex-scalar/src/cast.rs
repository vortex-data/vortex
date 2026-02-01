// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Scalar;
use crate::ScalarValue;

impl Scalar {
    /// Cast this scalar to another data type.
    pub fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        // If the types are the same, return a clone.
        if self.dtype() == dtype {
            return Ok(self.clone());
        }

        // Check for nullability casting.
        if self.dtype().eq_ignore_nullability(dtype) {
            // Cast from non-nullable to nullable or vice versa.
            // The try_new with check will handle nullability checks.
            return Scalar::try_new(dtype.clone(), self.value().clone());
        }

        match (self.dtype(), dtype) {
            (_, DType::Null) => {
                // Can cast anything to null if the value is null.
                if matches!(self.value(), ScalarValue::Null) {
                    return Ok(Scalar::null(dtype.clone()));
                }
                vortex_bail!("Cannot cast non-null value {} to null dtype", self);
            }
            _ => {
                vortex_bail!(
                    "Casting scalar from {} to {} is not supported",
                    self.dtype(),
                    dtype
                );
            }
        }
    }

    /// Cast the scalar into a nullable version of its current type.
    pub fn into_nullable(self) -> Scalar {
        let (dtype, value) = self.into_parts();
        Self::try_new(dtype.as_nullable(), value)
            .vortex_expect("Casting to nullable should always succeed")
    }
}
