// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar casting between [`DType`]s.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::scalar::Scalar;

impl Scalar {
    /// Cast this scalar to another data type.
    ///
    /// # Errors
    ///
    /// Returns an error if the cast is not supported or if a null value is cast to a non-nullable
    /// type.
    pub fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        // If the types are the same, return a clone.
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        // Check for solely nullability casting.
        if self.dtype().eq_ignore_nullability(target_dtype) {
            // Cast from non-nullable to nullable or vice versa.
            // The `try_new` will handle nullability checks.
            return Scalar::try_new(target_dtype.clone(), self.value().cloned());
        }

        // Null can be cast into any nullable type as null.
        // Note that the `matches` clause is technically unnecessary here, just protective.
        if self.value().is_none() || matches!(self.dtype(), DType::Null) {
            vortex_ensure!(
                target_dtype.is_nullable(),
                "Cannot cast null to {target_dtype}: target type is non-nullable"
            );

            return Scalar::try_new(target_dtype.clone(), self.value().cloned());
        }

        // TODO(connor): This isn't really correct for extension types.
        // If the target is an extension type, then we want to cast to its storage type.
        if let Some(ext_dtype) = target_dtype.as_extension_opt() {
            let cast_storage_scalar_value = self.cast(ext_dtype.storage_dtype())?.into_value();
            return Scalar::try_new(target_dtype.clone(), cast_storage_scalar_value);
        }

        match &self.dtype() {
            DType::Null => unreachable!("Handled by the if case above"),
            DType::Bool(_) => self.as_bool().cast(target_dtype),
            DType::Primitive(..) => self.as_primitive().cast(target_dtype),
            DType::Decimal(..) => self.as_decimal().cast(target_dtype),
            DType::Utf8(_) => self.as_utf8().cast(target_dtype),
            DType::Binary(_) => self.as_binary().cast(target_dtype),
            DType::Struct(..) => self.as_struct().cast(target_dtype),
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::List(..) | DType::FixedSizeList(..) => self.as_list().cast(target_dtype),
            DType::Extension(..) => self.as_extension().cast(target_dtype),
            DType::Variant(_) => vortex_bail!("Variant scalars can't be cast to {target_dtype}"),
        }
    }

    /// Cast the scalar into a nullable version of its current type.
    pub fn into_nullable(self) -> Scalar {
        let (dtype, value) = self.into_parts();
        Self::try_new(dtype.as_nullable(), value)
            .vortex_expect("Casting to nullable should always succeed")
    }
}
