// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar casting between [`DType`]s.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastOptions;

impl Scalar {
    /// Cast this scalar to another data type using by-name field matching (the default).
    ///
    /// For positional cast, use [`cast_opts`](Self::cast_opts) with
    /// [`CastOptions::by_position`].
    ///
    /// # Errors
    ///
    /// Returns an error if the cast is not supported or if a null value is cast to a non-nullable
    /// type.
    pub fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        self.cast_opts(target_dtype, CastOptions::by_name())
    }

    /// Cast this scalar to another data type, honoring the given [`CastOptions`].
    ///
    /// Only struct scalars actually use the options today; for all other source types the cast
    /// behavior is identical to [`Self::cast`].
    pub fn cast_opts(&self, target_dtype: &DType, options: CastOptions) -> VortexResult<Scalar> {
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        if self.dtype().eq_ignore_nullability(target_dtype) {
            return Scalar::try_new(target_dtype.clone(), self.value().cloned());
        }

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
            let cast_storage_scalar_value = self
                .cast_opts(ext_dtype.storage_dtype(), options)?
                .into_value();
            return Scalar::try_new(target_dtype.clone(), cast_storage_scalar_value);
        }

        match &self.dtype() {
            DType::Null => unreachable!("Handled by the if case above"),
            DType::Bool(_) => self.as_bool().cast(target_dtype),
            DType::Primitive(..) => self.as_primitive().cast(target_dtype),
            DType::Decimal(..) => self.as_decimal().cast(target_dtype),
            DType::Utf8(_) => self.as_utf8().cast(target_dtype),
            DType::Binary(_) => self.as_binary().cast(target_dtype),
            DType::Struct(..) => self.as_struct().cast_opts(target_dtype, options),
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
