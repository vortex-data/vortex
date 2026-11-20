// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar casting between [`DType`]s.

use num_traits::One;
use num_traits::Zero;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::cast::TemporalCast;
use crate::scalar_fn::builtin::cast::check_extension_cast;

impl Scalar {
    /// Cast this scalar to another data type.
    ///
    /// A cast can change the top-level nullability of an extension dtype, or perform an exact
    /// conversion between two temporal dtypes. A scalar of dtype [`DType::Null`] can also become a
    /// typed null of any nullable dtype.
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

        if let Some(cast) = TemporalCast::classify(self.dtype(), target_dtype) {
            return cast.cast_scalar(self, target_dtype);
        }
        check_extension_cast(self.dtype(), target_dtype)?;

        if let (Some(source), Some(target)) = (self.dtype().as_map_opt(), target_dtype.as_map_opt())
            && target.keys_sorted()
            && !source.keys_sorted()
        {
            return Err(vortex_err!(
                "Cannot cast {} to {target_dtype}: source does not assert sorted map keys",
                self.dtype()
            ));
        }

        // Cast nulls after checking the extension dtype restrictions.
        if self.value().is_none() || matches!(self.dtype(), DType::Null) {
            vortex_ensure!(
                target_dtype.is_nullable(),
                "Cannot cast null to {target_dtype}: target type is non-nullable"
            );

            return Scalar::try_new(target_dtype.clone(), self.value().cloned());
        }

        match &self.dtype() {
            DType::Null => unreachable!("Handled by the if case above"),
            DType::Bool(_) => {
                let DType::Primitive(ptype, nullability) = target_dtype else {
                    vortex_bail!(
                        "Cannot cast bool to {target_dtype}: boolean scalars can only be cast to boolean or primitive types"
                    );
                };
                let value = self
                    .as_bool()
                    .value()
                    .vortex_expect("null scalars are handled above");
                Ok(cast_bool_to_primitive(value, *ptype, *nullability))
            }
            DType::Primitive(..) => self.as_primitive().cast(target_dtype),
            DType::Decimal(..) => self.as_decimal().cast(target_dtype),
            DType::Utf8(_) => vortex_bail!(
                "Cannot cast utf8 to {target_dtype}: UTF-8 scalars can only be cast to UTF-8 types with different nullability"
            ),
            DType::Binary(_) => vortex_bail!(
                "Cannot cast binary to {target_dtype}: binary scalars can only be cast to binary types with different nullability"
            ),
            DType::List(..) | DType::FixedSizeList(..) => self.as_list().cast(target_dtype),
            DType::Map(..) => self.as_map().cast(target_dtype),
            DType::Struct(..) => self.as_struct().cast(target_dtype),
            DType::Union(..) => vortex_bail!(
                "union scalar cast from {} to {target_dtype} is not supported (yet)",
                self.dtype()
            ),
            DType::Variant(_) => {
                vortex_bail!("Variant scalars can't be cast to {target_dtype}")
            }
            DType::Extension(..) => unreachable!("Extension casts were checked above"),
        }
    }

    /// Cast the scalar into a nullable version of its current type.
    pub fn into_nullable(self) -> Scalar {
        let (dtype, value) = self.into_parts();
        Self::try_new(dtype.as_nullable(), value)
            .vortex_expect("Casting to nullable should always succeed")
    }
}

/// Casts `value` to `ptype` like the bool array cast: true is one, false is zero.
fn cast_bool_to_primitive(value: bool, ptype: PType, nullability: Nullability) -> Scalar {
    match_each_native_ptype!(ptype, |T| {
        Scalar::primitive(
            if value {
                <T as One>::one()
            } else {
                <T as Zero>::zero()
            },
            nullability,
        )
    })
}
