// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions for [`DecimalScalar`]s.

use vortex_dtype::i256;
use vortex_error::VortexError;
use vortex_error::vortex_err;

use crate::scalar::DecimalScalar;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

// TODO(connor): Ideally we remove this.
impl From<DecimalScalar<'_>> for Scalar {
    fn from(ds: DecimalScalar<'_>) -> Self {
        // SAFETY: `DecimalScalar` is already a valid `Scalar`.
        unsafe {
            Scalar::new_unchecked(
                ds.dtype().clone(),
                ds.decimal_value().map(ScalarValue::Decimal),
            )
        }
    }
}

/// Implements `TryFrom<DecimalScalar>` for the given type.
macro_rules! decimal_scalar_unpack {
    ($T:ident, $arm:ident) => {
        impl TryFrom<DecimalScalar<'_>> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                Ok(match value.decimal_value() {
                    None => None,
                    Some(DecimalValue::$arm(v)) => Some(v),
                    v => vortex_error::vortex_bail!(
                        "Cannot extract decimal {:?} as {}",
                        v,
                        stringify!($T)
                    ),
                })
            }
        }

        impl TryFrom<DecimalScalar<'_>> for $T {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                match value.decimal_value() {
                    None => vortex_error::vortex_bail!("Cannot extract value from null decimal"),
                    Some(DecimalValue::$arm(v)) => Ok(v),
                    v => vortex_error::vortex_bail!(
                        "Cannot extract decimal {:?} as {}",
                        v,
                        stringify!($T)
                    ),
                }
            }
        }
    };
}

/// Implements `Into<DecimalValue>` for the given type.
macro_rules! decimal_scalar_pack {
    ($from:ident, $to:ident, $arm:ident) => {
        impl From<$from> for DecimalValue {
            fn from(value: $from) -> Self {
                DecimalValue::$arm(value as $to)
            }
        }
    };
}

decimal_scalar_unpack!(i8, I8);
decimal_scalar_unpack!(i16, I16);
decimal_scalar_unpack!(i32, I32);
decimal_scalar_unpack!(i64, I64);
decimal_scalar_unpack!(i128, I128);
decimal_scalar_unpack!(i256, I256);

decimal_scalar_pack!(i8, i8, I8);
decimal_scalar_pack!(i16, i16, I16);
decimal_scalar_pack!(i32, i32, I32);
decimal_scalar_pack!(i64, i64, I64);
decimal_scalar_pack!(i128, i128, I128);
decimal_scalar_pack!(i256, i256, I256);

decimal_scalar_pack!(u8, i16, I16);
decimal_scalar_pack!(u16, i32, I32);
decimal_scalar_pack!(u32, i64, I64);
decimal_scalar_pack!(u64, i128, I128);

impl TryFrom<&Scalar> for DecimalValue {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        let decimal_scalar = scalar
            .as_decimal_opt()
            .ok_or_else(|| vortex_err!("Expected decimal scalar, found {}", scalar.dtype()))?;

        decimal_scalar
            .decimal_value()
            .as_ref()
            .cloned()
            .ok_or_else(|| vortex_err!("Cannot extract DecimalValue from null decimal"))
    }
}

impl TryFrom<Scalar> for DecimalValue {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        DecimalValue::try_from(&scalar)
    }
}

impl TryFrom<&Scalar> for Option<DecimalValue> {
    type Error = VortexError;

    fn try_from(scalar: &Scalar) -> Result<Self, Self::Error> {
        Ok(scalar
            .as_decimal_opt()
            .ok_or_else(|| vortex_err!("Expected decimal scalar, found {}", scalar.dtype()))?
            .decimal_value())
    }
}

impl TryFrom<Scalar> for Option<DecimalValue> {
    type Error = VortexError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        Option::<DecimalValue>::try_from(&scalar)
    }
}
