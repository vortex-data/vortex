// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Implements `TryFrom<DecimalScalar>` for the given type.
macro_rules! decimal_scalar_unpack {
    ($T:ident, $arm:ident) => {
        impl TryFrom<DecimalScalar<'_>> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                Ok(match value.value {
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
                match value.value {
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
pub(crate) use decimal_scalar_unpack;

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
pub(crate) use decimal_scalar_pack;
