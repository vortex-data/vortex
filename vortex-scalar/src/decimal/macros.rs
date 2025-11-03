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

/// Matches over each decimal value variant, binding the inner value to a variable.
///
/// # Example
///
/// ```ignore
/// match_each_decimal_value!(value, |v| {
///     println!("Value: {}", v);
/// });
/// ```
#[macro_export] // Used in `vortex-array`.
macro_rules! match_each_decimal_value {
    ($decimal_value:expr, | $ident:ident | $body:block) => {
        match $decimal_value {
            DecimalValue::I8($ident) => $body,
            DecimalValue::I16($ident) => $body,
            DecimalValue::I32($ident) => $body,
            DecimalValue::I64($ident) => $body,
            DecimalValue::I128($ident) => $body,
            DecimalValue::I256($ident) => $body,
        }
    };
}

/// Macro to match over each decimal value type, binding the corresponding native type (from
/// `DecimalType`)
#[macro_export] // Used in `vortex-array`.
macro_rules! match_each_decimal_value_type {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::DecimalType;
        match $self {
            DecimalType::I8 => {
                type $enc = i8;
                $body
            }
            DecimalType::I16 => {
                type $enc = i16;
                $body
            }
            DecimalType::I32 => {
                type $enc = i32;
                $body
            }
            DecimalType::I64 => {
                type $enc = i64;
                $body
            }
            DecimalType::I128 => {
                type $enc = i128;
                $body
            }
            DecimalType::I256 => {
                type $enc = $crate::i256;
                $body
            }
        }
    }};
}
