// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Implements `NativeDecimalType` for the given type.
macro_rules! impl_native_decimal_type {
    ($T:ty, $variant:ident) => {
        impl NativeDecimalType for $T {
            const VALUES_TYPE: DecimalValueType = DecimalValueType::$variant;

            fn maybe_from(decimal_type: DecimalValue) -> Option<Self> {
                if let DecimalValue::$variant(v) = decimal_type {
                    Some(v)
                } else {
                    None
                }
            }
        }
    };
}
pub(crate) use impl_native_decimal_type;

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
    ($self:expr, | $value:ident | $body:block) => {{
        match $self {
            DecimalValue::I8(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I16(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I32(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I64(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I128(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I256(v) => {
                let $value = v;
                $body
            }
        }
    }};
}

/// Macro to match over each decimal value type, binding the corresponding native type (from
/// `DecimalValueType`)
#[macro_export] // Used in `vortex-array`.
macro_rules! match_each_decimal_value_type {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::{DecimalValueType, i256};
        match $self {
            DecimalValueType::I8 => {
                type $enc = i8;
                $body
            }
            DecimalValueType::I16 => {
                type $enc = i16;
                $body
            }
            DecimalValueType::I32 => {
                type $enc = i32;
                $body
            }
            DecimalValueType::I64 => {
                type $enc = i64;
                $body
            }
            DecimalValueType::I128 => {
                type $enc = i128;
                $body
            }
            DecimalValueType::I256 => {
                type $enc = i256;
                $body
            }
            ty => unreachable!("unknown decimal value type {:?}", ty),
        }
    }};
}
