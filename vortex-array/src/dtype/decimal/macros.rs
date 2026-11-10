// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Dispatches a decimal storage type to a compile-time byte width for structural kernels.
///
/// Accepts an owned or borrowed [`DecimalType`](crate::dtype::DecimalType), evaluated once, and
/// specializes 1, 2, 4, 8, 16, and 32 bytes. Kernels can operate on `[u8; W]` records while the caller
/// preserves decimal storage type, precision, and scale.
///
/// The optional `widths = [...]` list and required fallback follow
/// [`match_each_power_of_two_width!`](crate::match_each_power_of_two_width).
///
/// # Examples
///
/// ```
/// use vortex_array::dtype::DecimalType;
/// use vortex_array::match_each_decimal_type_width;
///
/// let width = match_each_decimal_type_width!(
///     DecimalType::I128,
///     |W| { W },
///     _ => { unreachable!() },
/// );
/// assert_eq!(width, 16);
/// ```
///
/// Logical precision and scale do not determine an existing buffer's physical width:
///
/// ```compile_fail
/// use vortex_array::dtype::DecimalDType;
/// use vortex_array::match_each_decimal_type_width;
///
/// match_each_decimal_type_width!(DecimalDType::new(10, 2), |W| { W }, _ => { unreachable!() });
/// ```
///
/// Primitive types use [`match_each_ptype_width!`](crate::match_each_ptype_width):
///
/// ```compile_fail
/// use vortex_array::dtype::PType;
/// use vortex_array::match_each_decimal_type_width;
///
/// match_each_decimal_type_width!(PType::I64, |W| { W }, _ => { unreachable!() });
/// ```
#[macro_export]
macro_rules! match_each_decimal_type_width {
    (
        $decimal_type:expr,
        widths = [$($width:literal),+ $(,)?],
        |$W:ident| $body:block,
        $other:pat => $fallback:block $(,)?
    ) => {
        $crate::match_each_power_of_two_width!(
            $crate::dtype::DecimalType::byte_width(&$decimal_type),
            widths = [$($width),+],
            |$W| $body,
            $other => $fallback,
        )
    };
    ($decimal_type:expr, |$W:ident| $body:block, $other:pat => $fallback:block $(,)?) => {
        $crate::match_each_decimal_type_width!(
            $decimal_type,
            widths = [1, 2, 4, 8, 16, 32],
            |$W| $body,
            $other => $fallback,
        )
    };
}

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
        use $crate::dtype::DecimalType;
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
                type $enc = $crate::dtype::i256;
                $body
            }
        }
    }};
}
