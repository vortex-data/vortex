// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

/// Dispatches a decimal input and output type, instantiating only `In <= Out`.
/// This produces 21 type combinations instead of 36.
///
/// The caller must choose an output type at least as wide as the input storage type.
/// A decimal array's storage is no wider than
/// [`DecimalType::smallest_decimal_value_type`](crate::dtype::DecimalType::smallest_decimal_value_type)
/// for its precision, so using that type (or a wider working type) satisfies this contract.
/// Narrowing casts must use unrestricted dispatch instead. Invalid combinations panic.
/// Each type expression is evaluated once.
///
/// ```
/// use num_traits::AsPrimitive;
/// use vortex_array::dtype::DecimalType;
/// use vortex_array::match_decimal_unary;
///
/// let value = match_decimal_unary!(DecimalType::I8, DecimalType::I64, |In, Out| {
///     let input: In = 42i8.as_();
///     <In as AsPrimitive<Out>>::as_(input).to_string()
/// });
/// assert_eq!(value, "42");
/// ```
#[macro_export]
macro_rules! match_decimal_unary {
    ($input:expr, $output:expr, | $in:ident, $out:ident | $body:block) => {{
        let (input, output) = ($input, $output);
        match output {
            $crate::dtype::DecimalType::I8 => {
                type $out = i8;
                $crate::match_decimal_unary!(@input input, |$in| $body; I8 => i8)
            }
            $crate::dtype::DecimalType::I16 => {
                type $out = i16;
                $crate::match_decimal_unary!(@input input, |$in| $body; I8 => i8, I16 => i16)
            }
            $crate::dtype::DecimalType::I32 => {
                type $out = i32;
                $crate::match_decimal_unary!(@input input, |$in| $body; I8 => i8, I16 => i16, I32 => i32)
            }
            $crate::dtype::DecimalType::I64 => {
                type $out = i64;
                $crate::match_decimal_unary!(@input input, |$in| $body; I8 => i8, I16 => i16, I32 => i32, I64 => i64)
            }
            $crate::dtype::DecimalType::I128 => {
                type $out = i128;
                $crate::match_decimal_unary!(@input input, |$in| $body; I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128)
            }
            $crate::dtype::DecimalType::I256 => {
                type $out = $crate::dtype::i256;
                $crate::match_decimal_unary!(@input input, |$in| $body; I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128, I256 => $crate::dtype::i256)
            }
        }
    }};
    (@input $input:expr, | $in:ident | $body:block; $( $variant:ident => $native:ty ),+) => {{
        #[allow(unreachable_patterns)]
        match $input {
            $(
                $crate::dtype::DecimalType::$variant => {
                    type $in = $native;
                    $body
                }
            )+
            _ => panic!("decimal input storage type must be no wider than the output type"),
        }
    }};
}

/// Dispatches two decimal input types and an output type, instantiating only combinations
/// where both inputs are no wider than the output (`Left <= Out` and `Right <= Out`).
///
/// This produces 91 type combinations instead of the 216 from unrestricted nested dispatch.
/// See [`match_decimal_unary!`] for the storage contract. Invalid combinations panic.
/// Each type expression is evaluated once, from left to right.
///
/// ```
/// use num_traits::AsPrimitive;
/// use vortex_array::dtype::DecimalType;
/// use vortex_array::match_decimal_binary;
///
/// let value = match_decimal_binary!(
///     DecimalType::I8, DecimalType::I16, DecimalType::I64, |Left, Right, Out| {
///         let lhs: Left = 1i8.as_();
///         let rhs: Right = 2i8.as_();
///         (<Left as AsPrimitive<Out>>::as_(lhs) + <Right as AsPrimitive<Out>>::as_(rhs)).to_string()
///     }
/// );
/// assert_eq!(value, "3");
/// ```
#[macro_export]
macro_rules! match_decimal_binary {
    ($left:expr, $right:expr, $output:expr, | $lhs:ident, $rhs:ident, $out:ident | $body:block) => {{
        let (left, right, output) = ($left, $right, $output);
        match output {
            $crate::dtype::DecimalType::I8 => {
                type $out = i8;
                $crate::match_decimal_unary!(@input left, |$lhs| {
                    $crate::match_decimal_unary!(@input right, |$rhs| $body; I8 => i8)
                }; I8 => i8)
            }
            $crate::dtype::DecimalType::I16 => {
                type $out = i16;
                $crate::match_decimal_unary!(@input left, |$lhs| {
                    $crate::match_decimal_unary!(@input right, |$rhs| $body; I8 => i8, I16 => i16)
                }; I8 => i8, I16 => i16)
            }
            $crate::dtype::DecimalType::I32 => {
                type $out = i32;
                $crate::match_decimal_unary!(@input left, |$lhs| {
                    $crate::match_decimal_unary!(@input right, |$rhs| $body; I8 => i8, I16 => i16, I32 => i32)
                }; I8 => i8, I16 => i16, I32 => i32)
            }
            $crate::dtype::DecimalType::I64 => {
                type $out = i64;
                $crate::match_decimal_unary!(@input left, |$lhs| {
                    $crate::match_decimal_unary!(@input right, |$rhs| $body; I8 => i8, I16 => i16, I32 => i32, I64 => i64)
                }; I8 => i8, I16 => i16, I32 => i32, I64 => i64)
            }
            $crate::dtype::DecimalType::I128 => {
                type $out = i128;
                $crate::match_decimal_unary!(@input left, |$lhs| {
                    $crate::match_decimal_unary!(@input right, |$rhs| $body; I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128)
                }; I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128)
            }
            $crate::dtype::DecimalType::I256 => {
                type $out = $crate::dtype::i256;
                $crate::match_decimal_unary!(@input left, |$lhs| {
                    $crate::match_decimal_unary!(@input right, |$rhs| $body; I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128, I256 => $crate::dtype::i256)
                }; I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128, I256 => $crate::dtype::i256)
            }
        }
    }};
}

#[cfg(test)]
#[path = "macros_tests.rs"]
mod tests;
