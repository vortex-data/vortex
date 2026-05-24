// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow-rs reference kernels (interleaved array-of-structs layout).
//!
//! Arrow stores Decimal128/Decimal256 as contiguous little-endian values and
//! adds them element by element. `arrow_arith::numeric::add_wrapping` matches
//! the wrapping semantics of our kernels, so this is an apples-to-apples
//! comparison of layout + dispatch, not of overflow policy.

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::BooleanArray;
use arrow_array::Decimal128Array;
use arrow_array::Decimal256Array;
use arrow_buffer::i256;

/// Build an Arrow `Decimal128Array` from raw i128 values.
pub fn decimal128(values: &[i128], precision: u8, scale: i8) -> Decimal128Array {
    Decimal128Array::from_iter_values(values.iter().copied())
        .with_precision_and_scale(precision, scale)
        .expect("valid precision/scale")
}

/// Build an Arrow `Decimal256Array` from raw i256 values.
pub fn decimal256(values: &[i256], precision: u8, scale: i8) -> Decimal256Array {
    Decimal256Array::from_iter_values(values.iter().copied())
        .with_precision_and_scale(precision, scale)
        .expect("valid precision/scale")
}

/// Arrow's wrapping add for Decimal128 (interleaved kernel).
pub fn add_decimal128(a: &Decimal128Array, b: &Decimal128Array) -> ArrayRef {
    arrow_arith::numeric::add_wrapping(a, b).expect("arrow add")
}

/// Arrow's wrapping add for Decimal256 (interleaved kernel).
pub fn add_decimal256(a: &Decimal256Array, b: &Decimal256Array) -> ArrayRef {
    arrow_arith::numeric::add_wrapping(a, b).expect("arrow add")
}

/// Arrow's wrapping multiply for Decimal128 (low-128 product, interleaved).
pub fn mul_decimal128(a: &Decimal128Array, b: &Decimal128Array) -> ArrayRef {
    arrow_arith::numeric::mul_wrapping(a, b).expect("arrow mul")
}

/// Arrow's checked divide for Decimal128 (errors on zero divisor; with scale 0
/// this is integer division).
pub fn div_decimal128(a: &Decimal128Array, b: &Decimal128Array) -> ArrayRef {
    arrow_arith::numeric::div(a, b).expect("arrow div")
}

/// Pull the i128 values back out of an Arrow result for verification.
pub fn decimal128_values(arr: &ArrayRef) -> Vec<i128> {
    let arr = arr
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("decimal128 result");
    (0..arr.len()).map(|i| arr.value(i)).collect()
}

/// Pull the i256 values back out of an Arrow result for verification.
pub fn decimal256_values(arr: &ArrayRef) -> Vec<i256> {
    let arr = arr
        .as_any()
        .downcast_ref::<Decimal256Array>()
        .expect("decimal256 result");
    (0..arr.len()).map(|i| arr.value(i)).collect()
}

// ---- comparison --------------------------------------------------------------

/// Arrow's `lt` for Decimal128 (arrow_ord interleaved kernel).
pub fn lt_decimal128(a: &Decimal128Array, b: &Decimal128Array) -> BooleanArray {
    arrow_ord::cmp::lt(a, b).expect("arrow lt")
}

/// Arrow's `eq` for Decimal128.
pub fn eq_decimal128(a: &Decimal128Array, b: &Decimal128Array) -> BooleanArray {
    arrow_ord::cmp::eq(a, b).expect("arrow eq")
}

/// Arrow's `lt` for Decimal256.
pub fn lt_decimal256(a: &Decimal256Array, b: &Decimal256Array) -> BooleanArray {
    arrow_ord::cmp::lt(a, b).expect("arrow lt")
}

/// `true` at index `i` of an Arrow boolean result.
pub fn boolean_at(arr: &BooleanArray, i: usize) -> bool {
    arr.value(i)
}

// ---- aggregation -------------------------------------------------------------

/// Arrow's checked sum over a Decimal128 column (`None` on overflow). Arrow sums
/// into the same i128 width, so this overflows where Vortex's i256-widening sum
/// would not - a semantic difference worth showing.
pub fn sum_decimal128(a: &Decimal128Array) -> Option<i128> {
    arrow_arith::aggregate::sum(a)
}

/// Arrow's min/max over a Decimal128 column.
pub fn min_decimal128(a: &Decimal128Array) -> Option<i128> {
    arrow_arith::aggregate::min(a)
}

pub fn max_decimal128(a: &Decimal128Array) -> Option<i128> {
    arrow_arith::aggregate::max(a)
}
