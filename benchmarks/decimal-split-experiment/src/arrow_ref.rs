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
