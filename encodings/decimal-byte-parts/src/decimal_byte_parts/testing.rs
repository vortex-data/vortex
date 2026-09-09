// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared fixtures for decimal byte-parts tests.

use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::DecimalBytePartsArray;
use super::dbp_encode;

/// Encode a canonical decimal array as byte parts, splitting wide values into lower parts.
pub(crate) fn encode(decimal: &DecimalArray) -> VortexResult<DecimalBytePartsArray> {
    dbp_encode(decimal, &mut array_session().create_execution_ctx())
}

/// An `i128`-backed decimal array, encoded as byte parts with one lower part.
pub(crate) fn i128_parts(values: Vec<i128>, validity: Validity) -> DecimalBytePartsArray {
    encode(&DecimalArray::new(
        Buffer::from(values),
        DecimalDType::new(38, 2),
        validity,
    ))
    .vortex_expect("valid decimal byte parts")
}

/// An `i256`-backed decimal array, encoded as byte parts with three lower parts.
pub(crate) fn i256_parts(values: Vec<i256>, validity: Validity) -> DecimalBytePartsArray {
    encode(&DecimalArray::new(
        Buffer::from(values),
        DecimalDType::new(76, 2),
        validity,
    ))
    .vortex_expect("valid decimal byte parts")
}

/// Build an `i256` from a signed high `i128` and unsigned low `u128`.
pub(super) fn i256_of(high: i128, low: u128) -> i256 {
    i256::from_parts(low, high)
}

/// The largest unscaled value a `Decimal(38, _)` can hold: `10^38 - 1`.
const MAX_PRECISION_38: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

/// The largest unscaled value a `Decimal(76, _)` can hold: `10^76 - 1`.
fn max_precision_76() -> i256 {
    i256::from_i128(10).wrapping_pow(76) - i256::ONE
}

/// Values that exercise every 64-bit window of an `i128`, both signs, and the boundaries
/// where a lower part carries into the MSP.
pub(crate) fn wide_i128_values() -> Vec<i128> {
    vec![
        0,
        1,
        -1,
        (1 << 64) - 1,
        1 << 64,
        -(1 << 64),
        -((1 << 64) + 1),
        MAX_PRECISION_38,
        -MAX_PRECISION_38,
        1 << 100,
    ]
}

/// Values that exercise every 64-bit window of an `i256`.
pub(crate) fn wide_i256_values() -> Vec<i256> {
    vec![
        i256::ZERO,
        i256::ONE,
        i256::ZERO - i256::ONE,
        i256_of(0, u128::MAX),
        i256_of(1, 0),
        i256_of(-1, 0),
        i256_of(-1, u128::MAX - 1),
        i256_of(1 << 64, 12345),
        max_precision_76(),
        i256::ZERO - max_precision_76(),
    ]
}
