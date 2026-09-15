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
