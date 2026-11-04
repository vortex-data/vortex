// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared fixtures for decimal byte-parts tests.

use std::sync::LazyLock;

use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

use super::DecimalByteParts;
use super::DecimalBytePartsArray;
use super::MAX_I128_LOWER_PARTS;
use super::MAX_I256_LOWER_PARTS;
use super::split_decimal;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// An `i128`-backed decimal array, encoded as byte parts with one lower part.
pub(crate) fn i128_parts(values: Vec<i128>, validity: Validity) -> DecimalBytePartsArray {
    encode_with_lower_parts(
        &DecimalArray::new(Buffer::from(values), DecimalDType::new(38, 2), validity),
        MAX_I128_LOWER_PARTS,
    )
}

/// An `i256`-backed decimal array, encoded as byte parts with three lower parts.
pub(crate) fn i256_parts(values: Vec<i256>, validity: Validity) -> DecimalBytePartsArray {
    encode_with_lower_parts(
        &DecimalArray::new(Buffer::from(values), DecimalDType::new(76, 2), validity),
        MAX_I256_LOWER_PARTS,
    )
}

/// Encodes `decimal` as byte parts with `lower_part_count` lower parts, whatever its values need.
fn encode_with_lower_parts(
    decimal: &DecimalArray,
    lower_part_count: usize,
) -> DecimalBytePartsArray {
    let parts = split_decimal(
        decimal,
        lower_part_count,
        &mut SESSION.create_execution_ctx(),
    )
    .vortex_expect("the fixture splits into a supported shape");
    DecimalByteParts::try_new_with_lower_parts(
        parts.msp,
        parts.lower_parts,
        decimal.decimal_dtype(),
    )
    .vortex_expect("valid decimal byte parts")
}

/// The number of lower parts that holds every value of `values_type`.
pub(crate) fn full_width_lower_part_count(values_type: DecimalType) -> usize {
    match values_type {
        DecimalType::I128 => MAX_I128_LOWER_PARTS,
        DecimalType::I256 => MAX_I256_LOWER_PARTS,
        _ => 0,
    }
}
