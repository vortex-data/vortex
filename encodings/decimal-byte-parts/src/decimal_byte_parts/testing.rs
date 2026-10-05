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

use super::DecimalByteParts;
use super::DecimalBytePartsArray;

/// An `i128`-backed decimal array, encoded as byte parts with one lower part.
pub(crate) fn i128_parts(values: Vec<i128>, validity: Validity) -> DecimalBytePartsArray {
    DecimalByteParts::encode(
        &DecimalArray::new(Buffer::from(values), DecimalDType::new(38, 2), validity),
        &mut array_session().create_execution_ctx(),
    )
    .vortex_expect("valid decimal byte parts")
}

/// An `i256`-backed decimal array, encoded as byte parts with three lower parts.
pub(crate) fn i256_parts(values: Vec<i256>, validity: Validity) -> DecimalBytePartsArray {
    DecimalByteParts::encode(
        &DecimalArray::new(Buffer::from(values), DecimalDType::new(76, 2), validity),
        &mut array_session().create_execution_ctx(),
    )
    .vortex_expect("valid decimal byte parts")
}
