// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinData;
pub use vtable::VarBinArray;

pub(crate) mod compute;

mod vtable;
pub use vtable::VarBin;

pub mod builder;

mod accessor;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::scalar::Scalar;

pub fn varbin_scalar(value: ByteBuffer, dtype: &DType) -> Scalar {
    if matches!(dtype, DType::Utf8(_)) {
        Scalar::try_utf8(value, dtype.nullability())
            .map_err(|err| vortex_err!("Failed to create scalar from utf8 buffer: {}", err))
            .vortex_expect("UTF-8 scalar creation should succeed")
    } else {
        Scalar::binary(value, dtype.nullability())
    }
}

#[cfg(test)]
mod tests;
