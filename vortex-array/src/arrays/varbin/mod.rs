// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinArray;

mod compute;
pub(crate) use compute::varbin_compute_min_max;
// For use in `varbinview`.

mod vtable;
pub use vtable::VarBinVTable;

pub mod builder;

mod accessor;

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexUnwrap, vortex_err};
use vortex_scalar::Scalar;

pub fn varbin_scalar(value: ByteBuffer, dtype: &DType) -> Scalar {
    if matches!(dtype, DType::Utf8(_)) {
        Scalar::try_utf8(value, dtype.nullability())
            .map_err(|err| vortex_err!("Failed to create scalar from utf8 buffer: {}", err))
            .vortex_unwrap()
    } else {
        Scalar::binary(value, dtype.nullability())
    }
}

#[cfg(test)]
mod tests;
