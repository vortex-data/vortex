// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::VarBinArrayExt;
pub use array::VarBinArraySlotsExt;
pub use array::VarBinData;
pub use array::VarBinDataParts;
pub use array::VarBinSlots;
pub use array::VarBinSlotsView;
pub(crate) use array::offsets_tile_utf8;
pub use vtable::VarBinArray;

pub(crate) mod compute;
pub use compute::take_varbin;

mod vtable;
pub use vtable::VarBin;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}

pub mod builder;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

pub fn varbin_scalar(value: ByteBuffer, dtype: &DType) -> Scalar {
    if matches!(dtype, DType::Utf8(_)) {
        Scalar::try_utf8(value, dtype.nullability())
            .map_err(|err| vortex_err!("Failed to create scalar from utf8 buffer: {}", err))
            .vortex_expect("UTF-8 scalar creation should succeed")
    } else {
        Scalar::binary(value, dtype.nullability())
    }
}

/// Creates a non-null [`Scalar`] from `value` without checking UTF-8.
///
/// # Safety
///
/// `dtype` must be [`DType::Utf8`] or [`DType::Binary`]. If it is [`DType::Utf8`], `value` must be
/// valid UTF-8.
pub(crate) unsafe fn varbin_scalar_unchecked(value: ByteBuffer, dtype: &DType) -> Scalar {
    let value = match dtype {
        // SAFETY: The caller guarantees that `value` is valid UTF-8.
        DType::Utf8(_) => ScalarValue::Utf8(unsafe { BufferString::new_unchecked(value) }),
        _ => ScalarValue::Binary(value),
    };

    // SAFETY: The value variant matches the dtype.
    unsafe { Scalar::new_unchecked(dtype.clone(), Some(value)) }
}

#[cfg(test)]
mod tests;
