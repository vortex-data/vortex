// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small byte-table take with a fallback for unsupported targets and inputs.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;

use vortex_buffer::Buffer;

use super::FixedWidthTakeValue;
use super::take_values_fallback;
use crate::dtype::UnsignedPType;

pub(crate) fn take<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
) -> Buffer<T> {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if let Some(taken) = neon::take(values, indices) {
        return taken;
    }

    take_values_fallback(values, indices)
}
