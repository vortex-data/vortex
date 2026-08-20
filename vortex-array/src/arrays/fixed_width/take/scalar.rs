// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use crate::dtype::IntegerPType;

#[inline(always)]
pub(crate) fn take_values_scalar<T: Copy, I: IntegerPType>(
    values: &[T],
    indices: &[I],
) -> Buffer<T> {
    // The explicit pointer loop keeps the source length in a register and avoids a capacity check
    // for every output value.
    let mut result = BufferMut::with_capacity(indices.len());
    let result_ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for (output_index, index) in indices.iter().enumerate() {
        // SAFETY: `indices.len()` elements were reserved and each output position is written once.
        unsafe { result_ptr.add(output_index).write(values[index.as_()]) };
    }

    // SAFETY: the loop initialized every element in the reserved output range.
    unsafe { result.set_len(indices.len()) };
    result.freeze()
}
