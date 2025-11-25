// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::get_bit;
use vortex_dtype::UnsignedPType;

use crate::take::Take;

impl<I: UnsignedPType> Take<[I]> for &BitBuffer {
    type Output = BitBuffer;

    fn take(self, indices: &[I]) -> BitBuffer {
        // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
        // the overhead to convert to a `Vec<bool>`.
        if self.len() <= 4096 {
            let bools = self.iter().collect();
            take_byte_bool(bools, indices)
        } else {
            take_bool(self, indices)
        }
    }
}

fn take_byte_bool<I: UnsignedPType>(bools: Vec<bool>, indices: &[I]) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |idx| {
        // SAFETY: We are iterating within the bounds of the `indices` array, so we are always
        // within bounds of `indices`.
        let bool_idx = unsafe { indices.get_unchecked(idx).as_() };
        bools[bool_idx]
    })
}

fn take_bool<I: UnsignedPType>(bools: &BitBuffer, indices: &[I]) -> BitBuffer {
    // We dereference to the underlying buffer to avoid incurring an access cost on every index.
    let buffer = bools.inner().as_ref();
    let offset = bools.offset();

    BitBuffer::collect_bool(indices.len(), |idx| {
        // SAFETY: We are iterating within the bounds of the `indices` array, so we are always
        // within bounds.
        let bool_idx = unsafe { indices.get_unchecked(idx).as_() };
        get_bit(buffer, offset + bool_idx)
    })
}
