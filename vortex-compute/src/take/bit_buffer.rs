// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take operation on [`BitBuffer`].
//!
//! NB: We do NOT implement `impl<I: UnsignedPType> Take<PVector<I>> for &BitBuffer`, specifically
//! because there is a very similar implementation on `Mask` that has special logic for working with
//! null indices. That logic could also be implemented on `BitBuffer`, but since it is not
//! immediately clear what should happen in the case of a null index when taking a `BitBuffer` (do
//! you set it to true or false?), we do not implement this at all.

use vortex_buffer::BitBuffer;
use vortex_buffer::get_bit;
use vortex_dtype::UnsignedPType;

use crate::take::LINUX_PAGE_SIZE;
use crate::take::Take;

impl<I: UnsignedPType> Take<[I]> for &BitBuffer {
    type Output = BitBuffer;

    fn take(self, indices: &[I]) -> BitBuffer {
        // For boolean arrays that roughly fit into a single page (at least, on Linux), it's worth
        // the overhead to convert to a `Vec<bool>`.
        if self.len() <= LINUX_PAGE_SIZE {
            let bools = self.iter().collect();
            take_byte_bool(bools, indices)
        } else {
            take_bool(self, indices)
        }
    }
}

/// # Panics
///
/// Panics if an index is out of bounds.
fn take_byte_bool<I: UnsignedPType>(bools: Vec<bool>, indices: &[I]) -> BitBuffer {
    BitBuffer::collect_bool(indices.len(), |idx| {
        // SAFETY: We are iterating within the bounds of the `indices` array, so we are always
        // within bounds of `indices`.
        let bool_idx = unsafe { indices.get_unchecked(idx).as_() };
        bools[bool_idx]
    })
}

/// # Panics
///
/// Panics if an index is out of bounds.
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
