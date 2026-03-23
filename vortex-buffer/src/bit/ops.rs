// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::BitBuffer;
use crate::BitBufferMut;
use crate::Buffer;
use crate::trusted_len::TrustedLenExt;

#[inline]
pub(super) fn bitwise_unary_op<F: FnMut(u64) -> u64>(buffer: &BitBuffer, op: F) -> BitBuffer {
    let mut buf = buffer.clone().into_mut();
    bitwise_unary_op_mut(&mut buf, op);
    buf.freeze()
}

#[inline]
pub(super) fn bitwise_unary_op_mut<F: FnMut(u64) -> u64>(buffer: &mut BitBufferMut, mut op: F) {
    let slice_mut = buffer.as_mut_slice();

    // The number of complete u64 words in the buffer (unaligned)
    let u64_len = slice_mut.len() / 8;
    let remainder = slice_mut.len() % 8;

    // Create a pointer to the *unaligned* u64 words
    let mut ptr = slice_mut.as_mut_ptr() as *mut u64;
    for _ in 0..u64_len {
        let value = unsafe { ptr.read_unaligned() };
        let value = op(value);
        unsafe { ptr.write_unaligned(value) };
        ptr = unsafe { ptr.add(1) };
    }

    // Read remainder into a u64;
    let mut remainder_u64 = 0u64;
    let ptr = ptr as *mut u8;
    for i in 0..remainder {
        let byte = unsafe { ptr.add(i).read() };
        remainder_u64 |= (byte as u64) << (i * 8);
    }
    let remainder_u64 = op(remainder_u64);

    // Write back remainder
    for i in 0..remainder {
        let byte = ((remainder_u64 >> (i * 8)) & 0xFF) as u8;
        unsafe { ptr.add(i).write(byte) };
    }
}

pub(super) fn bitwise_binary_op<F: FnMut(u64, u64) -> u64>(
    left: &BitBuffer,
    right: &BitBuffer,
    mut op: F,
) -> BitBuffer {
    assert_eq!(left.len(), right.len());

    // If the buffers are aligned, we can use the fast path.
    if left.offset().is_multiple_of(8) && right.offset().is_multiple_of(8) {
        let left_chunks = left.unaligned_chunks();
        let right_chunks = right.unaligned_chunks();
        if left_chunks.lead_padding() == 0
            && left_chunks.trailing_padding() == 0
            && right_chunks.lead_padding() == 0
            && right_chunks.trailing_padding() == 0
        {
            let iter = left_chunks
                .iter()
                .zip(right_chunks.iter())
                .map(|(l, r)| op(l, r));
            let iter = unsafe { iter.trusted_len() };
            let result = Buffer::<u64>::from_trusted_len_iter(iter).into_byte_buffer();
            return BitBuffer::new(result, left.len());
        }
    }

    let iter = left
        .chunks()
        .iter_padded()
        .zip(right.chunks().iter_padded())
        .map(|(l, r)| op(l, r));
    let iter = unsafe { iter.trusted_len() };

    let result = Buffer::<u64>::from_trusted_len_iter(iter).into_byte_buffer();

    BitBuffer::new(result, left.len())
}

/// Applies a binary bitwise operation in-place on a [`BitBufferMut`], combining it with a
/// [`BitBuffer`]. The result is written back to the mutable buffer, avoiding a new allocation.
///
/// Both buffers must have the same length.
pub(super) fn bitwise_binary_op_mut<F: FnMut(u64, u64) -> u64>(
    left: &mut BitBufferMut,
    right: &BitBuffer,
    mut op: F,
) {
    assert_eq!(left.len(), right.len());

    let left_slice = left.as_mut_slice();
    let u64_len = left_slice.len() / 8;
    let remainder = left_slice.len() % 8;
    let mut left_ptr = left_slice.as_mut_ptr() as *mut u64;

    // Read right-hand side chunks, padded to u64 boundaries.
    let mut right_chunks = right.chunks().iter_padded();

    for _ in 0..u64_len {
        let left_val = unsafe { left_ptr.read_unaligned() };
        let right_val = right_chunks.next().unwrap_or(0);
        unsafe { left_ptr.write_unaligned(op(left_val, right_val)) };
        left_ptr = unsafe { left_ptr.add(1) };
    }

    // Handle remainder bytes.
    if remainder > 0 {
        let right_val = right_chunks.next().unwrap_or(0);
        let left_ptr = left_ptr as *mut u8;
        let mut left_u64 = 0u64;
        for i in 0..remainder {
            let byte = unsafe { left_ptr.add(i).read() };
            left_u64 |= (byte as u64) << (i * 8);
        }
        let result = op(left_u64, right_val);
        for i in 0..remainder {
            let byte = ((result >> (i * 8)) & 0xFF) as u8;
            unsafe { left_ptr.add(i).write(byte) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use super::*;
    use crate::bitbuffer;
    use crate::buffer;

    #[test]
    fn test_bitwise_unary_not() {
        let buffer = BitBuffer::new(buffer![0b10101010u8], 4);
        let result = bitwise_unary_op(&buffer, |x| !x);
        assert_eq!(result, bitbuffer![true, false, true, false]);
    }

    #[test]
    fn test_bitwise_binary_and() {
        // 0b1111 (15) & 0b1010 (10) = 0b1010 (10)
        let left = BitBuffer::new(buffer![15u8], 4);
        let right = BitBuffer::new(buffer![10u8], 4);
        let result = bitwise_binary_op(&left, &right, |l, r| l & r);
        assert_eq!(result, bitbuffer![false, true, false, true]);
    }

    #[test]
    fn test_bitwise_binary_or() {
        // 0b1010 (10) | 0b0101 (5) = 0b1111 (15)
        let left = BitBuffer::new(buffer![10u8], 4);
        let right = BitBuffer::new(buffer![5u8], 4);
        let result = bitwise_binary_op(&left, &right, |l, r| l | r);
        assert_eq!(result, bitbuffer![true; 4]);
    }

    #[test]
    fn test_bitwise_binary_xor() {
        // 0b1100 (12) ^ 0b1010 (10) = 0b0110 (6)
        let left = BitBuffer::new(buffer![12u8], 4);
        let right = BitBuffer::new(buffer![10u8], 4);
        let result = bitwise_binary_op(&left, &right, |l, r| l ^ r);
        assert_eq!(result, bitbuffer![false, true, true, false]);
    }

    /// Regression test for a bug where [`bitwise_unary_op`] produced corrupt results when
    /// the [`BitBuffer`]'s underlying byte pointer was not u64-aligned. Slicing a buffer by
    /// a non-multiple-of-8 number of bytes can cause this misalignment. The bug only
    /// manifested for buffers larger than 16 bytes (> 128 bits), because Arrow's
    /// `UnalignedBitChunk` switches from byte-copying to `align_to` at that threshold.
    ///
    /// Issue: <https://github.com/vortex-data/vortex/issues/6895>
    #[test]
    fn test_bitwise_unary_not_misaligned_buffer() {
        // Slice off 1 byte to shift the pointer off u64 alignment. Use 129 bits (17 bytes)
        // to exceed the 16-byte threshold where `UnalignedBitChunk` uses `align_to`.
        let padded = BitBuffer::new_set(8 + 129);
        let buf = padded.slice(8..8 + 129);

        let result = buf.not();
        assert_eq!(
            result.true_count(),
            0,
            "expected all-false after NOT of all-true"
        );
    }
}
