// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use crate::BitBuffer;
use crate::BitBufferMut;
use crate::Buffer;
use crate::ByteBufferMut;
use crate::trusted_len::TrustedLenExt;

/// Read up to 8 bytes as a little-endian `u64`, zero-padding the high bytes when fewer than 8 are
/// supplied. Using [`u64::from_le_bytes`] keeps the bit-numbering identical on little- and
/// big-endian targets; for a full 8-byte slice it lowers to a single word load.
#[inline]
fn read_u64_le(bytes: &[u8]) -> u64 {
    debug_assert!(bytes.len() <= 8);
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

trait BitWordTarget {
    fn byte_len(&self) -> usize;

    fn read_word(&self, byte_offset: usize, len: usize) -> u64;

    fn write_word(&mut self, byte_offset: usize, word: &[u8]);
}

impl BitWordTarget for &mut [u8] {
    #[inline]
    fn byte_len(&self) -> usize {
        (**self).len()
    }

    #[inline]
    fn read_word(&self, byte_offset: usize, len: usize) -> u64 {
        read_u64_le(&(**self)[byte_offset..byte_offset + len])
    }

    #[inline]
    fn write_word(&mut self, byte_offset: usize, word: &[u8]) {
        (**self)[byte_offset..byte_offset + word.len()].copy_from_slice(word);
    }
}

struct OutOfPlaceBitWordTarget<'a> {
    src: &'a [u8],
    dst: &'a mut [MaybeUninit<u8>],
}

impl<'a> OutOfPlaceBitWordTarget<'a> {
    #[inline]
    fn new(src: &'a [u8], dst: &'a mut [MaybeUninit<u8>]) -> Self {
        debug_assert!(dst.len() >= src.len());
        Self { src, dst }
    }
}

impl BitWordTarget for OutOfPlaceBitWordTarget<'_> {
    #[inline]
    fn byte_len(&self) -> usize {
        self.src.len()
    }

    #[inline]
    fn read_word(&self, byte_offset: usize, len: usize) -> u64 {
        read_u64_le(&self.src[byte_offset..byte_offset + len])
    }

    #[inline]
    fn write_word(&mut self, byte_offset: usize, word: &[u8]) {
        for (dst_byte, byte) in self.dst[byte_offset..byte_offset + word.len()]
            .iter_mut()
            .zip(word)
        {
            dst_byte.write(*byte);
        }
    }
}

/// Apply `op` to each little-endian `u64` word of `target`.
///
/// The target is split into full `u64` words, with the trailing `len % 8` bytes handled as
/// one final partial word (see [`read_u64_le`]).
#[inline]
fn map_u64_words<T: BitWordTarget, F: FnMut(u64) -> u64>(mut target: T, mut op: F) {
    let len = target.byte_len();
    let full_bytes = len - (len % 8);

    for byte_offset in (0..full_bytes).step_by(8) {
        let word = op(target.read_word(byte_offset, 8)).to_le_bytes();
        target.write_word(byte_offset, &word);
    }

    if full_bytes != len {
        let tail_len = len - full_bytes;
        let word = op(target.read_word(full_bytes, tail_len)).to_le_bytes();
        target.write_word(full_bytes, &word[..tail_len]);
    }
}

/// Combine each little-endian `u64` word of `dst` with the matching word of `src` via `op`,
/// writing the result back into `dst`. Processes `dst.len().min(src.len())` bytes, with the
/// trailing partial word handled like [`map_u64_words`].
#[inline]
fn zip_u64_words_in_place<F: FnMut(u64, u64) -> u64>(dst: &mut [u8], src: &[u8], mut op: F) {
    let n = dst.len().min(src.len());
    let (dst_words, dst_tail) = dst[..n].as_chunks_mut::<8>();
    let (src_words, src_tail) = src[..n].as_chunks::<8>();
    for (d, s) in dst_words.iter_mut().zip(src_words) {
        *d = op(u64::from_le_bytes(*d), u64::from_le_bytes(*s)).to_le_bytes();
    }
    // Both slices were truncated to `n`, so their tails are the same length.
    if !dst_tail.is_empty() {
        let word = op(read_u64_le(dst_tail), read_u64_le(src_tail)).to_le_bytes();
        dst_tail.copy_from_slice(&word[..dst_tail.len()]);
    }
}

/// Apply a unary operation to a [`BitBuffer`], always allocating a new output buffer.
#[inline]
pub(super) fn bitwise_unary_op_copy<F: FnMut(u64) -> u64>(buffer: &BitBuffer, op: F) -> BitBuffer {
    let src = buffer.inner().as_slice();
    let mut bytes = ByteBufferMut::with_capacity(src.len());
    map_u64_words(
        OutOfPlaceBitWordTarget::new(src, bytes.spare_capacity_mut()),
        op,
    );
    // SAFETY: `map_u64_words` initializes every byte in `0..src.len()` for
    // `OutOfPlaceU64WordTarget`.
    unsafe { bytes.set_len(src.len()) };
    BitBufferMut::from_buffer(bytes, buffer.offset(), buffer.len()).freeze()
}

/// Apply a unary operation to an owned [`BitBuffer`], mutating in-place when possible.
///
/// Tries to get exclusive access via `try_into_mut`. If the backing storage is shared
/// (Arc refcount > 1), falls back to [`bitwise_unary_op_copy`].
#[inline]
pub(super) fn bitwise_unary_op<F: FnMut(u64) -> u64>(buffer: BitBuffer, op: F) -> BitBuffer {
    match buffer.try_into_mut() {
        Ok(mut buf) => {
            bitwise_unary_op_mut(&mut buf, op);
            buf.freeze()
        }
        Err(buffer) => bitwise_unary_op_copy(&buffer, op),
    }
}

#[inline]
pub(super) fn bitwise_unary_op_mut<F: FnMut(u64) -> u64>(buffer: &mut BitBufferMut, op: F) {
    map_u64_words(buffer.as_mut_slice(), op);
}

/// Apply a binary operation with an owned left operand, mutating in-place when possible.
///
/// Tries `try_into_mut` on the left operand. If the backing storage has exclusive access,
/// the operation is performed in-place (zero allocation). Otherwise, falls back to
/// [`bitwise_binary_op`] which allocates a new buffer.
pub(super) fn bitwise_binary_op_lhs_owned<F: FnMut(u64, u64) -> u64>(
    left: BitBuffer,
    right: &BitBuffer,
    op: F,
) -> BitBuffer {
    assert_eq!(left.len(), right.len());

    // The in-place path combines the operands word-for-word, which only lines up the logical bits
    // when both share the same bit-to-byte alignment. When the offsets differ, fall back to the
    // offset-aware allocating path (`bitwise_binary_op`) rather than corrupting the result.
    if left.offset() != right.offset() {
        return bitwise_binary_op(&left, right, op);
    }

    match left.try_into_mut() {
        Ok(mut buf) => {
            zip_u64_words_in_place(buf.as_mut_slice(), right.inner().as_slice(), op);
            buf.freeze()
        }
        Err(left) => bitwise_binary_op(&left, right, op),
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

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use rstest::rstest;

    use super::*;
    use crate::ByteBufferMut;
    use crate::bitbuffer;
    use crate::buffer;

    #[test]
    fn test_bitwise_unary_not() {
        let buffer = BitBuffer::new(buffer![0b10101010u8], 4);
        let result = bitwise_unary_op(buffer, |x| !x);
        assert_eq!(result, bitbuffer![true, false, true, false]);
    }

    #[test]
    fn test_lhs_owned_offset_mismatch_regression() {
        use crate::buffer_mut;

        // `left` has bit offset 3 and uniquely-owned backing storage, so the in-place fast
        // path is taken. Byte 0b1111_1000 → logical bits [3..8) = [1,1,1,1,1].
        let left = BitBufferMut::from_buffer(buffer_mut![0b1111_1000u8], 3, 5).freeze();
        // `right` has bit offset 0. Byte 0b0001_1111 → logical bits [0..5) = [1,1,1,1,1].
        let right = BitBuffer::new(buffer![0b0001_1111u8], 5);

        // AND of two all-true ranges must be all-true. The naive byte-wise in-place path
        // ignores the differing offsets and yields the wrong answer.
        let got = bitwise_binary_op_lhs_owned(left, &right, |a, b| a & b);
        assert_eq!(got.true_count(), 5);
        assert_eq!(got, bitbuffer![true, true, true, true, true]);
    }

    /// The owned-LHS path (in-place when uniquely owned and the offsets match) must produce the
    /// same logical result as the always-correct allocating [`bitwise_binary_op`], for every
    /// combination of operand offsets and lengths.
    #[rstest]
    #[case::aligned(0, 0)]
    #[case::equal_nonzero(3, 3)]
    #[case::equal_seven(7, 7)]
    #[case::mismatch_lo(0, 3)]
    #[case::mismatch_hi(5, 2)]
    fn lhs_owned_matches_reference(#[case] left_offset: usize, #[case] right_offset: usize) {
        // Deterministic byte pattern, so the owned and borrowed inputs are bit-identical.
        #[allow(clippy::cast_possible_truncation)]
        let make = |offset: usize, len: usize, salt: u8| -> BitBuffer {
            let bytes: ByteBufferMut = (0..(offset + len).div_ceil(8).max(1))
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt))
                .collect();
            BitBufferMut::from_buffer(bytes, offset, len).freeze()
        };
        let ops: [fn(u64, u64) -> u64; 4] =
            [|a, b| a & b, |a, b| a | b, |a, b| a ^ b, |a, b| a & !b];

        for len in [1usize, 5, 8, 63, 64, 65, 129, 200] {
            let right = make(right_offset, len, 0x5A);
            for op in ops {
                // A fresh, uniquely-owned LHS triggers the in-place path when offsets match.
                let got = bitwise_binary_op_lhs_owned(make(left_offset, len, 0xC3), &right, op);
                let expected = bitwise_binary_op(&make(left_offset, len, 0xC3), &right, op);
                assert_eq!(
                    got, expected,
                    "loff={left_offset} roff={right_offset} len={len}"
                );
            }
        }
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
