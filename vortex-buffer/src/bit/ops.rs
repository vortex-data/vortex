// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use crate::BitBuffer;
use crate::BitBufferMut;
use crate::BufferMut;
use crate::ByteBufferMut;
use crate::read_u64_le;

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
    let len = left.len();
    if len == 0 {
        return BitBuffer::empty();
    }

    let n_bytes = len.div_ceil(8);
    let out = if left.offset().is_multiple_of(8) && right.offset().is_multiple_of(8) {
        // Byte-aligned operands: logical bits map onto physical `u64` words, so read the backing
        // bytes straight as words and build the result from a `TrustedLen` iterator.
        let l_start = left.offset() / 8;
        let r_start = right.offset() / 8;
        let lhs = &left.inner().as_slice()[l_start..l_start + n_bytes];
        let rhs = &right.inner().as_slice()[r_start..r_start + n_bytes];

        let (lhs_words, lhs_tail) = lhs.as_chunks::<8>();
        let (rhs_words, rhs_tail) = rhs.as_chunks::<8>();

        let mut out = BufferMut::<u64>::from_trusted_len_iter(
            lhs_words
                .iter()
                .zip(rhs_words)
                .map(|(l, r)| op(u64::from_le_bytes(*l), u64::from_le_bytes(*r))),
        );
        if !lhs_tail.is_empty() {
            out.push(op(read_u64_le(lhs_tail), read_u64_le(rhs_tail)));
        }
        out
    } else {
        // Sub-byte offset: `iter_padded` realigns the bits and appends one pad word, so take
        // exactly `ceil(len / 64)` words.
        let n_words = len.div_ceil(64);
        let mut out = BufferMut::<u64>::with_capacity(n_words);
        for (l, r) in left
            .chunks()
            .iter_padded()
            .zip(right.chunks().iter_padded())
            .take(n_words)
        {
            out.push(op(l, r));
        }
        out
    };

    let mut bytes = out.into_byte_buffer();
    bytes.truncate(n_bytes);
    BitBuffer::new(bytes.freeze(), len)
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

    /// `bitwise_binary_op` must match a naive per-bit reference for every op, offset and length,
    /// independent of the chunked kernels.
    #[rstest]
    #[case::aligned(0, 0)]
    #[case::byte_aligned(8, 16)]
    #[case::byte_aligned_mismatch(16, 0)]
    #[case::sub_byte(3, 3)]
    #[case::sub_byte_mismatch(0, 5)]
    fn binary_op_matches_naive(#[case] left_offset: usize, #[case] right_offset: usize) {
        #[allow(clippy::cast_possible_truncation)]
        let make = |offset: usize, len: usize, salt: u8| -> BitBuffer {
            let bytes: ByteBufferMut = (0..(offset + len).div_ceil(8).max(1))
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt))
                .collect();
            BitBufferMut::from_buffer(bytes, offset, len).freeze()
        };
        let ops: [fn(u64, u64) -> u64; 4] =
            [|a, b| a & b, |a, b| a | b, |a, b| a ^ b, |a, b| a & !b];

        for len in [1usize, 5, 8, 63, 64, 65, 127, 128, 200, 256] {
            let left = make(left_offset, len, 0xC3);
            let right = make(right_offset, len, 0x5A);
            for op in ops {
                let got = bitwise_binary_op(&left, &right, op);
                let expected: BitBuffer = (0..len)
                    .map(|i| op(u64::from(left.value(i)), u64::from(right.value(i))) & 1 == 1)
                    .collect();
                assert_eq!(
                    got, expected,
                    "loff={left_offset} roff={right_offset} len={len}"
                );
            }
        }
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
