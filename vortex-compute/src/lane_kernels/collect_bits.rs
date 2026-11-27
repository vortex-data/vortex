// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernels that collect bits into a [`BitBuffer`].

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;

use crate::lane_kernels::IndexedSourceExt;
use crate::lane_kernels::LaneZip;
use crate::lane_kernels::low_bits_mask;

/// Freeze `len` bits packed into `words` (LSB-first, 64 lanes per word) into a [`BitBuffer`].
pub fn bit_buffer_from_words(words: BufferMut<u64>, len: usize) -> BitBuffer {
    debug_assert!(words.len() * 64 >= len);

    let mut bytes = words.into_byte_buffer();
    bytes.truncate(len.div_ceil(8));
    BitBuffer::new(bytes.freeze(), len)
}

/// Build a bitmap filled with `fill`, with the opposite bit at each supplied index.
///
/// # Panics
///
/// Panics if an index is outside `0..len`.
pub fn collect_sparse_bits(
    len: usize,
    fill: bool,
    exceptions: impl IntoIterator<Item = usize>,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let mut words = BufferMut::full_in(
        if fill { u64::MAX } else { 0 },
        len.div_ceil(64),
        allocator.clone(),
    );
    let dst = words.as_mut_slice();
    for index in exceptions {
        assert!(index < len, "bit index {index} exceeds length {len}");
        let bit = 1 << (index % 64);
        if fill {
            dst[index / 64] &= !bit;
        } else {
            dst[index / 64] |= bit;
        }
    }
    if fill && !len.is_multiple_of(64) {
        dst[len / 64] &= low_bits_mask(len % 64);
    }
    bit_buffer_from_words(words, len)
}

/// Bit-pack the predicate `f(values[i])` over a slice into a [`BitBuffer`].
pub fn collect_bits<T: Copy>(
    values: &[T],
    f: impl Fn(T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let len = values.len();
    let mut words = BufferMut::<u64>::zeroed_in(len.div_ceil(64), allocator.clone());
    values.map_bits_into(words.as_mut_slice(), f);
    bit_buffer_from_words(words, len)
}

/// Bit-pack the predicate `f(lhs[i], rhs[i])` over two equal-length slices into a [`BitBuffer`].
pub fn collect_zip_bits<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let len = lhs.len();
    let mut words = BufferMut::<u64>::zeroed_in(len.div_ceil(64), allocator.clone());
    LaneZip::new(lhs, rhs).map_bits_into(words.as_mut_slice(), |(a, b)| f(a, b));
    bit_buffer_from_words(words, len)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BufferAllocatorRef;

    use super::collect_sparse_bits;

    #[rstest]
    fn sparse_exceptions_cross_words(#[values(false, true)] fill: bool) {
        let allocator = BufferAllocatorRef::statically_allocated();
        let bits = collect_sparse_bits(67, fill, [66, 0, 64, 0], &allocator);
        assert!(
            bits.iter()
                .eq((0..67).map(|i| fill ^ matches!(i, 0 | 64 | 66)))
        );
    }
}
