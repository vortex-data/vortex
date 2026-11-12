// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Allocating wrappers around [`IndexedSourceExt::map_bits_into`] that return a [`BitBuffer`].

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;

use crate::lane_kernels::IndexedSourceExt;
use crate::lane_kernels::LaneZip;

/// Freeze `len` bits packed into `words` (LSB-first, 64 lanes per word) into a [`BitBuffer`].
pub fn bit_buffer_from_words(words: BufferMut<u64>, len: usize) -> BitBuffer {
    debug_assert!(words.len() * 64 >= len);

    let mut bytes = words.into_byte_buffer();
    bytes.truncate(len.div_ceil(8));
    BitBuffer::new(bytes.freeze(), len)
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
