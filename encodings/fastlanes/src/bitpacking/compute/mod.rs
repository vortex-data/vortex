// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::mem::size_of;

use fastlanes::BitPacking;
use vortex_array::arrays::filter::ChunkDecoder;
use vortex_array::arrays::filter::FILTER_CHUNK_LEN;

use crate::FL_CHUNK_SIZE;

mod between;
mod cast;
mod compare;
mod compare_fused;
mod filter;
pub(crate) mod is_constant;
mod packed_range;
mod slice;
mod stream_predicate;
mod take;

const fn unpack_chunk_threshold<T>() -> usize {
    // FastLanes and Vortex benchmarks set conservative crossovers for each physical type.
    match size_of::<T>() {
        1 => 16,
        2 => 32,
        4 => 64,
        8 => 160,
        _ => unreachable!(),
    }
}

/// The packed FastLanes chunks of a bit-packed array.
pub(super) struct PackedChunks<'a, T> {
    packed: &'a [T],
    bit_width: usize,
    /// Number of `T` words that hold each packed chunk.
    packed_chunk_len: usize,
}

impl<'a, T> PackedChunks<'a, T> {
    /// The chunks of `packed`, at `bit_width` bits per value.
    pub(super) fn new(packed: &'a [T], bit_width: usize) -> Self {
        Self {
            packed,
            bit_width,
            packed_chunk_len: FL_CHUNK_SIZE * bit_width / (8 * size_of::<T>()),
        }
    }

    #[inline]
    fn chunk(&self, chunk_idx: usize) -> &[T] {
        &self.packed[chunk_idx * self.packed_chunk_len..][..self.packed_chunk_len]
    }
}

// SAFETY: the FastLanes unpack kernels initialize every value of `dst`.
unsafe impl<T: BitPacking> ChunkDecoder<T> for PackedChunks<'_, T> {
    #[inline]
    fn decode_chunk(&self, chunk_idx: usize, dst: &mut [MaybeUninit<T>; FILTER_CHUNK_LEN]) {
        // SAFETY: the unpack only writes to `dst`, and the packed chunk holds `FL_CHUNK_SIZE`
        // values of `bit_width` bits.
        unsafe {
            BitPacking::unchecked_unpack(
                self.bit_width,
                self.chunk(chunk_idx),
                dst.assume_init_mut(),
            );
        }
    }

    #[inline]
    fn decode_indices(&self, chunk_idx: usize, indices: &[usize], dst: &mut [MaybeUninit<T>]) {
        debug_assert!(indices.iter().all(|&index| index < FL_CHUNK_SIZE));
        // SAFETY: every index is within the chunk, and `dst` holds one value for each index.
        unsafe {
            BitPacking::unchecked_unpack_indices(
                self.bit_width,
                self.chunk(chunk_idx),
                indices,
                dst,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_session::VortexSession;

    use crate::BitPackedArray;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn bp(array: &PrimitiveArray, bit_width: u8) -> BitPackedArray {
        bitpack_encode(array, bit_width, None, &mut SESSION.create_execution_ctx()).unwrap()
    }

    #[rstest]
    // Basic integer arrays that can be bitpacked
    #[case::u8_small(bp(&PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]), 3))]
    #[case::u16_array(bp(&PrimitiveArray::from_iter([10u16, 20, 30, 40, 50]), 6))]
    #[case::u32_array(bp(&PrimitiveArray::from_iter([100u32, 200, 300, 400, 500]), 9))]
    // Arrays with nulls
    #[case::nullable_u8(bp(&PrimitiveArray::from_option_iter([Some(1u8), None, Some(3), Some(4), None]), 3))]
    #[case::nullable_u32(bp(&PrimitiveArray::from_option_iter([Some(100u32), None, Some(300), Some(400), None]), 9))]
    // Edge cases
    #[case::single_element(bp(&PrimitiveArray::from_iter([42u32]), 6))]
    #[case::all_zeros(bp(&PrimitiveArray::from_iter([0u16; 100]), 1))]
    // Large arrays (multiple chunks - fastlanes uses 1024-element chunks)
    #[case::large_u16(bp(&PrimitiveArray::from_iter((0..2048).map(|i| (i % 256) as u16)), 8))]
    #[case::large_u32(bp(&PrimitiveArray::from_iter((0..3000).map(|i| (i % 1024) as u32)), 10))]
    #[case::large_u8_many_chunks(bp(&PrimitiveArray::from_iter((0..5120).map(|i| (i % 128) as u8)), 7))] // 5 chunks
    #[case::large_nullable(bp(&PrimitiveArray::from_option_iter((0..2500).map(|i| if i % 10 == 0 { None } else { Some((i % 512) as u16) })), 9))]
    // Arrays with specific bit patterns
    #[case::max_value_for_bits(bp(&PrimitiveArray::from_iter([7u8, 7, 7, 7, 7]), 3))] // max value for 3 bits
    #[case::alternating_bits(bp(&PrimitiveArray::from_iter([0u16, 255, 0, 255, 0, 255]), 8))]

    fn test_bitpacked_consistency(#[case] array: BitPackedArray) {
        let ctx = &mut SESSION.create_execution_ctx();
        test_array_consistency(&array.into_array(), ctx);
    }

    #[rstest]
    #[case::u8_basic(bp(&PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]), 3))]
    #[case::u16_basic(bp(&PrimitiveArray::from_iter([10u16, 20, 30, 40, 50]), 6))]
    #[case::u32_basic(bp(&PrimitiveArray::from_iter([100u32, 200, 300, 400, 500]), 9))]
    #[case::u64_basic(bp(&PrimitiveArray::from_iter([1000u64, 2000, 3000, 4000, 5000]), 13))]
    #[case::i32_basic(bp(&PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]), 7))]
    #[case::large_u32(bp(&PrimitiveArray::from_iter((0..100).map(|i| i as u32)), 7))]
    fn test_bitpacked_binary_numeric(#[case] array: BitPackedArray) {
        test_binary_numeric_array(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
