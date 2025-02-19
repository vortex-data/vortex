use std::simd;

use num_traits::AsPrimitive;
use simd::num::SimdUint;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::{Alignment, BufferMut};
use vortex_dtype::{NativePType, Nullability};

/// Decodes a dictionary by mapping codes to their corresponding values.
///
/// # Type Parameters
/// * `C` - The type of the codes
/// * `V` - The type of the values
/// * `LANE_COUNT` - The number of SIMD lanes
///
/// # Parameters
/// * `codes` - Slice containing the dictionary codes/indices
/// * `values` - Slice containing the dictionary values to map to
///
/// # Returns
/// A `PrimitiveArray` containing the decoded values where each code has been replaced with its
/// corresponding value from the dictionary.
pub(crate) fn dict_decode_typed_primitive<C, V, const LANE_COUNT: usize>(
    codes: &[C],
    values: &[V],
    nullability: Nullability,
) -> PrimitiveArray
where
    C: simd::SimdElement + AsPrimitive<usize>,
    V: simd::SimdElement + NativePType,
    simd::LaneCount<LANE_COUNT>: simd::SupportedLaneCount,
    simd::Simd<C, LANE_COUNT>: SimdUint<Cast<usize> = simd::Simd<usize, LANE_COUNT>>,
{
    let codes_len = codes.len();

    let mut buffer = BufferMut::<V>::with_capacity_aligned(
        codes_len,
        Alignment::of::<simd::Simd<V, LANE_COUNT>>(),
    );

    for chunk_idx in 0..(codes_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let mask = simd::Mask::from_bitmask(u64::MAX);
        let codes_chunk = simd::Simd::<C, LANE_COUNT>::from_slice(&codes[offset..]);

        unsafe {
            let selection = simd::Simd::gather_select_unchecked(
                values,
                mask,
                codes_chunk.cast::<usize>(),
                simd::Simd::<V, LANE_COUNT>::default(),
            );

            selection.store_select_ptr(buffer.as_mut_ptr().add(offset), mask.cast());
        }
    }

    for idx in ((codes_len / LANE_COUNT) * LANE_COUNT)..codes_len {
        unsafe {
            *buffer.as_mut_ptr().add(idx) = values[codes[idx].as_()];
        }
    }

    unsafe {
        buffer.set_len(codes_len);
    }

    // TOOD(alex): handle nullable values & codes
    PrimitiveArray::new(buffer.freeze(), nullability.into())
}
