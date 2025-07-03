//! An implementation of the Take kernel for primitive Arrays that uses
//! the nightly-only `portable_simd` feature.
//!
//! This is only enabled on non-x86_64 platforms and when using the nightly compiler for builds.

#![allow(unused)]

use std::mem::{MaybeUninit, transmute};
use std::simd;
use std::simd::num::SimdUint;

use multiversion::multiversion;
use num_traits::AsPrimitive;
use vortex_buffer::{Alignment, Buffer, BufferMut};
use vortex_dtype::{
    NativePType, PType, match_each_native_simd_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::VortexResult;

use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray};

pub(super) struct TakeKernelPortableSimd;

// SIMD types larger than the SIMD register size are beneficial for
// performance as this leads to better instruction level parallelism.
const SIMD_WIDTH: usize = 64;

impl TakeImpl for TakeKernelPortableSimd {
    fn take(
        &self,
        array: &PrimitiveArray,
        unsigned_indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        if array.ptype() == PType::F16 {
            // Special handling for f16 to treat as opaque u16
            let decoded = match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                take_portable_simd::<C, u16, SIMD_WIDTH>(
                    unsigned_indices.as_slice(),
                    array.reinterpret_cast(PType::U16).as_slice(),
                )
            });
            Ok(PrimitiveArray::new(decoded, validity)
                .reinterpret_cast(PType::F16)
                .into_array())
        } else {
            match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                match_each_native_simd_ptype!(array.ptype(), |V| {
                    let decoded = take_portable_simd::<C, V, SIMD_WIDTH>(
                        unsigned_indices.as_slice(),
                        array.as_slice(),
                    );
                    Ok(PrimitiveArray::new(decoded, validity).into_array())
                })
            })
        }
    }
}

/// Takes elements from an array using SIMD indexing.
///
/// # Type Parameters
/// * `C` - Index type
/// * `V` - Value type
/// * `LANE_COUNT` - Number of SIMD lanes to process in parallel
///
/// # Parameters
/// * `indices` - Indices to gather values from
/// * `values` - Source values to index
///
/// # Returns
/// A `PrimitiveArray` containing the gathered values where each index has been replaced with
/// the corresponding value from the source array.
#[multiversion(targets("x86_64+avx2", "x86_64+avx", "aarch64+neon"))]
fn take_portable_simd<I, V, const LANE_COUNT: usize>(indices: &[I], values: &[V]) -> Buffer<V>
where
    I: simd::SimdElement + AsPrimitive<usize>,
    V: simd::SimdElement + NativePType,
    simd::LaneCount<LANE_COUNT>: simd::SupportedLaneCount,
    simd::Simd<I, LANE_COUNT>: SimdUint<Cast<usize> = simd::Simd<usize, LANE_COUNT>>,
{
    let indices_len = indices.len();

    let mut buffer = BufferMut::<V>::with_capacity_aligned(
        indices_len,
        Alignment::of::<simd::Simd<V, LANE_COUNT>>(),
    );

    let buf_slice = buffer.spare_capacity_mut();

    for chunk_idx in 0..(indices_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let mask = simd::Mask::from_bitmask(u64::MAX);
        let codes_chunk = simd::Simd::<I, LANE_COUNT>::from_slice(&indices[offset..]);

        let selection = simd::Simd::gather_select(
            values,
            mask,
            codes_chunk.cast::<usize>(),
            simd::Simd::<V, LANE_COUNT>::default(),
        );

        unsafe {
            selection.store_select_unchecked(
                transmute::<&mut [MaybeUninit<V>], &mut [V]>(&mut buf_slice[offset..][..64]),
                mask.cast(),
            );
        }
    }

    for idx in ((indices_len / LANE_COUNT) * LANE_COUNT)..indices_len {
        unsafe {
            buf_slice
                .get_unchecked_mut(idx)
                .write(values[indices[idx].as_()]);
        }
    }

    unsafe {
        buffer.set_len(indices_len);
    }

    buffer.freeze()
}

#[cfg(test)]
mod tests {
    use super::take_portable_simd;

    #[test]
    fn test_take_out_of_bounds() {
        let indices = vec![2_000_000u32; 64];
        let values = vec![1i32];

        let result = take_portable_simd::<u32, i32, 64>(&indices, &values);
        assert_eq!(result.as_slice(), [0i32; 64]);
    }
}
