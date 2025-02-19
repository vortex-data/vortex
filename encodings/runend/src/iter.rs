use std::cmp::min;
use std::ops::{Add, Sub};
use std::simd::cmp::SimdOrd;
use std::simd::{Simd, SimdElement};

use num_traits::{AsPrimitive, FromPrimitive};
use vortex_array::arrays::PrimitiveArray;
use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

#[inline]
pub fn trimmed_ends_iter<E: NativePType + FromPrimitive + AsPrimitive<usize> + Ord>(
    run_ends: &[E],
    offset: usize,
    length: usize,
) -> impl Iterator<Item = usize> + use<'_, E> {
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });
    run_ends
        .iter()
        .copied()
        .map(move |v| v - offset_e)
        .map(move |v| min(v, length_e))
        .map(|v| v.as_())
}

const LANE_COUNT: usize = 64;

#[inline(always)]
pub fn trimmed_ends<E>(run_ends: PrimitiveArray, offset: usize, length: usize) -> Vec<E>
where
    E: NativePType
        + Ord
        + SimdElement
        + AsPrimitive<usize>
        + Add<Output = E>
        + Sub<Output = E>
        + Copy,
    Simd<E, LANE_COUNT>: SimdOrd + Sub<Output = Simd<E, LANE_COUNT>>,
{
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });

    let slice = run_ends.as_slice::<E>();
    let len = slice.len();
    let mut result = Vec::with_capacity(len);
    let offset_simd = Simd::splat(offset_e);
    let length_simd = Simd::splat(length_e);

    for chunk_idx in 0..(len / LANE_COUNT) {
        let v = Simd::from_slice(&slice[chunk_idx * LANE_COUNT..]);
        result.extend(Simd::simd_min(v - offset_simd, length_simd).as_array());
    }

    for idx in ((len / LANE_COUNT) * LANE_COUNT)..len {
        result.push(min(slice[idx] - offset_e, length_e));
    }

    result
}
