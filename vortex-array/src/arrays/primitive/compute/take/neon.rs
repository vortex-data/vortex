// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AArch64 NEON implementation of take operation.
//!
//! NEON has no native gather instruction, so the best stable fallback is to batch scalar indexed
//! loads and use full-width NEON loads/stores for the 32-bit and 64-bit value paths.

use core::arch::aarch64::uint32x4_t;
use core::arch::aarch64::uint64x2_t;
use core::arch::aarch64::vld1q_u32;
use core::arch::aarch64::vld1q_u64;
use core::arch::aarch64::vst1q_u32;
use core::arch::aarch64::vst1q_u64;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;

use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TypedTakeImpl;
use crate::arrays::primitive::compute::take::finish_simd_buffer;
use crate::arrays::primitive::compute::take::new_simd_buffer;
use crate::arrays::primitive::compute::take::take_primitive_scalar;
use crate::arrays::primitive::compute::take::take_primitive_with_validity;
use crate::arrays::primitive::compute::take::take_reinterpreted;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::validity::Validity;

const NEON32_LANES: usize = 4;
const NEON64_LANES: usize = 2;

#[allow(unused)]
pub(super) struct TakeKernelNEON;

impl TypedTakeImpl for TakeKernelNEON {
    #[inline(always)]
    unsafe fn take_typed<V, I>(
        &self,
        values: &[V],
        indices: &[I],
        validity: Validity,
    ) -> PrimitiveArray
    where
        V: NativePType,
        I: UnsignedPType,
    {
        unsafe { take_primitive_with_validity(values, indices, validity, take_neon) }
    }
}

fn take_neon<V: NativePType, I: UnsignedPType>(values: &[V], indices: &[I]) -> Buffer<V> {
    match V::PTYPE {
        PType::U32 | PType::I32 | PType::F32 => unsafe {
            take_reinterpreted(values, indices, take_neon_u32::<I>)
        },
        PType::U64 | PType::I64 | PType::F64 => unsafe {
            take_reinterpreted(values, indices, take_neon_u64::<I>)
        },
        _ => {
            tracing::trace!(
                "take NEON kernel missing for indices {} values {}, falling back to scalar",
                I::PTYPE,
                V::PTYPE
            );

            take_primitive_scalar(values, indices)
        }
    }
}

fn take_neon_u32<I: UnsignedPType>(values: &[u32], indices: &[I]) -> Buffer<u32> {
    let len = indices.len();
    let mut buffer = new_simd_buffer::<u32>(len, Alignment::of::<uint32x4_t>());
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u32>();

    let mut offset = 0;
    while offset + NEON32_LANES <= len {
        let mut lanes = [0u32; NEON32_LANES];
        for lane in 0..NEON32_LANES {
            let index = indices[offset + lane].as_();
            if index < values.len() {
                lanes[lane] = values[index];
            }
        }

        unsafe {
            let gathered = vld1q_u32(lanes.as_ptr());
            vst1q_u32(out.add(offset), gathered);
        }
        offset += NEON32_LANES;
    }

    while offset < len {
        let index = indices[offset].as_();
        unsafe { out.add(offset).write(values[index]) };
        offset += 1;
    }

    finish_simd_buffer(buffer, len)
}

fn take_neon_u64<I: UnsignedPType>(values: &[u64], indices: &[I]) -> Buffer<u64> {
    let len = indices.len();
    let mut buffer = new_simd_buffer::<u64>(len, Alignment::of::<uint64x2_t>());
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    let mut offset = 0;
    while offset + NEON64_LANES <= len {
        let mut lanes = [0u64; NEON64_LANES];
        for lane in 0..NEON64_LANES {
            let index = indices[offset + lane].as_();
            if index < values.len() {
                lanes[lane] = values[index];
            }
        }

        unsafe {
            let gathered = vld1q_u64(lanes.as_ptr());
            vst1q_u64(out.add(offset), gathered);
        }
        offset += NEON64_LANES;
    }

    while offset < len {
        let index = indices[offset].as_();
        unsafe { out.add(offset).write(values[index]) };
        offset += 1;
    }

    finish_simd_buffer(buffer, len)
}

#[cfg(test)]
mod neon_tests {
    use super::*;
    use crate::arrays::primitive::compute::take::take_primitive_scalar;

    macro_rules! test_cases {
        (index_type => $IDX:ty, value_types => $($VAL:ty),+ $(,)?) => {
            paste::paste! {
                $(
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_neon_take_matches_scalar_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = (0..257).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (0..67).map(|i| ((i * 5) % 257) as $IDX).collect();

                        let expected = take_primitive_scalar(&values, &indices);
                        let actual = take_neon(&values, &indices);

                        assert_eq!(actual.as_slice(), expected.as_slice());
                    }
                )+
            }
        };
    }

    #[test]
    fn test_neon_take_zero_fills_out_of_bounds() {
        let values = vec![11u32, 22, 33];
        let indices = vec![0u32, 2, 9, 1];

        let actual = take_neon(&values, &indices);

        assert_eq!(actual.as_slice(), &[11, 33, 0, 22]);
    }

    test_cases!(
        index_type => u8,
        value_types => u32, i32, u64, i64, f32, f64
    );
    test_cases!(
        index_type => u16,
        value_types => u32, i32, u64, i64, f32, f64
    );
    test_cases!(
        index_type => u32,
        value_types => u32, i32, u64, i64, f32, f64
    );
    test_cases!(
        index_type => u64,
        value_types => u32, i32, u64, i64, f32, f64
    );
}
