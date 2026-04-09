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
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::arrays::primitive::compute::take::take_primitive_scalar;
use crate::arrays::primitive::vtable::Primitive;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

const NEON32_LANES: usize = 4;
const NEON64_LANES: usize = 2;

#[allow(unused)]
pub(super) struct TakeKernelNEON;

impl TakeImpl for TakeKernelNEON {
    #[inline(always)]
    fn take(
        &self,
        values: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        assert!(indices.ptype().is_unsigned_int());

        Ok(match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            match_each_native_ptype!(values.ptype(), |V| {
                unsafe {
                    take_primitive_neon(values.as_slice::<V>(), indices.as_slice::<I>(), validity)
                }
            })
        })
        .into_array())
    }
}

/// # Safety
///
/// The caller must ensure that if the validity has a length, it is the same length as the
/// indices.
unsafe fn take_primitive_neon<V, I>(
    values: &[V],
    indices: &[I],
    validity: Validity,
) -> PrimitiveArray
where
    V: NativePType,
    I: UnsignedPType,
{
    let buffer = take_neon(values, indices);

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}

fn take_neon<V: NativePType, I: UnsignedPType>(values: &[V], indices: &[I]) -> Buffer<V> {
    macro_rules! take_reinterpreted {
        ($cast:ty, $kernel:ident) => {{
            let values = unsafe { cast_slice::<V, $cast>(values) };
            let taken = $kernel(values, indices);
            unsafe { taken.transmute::<V>() }
        }};
    }

    match V::PTYPE {
        PType::U32 | PType::I32 | PType::F32 => take_reinterpreted!(u32, take_neon_u32),
        PType::U64 | PType::I64 | PType::F64 => take_reinterpreted!(u64, take_neon_u64),
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

#[inline(always)]
unsafe fn cast_slice<T, U>(slice: &[T]) -> &[U] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<U>(), slice.len()) }
}

#[inline(always)]
fn finish_buffer<T>(mut buffer: BufferMut<T>, len: usize) -> Buffer<T> {
    unsafe { buffer.set_len(len) };
    buffer = buffer.aligned(Alignment::of::<T>());
    buffer.freeze()
}

fn take_neon_u32<I: UnsignedPType>(values: &[u32], indices: &[I]) -> Buffer<u32> {
    let len = indices.len();
    let mut buffer = BufferMut::<u32>::with_capacity_aligned(len, Alignment::of::<uint32x4_t>());
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

    finish_buffer(buffer, len)
}

fn take_neon_u64<I: UnsignedPType>(values: &[u64], indices: &[I]) -> Buffer<u64> {
    let len = indices.len();
    let mut buffer = BufferMut::<u64>::with_capacity_aligned(len, Alignment::of::<uint64x2_t>());
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

    finish_buffer(buffer, len)
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
