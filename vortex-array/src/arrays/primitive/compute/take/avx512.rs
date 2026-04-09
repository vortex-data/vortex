// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX-512 implementation of take operation using gather instructions.
//!
//! We only specialize by value width here:
//! - 32-bit values use the 16-lane AVX-512 gather path.
//! - 64-bit values use the 8-lane AVX-512 gather path.
//!
//! That keeps the logic much smaller than the AVX2 matrix while still using the full AVX-512F
//! gather width the hardware exposes.

use std::arch::x86_64::__m512i;
use std::arch::x86_64::_mm_loadl_epi64;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::arch::x86_64::_mm512_cmplt_epu32_mask;
use std::arch::x86_64::_mm512_cmplt_epu64_mask;
use std::arch::x86_64::_mm512_cvtepu8_epi32;
use std::arch::x86_64::_mm512_cvtepu8_epi64;
use std::arch::x86_64::_mm512_cvtepu16_epi32;
use std::arch::x86_64::_mm512_cvtepu16_epi64;
use std::arch::x86_64::_mm512_cvtepu32_epi64;
use std::arch::x86_64::_mm512_loadu_si512;
use std::arch::x86_64::_mm512_mask_i32gather_epi32;
use std::arch::x86_64::_mm512_mask_i64gather_epi32;
use std::arch::x86_64::_mm512_mask_i64gather_epi64;
use std::arch::x86_64::_mm512_set1_epi32;
use std::arch::x86_64::_mm512_set1_epi64;
use std::arch::x86_64::_mm512_setzero_si512;
use std::arch::x86_64::_mm512_storeu_si512;

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

const GATHER32_LANES: usize = 16;
const GATHER64_LANES: usize = 8;

#[allow(unused)]
pub(super) struct TakeKernelAVX512;

impl TakeImpl for TakeKernelAVX512 {
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
                // SAFETY: This kernel is only selected when avx512f cpu-feature is detected.
                unsafe {
                    take_primitive_avx512(values.as_slice::<V>(), indices.as_slice::<I>(), validity)
                }
            })
        })
        .into_array())
    }
}

/// # Safety
///
/// The caller must ensure that if the validity has a length, it is the same length as the
/// indices, and that the `avx512f` feature is enabled.
#[target_feature(enable = "avx512f")]
unsafe fn take_primitive_avx512<V, I>(
    values: &[V],
    indices: &[I],
    validity: Validity,
) -> PrimitiveArray
where
    V: NativePType,
    I: UnsignedPType,
{
    let buffer = unsafe { take_avx512(values, indices) };

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}

/// # Safety
///
/// The caller must ensure the `avx512f` feature is enabled.
#[target_feature(enable = "avx512f")]
unsafe fn take_avx512<V: NativePType, I: UnsignedPType>(values: &[V], indices: &[I]) -> Buffer<V> {
    macro_rules! take_reinterpreted {
        ($cast:ty, $kernel:ident) => {{
            let values = unsafe { cast_slice::<V, $cast>(values) };
            let taken = $kernel::<I>(values, indices);
            unsafe { taken.transmute::<V>() }
        }};
    }

    match V::PTYPE {
        PType::U32 | PType::I32 | PType::F32 => take_reinterpreted!(u32, take_avx512_u32),
        PType::U64 | PType::I64 | PType::F64 => take_reinterpreted!(u64, take_avx512_u64),
        _ => {
            tracing::trace!(
                "take AVX-512 kernel missing for indices {} values {}, falling back to scalar",
                I::PTYPE,
                V::PTYPE
            );

            take_primitive_scalar(values, indices)
        }
    }
}

fn take_avx512_u32<I: UnsignedPType>(values: &[u32], indices: &[I]) -> Buffer<u32> {
    match I::PTYPE {
        PType::U8 => unsafe { take_avx512_u32_u8(values, cast_slice::<I, u8>(indices)) },
        PType::U16 => unsafe { take_avx512_u32_u16(values, cast_slice::<I, u16>(indices)) },
        PType::U32 => unsafe { take_avx512_u32_u32(values, cast_slice::<I, u32>(indices)) },
        PType::U64 => unsafe { take_avx512_u32_u64(values, cast_slice::<I, u64>(indices)) },
        _ => unreachable!("unsigned take indices are always u8/u16/u32/u64"),
    }
}

fn take_avx512_u64<I: UnsignedPType>(values: &[u64], indices: &[I]) -> Buffer<u64> {
    match I::PTYPE {
        PType::U8 => unsafe { take_avx512_u64_u8(values, cast_slice::<I, u8>(indices)) },
        PType::U16 => unsafe { take_avx512_u64_u16(values, cast_slice::<I, u16>(indices)) },
        PType::U32 => unsafe { take_avx512_u64_u32(values, cast_slice::<I, u32>(indices)) },
        PType::U64 => unsafe { take_avx512_u64_u64(values, cast_slice::<I, u64>(indices)) },
        _ => unreachable!("unsigned take indices are always u8/u16/u32/u64"),
    }
}

#[inline(always)]
unsafe fn cast_slice<T, U>(slice: &[T]) -> &[U] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<U>(), slice.len()) }
}

#[inline(always)]
fn new_buffer<T>(len: usize) -> BufferMut<T> {
    BufferMut::with_capacity_aligned(len, Alignment::of::<__m512i>())
}

#[inline(always)]
fn finish_buffer<T>(mut buffer: BufferMut<T>, len: usize) -> Buffer<T> {
    unsafe { buffer.set_len(len) };
    buffer = buffer.aligned(Alignment::of::<T>());
    buffer.freeze()
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u32_u8(values: &[u32], indices: &[u8]) -> Buffer<u32> {
    let len = indices.len();
    let max_index = u32::try_from(values.len()).unwrap_or(u32::MAX);
    let max_index_vec = _mm512_set1_epi32(max_index as i32);
    let mut buffer = new_buffer::<u32>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u32>();

    let mut offset = 0;
    while offset + GATHER32_LANES <= len {
        let index_vec = _mm512_cvtepu8_epi32(_mm_loadu_si128(indices.as_ptr().add(offset).cast()));
        let mask = _mm512_cmplt_epu32_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i32gather_epi32::<4>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER32_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u32_u16(values: &[u32], indices: &[u16]) -> Buffer<u32> {
    let len = indices.len();
    let max_index = u32::try_from(values.len()).unwrap_or(u32::MAX);
    let max_index_vec = _mm512_set1_epi32(max_index as i32);
    let mut buffer = new_buffer::<u32>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u32>();

    let mut offset = 0;
    while offset + GATHER32_LANES <= len {
        let index_vec =
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(indices.as_ptr().add(offset).cast()));
        let mask = _mm512_cmplt_epu32_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i32gather_epi32::<4>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER32_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u32_u32(values: &[u32], indices: &[u32]) -> Buffer<u32> {
    let len = indices.len();
    let max_index = u32::try_from(values.len()).unwrap_or(u32::MAX);
    let max_index_vec = _mm512_set1_epi32(max_index as i32);
    let mut buffer = new_buffer::<u32>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u32>();

    let mut offset = 0;
    while offset + GATHER32_LANES <= len {
        let index_vec = _mm512_loadu_si512(indices.as_ptr().add(offset).cast());
        let mask = _mm512_cmplt_epu32_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i32gather_epi32::<4>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER32_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u32_u64(values: &[u32], indices: &[u64]) -> Buffer<u32> {
    let len = indices.len();
    let max_index = values.len() as u64;
    let max_index_vec = _mm512_set1_epi64(max_index as i64);
    let mut buffer = new_buffer::<u32>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u32>();

    let mut offset = 0;
    while offset + GATHER64_LANES <= len {
        let index_vec = _mm512_loadu_si512(indices.as_ptr().add(offset).cast());
        let mask = _mm512_cmplt_epu64_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i64gather_epi32::<4>(
            _mm256_setzero_si256(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm256_storeu_si256(out.add(offset).cast(), gathered);
        offset += GATHER64_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u64_u8(values: &[u64], indices: &[u8]) -> Buffer<u64> {
    let len = indices.len();
    let max_index = values.len() as u64;
    let max_index_vec = _mm512_set1_epi64(max_index as i64);
    let mut buffer = new_buffer::<u64>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    let mut offset = 0;
    while offset + GATHER64_LANES <= len {
        let index_vec = _mm512_cvtepu8_epi64(_mm_loadl_epi64(indices.as_ptr().add(offset).cast()));
        let mask = _mm512_cmplt_epu64_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i64gather_epi64::<8>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER64_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u64_u16(values: &[u64], indices: &[u16]) -> Buffer<u64> {
    let len = indices.len();
    let max_index = values.len() as u64;
    let max_index_vec = _mm512_set1_epi64(max_index as i64);
    let mut buffer = new_buffer::<u64>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    let mut offset = 0;
    while offset + GATHER64_LANES <= len {
        let index_vec = _mm512_cvtepu16_epi64(_mm_loadu_si128(indices.as_ptr().add(offset).cast()));
        let mask = _mm512_cmplt_epu64_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i64gather_epi64::<8>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER64_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u64_u32(values: &[u64], indices: &[u32]) -> Buffer<u64> {
    let len = indices.len();
    let max_index = values.len() as u64;
    let max_index_vec = _mm512_set1_epi64(max_index as i64);
    let mut buffer = new_buffer::<u64>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    let mut offset = 0;
    while offset + GATHER64_LANES <= len {
        let index_vec =
            _mm512_cvtepu32_epi64(_mm256_loadu_si256(indices.as_ptr().add(offset).cast()));
        let mask = _mm512_cmplt_epu64_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i64gather_epi64::<8>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER64_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn take_avx512_u64_u64(values: &[u64], indices: &[u64]) -> Buffer<u64> {
    let len = indices.len();
    let max_index = values.len() as u64;
    let max_index_vec = _mm512_set1_epi64(max_index as i64);
    let mut buffer = new_buffer::<u64>(len);
    let out = buffer.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    let mut offset = 0;
    while offset + GATHER64_LANES <= len {
        let index_vec = _mm512_loadu_si512(indices.as_ptr().add(offset).cast());
        let mask = _mm512_cmplt_epu64_mask(index_vec, max_index_vec);
        let gathered = _mm512_mask_i64gather_epi64::<8>(
            _mm512_setzero_si512(),
            mask,
            index_vec,
            values.as_ptr().cast(),
        );

        _mm512_storeu_si512(out.add(offset).cast(), gathered);
        offset += GATHER64_LANES;
    }

    while offset < len {
        out.add(offset).write(values[indices[offset] as usize]);
        offset += 1;
    }

    finish_buffer(buffer, len)
}

#[cfg(test)]
#[cfg_attr(miri, ignore)]
#[cfg(target_arch = "x86_64")]
mod avx512_tests {
    use super::*;
    use crate::arrays::primitive::compute::take::take_primitive_scalar;

    macro_rules! test_cases {
        (index_type => $IDX:ty, value_types => $($VAL:ty),+ $(,)?) => {
            paste::paste! {
                $(
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_matches_scalar_ $IDX _ $VAL>]() {
                        if !std::is_x86_feature_detected!("avx512f") {
                            return;
                        }

                        let values: Vec<$VAL> = (0..257).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (0..64).map(|i| ((i * 3) % 257) as $IDX).collect();

                        let expected = take_primitive_scalar(&values, &indices);
                        let actual = unsafe { take_avx512(&values, &indices) };

                        assert_eq!(actual.as_slice(), expected.as_slice());
                    }
                )+
            }
        };
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
