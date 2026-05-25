// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! EXPERIMENTAL AVX-512 take kernel, used to benchmark AVX-512 gather against the scalar and
//! AVX2 paths. Only specializes u32 indices into 32- and 64-bit value types; everything else
//! falls back to the scalar kernel.

#![allow(
    unused,
    clippy::many_single_char_names,
    clippy::cognitive_complexity,
    clippy::cast_possible_truncation
)]

use std::arch::x86_64::*;
use std::mem::size_of;

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
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

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

        // Only the u32-index path is specialized; other index widths use scalar.
        if indices.ptype() != PType::U32 {
            return match_each_native_ptype!(values.ptype(), |V| {
                match_each_integer_ptype!(indices.ptype(), |I| {
                    let buffer =
                        take_primitive_scalar(values.as_slice::<V>(), indices.as_slice::<I>());
                    Ok(PrimitiveArray::new(buffer, validity).into_array())
                })
            });
        }

        let idx = indices.as_slice::<u32>();
        Ok(match_each_native_ptype!(values.ptype(), |V| {
            let vals = values.as_slice::<V>();
            let buffer: Buffer<V> = match size_of::<V>() {
                4 => {
                    // SAFETY: slice references are `(ptr, len)`; the gather only runs for 4-byte V.
                    let v32 = unsafe { std::mem::transmute::<&[V], &[u32]>(vals) };
                    // SAFETY: kernel selected only when avx512f is available.
                    unsafe { gather32(v32, idx).transmute::<V>() }
                }
                8 => {
                    // SAFETY: see above; only runs for 8-byte V.
                    let v64 = unsafe { std::mem::transmute::<&[V], &[u64]>(vals) };
                    // SAFETY: kernel selected only when avx512f+avx512vl are available.
                    unsafe { gather64(v64, idx).transmute::<V>() }
                }
                _ => take_primitive_scalar(vals, idx),
            };
            PrimitiveArray::new(buffer, validity).into_array()
        }))
    }
}

/// 16-wide AVX-512 gather of 32-bit values using u32 indices.
///
/// Out-of-bounds indices yield zero, matching the AVX2 kernel's masked-gather semantics.
///
/// # Safety
/// The caller must ensure the `avx512f` feature is enabled.
#[target_feature(enable = "avx512f")]
unsafe fn gather32(values: &[u32], indices: &[u32]) -> Buffer<u32> {
    let n = indices.len();
    let mut out = BufferMut::<u32>::with_capacity_aligned(n, Alignment::of::<__m512i>());
    let dst = out.spare_capacity_mut().as_mut_ptr().cast::<u32>();

    let base = values.as_ptr() as *const i32;
    let len_vec = _mm512_set1_epi32(values.len() as i32);
    let zero = _mm512_setzero_si512();

    unsafe {
        let mut i = 0usize;
        while i + 16 <= n {
            let vindex = _mm512_loadu_epi32(indices.as_ptr().add(i) as *const i32);
            // unsigned compare: lane valid when index < len
            let k = _mm512_cmplt_epu32_mask(vindex, len_vec);
            let gathered = _mm512_mask_i32gather_epi32::<4>(zero, k, vindex, base);
            _mm512_storeu_epi32(dst.add(i) as *mut i32, gathered);
            i += 16;
        }
        while i < n {
            let ix = indices[i] as usize;
            let v = if ix < values.len() { values[ix] } else { 0 };
            dst.add(i).write(v);
            i += 1;
        }

        out.set_len(n);
    }
    out = out.aligned(Alignment::of::<u32>());
    out.freeze()
}

/// 8-wide AVX-512 gather of 64-bit values using u32 indices.
///
/// # Safety
/// The caller must ensure the `avx512f` and `avx512vl` features are enabled.
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn gather64(values: &[u64], indices: &[u32]) -> Buffer<u64> {
    let n = indices.len();
    let mut out = BufferMut::<u64>::with_capacity_aligned(n, Alignment::of::<__m512i>());
    let dst = out.spare_capacity_mut().as_mut_ptr().cast::<u64>();

    let base = values.as_ptr() as *const i64;
    let len_vec = _mm256_set1_epi32(values.len() as i32);
    let zero = _mm512_setzero_si512();

    unsafe {
        let mut i = 0usize;
        while i + 8 <= n {
            let vindex = _mm256_loadu_si256(indices.as_ptr().add(i) as *const __m256i);
            let k = _mm256_cmplt_epu32_mask(vindex, len_vec);
            let gathered = _mm512_mask_i32gather_epi64::<8>(zero, k, vindex, base);
            _mm512_storeu_epi64(dst.add(i) as *mut i64, gathered);
            i += 8;
        }
        while i < n {
            let ix = indices[i] as usize;
            let v = if ix < values.len() { values[ix] } else { 0 };
            dst.add(i).write(v);
            i += 1;
        }

        out.set_len(n);
    }
    out = out.aligned(Alignment::of::<u64>());
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference32(values: &[u32], indices: &[u32]) -> Vec<u32> {
        indices
            .iter()
            .map(|&i| values.get(i as usize).copied().unwrap_or(0))
            .collect()
    }

    fn reference64(values: &[u64], indices: &[u32]) -> Vec<u64> {
        indices
            .iter()
            .map(|&i| values.get(i as usize).copied().unwrap_or(0))
            .collect()
    }

    // Exercise vector body + tail across many lengths, plus out-of-bounds indices.
    #[test]
    fn gather32_matches_reference() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let values: Vec<u32> = (0..1000u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        for n in [0usize, 1, 7, 15, 16, 17, 31, 33, 100, 1000] {
            let mut indices: Vec<u32> =
                (0..n as u32).map(|i| (i.wrapping_mul(37)) % 1000).collect();
            // Inject out-of-bounds indices to verify masked-gather zeroing.
            if n > 5 {
                indices[3] = 1000; // == len, OOB
                indices[n - 1] = 5000; // OOB
            }
            let got = unsafe { gather32(&values, &indices) };
            assert_eq!(
                got.as_slice(),
                reference32(&values, &indices).as_slice(),
                "n={n}"
            );
        }
    }

    #[test]
    fn gather64_matches_reference() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
            return;
        }
        let values: Vec<u64> = (0..1000u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 100, 1000] {
            let mut indices: Vec<u32> =
                (0..n as u32).map(|i| (i.wrapping_mul(37)) % 1000).collect();
            if n > 5 {
                indices[2] = 1000;
                indices[n - 1] = 9999;
            }
            let got = unsafe { gather64(&values, &indices) };
            assert_eq!(
                got.as_slice(),
                reference64(&values, &indices).as_slice(),
                "n={n}"
            );
        }
    }
}
