// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX-512 implementation of take operation using gather instructions.
//!
//! Only enabled for x86_64 hosts and gated at runtime behind feature detection to ensure the
//! AVX-512 instruction subsets we need (`avx512f`, `avx512bw`, `avx512dq`, `avx512vl`) are
//! available.
//!
//! Compared with the AVX2 equivalent, every 32-bit gather processes 16 lanes per instruction
//! (vs. 8 with `_mm256_mask_i32gather_epi32`) and every 64-bit gather processes 8 lanes per
//! instruction (vs. 4 with `_mm256_mask_i64gather_epi64`).

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use std::arch::x86_64::__m512i;
use std::arch::x86_64::_mm_loadl_epi64;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::arch::x86_64::_mm512_cmple_epi32_mask;
use std::arch::x86_64::_mm512_cmple_epi64_mask;
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
use std::convert::identity;
use std::mem::MaybeUninit;

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
                // SAFETY: This kernel is only selected when the required AVX-512 features
                // are detected at runtime.
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
/// indices, and that the AVX-512 features (`avx512f`, `avx512bw`, `avx512dq`, `avx512vl`) are
/// enabled.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
#[allow(unused)]
unsafe fn take_primitive_avx512<V, I>(
    values: &[V],
    indices: &[I],
    validity: Validity,
) -> PrimitiveArray
where
    V: NativePType,
    I: UnsignedPType,
{
    // SAFETY: The caller guarantees that the required AVX-512 features are enabled.
    let buffer = unsafe { take_avx512(values, indices) };

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    // SAFETY: The caller ensures that the validity and indices have the same length, so the
    // taken buffer and the validity must have the same length.
    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}

// ---------------------------------------------------------------------------
// AVX-512 SIMD take algorithm
// ---------------------------------------------------------------------------

/// Takes the specified `indices` into a freshly allocated [`Buffer`] using AVX-512 SIMD.
///
/// Out-of-bounds indices produce a zero in the output rather than reading past the end of
/// `values`.
///
/// # Safety
///
/// The caller must ensure that the AVX-512 features (`avx512f`, `avx512bw`, `avx512dq`,
/// `avx512vl`) are enabled.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
#[doc(hidden)]
pub unsafe fn take_avx512<V: NativePType, I: UnsignedPType>(
    values: &[V],
    indices: &[I],
) -> Buffer<V> {
    if values.is_empty() {
        return Buffer::zeroed(indices.len());
    }

    let mut buffer =
        BufferMut::<V>::with_capacity_aligned(indices.len(), Alignment::of::<__m512i>());
    let buf_uninit = buffer.spare_capacity_mut();

    // SAFETY: Required AVX-512 features are enabled by the caller; `buf_uninit` has at least
    // `indices.len()` slots because we just reserved that capacity.
    unsafe { take_avx512_into(values, indices, buf_uninit) };

    // SAFETY: `take_avx512_into` initializes exactly `indices.len()` slots.
    unsafe { buffer.set_len(indices.len()) };

    // Reset the buffer alignment to the Value type so callers can slice it at value boundaries.
    buffer = buffer.aligned(Alignment::of::<V>());

    buffer.freeze()
}

/// Takes the specified `indices` into the provided uninitialized `dst` slice using AVX-512.
///
/// On return, the first `indices.len()` slots of `dst` are initialized.
///
/// # Panics
///
/// Panics if `dst.len() < indices.len()`.
///
/// # Safety
///
/// The caller must ensure that the AVX-512 features (`avx512f`, `avx512bw`, `avx512dq`,
/// `avx512vl`) are enabled.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
#[doc(hidden)]
pub unsafe fn take_avx512_into<V: NativePType, I: UnsignedPType>(
    values: &[V],
    indices: &[I],
    dst: &mut [MaybeUninit<V>],
) {
    assert!(
        dst.len() >= indices.len(),
        "dst must have room for at least indices.len() elements"
    );

    macro_rules! dispatch_avx512 {
        ($indices:ty, $values:ty) => {
            { dispatch_avx512!($indices, $values, cast: $values); }
        };
        ($indices:ty, $values:ty, cast: $cast:ty) => {{
            // SAFETY: The runtime ptype match guarantees that the slice element types match.
            let indices = unsafe { std::mem::transmute::<&[I], &[$indices]>(indices) };
            let values = unsafe { std::mem::transmute::<&[V], &[$cast]>(values) };
            let dst = unsafe {
                std::mem::transmute::<&mut [MaybeUninit<V>], &mut [MaybeUninit<$cast>]>(dst)
            };
            // SAFETY: AVX-512 features are enabled by precondition of this function.
            unsafe {
                exec_take_into_avx512::<$cast, $indices, AVX512Gather>(values, indices, dst)
            }
        }};
    }

    match (I::PTYPE, V::PTYPE) {
        // Int value types. Only 32- and 64-bit values are supported.
        (PType::U8, PType::I32) => dispatch_avx512!(u8, i32),
        (PType::U8, PType::U32) => dispatch_avx512!(u8, u32),
        (PType::U8, PType::I64) => dispatch_avx512!(u8, i64),
        (PType::U8, PType::U64) => dispatch_avx512!(u8, u64),
        (PType::U16, PType::I32) => dispatch_avx512!(u16, i32),
        (PType::U16, PType::U32) => dispatch_avx512!(u16, u32),
        (PType::U16, PType::I64) => dispatch_avx512!(u16, i64),
        (PType::U16, PType::U64) => dispatch_avx512!(u16, u64),
        (PType::U32, PType::I32) => dispatch_avx512!(u32, i32),
        (PType::U32, PType::U32) => dispatch_avx512!(u32, u32),
        (PType::U32, PType::I64) => dispatch_avx512!(u32, i64),
        (PType::U32, PType::U64) => dispatch_avx512!(u32, u64),
        (PType::U64, PType::I32) => dispatch_avx512!(u64, i32),
        (PType::U64, PType::U32) => dispatch_avx512!(u64, u32),
        (PType::U64, PType::I64) => dispatch_avx512!(u64, i64),
        (PType::U64, PType::U64) => dispatch_avx512!(u64, u64),

        // Float value types reuse integer gathers of the same width.
        (PType::U8, PType::F32) => dispatch_avx512!(u8, f32, cast: u32),
        (PType::U16, PType::F32) => dispatch_avx512!(u16, f32, cast: u32),
        (PType::U32, PType::F32) => dispatch_avx512!(u32, f32, cast: u32),
        (PType::U64, PType::F32) => dispatch_avx512!(u64, f32, cast: u32),

        (PType::U8, PType::F64) => dispatch_avx512!(u8, f64, cast: u64),
        (PType::U16, PType::F64) => dispatch_avx512!(u16, f64, cast: u64),
        (PType::U32, PType::F64) => dispatch_avx512!(u32, f64, cast: u64),
        (PType::U64, PType::F64) => dispatch_avx512!(u64, f64, cast: u64),

        // Scalar fallback for unsupported value types (e.g. 8/16-bit values).
        _ => {
            tracing::trace!(
                "take AVX-512 kernel missing for indices {} values {}, falling back to scalar",
                I::PTYPE,
                V::PTYPE
            );
            let fallback = take_primitive_scalar(values, indices);
            for (slot, value) in dst.iter_mut().zip(fallback.as_slice().iter().copied()) {
                slot.write(value);
            }
        }
    }
}

/// The main gather function used by the inner loop kernel for AVX-512 gather.
trait GatherFn<Idx, Values> {
    /// The number of data elements written to `dst` on each loop iteration.
    const WIDTH: usize;
    /// The number of indices read from `indices` on each loop iteration. With AVX-512 we size
    /// the index load to exactly the gather width, so `STRIDE == WIDTH` for every impl below.
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into `dst` using `indices`.
    ///
    /// # Safety
    ///
    /// This function reads up to `STRIDE` elements through `indices`, and reads/writes up to
    /// `WIDTH` elements through `src` and `dst` respectively. The caller must guarantee the
    /// required AVX-512 features are enabled.
    unsafe fn gather(indices: *const Idx, max_idx: Idx, src: *const Values, dst: *mut Values);
}

/// AVX-512 version of [`GatherFn`] defined for 32- and 64-bit value types.
enum AVX512Gather {}

macro_rules! impl_gather {
    ($idx:ty, $({$value:ty =>
        load: $load:ident,
        extend: $extend:ident,
        splat: $splat:ident,
        zero_vec: $zero_vec:ident,
        mask_cmp: $mask_cmp:ident,
        gather: $masked_gather:ident,
        store: $store:ident,
        WIDTH = $WIDTH:literal }),+) => {
        $(
            impl_gather!(single; $idx, $value,
                load: $load,
                extend: $extend,
                splat: $splat,
                zero_vec: $zero_vec,
                mask_cmp: $mask_cmp,
                gather: $masked_gather,
                store: $store,
                WIDTH = $WIDTH);
        )*
    };
    (single; $idx:ty, $value:ty,
        load: $load:ident,
        extend: $extend:ident,
        splat: $splat:ident,
        zero_vec: $zero_vec:ident,
        mask_cmp: $mask_cmp:ident,
        gather: $masked_gather:ident,
        store: $store:ident,
        WIDTH = $WIDTH:literal) => {
        impl GatherFn<$idx, $value> for AVX512Gather {
            const WIDTH: usize = $WIDTH;

            #[allow(unused_unsafe, clippy::cast_possible_truncation)]
            #[inline(always)]
            unsafe fn gather(
                indices: *const $idx,
                max_idx: $idx,
                src: *const $value,
                dst: *mut $value,
            ) {
                const SCALE: i32 = std::mem::size_of::<$value>() as i32;

                // Load raw indices, then extend to the full-width index vector.
                let indices_raw = unsafe { $load(indices.cast()) };
                let indices_vec = unsafe { $extend(indices_raw) };

                // Splat the upper bound and produce a positive mask (`lane <= max_idx`)
                // selecting the lanes the gather is allowed to read. Out-of-bounds lanes pick
                // up the zero source operand instead.
                let max_idx_vec = unsafe { $splat(max_idx as _) };
                let valid_mask = unsafe { $mask_cmp(indices_vec, max_idx_vec) };
                let zero = unsafe { $zero_vec() };

                // Gather and store. AVX-512 gathers take the base pointer last and the mask
                // as a `__mmaskN` bitmask rather than a vector mask.
                let values_vec = unsafe {
                    $masked_gather::<SCALE>(zero, valid_mask, indices_vec, src.cast())
                };
                unsafe { $store(dst.cast(), values_vec) };
            }
        }
    };
}

// kernels for u8 indices
impl_gather!(u8,
    // 32-bit values: gather 16 lanes per call.
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm512_cvtepu8_epi32,
        splat: _mm512_set1_epi32,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi32_mask,
        gather: _mm512_mask_i32gather_epi32,
        store: _mm512_storeu_si512,
        WIDTH = 16
    },
    { i32 =>
        load: _mm_loadu_si128,
        extend: _mm512_cvtepu8_epi32,
        splat: _mm512_set1_epi32,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi32_mask,
        gather: _mm512_mask_i32gather_epi32,
        store: _mm512_storeu_si512,
        WIDTH = 16
    },

    // 64-bit values: gather 8 lanes per call.
    { u64 =>
        load: _mm_loadl_epi64,
        extend: _mm512_cvtepu8_epi64,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    },
    { i64 =>
        load: _mm_loadl_epi64,
        extend: _mm512_cvtepu8_epi64,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    }
);

// kernels for u16 indices
impl_gather!(u16,
    // 32-bit values: gather 16 lanes per call.
    { u32 =>
        load: _mm256_loadu_si256,
        extend: _mm512_cvtepu16_epi32,
        splat: _mm512_set1_epi32,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi32_mask,
        gather: _mm512_mask_i32gather_epi32,
        store: _mm512_storeu_si512,
        WIDTH = 16
    },
    { i32 =>
        load: _mm256_loadu_si256,
        extend: _mm512_cvtepu16_epi32,
        splat: _mm512_set1_epi32,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi32_mask,
        gather: _mm512_mask_i32gather_epi32,
        store: _mm512_storeu_si512,
        WIDTH = 16
    },

    // 64-bit values: gather 8 lanes per call.
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm512_cvtepu16_epi64,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    },
    { i64 =>
        load: _mm_loadu_si128,
        extend: _mm512_cvtepu16_epi64,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    }
);

// kernels for u32 indices
impl_gather!(u32,
    // 32-bit values: load 16 u32 directly, gather 16 lanes.
    { u32 =>
        load: _mm512_loadu_si512,
        extend: identity,
        splat: _mm512_set1_epi32,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi32_mask,
        gather: _mm512_mask_i32gather_epi32,
        store: _mm512_storeu_si512,
        WIDTH = 16
    },
    { i32 =>
        load: _mm512_loadu_si512,
        extend: identity,
        splat: _mm512_set1_epi32,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi32_mask,
        gather: _mm512_mask_i32gather_epi32,
        store: _mm512_storeu_si512,
        WIDTH = 16
    },

    // 64-bit values: load 8 u32, extend to 8 i64, gather 8 lanes.
    { u64 =>
        load: _mm256_loadu_si256,
        extend: _mm512_cvtepu32_epi64,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    },
    { i64 =>
        load: _mm256_loadu_si256,
        extend: _mm512_cvtepu32_epi64,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    }
);

// kernels for u64 indices.
//
// 32-bit values use `_mm512_mask_i64gather_epi32`, which gathers 8 i32s into a __m256i using 8
// i64 indices in a __m512i. 64-bit values use `_mm512_mask_i64gather_epi64`.
impl_gather!(u64,
    { u32 =>
        load: _mm512_loadu_si512,
        extend: identity,
        splat: _mm512_set1_epi64,
        zero_vec: _mm256_setzero_si256,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8
    },
    { i32 =>
        load: _mm512_loadu_si512,
        extend: identity,
        splat: _mm512_set1_epi64,
        zero_vec: _mm256_setzero_si256,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8
    },

    { u64 =>
        load: _mm512_loadu_si512,
        extend: identity,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    },
    { i64 =>
        load: _mm512_loadu_si512,
        extend: identity,
        splat: _mm512_set1_epi64,
        zero_vec: _mm512_setzero_si512,
        mask_cmp: _mm512_cmple_epi64_mask,
        gather: _mm512_mask_i64gather_epi64,
        store: _mm512_storeu_si512,
        WIDTH = 8
    }
);

/// AVX-512 core inner loop for a given `Idx` and `Value` type. Writes `indices.len()` elements
/// into `dst` starting at offset 0.
///
/// # Safety
///
/// The caller must ensure that the AVX-512 features (`avx512f`, `avx512bw`, `avx512dq`,
/// `avx512vl`) are enabled and that `dst.len() >= indices.len()`.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn exec_take_into_avx512<Value, Idx, Gather>(
    values: &[Value],
    indices: &[Idx],
    dst: &mut [MaybeUninit<Value>],
) where
    Value: Copy,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Value>,
{
    let indices_len = indices.len();
    let max_index = Idx::from(values.len()).unwrap_or_else(|| Idx::max_value());

    let mut offset = 0;
    // Loop terminates STRIDE elements before the end of the indices array because `GatherFn`
    // may read up to STRIDE index elements per call and write up to WIDTH dst elements.
    while offset + Gather::STRIDE < indices_len {
        // SAFETY:
        //   1. `indices + offset .. + STRIDE` is in-bounds of the indices allocation.
        //   2. `dst + offset + WIDTH` is in-bounds because `dst.len() >= indices_len`.
        unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr(),
                dst.as_mut_ptr().add(offset).cast(),
            )
        };
        offset += Gather::WIDTH;
    }

    // Scalar remainder.
    while offset < indices_len {
        dst[offset].write(values[indices[offset].as_()]);
        offset += 1;
    }

    debug_assert_eq!(offset, indices_len);
}

#[cfg(test)]
#[cfg_attr(miri, ignore)]
#[cfg(target_arch = "x86_64")]
mod avx512_tests {
    use super::*;

    fn avx512_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
    }

    macro_rules! test_cases {
        (index_type => $IDX:ty, value_types => $($VAL:ty),+) => {
            paste::paste! {
                $(
                    // Happy path: valid indices into a populated array.
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_simple_ $IDX _ $VAL>]() {
                        if !avx512_available() {
                            return;
                        }
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (0..127).collect();

                        let result = unsafe { take_avx512(&values, &indices) };
                        assert_eq!(&values, result.as_slice());
                    }

                    // Take from an empty values array: returns indices.len() zeros, so the
                    // expected slice length mismatch triggers the should_panic.
                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_empty_ $IDX _ $VAL>]() {
                        if !avx512_available() {
                            // Force a panic so #[should_panic] is satisfied on machines without
                            // AVX-512.
                            panic!("avx512 unavailable");
                        }
                        let values: Vec<$VAL> = vec![];
                        let indices: Vec<$IDX> = (0..127).collect();
                        let result = unsafe { take_avx512(&values, &indices) };
                        assert!(result.is_empty());
                    }

                    // All-invalid indices map to zeros; the expected slice length mismatch
                    // triggers the should_panic.
                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_invalid_ $IDX _ $VAL>]() {
                        if !avx512_available() {
                            panic!("avx512 unavailable");
                        }
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (127..=254).collect();

                        let result = unsafe { take_avx512(&values, &indices) };
                        assert_eq!(&[0 as $VAL; 127], result.as_slice());
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

    #[test]
    fn test_avx512_take_last_valid_index_u8() {
        if !avx512_available() {
            return;
        }
        let values: Vec<i64> = (0..(255 + 1)).collect();
        let indices: Vec<u8> = vec![255; 20];

        let result = unsafe { take_avx512(&values, &indices) };
        assert_eq!(&vec![255; indices.len()], result.as_slice());
    }

    #[test]
    fn test_avx512_take_last_valid_index_u16() {
        if !avx512_available() {
            return;
        }
        let values: Vec<i64> = (0..(65535 + 1)).collect();
        let indices: Vec<u16> = vec![65535; 20];

        let result = unsafe { take_avx512(&values, &indices) };
        assert_eq!(&vec![65535; indices.len()], result.as_slice());
    }

    /// `take_avx512_into` should populate the caller-supplied buffer in-place.
    #[test]
    fn test_avx512_take_into_buffer_u32_i32() {
        if !avx512_available() {
            return;
        }
        let values: Vec<i32> = (10..(10 + 200)).collect();
        let indices: Vec<u32> = (0..200).rev().collect();
        let mut dst: Vec<MaybeUninit<i32>> =
            (0..indices.len()).map(|_| MaybeUninit::uninit()).collect();

        unsafe { take_avx512_into(&values, &indices, &mut dst) };

        let initialized: Vec<i32> = dst
            .into_iter()
            .map(|s| unsafe { s.assume_init() })
            .collect();
        let expected: Vec<i32> = indices.iter().map(|&i| values[i as usize]).collect();
        assert_eq!(initialized, expected);
    }
}
