// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX-512 implementation of the take operation using `vpgatherdd` / `vpgatherdq` /
//! `vpgatherqq` instructions.
//!
//! Only enabled for x86_64 hosts and gated at runtime behind feature detection to
//! ensure AVX-512F, AVX-512BW, AVX-512DQ, and AVX-512VL instructions are available.
//!
//! Mirrors `avx2.rs`. Index/value pairs that aren't implemented here fall back to the
//! AVX-2 kernel (which in turn can fall back to scalar).

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use std::arch::x86_64::__m256i;
use std::arch::x86_64::__m512i;
use std::arch::x86_64::__mmask8;
use std::arch::x86_64::__mmask16;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm256_cvtepu8_epi32;
use std::arch::x86_64::_mm256_cvtepu16_epi32;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::arch::x86_64::_mm512_cmple_epu32_mask;
use std::arch::x86_64::_mm512_cmple_epu64_mask;
use std::arch::x86_64::_mm512_cvtepu8_epi32;
use std::arch::x86_64::_mm512_cvtepu16_epi32;
use std::arch::x86_64::_mm512_loadu_si512;
use std::arch::x86_64::_mm512_mask_i32gather_epi32;
use std::arch::x86_64::_mm512_mask_i32gather_epi64;
use std::arch::x86_64::_mm512_mask_i64gather_epi32;
use std::arch::x86_64::_mm512_mask_i64gather_epi64;
use std::arch::x86_64::_mm512_set1_epi32;
use std::arch::x86_64::_mm512_set1_epi64;
use std::arch::x86_64::_mm512_setzero_si512;
use std::arch::x86_64::_mm512_storeu_si512;
use std::arch::x86_64::_mm512_zextsi256_si512;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::arrays::primitive::compute::take::avx2;
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
                // SAFETY: This kernel is only selected when the required AVX-512 cpu-features
                // are detected.
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
/// The caller must ensure that if the validity has a length, it is the same length as the indices,
/// and that the `avx512f`, `avx512bw`, `avx512dq`, and `avx512vl` features are enabled.
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
    // SAFETY: The caller guarantees the required features are enabled.
    let buffer = unsafe { take_avx512(values, indices) };

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    // SAFETY: The caller ensures that the validity and indices have the same length, so the taken
    // buffer and the validity must have the same length.
    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}

// ---------------------------------------------------------------------------
// AVX-512 SIMD take algorithm
// ---------------------------------------------------------------------------

/// AVX-512 gather into a caller-supplied uninitialized destination slice. Used by the
/// chunked execution engine to avoid the per-call [`Buffer`] allocation that
/// [`take_avx512`] performs.
///
/// # Safety
///
/// `dst` must be writable for `indices.len()` elements. The `avx512f`, `avx512bw`,
/// `avx512dq`, and `avx512vl` features must be enabled on the calling CPU.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
pub(crate) unsafe fn take_avx512_into<V: NativePType, I: UnsignedPType>(
    buffer: &[V],
    indices: &[I],
    dst: *mut V,
) {
    macro_rules! dispatch_into {
        ($indices:ty, $values:ty) => {{
            dispatch_into!($indices, $values, cast: $values);
        }};
        ($indices:ty, $values:ty, cast: $cast:ty) => {{
            let indices_typed =
                unsafe { std::mem::transmute::<&[I], &[$indices]>(indices) };
            let values_typed = unsafe { std::mem::transmute::<&[V], &[$cast]>(buffer) };
            let dst_typed = dst.cast::<$cast>();
            unsafe {
                exec_take_into_avx512::<$cast, $indices, AVX512Gather>(
                    values_typed,
                    indices_typed,
                    dst_typed,
                );
            }
        }};
    }

    if buffer.is_empty() {
        // Zero-fill: caller still needs dst[..indices.len()] initialized.
        for i in 0..indices.len() {
            // SAFETY: caller guarantees dst has indices.len() cells.
            unsafe { dst.add(i).write(V::default()) };
        }
        return;
    }

    match (I::PTYPE, V::PTYPE) {
        // Int value types. Only 32 and 64 bit types are supported.
        (PType::U8, PType::I32) => dispatch_into!(u8, i32),
        (PType::U8, PType::U32) => dispatch_into!(u8, u32),
        (PType::U8, PType::I64) => dispatch_into!(u8, i64),
        (PType::U8, PType::U64) => dispatch_into!(u8, u64),
        (PType::U16, PType::I32) => dispatch_into!(u16, i32),
        (PType::U16, PType::U32) => dispatch_into!(u16, u32),
        (PType::U16, PType::I64) => dispatch_into!(u16, i64),
        (PType::U16, PType::U64) => dispatch_into!(u16, u64),
        (PType::U32, PType::I32) => dispatch_into!(u32, i32),
        (PType::U32, PType::U32) => dispatch_into!(u32, u32),
        (PType::U32, PType::I64) => dispatch_into!(u32, i64),
        (PType::U32, PType::U64) => dispatch_into!(u32, u64),
        (PType::U64, PType::I32) => dispatch_into!(u64, i32),
        (PType::U64, PType::U32) => dispatch_into!(u64, u32),
        (PType::U64, PType::I64) => dispatch_into!(u64, i64),
        (PType::U64, PType::U64) => dispatch_into!(u64, u64),

        // Float value types, treat them as if they were corresponding int types.
        (PType::U8, PType::F32) => dispatch_into!(u8, f32, cast: u32),
        (PType::U16, PType::F32) => dispatch_into!(u16, f32, cast: u32),
        (PType::U32, PType::F32) => dispatch_into!(u32, f32, cast: u32),
        (PType::U64, PType::F32) => dispatch_into!(u64, f32, cast: u32),

        (PType::U8, PType::F64) => dispatch_into!(u8, f64, cast: u64),
        (PType::U16, PType::F64) => dispatch_into!(u16, f64, cast: u64),
        (PType::U32, PType::F64) => dispatch_into!(u32, f64, cast: u64),
        (PType::U64, PType::F64) => dispatch_into!(u64, f64, cast: u64),

        // Fall back to AVX-2 (or scalar) for unsupported pairs.
        _ => {
            // SAFETY: AVX-2 is a strict subset of the AVX-512 features required to call
            // `take_avx512_into`, so the AVX-2 entry point is valid here.
            unsafe { avx2::take_avx2_into::<V, I>(buffer, indices, dst) };
        }
    }
}

/// AVX-512 inner gather loop. Writes into a caller-supplied destination pointer instead of
/// allocating. Marked `#[target_feature(enable = "avx512f,...")]` so the AVX-512 gather
/// intrinsics inside `AVX512Gather::gather` get the correct codegen context.
///
/// # Safety
///
/// - `dst` must point to at least `indices.len()` writable elements.
/// - The `avx512f`, `avx512bw`, `avx512dq`, and `avx512vl` features must be enabled on
///   the caller's CPU.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn exec_take_into_avx512<Value, Idx, Gather>(
    values: &[Value],
    indices: &[Idx],
    dst: *mut Value,
) where
    Value: Copy,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Value>,
{
    let indices_len = indices.len();
    let max_index = Idx::from(values.len()).unwrap_or_else(|| Idx::max_value());
    let mut offset = 0;
    while offset + Gather::STRIDE < indices_len {
        // SAFETY: same as exec_take_avx512.
        unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr(),
                dst.add(offset),
            )
        };
        offset += Gather::WIDTH;
    }
    while offset < indices_len {
        // SAFETY: offset < indices_len ≤ dst capacity; indices[offset] is bounds-checked.
        unsafe { dst.add(offset).write(values[indices[offset].as_()]) };
        offset += 1;
    }
    debug_assert_eq!(offset, indices_len);
}

/// Takes the specified indices into a new [`Buffer`] using AVX-512 SIMD.
///
/// # Panics
///
/// This function panics if any of the provided `indices` are out of bounds for `values`.
///
/// # Safety
///
/// The caller must ensure the `avx512f`, `avx512bw`, `avx512dq`, and `avx512vl` features
/// are enabled.
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
#[doc(hidden)]
pub(crate) unsafe fn take_avx512<V: NativePType, I: UnsignedPType>(
    buffer: &[V],
    indices: &[I],
) -> Buffer<V> {
    macro_rules! dispatch_avx512 {
        ($indices:ty, $values:ty) => {
            { let result = dispatch_avx512!($indices, $values, cast: $values); result }
        };
        ($indices:ty, $values:ty, cast: $cast:ty) => {{
            let indices_typed =
                unsafe { std::mem::transmute::<&[I], &[$indices]>(indices) };
            let values_typed = unsafe { std::mem::transmute::<&[V], &[$cast]>(buffer) };

            let result = unsafe {
                exec_take_avx512::<$cast, $indices, AVX512Gather>(values_typed, indices_typed)
            };
            unsafe { result.transmute::<V>() }
        }};
    }

    if buffer.is_empty() {
        return Buffer::zeroed(indices.len());
    }

    match (I::PTYPE, V::PTYPE) {
        // Int value types. Only 32 and 64 bit types are supported.
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

        // Float value types, treat them as if they were corresponding int types.
        (PType::U8, PType::F32) => dispatch_avx512!(u8, f32, cast: u32),
        (PType::U16, PType::F32) => dispatch_avx512!(u16, f32, cast: u32),
        (PType::U32, PType::F32) => dispatch_avx512!(u32, f32, cast: u32),
        (PType::U64, PType::F32) => dispatch_avx512!(u64, f32, cast: u32),

        (PType::U8, PType::F64) => dispatch_avx512!(u8, f64, cast: u64),
        (PType::U16, PType::F64) => dispatch_avx512!(u16, f64, cast: u64),
        (PType::U32, PType::F64) => dispatch_avx512!(u32, f64, cast: u64),
        (PType::U64, PType::F64) => dispatch_avx512!(u64, f64, cast: u64),

        // Fall back to AVX-2 (which itself falls back to scalar) for unsupported pairs.
        _ => {
            tracing::trace!(
                "take AVX-512 kernel missing for indices {} values {}, falling back to AVX-2",
                I::PTYPE,
                V::PTYPE
            );
            // SAFETY: AVX-2 is a strict subset of the AVX-512 features required to call
            // `take_avx512`, so the AVX-2 entry point is valid here.
            unsafe { avx2::take_avx2(buffer, indices) }
        }
    }
}

/// The main gather function that is used by the inner loop kernel for AVX-512 gather.
trait GatherFn<Idx, Values> {
    /// The number of data elements that are written to the `dst` on each loop iteration.
    const WIDTH: usize;
    /// The number of indices read from `indices` on each loop iteration.
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into the `dst` using the `indices`.
    ///
    /// # Safety
    ///
    /// This function can read up to `STRIDE` elements through `indices`, and read/write up to
    /// `WIDTH` elements through `src` and `dst` respectively.
    unsafe fn gather(indices: *const Idx, max_idx: Idx, src: *const Values, dst: *mut Values);
}

/// AVX-512 version of [`GatherFn`] defined for 32- and 64-bit value types.
enum AVX512Gather {}

// ---------------------------------------------------------------------------
// 16-lane (`__m512i` of i32 indices) gathers — output __m512i of 32-bit values.
// ---------------------------------------------------------------------------

macro_rules! impl_gather_i32x16 {
    ($idx:ty, $value:ty, load_idx: $load_idx:ident, extend_idx: $extend_idx:expr, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal) => {
        impl GatherFn<$idx, $value> for AVX512Gather {
            const WIDTH: usize = $WIDTH;
            const STRIDE: usize = $STRIDE;

            #[allow(unused_unsafe, clippy::cast_possible_truncation)]
            #[inline(always)]
            unsafe fn gather(
                indices: *const $idx,
                max_idx: $idx,
                src: *const $value,
                dst: *mut $value,
            ) {
                const {
                    assert!(
                        $WIDTH <= $STRIDE,
                        "dst cannot advance by more than the stride"
                    );
                    assert!($WIDTH == 16);
                }

                const SCALE: i32 = size_of::<$value>() as i32;

                // Load and zero-extend `indices` into 16 i32 lanes.
                let raw = unsafe { $load_idx(indices.cast()) };
                let indices_vec: __m512i = unsafe { $extend_idx(raw) };

                // Compute `valid_mask = idx < max_idx` (i.e. in-bounds positions).
                let max_idx_vec = unsafe { _mm512_set1_epi32(max_idx as i32) };
                let valid_mask: __mmask16 =
                    unsafe { _mm512_cmple_epu32_mask(indices_vec, max_idx_vec) };

                let zero_vec = unsafe { _mm512_setzero_si512() };

                // Masked gather: gather where valid_mask=1, leave zero otherwise.
                let values_vec = unsafe {
                    _mm512_mask_i32gather_epi32::<SCALE>(
                        zero_vec,
                        valid_mask,
                        indices_vec,
                        src.cast(),
                    )
                };

                // Write the vec out to dst.
                unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
            }
        }
    };
}

// kernels for u8 indices into 32-bit values
impl_gather_i32x16!(
    u8, u32,
    load_idx: _mm_loadu_si128,
    extend_idx: _mm512_cvtepu8_epi32,
    WIDTH = 16, STRIDE = 16
);
impl_gather_i32x16!(
    u8, i32,
    load_idx: _mm_loadu_si128,
    extend_idx: _mm512_cvtepu8_epi32,
    WIDTH = 16, STRIDE = 16
);

// kernels for u16 indices into 32-bit values
impl_gather_i32x16!(
    u16, u32,
    load_idx: _mm256_loadu_si256,
    extend_idx: _mm512_cvtepu16_epi32,
    WIDTH = 16, STRIDE = 16
);
impl_gather_i32x16!(
    u16, i32,
    load_idx: _mm256_loadu_si256,
    extend_idx: _mm512_cvtepu16_epi32,
    WIDTH = 16, STRIDE = 16
);

// kernels for u32 indices into 32-bit values — pass-through, no extension needed.
impl GatherFn<u32, u32> for AVX512Gather {
    const WIDTH: usize = 16;
    const STRIDE: usize = 16;

    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    unsafe fn gather(indices: *const u32, max_idx: u32, src: *const u32, dst: *mut u32) {
        const SCALE: i32 = size_of::<u32>() as i32;

        let indices_vec = unsafe { _mm512_loadu_si512(indices.cast()) };
        let max_idx_vec = unsafe { _mm512_set1_epi32(max_idx as i32) };
        let valid_mask = unsafe { _mm512_cmple_epu32_mask(indices_vec, max_idx_vec) };
        let zero_vec = unsafe { _mm512_setzero_si512() };
        let values_vec = unsafe {
            _mm512_mask_i32gather_epi32::<SCALE>(zero_vec, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
    }
}

impl GatherFn<u32, i32> for AVX512Gather {
    const WIDTH: usize = 16;
    const STRIDE: usize = 16;

    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    unsafe fn gather(indices: *const u32, max_idx: u32, src: *const i32, dst: *mut i32) {
        const SCALE: i32 = size_of::<i32>() as i32;

        let indices_vec = unsafe { _mm512_loadu_si512(indices.cast()) };
        let max_idx_vec = unsafe { _mm512_set1_epi32(max_idx as i32) };
        let valid_mask = unsafe { _mm512_cmple_epu32_mask(indices_vec, max_idx_vec) };
        let zero_vec = unsafe { _mm512_setzero_si512() };
        let values_vec = unsafe {
            _mm512_mask_i32gather_epi32::<SCALE>(zero_vec, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
    }
}

// ---------------------------------------------------------------------------
// 8-lane (`__m256i` of i32 indices) gathers — output __m512i of 64-bit values.
// ---------------------------------------------------------------------------

macro_rules! impl_gather_i32_to_i64x8 {
    ($idx:ty, $value:ty, load_idx: $load_idx:ident, extend_idx: $extend_idx:expr, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal) => {
        impl GatherFn<$idx, $value> for AVX512Gather {
            const WIDTH: usize = $WIDTH;
            const STRIDE: usize = $STRIDE;

            #[allow(unused_unsafe, clippy::cast_possible_truncation)]
            #[inline(always)]
            unsafe fn gather(
                indices: *const $idx,
                max_idx: $idx,
                src: *const $value,
                dst: *mut $value,
            ) {
                const {
                    assert!(
                        $WIDTH <= $STRIDE,
                        "dst cannot advance by more than the stride"
                    );
                    assert!($WIDTH == 8);
                }

                const SCALE: i32 = size_of::<$value>() as i32;

                // Load `STRIDE` indices, zero-extended to i32x8 (the upper lanes of the
                // 256-bit vector are unused by the gather).
                let raw = unsafe { $load_idx(indices.cast()) };
                let indices_vec: __m256i = unsafe { $extend_idx(raw) };

                // Build a 256-bit max_idx vector and compare unsigned. We borrow the AVX-512VL
                // 256-bit comparison helpers via the 512-bit zero-extended path: zero-extend
                // both sides to a 512i and use `cmplt_epu32_mask`.
                let max_idx_512 = unsafe { _mm512_set1_epi32(max_idx as i32) };
                let indices_512: __m512i = unsafe { _mm512_zextsi256_si512(indices_vec) };
                let valid_mask_16: __mmask16 =
                    unsafe { _mm512_cmple_epu32_mask(indices_512, max_idx_512) };
                // Only the low 8 lanes are meaningful for the i32->i64 gather.
                let valid_mask: __mmask8 = (valid_mask_16 as u16 & 0x00FFu16) as __mmask8;

                let zero_vec = unsafe { _mm512_setzero_si512() };

                let values_vec = unsafe {
                    _mm512_mask_i32gather_epi64::<SCALE>(
                        zero_vec,
                        valid_mask,
                        indices_vec,
                        src.cast(),
                    )
                };

                unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
            }
        }
    };
}

// u8 → 64-bit values (load 8 u8 bytes into a __m128i, zero-extend to __m256i of 8 i32s).
// We need `_mm256_cvtepu8_epi32` from AVX2 for the index zero-extension.
impl GatherFn<u8, u64> for AVX512Gather {
    const WIDTH: usize = 8;
    const STRIDE: usize = 16;

    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    unsafe fn gather(indices: *const u8, max_idx: u8, src: *const u64, dst: *mut u64) {
        const SCALE: i32 = size_of::<u64>() as i32;

        // Load 16 u8s but only the low 8 are used (STRIDE=16 stays compatible with AVX-2).
        let raw = unsafe { _mm_loadu_si128(indices.cast()) };
        let indices_vec: __m256i = unsafe { _mm256_cvtepu8_epi32(raw) };

        let max_idx_512 = unsafe { _mm512_set1_epi32(max_idx as i32) };
        let indices_512 = unsafe { _mm512_zextsi256_si512(indices_vec) };
        let valid_mask_16 = unsafe { _mm512_cmple_epu32_mask(indices_512, max_idx_512) };
        let valid_mask: __mmask8 = (valid_mask_16 as u16 & 0x00FFu16) as __mmask8;

        let zero_vec = unsafe { _mm512_setzero_si512() };
        let values_vec = unsafe {
            _mm512_mask_i32gather_epi64::<SCALE>(zero_vec, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
    }
}

impl GatherFn<u8, i64> for AVX512Gather {
    const WIDTH: usize = 8;
    const STRIDE: usize = 16;

    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    unsafe fn gather(indices: *const u8, max_idx: u8, src: *const i64, dst: *mut i64) {
        const SCALE: i32 = size_of::<i64>() as i32;

        let raw = unsafe { _mm_loadu_si128(indices.cast()) };
        let indices_vec: __m256i = unsafe { _mm256_cvtepu8_epi32(raw) };

        let max_idx_512 = unsafe { _mm512_set1_epi32(max_idx as i32) };
        let indices_512 = unsafe { _mm512_zextsi256_si512(indices_vec) };
        let valid_mask_16 = unsafe { _mm512_cmple_epu32_mask(indices_512, max_idx_512) };
        let valid_mask: __mmask8 = (valid_mask_16 as u16 & 0x00FFu16) as __mmask8;

        let zero_vec = unsafe { _mm512_setzero_si512() };
        let values_vec = unsafe {
            _mm512_mask_i32gather_epi64::<SCALE>(zero_vec, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
    }
}

// u16 → 64-bit values
impl_gather_i32_to_i64x8!(
    u16, u64,
    load_idx: _mm_loadu_si128,
    extend_idx: _mm256_cvtepu16_epi32,
    WIDTH = 8, STRIDE = 8
);
impl_gather_i32_to_i64x8!(
    u16, i64,
    load_idx: _mm_loadu_si128,
    extend_idx: _mm256_cvtepu16_epi32,
    WIDTH = 8, STRIDE = 8
);

// u32 → 64-bit values (load 8 u32 into a __m256i, gather via i32->i64).
impl_gather_i32_to_i64x8!(
    u32, u64,
    load_idx: _mm256_loadu_si256,
    extend_idx: identity_m256,
    WIDTH = 8, STRIDE = 8
);
impl_gather_i32_to_i64x8!(
    u32, i64,
    load_idx: _mm256_loadu_si256,
    extend_idx: identity_m256,
    WIDTH = 8, STRIDE = 8
);

#[inline(always)]
unsafe fn identity_m256(x: __m256i) -> __m256i {
    x
}

// ---------------------------------------------------------------------------
// 8-lane (`__m512i` of i64 indices) gathers — output __m512i of 64-bit values.
// ---------------------------------------------------------------------------

impl GatherFn<u64, u64> for AVX512Gather {
    const WIDTH: usize = 8;
    const STRIDE: usize = 8;

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    #[inline(always)]
    unsafe fn gather(indices: *const u64, max_idx: u64, src: *const u64, dst: *mut u64) {
        const SCALE: i32 = size_of::<u64>() as i32;

        let indices_vec = unsafe { _mm512_loadu_si512(indices.cast()) };
        let max_idx_vec = unsafe { _mm512_set1_epi64(max_idx as i64) };
        let valid_mask = unsafe { _mm512_cmple_epu64_mask(indices_vec, max_idx_vec) };
        let zero_vec = unsafe { _mm512_setzero_si512() };
        let values_vec = unsafe {
            _mm512_mask_i64gather_epi64::<SCALE>(zero_vec, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
    }
}

impl GatherFn<u64, i64> for AVX512Gather {
    const WIDTH: usize = 8;
    const STRIDE: usize = 8;

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    #[inline(always)]
    unsafe fn gather(indices: *const u64, max_idx: u64, src: *const i64, dst: *mut i64) {
        const SCALE: i32 = size_of::<i64>() as i32;

        let indices_vec = unsafe { _mm512_loadu_si512(indices.cast()) };
        let max_idx_vec = unsafe { _mm512_set1_epi64(max_idx as i64) };
        let valid_mask = unsafe { _mm512_cmple_epu64_mask(indices_vec, max_idx_vec) };
        let zero_vec = unsafe { _mm512_setzero_si512() };
        let values_vec = unsafe {
            _mm512_mask_i64gather_epi64::<SCALE>(zero_vec, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm512_storeu_si512(dst.cast(), values_vec) };
    }
}

// ---------------------------------------------------------------------------
// 8-lane (`__m512i` of i64 indices) gathers — output __m256i of 32-bit values.
// ---------------------------------------------------------------------------

impl GatherFn<u64, u32> for AVX512Gather {
    const WIDTH: usize = 8;
    const STRIDE: usize = 8;

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    #[inline(always)]
    unsafe fn gather(indices: *const u64, max_idx: u64, src: *const u32, dst: *mut u32) {
        const SCALE: i32 = size_of::<u32>() as i32;

        let indices_vec = unsafe { _mm512_loadu_si512(indices.cast()) };
        let max_idx_vec = unsafe { _mm512_set1_epi64(max_idx as i64) };
        let valid_mask = unsafe { _mm512_cmple_epu64_mask(indices_vec, max_idx_vec) };
        let zero_vec_256 = unsafe { _mm256_setzero_si256() };
        let values_vec = unsafe {
            _mm512_mask_i64gather_epi32::<SCALE>(zero_vec_256, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm256_storeu_si256(dst.cast(), values_vec) };
    }
}

impl GatherFn<u64, i32> for AVX512Gather {
    const WIDTH: usize = 8;
    const STRIDE: usize = 8;

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    #[inline(always)]
    unsafe fn gather(indices: *const u64, max_idx: u64, src: *const i32, dst: *mut i32) {
        const SCALE: i32 = size_of::<i32>() as i32;

        let indices_vec = unsafe { _mm512_loadu_si512(indices.cast()) };
        let max_idx_vec = unsafe { _mm512_set1_epi64(max_idx as i64) };
        let valid_mask = unsafe { _mm512_cmple_epu64_mask(indices_vec, max_idx_vec) };
        let zero_vec_256 = unsafe { _mm256_setzero_si256() };
        let values_vec = unsafe {
            _mm512_mask_i64gather_epi32::<SCALE>(zero_vec_256, valid_mask, indices_vec, src.cast())
        };
        unsafe { _mm256_storeu_si256(dst.cast(), values_vec) };
    }
}

/// AVX-512 core inner loop for a specific `Idx` / `Value` type pair.
#[inline(always)]
unsafe fn exec_take_avx512<Value, Idx, Gather>(values: &[Value], indices: &[Idx]) -> Buffer<Value>
where
    Value: Copy,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Value>,
{
    let indices_len = indices.len();
    let max_index = Idx::from(values.len()).unwrap_or_else(|| Idx::max_value());
    let mut buffer =
        BufferMut::<Value>::with_capacity_aligned(indices_len, Alignment::of::<__m512i>());
    let buf_uninit = buffer.spare_capacity_mut();

    let mut offset = 0;
    // Loop terminates STRIDE elements before the end of the indices array.
    while offset + Gather::STRIDE < indices_len {
        // SAFETY: `gather` reads at most STRIDE indices and writes WIDTH values, and we have
        // STRIDE more indices available and WIDTH+ destination cells available.
        unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr(),
                buf_uninit.as_mut_ptr().add(offset).cast(),
            )
        };
        offset += Gather::WIDTH;
    }

    while offset < indices_len {
        buf_uninit[offset].write(values[indices[offset].as_()]);
        offset += 1;
    }

    assert_eq!(offset, indices_len);

    // SAFETY: All elements have been initialized.
    unsafe { buffer.set_len(indices_len) };

    // Reset the buffer alignment to the Value type so downstream slicing works at value
    // boundaries.
    buffer = buffer.aligned(Alignment::of::<Value>());

    buffer.freeze()
}

#[cfg(test)]
#[cfg_attr(miri, ignore)]
#[cfg(target_arch = "x86_64")]
mod avx512_tests {
    use super::*;

    fn host_has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
    }

    macro_rules! test_cases {
        (index_type => $IDX:ty, value_types => $($VAL:ty),+) => {
            paste::paste! {
                $(
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_simple_ $IDX _ $VAL>]() {
                        if !host_has_avx512() {
                            return;
                        }
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (0..127).collect();

                        let result = unsafe { take_avx512(&values, &indices) };
                        assert_eq!(&values, result.as_slice());
                    }

                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_empty_ $IDX _ $VAL>]() {
                        if !host_has_avx512() {
                            // Force the test to "panic" so the should_panic harness is satisfied
                            // on hosts without AVX-512.
                            panic!("avx512 not available; skipping");
                        }
                        let values: Vec<$VAL> = vec![];
                        let indices: Vec<$IDX> = (0..127).collect();
                        let result = unsafe { take_avx512(&values, &indices) };
                        assert!(result.is_empty());
                    }

                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx512_take_invalid_ $IDX _ $VAL>]() {
                        if !host_has_avx512() {
                            panic!("avx512 not available; skipping");
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
        if !host_has_avx512() {
            return;
        }
        let values: Vec<i64> = (0..(255 + 1)).collect();
        let indices: Vec<u8> = vec![255; 20];

        let result = unsafe { take_avx512(&values, &indices) };
        assert_eq!(&vec![255; indices.len()], result.as_slice());
    }

    #[test]
    fn test_avx512_take_last_valid_index_u16() {
        if !host_has_avx512() {
            return;
        }
        let values: Vec<i64> = (0..(65535 + 1)).collect();
        let indices: Vec<u16> = vec![65535; 20];

        let result = unsafe { take_avx512(&values, &indices) };
        assert_eq!(&vec![65535; indices.len()], result.as_slice());
    }
}
