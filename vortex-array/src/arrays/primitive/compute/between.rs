// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::between::BetweenKernel;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;

impl BetweenKernel for Primitive {
    fn between(
        arr: ArrayView<'_, Primitive>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        // Note, we know that have checked before that the lower and upper bounds are not constant
        // null values

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        Ok(Some(match_each_native_ptype!(arr.ptype(), |P| {
            between_impl::<P>(
                arr,
                P::try_from(&lower)?,
                P::try_from(&upper)?,
                nullability,
                options,
            )
        })))
    }
}

fn between_impl<T: NativePType + Copy>(
    arr: ArrayView<'_, Primitive>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
) -> ArrayRef {
    // Inclusive-inclusive on unsigned takes the wraparound subtract fast path:
    //   `lo <= v <= hi`  ≡  `v.wrapping_sub(lo) <= hi.wrapping_sub(lo)`
    // One subtract + one compare per element instead of two compares + AND.
    if matches!(options.lower_strict, StrictComparison::NonStrict)
        && matches!(options.upper_strict, StrictComparison::NonStrict)
        && T::PTYPE.is_unsigned_int()
    {
        if let Some(arr_ref) = between_wraparound::<T>(arr, lower, upper, nullability) {
            return arr_ref;
        }
    }

    match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_lt,
            nullability,
        ),
        (StrictComparison::Strict, StrictComparison::NonStrict) => between_impl_(
            arr,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_le,
            nullability,
        ),
        (StrictComparison::NonStrict, StrictComparison::Strict) => between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_lt,
            nullability,
        ),
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => between_impl_(
            arr,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_le,
            nullability,
        ),
    }
}

/// Wraparound-subtract fast path for unsigned inclusive between.
///
/// `lo <= v <= hi` is equivalent to `v.wrapping_sub(lo) <= hi.wrapping_sub(lo)` on
/// unsigned types — one subtract + one compare per element instead of two compares
/// plus an AND. The inner loop is more SIMD-friendly because there's only one
/// dependency chain feeding the bit-pack.
fn between_wraparound<T: NativePType + Copy>(
    arr: ArrayView<'_, Primitive>,
    lower: T,
    upper: T,
    nullability: Nullability,
) -> Option<ArrayRef> {
    use crate::dtype::PType;
    let slice = arr.as_slice::<T>();
    let len = slice.len();
    let bits = match T::PTYPE {
        PType::U8 => {
            // SAFETY: T == u8 here by the match guard.
            let s: &[u8] = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) };
            let lo: u8 = unsafe { *(&lower as *const T as *const u8) };
            let hi: u8 = unsafe { *(&upper as *const T as *const u8) };
            wraparound_chunked_u8(s, lo, hi)
        }
        PType::U16 => {
            let s: &[u16] = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) };
            let lo: u16 = unsafe { *(&lower as *const T as *const u16) };
            let hi: u16 = unsafe { *(&upper as *const T as *const u16) };
            wraparound_chunked_u16(s, lo, hi)
        }
        PType::U32 => {
            let s: &[u32] = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) };
            let lo: u32 = unsafe { *(&lower as *const T as *const u32) };
            let hi: u32 = unsafe { *(&upper as *const T as *const u32) };
            wraparound_chunked_u32(s, lo, hi)
        }
        PType::U64 => {
            let s: &[u64] = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) };
            let lo: u64 = unsafe { *(&lower as *const T as *const u64) };
            let hi: u64 = unsafe { *(&upper as *const T as *const u64) };
            wraparound_chunked_u64(s, lo, hi)
        }
        _ => return None,
    };
    Some(
        BoolArray::new(
            bits,
            arr.validity()
                .vortex_expect("validity should be derivable")
                .union_nullability(nullability),
        )
        .into_array(),
    )
}

macro_rules! impl_wraparound_chunked {
    ($name:ident, $t:ty) => {
        #[inline]
        fn $name(slice: &[$t], lo: $t, hi: $t) -> BitBuffer {
            use vortex_buffer::ByteBufferMut;
            let len = slice.len();
            let range = hi.wrapping_sub(lo);
            let bytes_len = len.div_ceil(8);
            let mut bytes = ByteBufferMut::zeroed(bytes_len);
            let dst = bytes.as_mut_slice();
            let full = len / 8;
            for chunk_idx in 0..full {
                let base = chunk_idx * 8;
                let mut b = 0u8;
                for j in 0..8 {
                    // SAFETY: base + j < full*8 <= len.
                    let v = unsafe { *slice.get_unchecked(base + j) };
                    b |= u8::from(v.wrapping_sub(lo) <= range) << j;
                }
                // SAFETY: chunk_idx < full <= bytes_len.
                unsafe { *dst.get_unchecked_mut(chunk_idx) = b };
            }
            let tail = full * 8;
            if tail < len {
                let mut b = 0u8;
                for j in 0..(len - tail) {
                    let v = unsafe { *slice.get_unchecked(tail + j) };
                    b |= u8::from(v.wrapping_sub(lo) <= range) << j;
                }
                unsafe { *dst.get_unchecked_mut(full) = b };
            }
            BitBuffer::new(bytes.freeze(), len)
        }
    };
}
impl_wraparound_chunked!(wraparound_chunked_u8, u8);
impl_wraparound_chunked!(wraparound_chunked_u16, u16);
impl_wraparound_chunked!(wraparound_chunked_u32, u32);
impl_wraparound_chunked!(wraparound_chunked_u64, u64);

fn between_impl_<T>(
    arr: ArrayView<'_, Primitive>,
    lower: T,
    lower_fn: impl Fn(T, T) -> bool,
    upper: T,
    upper_fn: impl Fn(T, T) -> bool,
    nullability: Nullability,
) -> ArrayRef
where
    T: NativePType + Copy,
{
    let slice = arr.as_slice::<T>();
    let bits = chunked_between::<T>(slice, lower, lower_fn, upper, upper_fn);
    BoolArray::new(
        bits,
        arr.validity()
            .vortex_expect("validity should be derivable")
            .union_nullability(nullability),
    )
    .into_array()
}

/// Pack 8 between-checks into a byte at a time. Matches the chunked SIMD pattern in
/// `compare::cmp_chunked` — Vortex's `BitBuffer::collect_bool` builder serializes the
/// bit writes one at a time and defeats auto-vectorization (~2-3× slower than this).
#[inline]
fn chunked_between<T>(
    slice: &[T],
    lower: T,
    lower_fn: impl Fn(T, T) -> bool,
    upper: T,
    upper_fn: impl Fn(T, T) -> bool,
) -> BitBuffer
where
    T: NativePType + Copy,
{
    use vortex_buffer::ByteBufferMut;
    let len = slice.len();
    let bytes_len = len.div_ceil(8);
    let mut bytes = ByteBufferMut::zeroed(bytes_len);
    let dst = bytes.as_mut_slice();
    let full = len / 8;
    for chunk_idx in 0..full {
        let base = chunk_idx * 8;
        let mut b = 0u8;
        for j in 0..8 {
            // SAFETY: base + j < full*8 <= len.
            let v = unsafe { *slice.get_unchecked(base + j) };
            b |= u8::from(lower_fn(lower, v) & upper_fn(v, upper)) << j;
        }
        // SAFETY: chunk_idx < full <= bytes_len.
        unsafe { *dst.get_unchecked_mut(chunk_idx) = b };
    }
    let tail = full * 8;
    if tail < len {
        let mut b = 0u8;
        for j in 0..(len - tail) {
            // SAFETY: tail + j < len.
            let v = unsafe { *slice.get_unchecked(tail + j) };
            b |= u8::from(lower_fn(lower, v) & upper_fn(v, upper)) << j;
        }
        // SAFETY: full < bytes_len when len % 8 != 0.
        unsafe { *dst.get_unchecked_mut(full) = b };
    }
    BitBuffer::new(bytes.freeze(), len)
}
