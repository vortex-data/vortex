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
    match (options.lower_strict, options.upper_strict) {
        // Note: these comparisons are explicitly passed in to allow function impl inlining
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
