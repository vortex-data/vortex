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
use crate::arrays::primitive::compute::chunked_pack::chunked_pack;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
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

        let nullability =
            arr.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        // Inclusive-inclusive on unsigned has a wraparound fast path: `lo <= v <= hi` is
        // equivalent to `v.wrapping_sub(lo) <= hi - lo` — one sub + one cmp per element.
        if matches!(options.lower_strict, StrictComparison::NonStrict)
            && matches!(options.upper_strict, StrictComparison::NonStrict)
            && arr.ptype().is_unsigned_int()
        {
            let bits = match_each_unsigned_integer_ptype!(arr.ptype(), |P| {
                wraparound_unsigned::<P>(arr.as_slice::<P>(), P::try_from(&lower)?, P::try_from(&upper)?)
            });
            return Ok(Some(into_bool_array(arr, bits, nullability)));
        }

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

#[inline]
fn wraparound_unsigned<T>(slice: &[T], lo: T, hi: T) -> BitBuffer
where
    T: Copy + num_traits::WrappingSub + PartialOrd,
{
    let range = hi.wrapping_sub(&lo);
    chunked_pack(slice, |v| v.wrapping_sub(&lo) <= range)
}

fn into_bool_array(
    arr: ArrayView<'_, Primitive>,
    bits: BitBuffer,
    nullability: Nullability,
) -> ArrayRef {
    BoolArray::new(
        bits,
        arr.validity()
            .vortex_expect("validity should be derivable")
            .union_nullability(nullability),
    )
    .into_array()
}

fn between_impl<T: NativePType + Copy>(
    arr: ArrayView<'_, Primitive>,
    lower: T,
    upper: T,
    nullability: Nullability,
    options: &BetweenOptions,
) -> ArrayRef {
    let bits = match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => {
            chunked_pack(arr.as_slice::<T>(), |v| {
                NativePType::is_lt(lower, v) & NativePType::is_lt(v, upper)
            })
        }
        (StrictComparison::Strict, StrictComparison::NonStrict) => {
            chunked_pack(arr.as_slice::<T>(), |v| {
                NativePType::is_lt(lower, v) & NativePType::is_le(v, upper)
            })
        }
        (StrictComparison::NonStrict, StrictComparison::Strict) => {
            chunked_pack(arr.as_slice::<T>(), |v| {
                NativePType::is_le(lower, v) & NativePType::is_lt(v, upper)
            })
        }
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => {
            chunked_pack(arr.as_slice::<T>(), |v| {
                NativePType::is_le(lower, v) & NativePType::is_le(v, upper)
            })
        }
    };
    into_bool_array(arr, bits, nullability)
}
