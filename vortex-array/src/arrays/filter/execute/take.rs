// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod fixed_width;
mod rank;

#[cfg(test)]
mod tests;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use self::fixed_width::take_decimal;
use self::fixed_width::take_primitive;
use self::rank::contiguous_sequential_take_range_indices;
use self::rank::sequential_take_len;
use self::rank::translate_indices;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::filter::FilterArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::scalar::Scalar;

const BIG_TAKE_FALLBACK_LEN: usize = 4096;
const BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_TAKE_LEN: usize = 25_000;
const BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_RATIO: usize = 10;
const SMALL_TAKE_RANK_LOOKUP_DIVISOR: usize = 80;
const SMALL_TAKE_RANK_LOOKUP_MAX: usize = 256;

#[inline]
fn small_take_rank_lookup_len(filter: &Mask) -> usize {
    (filter.true_count() / SMALL_TAKE_RANK_LOOKUP_DIVISOR).clamp(1, SMALL_TAKE_RANK_LOOKUP_MAX)
}

fn take_impl(
    array: ArrayView<'_, Filter>,
    indices: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if array.child().dtype().is_primitive() {
        return take_primitive(array, indices, ctx);
    }

    if array.child().dtype().is_decimal() {
        return take_decimal(array, indices, ctx);
    }

    let indices_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;

    match indices_validity.bit_buffer() {
        AllOr::All => {
            let result_dtype = array
                .dtype()
                .union_nullability(indices.dtype().nullability());

            if let Some((start, end)) =
                contiguous_sequential_take_range_indices(array.filter_mask(), indices)?
            {
                return array.child().slice(start..end)?.cast(result_dtype);
            }

            if let Some(take_len) = sequential_take_len(indices, array.len())? {
                if take_len == 0 {
                    return Ok(Canonical::empty(&result_dtype).into_array());
                }
                let rank_mask = Mask::from_slices(array.len(), vec![(0, take_len)]);
                let mask = array.filter_mask().intersect_by_rank(&rank_mask);
                return array.child().filter(mask)?.cast(result_dtype);
            }

            let translated = translate_indices(array.filter_mask(), indices, None)?;
            let translated_indices =
                PrimitiveArray::new(translated, indices.validity()?).into_array();

            array.child().take(translated_indices)
        }
        AllOr::None => Ok(ConstantArray::new(
            Scalar::null(array.dtype().as_nullable()),
            indices.len(),
        )
        .into_array()),
        AllOr::Some(buf) => {
            let translated = translate_indices(array.filter_mask(), indices, Some(buf))?;
            let translated_indices =
                PrimitiveArray::new(translated, indices.validity()?).into_array();

            array.child().take(translated_indices)
        }
    }
}

fn should_materialize_big_take(array: ArrayView<'_, Filter>, indices: &ArrayRef) -> bool {
    let filtered_len = array.len();
    let take_len = indices.len();

    if take_len < filtered_len {
        return false;
    }

    if filtered_len >= BIG_TAKE_FALLBACK_LEN {
        return true;
    }

    let child_dtype = array.child().dtype();
    let fixed_width_child = child_dtype.is_primitive() || child_dtype.is_decimal();
    fixed_width_child
        && take_len >= BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_TAKE_LEN
        && take_len >= filtered_len.saturating_mul(BIG_TAKE_FALLBACK_MIN_FIXED_WIDTH_RATIO)
}

impl TakeExecute for Filter {
    fn take(
        array: ArrayView<'_, Filter>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Bool filtering is already very cheap. Translating take indices through the filter adds
        // overhead without improving the downstream bool take, so leave bool children on the
        // regular filter path.
        if array.child().dtype().is_boolean() {
            return Ok(None);
        }

        let DType::Primitive(ptype, nullability) = indices.dtype() else {
            vortex_bail!("Invalid indices dtype: {}", indices.dtype())
        };

        if should_materialize_big_take(array, indices) {
            return Ok(None);
        }

        let unsigned_indices = if ptype.is_unsigned_int() {
            indices.clone().execute::<PrimitiveArray>(ctx)?
        } else {
            indices
                .clone()
                .cast(DType::Primitive(ptype.to_unsigned(), *nullability))?
                .execute::<PrimitiveArray>(ctx)?
        };

        take_impl(array, &unsigned_indices, ctx).map(Some)
    }
}
