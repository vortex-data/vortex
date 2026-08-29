// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::evaluate_prepared_integer_membership;
use vortex_array::arrays::primitive::integer_membership_binary_search_min;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::list_contains::IntegerMembership;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use super::compare_fused::stream_predicate_fused;
use crate::BitPacked;
use crate::unpack_iter::BitPacked as BitPackedIter;

const MAX_FUSED_DISTINCT_MEMBERS: usize = 4;
const SHORT_ARRAY_MAX_ROWS_8_16: usize = 8_192;
const SHORT_ARRAY_MAX_ROWS_32: usize = 16_384;
fn min_decode_source_members(ptype: PType, len: usize) -> usize {
    // The generic fallback scans the packed child once per source member. Decode before repeated
    // packed scans become more expensive than one decode plus Primitive membership evaluation.
    let short_array_max_rows = if ptype.bit_width() == 32 {
        SHORT_ARRAY_MAX_ROWS_32
    } else {
        SHORT_ARRAY_MAX_ROWS_8_16
    };
    if len <= short_array_max_rows && ptype.bit_width() < 64 {
        return integer_membership_binary_search_min(ptype);
    }
    match ptype.bit_width() {
        8 => 30,
        16 => 25,
        32 => 13,
        64 => 5,
        _ => 5,
    }
}

impl ListContainsElementKernel for BitPacked {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        list_contains_compressed(list, element, ctx)
    }
}

fn list_contains_compressed(
    list: &ArrayRef,
    element: ArrayView<'_, BitPacked>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let nullability = list.dtype().nullability() | element.dtype().nullability();

    match_each_integer_ptype!(element.dtype().as_ptype(), |T| {
        list_contains_typed::<T>(list, element, nullability, ctx)
    })
}

fn list_contains_typed<T>(
    list: &ArrayRef,
    element: ArrayView<'_, BitPacked>,
    nullability: vortex_array::dtype::Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: IntegerPType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking + NativePType + BitPackingCompare,
{
    let Some(membership) = IntegerMembership::<T>::try_from_constant_list(list, element.dtype())?
    else {
        return Ok(None);
    };
    if membership.members().len() > MAX_FUSED_DISTINCT_MEMBERS {
        if membership.non_null_source_len()
            < min_decode_source_members(element.dtype().as_ptype(), element.len())
        {
            return Ok(None);
        }
        // The generic list implementation expands membership into one comparison per source
        // member. Each comparison scans the packed child. Decode once before applying the
        // Primitive membership policy when repeated packed scans become more expensive.
        let primitive = element.array().clone().execute::<PrimitiveArray>(ctx)?;
        return evaluate_prepared_integer_membership(membership, primitive.as_view(), nullability)
            .map(Some);
    }

    let result = match membership.members() {
        [] => BoolArray::new(
            BitBuffer::new_unset(element.len()),
            element.validity()?.union_nullability(nullability),
        )
        .into_array(),
        [member] => {
            let member = *member;
            stream_predicate_fused::<T, _>(
                element,
                nullability,
                move |value| value.is_eq(member),
                ctx,
            )?
        }
        [first, second] => {
            let (first, second) = (*first, *second);
            stream_predicate_fused::<T, _>(
                element,
                nullability,
                move |value| value.is_eq(first) | value.is_eq(second),
                ctx,
            )?
        }
        [first, second, third] => {
            let (first, second, third) = (*first, *second, *third);
            stream_predicate_fused::<T, _>(
                element,
                nullability,
                move |value| value.is_eq(first) | value.is_eq(second) | value.is_eq(third),
                ctx,
            )?
        }
        [first, second, third, fourth] => {
            let (first, second, third, fourth) = (*first, *second, *third, *fourth);
            stream_predicate_fused::<T, _>(
                element,
                nullability,
                move |value| {
                    value.is_eq(first)
                        | value.is_eq(second)
                        | value.is_eq(third)
                        | value.is_eq(fourth)
                },
                ctx,
            )?
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}

#[cfg(test)]
mod tests;
