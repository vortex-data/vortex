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
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::list_contains::IntegerMembership;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_array::scalar_fn::fns::list_contains::evaluate_constant_list_generic;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use super::compare_fused::stream_predicate_fused;
use crate::BitPacked;
use crate::unpack_iter::BitPacked as BitPackedIter;

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
        return evaluate_constant_list_generic(list, element.array(), nullability);
    };
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
