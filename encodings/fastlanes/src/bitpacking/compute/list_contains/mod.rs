// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::list_contains::IntegerMembership;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::compare_fused::stream_compare_fused;
use crate::BitPacked;

// Decode short batches once because their fixed fusion overhead exceeds the saved materialization.
const MIN_DENSE_FUSION_LEN: usize = 2_048;

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
    let Some(list_scalar) = list.as_constant() else {
        return Ok(None);
    };
    let DType::List(member_dtype, _) = list.dtype() else {
        return Ok(None);
    };
    if !member_dtype.eq_ignore_nullability(element.dtype()) {
        return Ok(None);
    }

    let nullability = list.dtype().nullability() | element.dtype().nullability();
    let Some(elements) = list_scalar.as_list().elements() else {
        return Ok(None);
    };
    if elements.is_empty() {
        return Ok(None);
    }

    let result = match_each_integer_ptype!(element.dtype().as_ptype(), |T| {
        let members = elements
            .iter()
            .map(|value| {
                value
                    .as_primitive_opt()
                    .ok_or_else(|| vortex_err!("List member is not a primitive scalar"))?
                    .try_typed_value::<T>()
            })
            .collect::<VortexResult<Vec<Option<T>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let membership = IntegerMembership::new(members);

        match membership.members() {
            [] => BoolArray::new(
                BitBuffer::new_unset(element.len()),
                element.validity()?.union_nullability(nullability),
            )
            .into_array(),
            [member] => {
                let member = *member;
                stream_compare_fused::<T, _>(element, member, nullability, NativePType::is_eq, ctx)?
            }
            [first, second] => {
                let (first, second) = (*first, *second);
                stream_compare_fused::<T, _>(
                    element,
                    first,
                    nullability,
                    move |value, _| value.is_eq(first) | value.is_eq(second),
                    ctx,
                )?
            }
            [first, second, third] => {
                let (first, second, third) = (*first, *second, *third);
                stream_compare_fused::<T, _>(
                    element,
                    first,
                    nullability,
                    move |value, _| value.is_eq(first) | value.is_eq(second) | value.is_eq(third),
                    ctx,
                )?
            }
            [first, second, third, fourth] => {
                let (first, second, third, fourth) = (*first, *second, *third, *fourth);
                stream_compare_fused::<T, _>(
                    element,
                    first,
                    nullability,
                    move |value, _| {
                        value.is_eq(first)
                            | value.is_eq(second)
                            | value.is_eq(third)
                            | value.is_eq(fourth)
                    },
                    ctx,
                )?
            }
            _ => {
                if membership.uses_dense_table() && element.len() >= MIN_DENSE_FUSION_LEN {
                    stream_compare_fused::<T, _>(
                        element,
                        membership.members()[0],
                        nullability,
                        |value, _| membership.contains(value),
                        ctx,
                    )?
                } else {
                    let primitive = element
                        .into_owned()
                        .into_array()
                        .execute::<PrimitiveArray>(ctx)?;
                    membership.evaluate_primitive(primitive.as_view(), nullability)?
                }
            }
        }
    });
    Ok(Some(result))
}

#[cfg(test)]
mod tests;
