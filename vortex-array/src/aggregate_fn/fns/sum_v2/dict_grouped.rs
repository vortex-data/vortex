// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sum dictionary-encoded i32 groups without expanding or widening their elements.

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use super::SumV2;
use super::sum_v2_partial_fields;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::BoolArray;
use crate::arrays::Dict;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

#[derive(Debug)]
pub(crate) struct DictGroupedSumV2EncodingKernel;

impl DynGroupedAggregateKernel for DictGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if aggregate.as_opt::<SumV2>().is_none()
            || !matches!(groups.elements().dtype(), DType::Primitive(PType::I32, _))
        {
            return Ok(None);
        }
        let Some(dict) = groups.elements().as_opt::<Dict>() else {
            return Ok(None);
        };
        let ranges = groups.group_ranges(ctx)?;
        // At most u32::MAX i32 values fit in i64 without intermediate overflow.
        // Longer groups retain the existing overflow-aware fallback.
        if ranges.iter().any(|(_, len)| len > u32::MAX as usize) {
            return Ok(None);
        }
        let group_mask = groups.group_validity(ctx)?;
        let code_mask = dict.codes().validity()?.execute_mask(dict.codes().len(), ctx)?;
        let value_mask = dict.values().validity()?.execute_mask(dict.values().len(), ctx)?;
        let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
        let values = dict.values().clone().execute::<PrimitiveArray>(ctx)?;
        let values = values.as_slice::<i32>();
        let mut sums = Vec::with_capacity(groups.len());
        let mut empty = Vec::with_capacity(groups.len());
        match_each_integer_ptype!(codes.ptype(), |C| {
            let codes = codes.as_slice::<C>();
            for (group_index, (offset, len)) in ranges.iter().enumerate() {
                let mut sum = 0_i64;
                let mut any_valid = false;
                if group_mask.value(group_index) {
                    for index in offset..offset + len {
                        if !code_mask.value(index) {
                            continue;
                        }
                        let code = codes[index].as_();
                        if value_mask.value(code) {
                            sum += i64::from(values[code]);
                            any_valid = true;
                        }
                    }
                }
                sums.push(sum);
                empty.push(!any_valid);
            }
        });
        let sums = PrimitiveArray::from_iter(sums);
        let fields = sum_v2_partial_fields(sums.dtype().clone());
        // SAFETY: every child has one correctly typed partial state per group.
        Ok(Some(unsafe {
            StructArray::new_unchecked(
                vec![
                    sums.into_array(),
                    BoolArray::new(BitBuffer::new_unset(groups.len()), Validity::NonNullable)
                        .into_array(),
                    BoolArray::from_iter(empty).into_array(),
                ],
                fields,
                groups.len(),
                Validity::from_mask(group_mask, Nullability::Nullable),
            )
        }.into_array()))
    }
}
