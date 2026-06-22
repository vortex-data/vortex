// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Sum;
use super::SumPartial;
use super::SumState;
use super::checked_add_i64;
use super::checked_add_u64;
use super::primitive::sum_float_all;
use super::primitive::sum_signed_all;
use super::primitive::sum_unsigned_all;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::GroupIds;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::aggregate_fn::kernels::GroupedAggregateKernelResult;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

const MIN_AVG_RUN_LENGTH_FOR_GROUPED_SUM_RUNS: usize = 4;

#[derive(Debug)]
pub(crate) struct SumGroupedKernel;

impl DynGroupedAggregateKernel for SumGroupedKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        group_ids: &GroupIds,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<GroupedAggregateKernelResult>> {
        let Some(options) = aggregate_fn.as_opt::<Sum>() else {
            return Ok(None);
        };

        let columnar = batch.clone().execute::<Columnar>(ctx)?;
        match &columnar {
            Columnar::Canonical(Canonical::Primitive(_))
            | Columnar::Canonical(Canonical::Bool(_)) => {}
            // Decimal and constants still use the universal grouped fallback.
            Columnar::Canonical(Canonical::Decimal(_)) | Columnar::Constant(_) => return Ok(None),
            Columnar::Canonical(_) => {
                vortex_bail!("Unsupported canonical type for sum: {}", columnar.dtype())
            }
        }

        let partial_dtype = Sum
            .partial_dtype(options, batch.dtype())
            .ok_or_else(|| vortex_error::vortex_err!("Unsupported sum dtype: {}", batch.dtype()))?;
        let ids = group_ids.validated_ids(ctx)?;
        let mut partials = (0..group_ids.num_groups())
            .map(|_| Sum.empty_partial(options, batch.dtype()))
            .collect::<VortexResult<Vec<_>>>()?;

        match &columnar {
            Columnar::Canonical(Canonical::Primitive(p)) => {
                accumulate_grouped_primitive(&mut partials, p, ids.as_ref(), ctx)?;
            }
            Columnar::Canonical(Canonical::Bool(b)) => {
                accumulate_grouped_bool(&mut partials, b, ids.as_ref(), ctx)?;
            }
            Columnar::Canonical(Canonical::Decimal(_)) | Columnar::Constant(_) => unreachable!(),
            Columnar::Canonical(_) => unreachable!(),
        }

        let Some(partials) = Sum.partials_to_array(&partials, &partial_dtype)? else {
            return Ok(None);
        };
        Ok(Some(GroupedAggregateKernelResult::dense(
            partials,
            group_ids.num_groups(),
        )?))
    }
}

fn for_each_valid_idx(validity: &Mask, len: usize, mut f: impl FnMut(usize)) {
    match validity.indices() {
        AllOr::All => {
            for idx in 0..len {
                f(idx);
            }
        }
        AllOr::None => {}
        AllOr::Some(indices) => {
            for &idx in indices {
                f(idx);
            }
        }
    }
}

fn should_accumulate_group_runs(group_ids: &[u32]) -> bool {
    let Some((&first, rest)) = group_ids.split_first() else {
        return false;
    };

    let mut run_count = 1usize;
    let mut group_id = first;
    for &next_group_id in rest {
        if next_group_id != group_id {
            run_count += 1;
            group_id = next_group_id;
        }
    }

    run_count * MIN_AVG_RUN_LENGTH_FOR_GROUPED_SUM_RUNS <= group_ids.len()
}

fn for_each_group_run(group_ids: &[u32], mut f: impl FnMut(u32, usize, usize)) {
    let Some((&first, rest)) = group_ids.split_first() else {
        return;
    };

    let mut group_id = first;
    let mut start = 0usize;
    for (idx, &next_group_id) in rest.iter().enumerate() {
        let idx = idx + 1;
        if next_group_id != group_id {
            f(group_id, start, idx);
            group_id = next_group_id;
            start = idx;
        }
    }
    f(group_id, start, group_ids.len());
}

fn accumulate_grouped_unsigned(partials: &mut [SumPartial], group_id: u32, value: u64) {
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Unsigned(acc)) => checked_add_u64(acc, value),
        Some(_) => vortex_panic!("unsigned sum state with non-unsigned input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_unsigned_run<T>(partials: &mut [SumPartial], group_id: u32, values: &[T])
where
    T: NativePType + AsPrimitive<u64>,
{
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Unsigned(acc)) => sum_unsigned_all(acc, values),
        Some(_) => vortex_panic!("unsigned sum state with non-unsigned input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_signed(partials: &mut [SumPartial], group_id: u32, value: i64) {
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Signed(acc)) => checked_add_i64(acc, value),
        Some(_) => vortex_panic!("signed sum state with non-signed input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_signed_run<T>(partials: &mut [SumPartial], group_id: u32, values: &[T])
where
    T: NativePType + AsPrimitive<i64>,
{
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Signed(acc)) => sum_signed_all(acc, values),
        Some(_) => vortex_panic!("signed sum state with non-signed input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_float(partials: &mut [SumPartial], group_id: u32, value: f64) {
    if value.is_nan() {
        return;
    }

    match partials[group_id as usize].current.as_mut() {
        None => {}
        Some(SumState::Float(acc)) => *acc += value,
        Some(_) => vortex_panic!("float sum state with non-float input"),
    }
}

fn accumulate_grouped_float_run<T: NativePType>(
    partials: &mut [SumPartial],
    group_id: u32,
    values: &[T],
) {
    match partials[group_id as usize].current.as_mut() {
        None => {}
        Some(SumState::Float(acc)) => sum_float_all(acc, values),
        Some(_) => vortex_panic!("float sum state with non-float input"),
    }
}

pub(super) fn accumulate_grouped_primitive(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;
    let use_runs =
        matches!(validity.slices(), AllOr::All) && should_accumulate_group_runs(group_ids);

    match_each_native_ptype!(primitive.ptype(),
        unsigned: |T| {
            if use_runs {
                accumulate_grouped_primitive_unsigned_runs::<T>(partials, primitive, group_ids);
            } else {
                accumulate_grouped_primitive_unsigned::<T>(partials, primitive, group_ids, &validity);
            }
            Ok(())
        },
        signed: |T| {
            if use_runs {
                accumulate_grouped_primitive_signed_runs::<T>(partials, primitive, group_ids);
            } else {
                accumulate_grouped_primitive_signed::<T>(partials, primitive, group_ids, &validity);
            }
            Ok(())
        },
        floating: |T| {
            if use_runs {
                accumulate_grouped_primitive_float_runs::<T>(partials, primitive, group_ids);
            } else {
                accumulate_grouped_primitive_float::<T>(partials, primitive, group_ids, &validity);
            }
            Ok(())
        }
    )
}

fn accumulate_grouped_primitive_unsigned<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativePType + AsPrimitive<u64>,
{
    let values = primitive.as_slice::<T>();
    for_each_valid_idx(validity, values.len(), |idx| {
        accumulate_grouped_unsigned(partials, group_ids[idx], values[idx].as_());
    });
}

fn accumulate_grouped_primitive_unsigned_runs<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
) where
    T: NativePType + AsPrimitive<u64>,
{
    let values = primitive.as_slice::<T>();
    for_each_group_run(group_ids, |group_id, start, end| {
        accumulate_grouped_unsigned_run(partials, group_id, &values[start..end]);
    });
}

fn accumulate_grouped_primitive_signed<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativePType + AsPrimitive<i64>,
{
    let values = primitive.as_slice::<T>();
    for_each_valid_idx(validity, values.len(), |idx| {
        accumulate_grouped_signed(partials, group_ids[idx], values[idx].as_());
    });
}

fn accumulate_grouped_primitive_signed_runs<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
) where
    T: NativePType + AsPrimitive<i64>,
{
    let values = primitive.as_slice::<T>();
    for_each_group_run(group_ids, |group_id, start, end| {
        accumulate_grouped_signed_run(partials, group_id, &values[start..end]);
    });
}

fn accumulate_grouped_primitive_float<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativePType + ToPrimitive,
{
    let values = primitive.as_slice::<T>();
    for_each_valid_idx(validity, values.len(), |idx| {
        let value = values[idx].to_f64().vortex_expect("float to f64");
        accumulate_grouped_float(partials, group_ids[idx], value);
    });
}

fn accumulate_grouped_primitive_float_runs<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
) where
    T: NativePType,
{
    let values = primitive.as_slice::<T>();
    for_each_group_run(group_ids, |group_id, start, end| {
        accumulate_grouped_float_run(partials, group_id, &values[start..end]);
    });
}

pub(super) fn accumulate_grouped_bool(
    partials: &mut [SumPartial],
    bools: &BoolArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = bools
        .as_ref()
        .validity()?
        .execute_mask(bools.as_ref().len(), ctx)?;
    let values = bools.to_bit_buffer();
    for_each_valid_idx(&validity, values.len(), |idx| {
        if values.value(idx) {
            accumulate_grouped_unsigned(partials, group_ids[idx], 1);
        }
    });
    Ok(())
}
