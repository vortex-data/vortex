// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::expr::stats::Precision;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar::Scalar;
use crate::validity::Validity;

pub trait TakeReduce: VTable {
    /// Take elements from an array at the given indices without reading buffers.
    ///
    /// This trait is for take implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if taking requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

pub trait TakeExecute: VTable {
    /// Take elements from an array at the given indices, potentially reading buffers.
    ///
    /// Unlike [`TakeReduce`], this trait is for take implementations that may need to read
    /// and execute on the underlying buffers to produce the result.
    ///
    /// # Preconditions
    ///
    /// The indices are guaranteed to be non-empty.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Short-circuits take for the inputs that need no encoding-specific work.
///
/// Returns `Some(result)` when the answer is already known, or `None` when take must proceed
/// normally.
fn short_circuit<V: VTable>(array: ArrayView<'_, V>, indices: &ArrayRef) -> Option<ArrayRef> {
    // Fast-path for empty indices.
    if indices.is_empty() {
        let result_dtype = array
            .dtype()
            .clone()
            .union_nullability(indices.dtype().nullability());
        return Some(Canonical::empty(&result_dtype).into_array());
    }

    // Fast-path for empty arrays: all indices must be null, return all-invalid result.
    if array.is_empty() {
        return Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    None
}

#[derive(Default, Debug)]
pub struct TakeReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for TakeReduceAdaptor<V>
where
    V: TakeReduce,
{
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = short_circuit::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeReduce>::take(array, parent.codes())?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes(), None)?;
        }
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct TakeExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for TakeExecuteAdaptor<V>
where
    V: TakeExecute,
{
    type Parent = Dict;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle the values child (index 1), not the codes child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        if let Some(result) = short_circuit::<V>(array, parent.codes()) {
            return Ok(Some(result));
        }
        let result = <V as TakeExecute>::take(array, parent.codes(), ctx)?;
        if let Some(taken) = &result {
            propagate_take_stats(array.array(), taken, parent.codes(), Some(ctx))?;
        }
        Ok(result)
    }
}

pub(crate) fn propagate_take_stats(
    source: &ArrayRef,
    target: &ArrayRef,
    indices: &ArrayRef,
    ctx: Option<&ExecutionCtx>,
) -> VortexResult<()> {
    let get_result = |aggregate: &AggregateFnRef| {
        ctx.map_or_else(
            || source.aggregations().get_result(aggregate),
            |ctx| ctx.aggregate_result(source, aggregate),
        )
    };
    let insert_result = |aggregate, result| match ctx {
        Some(ctx) => ctx.insert_aggregate_result(target, aggregate, result),
        None => target.aggregations().insert_result(aggregate, result),
    };
    let indices_all_valid = matches!(
        indices.validity()?,
        Validity::NonNullable | Validity::AllValid
    );
    let is_constant = AggregateFn::new(IsConstant, EmptyOptions).erased();
    if indices_all_valid
        && !target.is_empty()
        && matches!(
            get_result(&is_constant)
                .map(|value| bool::try_from(&value))
                .transpose()?,
            Precision::Exact(true)
        )
    {
        // Taking valid indices preserves a non-empty constant input's constantness.
        insert_result(is_constant, Precision::Exact(true.into()))?;
    }

    for aggregate in [
        AggregateFn::new(Min, NumericalAggregateOpts::default()).erased(),
        AggregateFn::new(Max, NumericalAggregateOpts::default()).erased(),
    ] {
        let bound = get_result(&aggregate).into_inexact();
        if bound
            .as_ref()
            .into_inner()
            .is_some_and(|value| !value.is_null())
        {
            insert_result(aggregate, bound)?;
        }
    }

    Ok(())
}
