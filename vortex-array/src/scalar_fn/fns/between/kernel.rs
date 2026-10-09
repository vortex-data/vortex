// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Between;
use super::BetweenOptions;
use super::short_circuit;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::builtins::ArrayBuiltins;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

/// Reduce rule for between: restructure the array without reading buffers.
///
/// Returns `Ok(None)` if the rule doesn't apply or buffer access is needed.
pub trait BetweenReduce: VTable {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Execute kernel for between: perform the actual between check, potentially reading buffers.
///
/// Returns `Ok(None)` if this kernel cannot handle the given inputs.
pub trait BetweenKernel: VTable {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapts a [`BetweenReduce`] impl into an [`ArrayParentReduceRule`] for `ScalarFnArray(Between, ...)`.
#[derive(Default, Debug)]
pub struct BetweenReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for BetweenReduceAdaptor<V>
where
    V: BetweenReduce,
{
    type Parent = ExactScalarFn<Between>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Between>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the main array child (index 0), not lower (1) or upper (2).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let children = scalar_fn_array.children();
        let lower = &children[1];
        let upper = &children[2];
        let arr = array.array().clone();
        if let Some(result) = short_circuit(&arr, lower, upper, parent.options)? {
            return Ok(Some(result));
        }
        <V as BetweenReduce>::between(array, lower, upper, parent.options)
    }
}

/// Adapts a [`BetweenKernel`] impl into an [`ExecuteParentKernel`] for `ScalarFnArray(Between, ...)`.
#[derive(Default, Debug)]
pub struct BetweenExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for BetweenExecuteAdaptor<V>
where
    V: BetweenKernel,
{
    type Parent = ExactScalarFn<Between>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Between>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the main array child (index 0), not lower (1) or upper (2).
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let children = scalar_fn_array.children();
        let lower = &children[1];
        let upper = &children[2];
        let arr = array.array().clone();
        if let Some(result) = short_circuit(&arr, lower, upper, parent.options)? {
            // TODO(joe): return the lazy array directly, blocked on the same executor support as
            // the fallback in `between_canonical`. The reduce adaptor above already passes it
            // through unexecuted, since a reduce rule can return a lazy array.
            return result.execute::<ArrayRef>(ctx).map(Some);
        }
        <V as BetweenKernel>::between(array, lower, upper, parent.options, ctx)
    }
}

/// Uses an encoding's comparison kernels when it has no fused between kernel.
///
/// Both comparisons must be handled by the encoding. Otherwise this declines so the
/// normal between fallback can canonicalize the input once.
#[derive(Default, Debug)]
pub struct BetweenCompareAdaptor<V>(pub V);

impl<V: CompareKernel> ExecuteParentKernel<V> for BetweenCompareAdaptor<V> {
    type Parent = ExactScalarFn<Between>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Between>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let children = parent.children();
        let lower = &children[1];
        let upper = &children[2];
        if let Some(result) = short_circuit(array.array(), lower, upper, parent.options)? {
            return result.execute::<ArrayRef>(ctx).map(Some);
        }
        between_compare(array, lower, upper, parent.options, ctx)
    }
}

fn between_compare<V: CompareKernel>(
    array: ArrayView<'_, V>,
    lower: &ArrayRef,
    upper: &ArrayRef,
    options: &BetweenOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let lower_op = if options.lower_strict.is_strict() {
        CompareOperator::Gt
    } else {
        CompareOperator::Gte
    };
    let upper_op = if options.upper_strict.is_strict() {
        CompareOperator::Lt
    } else {
        CompareOperator::Lte
    };
    let Some(lower_cmp) = V::compare(array, lower, lower_op, ctx)? else {
        return Ok(None);
    };
    let Some(upper_cmp) = V::compare(array, upper, upper_op, ctx)? else {
        return Ok(None);
    };
    lower_cmp
        .binary(upper_cmp, Operator::And)?
        .execute::<ArrayRef>(ctx)
        .map(Some)
}

#[cfg(test)]
mod tests;
