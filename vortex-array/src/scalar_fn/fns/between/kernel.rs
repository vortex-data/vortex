// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Between;
use super::BetweenOptions;
use super::precondition;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ScalarFnVTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;

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
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let children = scalar_fn_array.children();
        let lower = &children[1];
        let upper = &children[2];
        let arr = array.array().clone();
        if let Some(result) = precondition(&arr, lower, upper)? {
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
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let children = scalar_fn_array.children();
        let lower = &children[1];
        let upper = &children[2];
        let arr = array.array().clone();
        if let Some(result) = precondition(&arr, lower, upper)? {
            return Ok(Some(result));
        }
        <V as BetweenKernel>::between(array, lower, upper, parent.options, ctx)
    }
}
