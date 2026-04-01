// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::not::Not as NotExpr;

/// Invert a boolean array without reading buffers.
///
/// This trait is for invert implementations that can operate purely on array metadata
/// and structure without needing to read or execute on the underlying buffers.
/// Implementations should return `None` if the operation requires buffer access.
pub trait NotReduce: VTable {
    fn invert(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>>;
}

/// Invert a boolean array, potentially reading buffers.
///
/// Unlike [`NotReduce`], this trait is for invert implementations that may need
/// to read and execute on the underlying buffers to produce the result.
pub trait NotKernel: VTable {
    fn invert(array: ArrayView<'_, Self>, ctx: &mut ExecutionCtx)
    -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`NotReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct NotReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for NotReduceAdaptor<V>
where
    V: NotReduce,
{
    type Parent = ExactScalarFn<NotExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: ScalarFnArrayView<'_, NotExpr>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        <V as NotReduce>::invert(array)
    }
}

/// Adaptor that wraps a [`NotKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct NotExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for NotExecuteAdaptor<V>
where
    V: NotKernel,
{
    type Parent = ExactScalarFn<NotExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: ScalarFnArrayView<'_, NotExpr>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <V as NotKernel>::invert(array, ctx)
    }
}
