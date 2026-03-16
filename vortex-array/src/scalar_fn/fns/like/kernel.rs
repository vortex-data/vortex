// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ScalarFnVTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::like::Like as LikeExpr;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::vtable::VTable;

/// Like pattern matching on an array without reading buffers.
///
/// This trait is for like implementations that can operate purely on array metadata
/// and structure without needing to read or execute on the underlying buffers.
/// Implementations should return `None` if the operation requires buffer access.
///
/// Dispatch is on child 0 (the input). The `pattern` and `options` are extracted from
/// the parent `ScalarFnArray`.
pub trait LikeReduce: VTable {
    fn like(
        array: &Self::Array,
        pattern: &ArrayRef,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Like pattern matching on an array, potentially reading buffers.
///
/// Unlike [`LikeReduce`], this trait is for like implementations that may need
/// to read and execute on the underlying buffers to produce the result.
///
/// Dispatch is on child 0 (the input). The `pattern` and `options` are extracted from
/// the parent `ScalarFnArray`.
pub trait LikeKernel: VTable {
    fn like(
        array: &Self::Array,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`LikeReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct LikeReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for LikeReduceAdaptor<V>
where
    V: LikeReduce,
{
    type Parent = ExactScalarFn<LikeExpr>;

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, LikeExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let pattern = scalar_fn_array.get_child(1);
        let options = *parent.options;
        <V as LikeReduce>::like(array, pattern, options)
    }
}

/// Adaptor that wraps a [`LikeKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct LikeExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for LikeExecuteAdaptor<V>
where
    V: LikeKernel,
{
    type Parent = ExactScalarFn<LikeExpr>;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, LikeExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let pattern = scalar_fn_array.get_child(1);
        let options = *parent.options;
        <V as LikeKernel>::like(array, pattern, options, ctx)
    }
}
