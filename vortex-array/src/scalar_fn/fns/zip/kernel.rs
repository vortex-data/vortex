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
use crate::scalar_fn::fns::zip::Zip as ZipExpr;
use crate::vtable::VTable;

/// Zip two arrays using a mask without reading buffers.
///
/// This trait is for zip implementations that can operate purely on array metadata
/// and structure without needing to read or execute on the underlying buffers.
/// Implementations should return `None` if the operation requires buffer access.
///
/// Dispatch is on child 0 (if_true). The `if_false` and `mask` are extracted from
/// the parent `ScalarFnArray`.
pub trait ZipReduce: VTable {
    fn zip(
        array: &Self::Array,
        if_false: &ArrayRef,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Zip two arrays using a mask, potentially reading buffers.
///
/// Unlike [`ZipReduce`], this trait is for zip implementations that may need
/// to read and execute on the underlying buffers to produce the result.
///
/// Dispatch is on child 0 (if_true). The `if_false` and `mask` are extracted from
/// the parent `ScalarFnArray`.
pub trait ZipKernel: VTable {
    fn zip(
        array: &Self::Array,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`ZipReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ZipReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ZipReduceAdaptor<V>
where
    V: ZipReduce,
{
    type Parent = ExactScalarFn<ZipExpr>;

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, ZipExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let if_false = scalar_fn_array.get_child(1);
        let mask_array = scalar_fn_array.get_child(2);
        <V as ZipReduce>::zip(array, if_false, mask_array)
    }
}

/// Adaptor that wraps a [`ZipKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ZipExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ZipExecuteAdaptor<V>
where
    V: ZipKernel,
{
    type Parent = ExactScalarFn<ZipExpr>;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, ZipExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let if_false = scalar_fn_array.get_child(1);
        let mask_array = scalar_fn_array.get_child(2);
        <V as ZipKernel>::zip(array, if_false, mask_array, ctx)
    }
}
