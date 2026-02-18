// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ExactScalarFn;
use crate::arrays::ScalarFnArrayView;
use crate::arrays::ScalarFnVTable;
use crate::expr::ListContains as ListContainsExpr;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::vtable::VTable;

/// Check list-contains without reading buffers (metadata-only).
///
/// Implementations dispatch on the **element** (needle) child at index 1.
/// Return `None` if the operation requires buffer access.
pub trait ListContainsElementReduce: VTable {
    fn list_contains_element(
        list: &dyn Array,
        element: &Self::Array,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Unlike [`ListContainsElementReduce`], this trait is for implementations that may need
/// to read and execute on the underlying buffers to produce the result.
pub trait ListContainsElementKernel: VTable {
    fn list_contains_element(
        list: &dyn Array,
        element: &Self::Array,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`ListContainsElementReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct ListContainsElementReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for ListContainsElementReduceAdaptor<V>
where
    V: ListContainsElementReduce,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = &scalar_fn_array.children()[0];
        <V as ListContainsElementReduce>::list_contains_element(list.as_ref(), array)
    }
}

/// Adaptor that wraps a [`ListContainsElementKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct ListContainsElementExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for ListContainsElementExecuteAdaptor<V>
where
    V: ListContainsElementKernel,
{
    type Parent = ExactScalarFn<ListContainsExpr>;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFnVTable>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = &scalar_fn_array.children()[0];
        <V as ListContainsElementKernel>::list_contains_element(list.as_ref(), array, ctx)
    }
}
