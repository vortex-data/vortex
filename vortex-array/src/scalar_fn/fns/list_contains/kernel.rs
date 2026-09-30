// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::list_contains::ListContains as ListContainsExpr;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;

/// Check list-contains without reading buffers (metadata-only).
///
/// This trait dispatches on the **element** (needle) child at index 1 of the `ListContains`
/// expression. `Self` is the concrete element encoding, while the list (haystack) is passed as an
/// opaque `&ArrayRef`.
///
/// Return `None` if the operation cannot be resolved from metadata alone.
pub trait ListContainsElementReduce: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Check list-contains, potentially reading buffers.
///
/// Like [`ListContainsElementReduce`], this dispatches on the **element** (needle) child at
/// index 1. Unlike the reduce variant, implementations may read and execute on buffers via
/// the provided [`ExecutionCtx`].
///
/// For a needle that is not canonical, execution prepares a constant list into a
/// [`PreparedSetArray`] and runs the kernels again. Thus a kernel can get the prepared set with
/// `list.as_opt::<PreparedSet>()`, and probe its own values with [`PreparedSetData::contains`].
/// For example, a dictionary probes only its values. A single value, such as the fill value of a
/// sparse needle, is probed with [`PreparedSetData::contains_scalar`]. A kernel that finds its
/// matches in another way makes its result with [`PreparedSetData::result_from_bits`], which applies
/// the same null semantics.
///
/// [`PreparedSetArray`]: crate::scalar_fn::fns::list_contains::PreparedSetArray
/// [`PreparedSetData::contains`]: crate::scalar_fn::fns::list_contains::PreparedSetData::contains
/// [`PreparedSetData::contains_scalar`]: crate::scalar_fn::fns::list_contains::PreparedSetData::contains_scalar
/// [`PreparedSetData::result_from_bits`]: crate::scalar_fn::fns::list_contains::PreparedSetData::result_from_bits
pub trait ListContainsElementKernel: VTable {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
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
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        <V as ListContainsElementReduce>::list_contains(list, array, parent.options)
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
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, ListContainsExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only process the element/needle child (index 1), not the list child (index 0).
        if child_idx != 1 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let list = scalar_fn_array.get_child(0);
        <V as ListContainsElementKernel>::list_contains(list, array, parent.options, ctx)
    }
}
