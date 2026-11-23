// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::list_contains::ListContains as ListContainsExpr;
use crate::list_contains::ListContainsOptions;

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
/// The list of an optimized node is a [`PreparedSet`], so a kernel can get the prepared set with
/// [`PreparedSet::set_of`] and look up its own values with
/// [`PreparedSetData::contains`] or [`PreparedSetData::contains_scalar`], or make its result from
/// matches it found another way with [`PreparedSetData::result_from_bits`].
///
/// [`PreparedSet`]: crate::list_contains::PreparedSet
/// [`PreparedSet::set_of`]: crate::list_contains::PreparedSet::set_of
/// [`PreparedSetData::contains`]: crate::list_contains::PreparedSetData::contains
/// [`PreparedSetData::contains_scalar`]: crate::list_contains::PreparedSetData::contains_scalar
/// [`PreparedSetData::result_from_bits`]: crate::list_contains::PreparedSetData::result_from_bits
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
