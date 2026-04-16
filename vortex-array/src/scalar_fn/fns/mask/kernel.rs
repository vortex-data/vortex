// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Bool;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;

/// Mask an array without reading buffers.
///
/// This trait is for mask implementations that can operate purely on array metadata and
/// structure without needing to read or execute on the underlying buffers. Implementations
/// should return `None` if masking requires buffer access.
///
/// The `mask` parameter is a boolean array where true=keep/valid, false=null-out.
///
/// # Preconditions
///
/// The mask is guaranteed to have the same length as the array. Trivial cases
/// (`AllValid`, `AllInvalid`, `NonNullable`) are handled by the caller before dispatch.
pub trait MaskReduce: VTable {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

/// Mask an array, potentially reading buffers.
///
/// Unlike [`MaskReduce`], this trait is for mask implementations that may need to read
/// and execute on the underlying buffers to produce the masked result.
///
/// The `mask` parameter is a boolean array where true=keep/valid, false=null-out.
///
/// # Preconditions
///
/// The mask is guaranteed to have the same length as the array. Trivial cases
/// (`AllValid`, `AllInvalid`, `NonNullable`) are handled by the caller before dispatch.
pub trait MaskKernel: VTable {
    fn mask(
        array: ArrayView<'_, Self>,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`MaskReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct MaskReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for MaskReduceAdaptor<V>
where
    V: MaskReduce,
{
    type Parent = ExactScalarFn<MaskExpr>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, MaskExpr>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only reduce the input child (index 0), not the mask child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        // The mask child (child 1) is a non-nullable BoolArray where true=keep.
        // If it's not yet a BoolArray, we can't reduce without execution.
        let parent_ref: ArrayRef = (*parent).clone();
        let mask_child = parent_ref
            .nth_child(1)
            .ok_or_else(|| vortex_err!("Mask expression must have 2 children"))?;
        if mask_child.as_opt::<Bool>().is_none() {
            return Ok(None);
        };
        <V as MaskReduce>::mask(array, &mask_child)
    }
}

/// Adaptor that wraps a [`MaskKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct MaskExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for MaskExecuteAdaptor<V>
where
    V: MaskKernel,
{
    type Parent = ExactScalarFn<MaskExpr>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, MaskExpr>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only execute the input child (index 0), not the mask child (index 1).
        if child_idx != 0 {
            return Ok(None);
        }
        let mask_child = parent
            .nth_child(1)
            .ok_or_else(|| vortex_err!("Mask expression must have 2 children"))?;
        <V as MaskKernel>::mask(array, &mask_child, ctx)
    }
}
