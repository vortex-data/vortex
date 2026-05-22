// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce and execute adaptors for filter operations.
//!
//! Encodings that know how to filter themselves implement [`FilterReduce`] (metadata-only)
//! or [`FilterKernel`] (buffer-reading). The adaptors [`FilterReduceAdaptor`] and
//! [`FilterExecuteAdaptor`] bridge these into the execution model as
//! [`ArrayParentReduceRule`] and [`ExecuteParentKernel`] respectively.

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Filter;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;

pub(super) const PARENT_KERNELS: ParentKernelSet<Filter> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(Filter))]);

pub trait FilterReduce: VTable {
    /// Filter an array with the provided mask without reading buffers.
    ///
    /// This trait is for filter implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if filtering requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The mask is guaranteed to have the same length as the array.
    ///
    /// Additionally, the mask is guaranteed to be a `Mask::Values` variant (i.e., neither
    /// `Mask::AllTrue` nor `Mask::AllFalse`).
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>>;
}

pub trait FilterKernel: VTable {
    /// Filter an array with the provided mask, potentially reading buffers.
    ///
    /// Unlike [`FilterReduce`], this trait is for filter implementations that may need to read
    /// and execute on the underlying buffers to produce the filtered result.
    ///
    /// # Preconditions
    ///
    /// The mask is guaranteed to have the same length as the array.
    ///
    /// Additionally, the mask is guaranteed to be a `Mask::Values` variant (i.e., neither
    /// `Mask::AllTrue` nor `Mask::AllFalse`).
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Common preconditions for filter operations that apply to all arrays.
///
/// Returns `Some(result)` if the precondition short-circuits the filter operation,
/// or `None` if the filter should proceed normally.
fn precondition<V: VTable>(array: ArrayView<'_, V>, mask: &Mask) -> Option<ArrayRef> {
    let true_count = mask.true_count();

    // Fast-path for empty mask (all false).
    if true_count == 0 {
        return Some(Canonical::empty(array.dtype()).into_array());
    }

    // Fast-path for full mask (all true).
    if true_count == mask.len() {
        return Some(array.array().clone());
    }

    None
}

/// Adaptor that wraps a [`FilterReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct FilterReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for FilterReduceAdaptor<V>
where
    V: FilterReduce,
{
    type Parent = Filter;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ArrayView<'_, Filter>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        if let Some(result) = precondition::<V>(array, parent.filter_mask()) {
            return Ok(Some(result));
        }
        <V as FilterReduce>::filter(array, parent.filter_mask())
    }
}

/// Adaptor that wraps a [`FilterKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct FilterExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for FilterExecuteAdaptor<V>
where
    V: FilterKernel,
{
    type Parent = Filter;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        if let Some(result) = precondition::<V>(array, parent.filter_mask()) {
            return Ok(Some(result));
        }
        <V as FilterKernel>::filter(array, parent.filter_mask(), ctx)
    }
}
