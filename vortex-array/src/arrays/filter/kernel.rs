// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce and execute adaptors for filter operations.
//!
//! Encodings that know how to filter themselves implement [`FilterReduce`] (metadata-only)
//! or [`FilterKernel`] (buffer-reading). The adaptors [`FilterReduceAdaptor`] and
//! [`FilterExecuteAdaptor`] bridge these into the execution model as
//! [`ArrayParentReduceRule`] and [`ExecuteParentKernel`] respectively.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::FilterArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterSlots;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::execute_parent_for_child;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::ScalarFnPlugin;
use crate::scalar_fn::fns::binary::Binary;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Dict.id(), Filter, TakeExecuteAdaptor(Filter));
    kernels.register_execute_parent_kernel(Binary.id(), Filter, FilterScalarFnUnaryPushDownRule);
}

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

/// Short-circuits filter for the inputs that need no encoding-specific work.
///
/// Returns `Some(result)` when the answer is already known, or `None` when the filter must proceed
/// normally.
fn short_circuit<V: VTable>(array: ArrayView<'_, V>, mask: &Mask) -> Option<ArrayRef> {
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
        if let Some(result) = short_circuit::<V>(array, parent.filter_mask()) {
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
        if let Some(result) = short_circuit::<V>(array, parent.filter_mask()) {
            return Ok(Some(result));
        }
        <V as FilterKernel>::filter(array, parent.filter_mask(), ctx)
    }
}

#[derive(Debug)]
struct FilterScalarFnUnaryPushDownRule;

impl ExecuteParentKernel<Filter> for FilterScalarFnUnaryPushDownRule {
    type Parent = ScalarFn;

    fn execute_parent(
        &self,
        child: ArrayView<'_, Filter>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we have one non-constant child, and ScalarFn has a registered
        // kernel for given encoding id (e.g. can operate on compressed data),
        // it's faster to pull Filter up so that it doesn't canonicalize its
        // child.
        //
        // Fn(Filter(x), consts) -> Filter(Fn(x, consts))
        //
        // Example: clickbench q1, SELECT * FROM hits WHERE AdvEngineID <> 0;
        // AdvEngineID <> 0 is Binary over Sparse, but FlatReader applies
        // Filter over Sparse, so we get Binary(Filter(Sparse)). Filter(Sparse)
        // canonicalizes.
        if parent
            .iter_children()
            .filter(|c| !c.is::<Constant>())
            .count()
            != 1
        {
            return Ok(None);
        }

        // "x" in above formula
        let new_non_const_grandchild = child.slots()[FilterSlots::CHILD]
            .as_ref()
            .vortex_expect("no child for Filter");

        let unfiltered_len = new_non_const_grandchild.len();
        // (x, consts)
        let new_grandchildren: Vec<_> = parent
            .iter_children()
            .map(|c| match c.as_constant() {
                Some(scalar) => ConstantArray::new(scalar, unfiltered_len).into_array(),
                // by above check this is the only non-const argument
                None => new_non_const_grandchild.clone(),
            })
            .collect();

        // Fn(x, consts)
        let new_child =
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_grandchildren)?.into_array();

        // Eagerly execute swapped Fn to avoid infinite runtime with
        // ScalarFnUnaryFilterPushDownRule.
        //
        // This is Res = Fn(x, consts)
        let new_child = execute_parent_for_child(
            "filter_scalar_fn_pushdown",
            &new_child,               // parent, Fn
            new_non_const_grandchild, // child, x
            child_idx,
            &ctx.execute_parent_kernels.clone(),
            ctx,
        )?;
        let Some(new_child) = new_child else {
            // All child kernels rejected, can't proceed
            return Ok(None);
        };

        let mask = child.filter_mask().clone();
        // Filter(Res)
        let new_parent = FilterArray::try_new(new_child.clone(), mask)?;
        Ok(Some(new_parent.into_array()))
    }
}
