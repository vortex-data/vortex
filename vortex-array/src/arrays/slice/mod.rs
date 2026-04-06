// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce and execute adaptors for slice operations.
//!
//! Encodings that know how to slice themselves implement [`SliceReduce`] (metadata-only)
//! or [`SliceKernel`] (buffer-reading). The adaptors [`SliceReduceAdaptor`] and
//! [`SliceExecuteAdaptor`] bridge these into the execution model as
//! [`ArrayParentReduceRule`] and [`ExecuteParentKernel`] respectively.

mod array;
mod rules;
mod slice_;
mod vtable;

use std::ops::Range;

pub use array::SliceArrayParts;
pub use array::SliceData;
use vortex_error::VortexResult;
pub use vtable::*;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;

pub trait SliceReduce: VTable {
    /// Slice an array with the provided range without reading buffers.
    ///
    /// This trait is for slice implementations that can operate purely on array metadata and
    /// structure without needing to read or execute on the underlying buffers. Implementations
    /// should return `None` if slicing requires buffer access.
    ///
    /// # Preconditions
    ///
    /// The range is guaranteed to be within bounds of the array (i.e., `range.end <= array.len()`).
    ///
    /// Additionally, the range is guaranteed to be non-empty (i.e., `range.start < range.end`).
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>>;
}

pub trait SliceKernel: VTable {
    /// Slice an array with the provided range, potentially reading buffers.
    ///
    /// Unlike [`SliceReduce`], this trait is for slice implementations that may need to read
    /// and execute on the underlying buffers to produce the sliced result.
    ///
    /// # Preconditions
    ///
    /// The range is guaranteed to be within bounds of the array (i.e., `range.end <= array.len()`).
    ///
    /// Additionally, the range is guaranteed to be non-empty (i.e., `range.start < range.end`).
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

fn precondition<V: VTable>(array: ArrayView<'_, V>, range: &Range<usize>) -> Option<ArrayRef> {
    if range.start == 0 && range.end == array.len() {
        return Some(array.array().clone());
    };
    if range.start == range.end {
        return Some(Canonical::empty(array.dtype()).into_array());
    }
    None
}

/// Adaptor that wraps a [`SliceReduce`] impl as an [`ArrayParentReduceRule`].
#[derive(Default, Debug)]
pub struct SliceReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for SliceReduceAdaptor<V>
where
    V: SliceReduce,
{
    type Parent = Slice;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        if let Some(result) = precondition::<V>(array, &parent.range) {
            return Ok(Some(result));
        }
        <V as SliceReduce>::slice(array, parent.range.clone())
    }
}

/// Adaptor that wraps a [`SliceKernel`] impl as an [`ExecuteParentKernel`].
#[derive(Default, Debug)]
pub struct SliceExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for SliceExecuteAdaptor<V>
where
    V: SliceKernel,
{
    type Parent = Slice;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        if let Some(result) = precondition::<V>(array, &parent.range) {
            return Ok(Some(result));
        }
        <V as SliceKernel>::slice(array, parent.range.clone(), ctx)
    }
}
