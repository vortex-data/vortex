// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod rules;
mod slice_;
mod vtable;

use std::ops::Range;

pub use array::*;
use vortex_error::VortexResult;
pub use vtable::*;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::vtable::VTable;

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
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>>;
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
        array: &Self::Array,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Default, Debug)]
pub struct SliceReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for SliceReduceAdaptor<V>
where
    V: SliceReduce,
{
    type Parent = SliceVTable;

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        <V as SliceReduce>::slice(array, parent.range.clone())
    }
}

#[derive(Default, Debug)]
pub struct SliceExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for SliceExecuteAdaptor<V>
where
    V: SliceKernel,
{
    type Parent = SliceVTable;

    fn execute_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        <V as SliceKernel>::slice(array, parent.range.clone(), ctx)
    }
}
