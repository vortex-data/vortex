// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels: child-driven fused execution of parent arrays.
//!
//! A parent kernel allows a child encoding to provide a specialized execution path for its
//! parent array. This is Layer 3 of the [execution model](https://docs.vortex.dev/developer-guide/internals/execution).
//!
//! For example, a `RunEndArray` child of a `SliceArray` can perform a binary search on its
//! run ends rather than decoding the entire array and slicing the result.
//!
//! Encodings declare their parent kernels by implementing [`ExecuteParentKernel`] and
//! registering them with the session's optimizer-kernel registry. Each kernel specifies which
//! parent types it handles via a [`Matcher`].

use std::fmt::Debug;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::matcher::Matcher;

/// A kernel that allows a child encoding `V` to execute its parent array in a fused manner.
///
/// This is the typed trait that encoding authors implement. The associated `Parent` type
/// specifies which parent array types this kernel can handle. When the parent matches,
/// [`execute_parent`](Self::execute_parent) is called with the strongly-typed child and parent views.
///
/// Unlike reduce rules, parent kernels may read buffers and perform real computation.
///
/// Return `Ok(None)` to decline handling (the scheduler will try the next kernel or fall
/// through to the encoding's own `execute`).
pub trait ExecuteParentKernel<V: VTable>: Debug + Send + Sync + 'static {
    /// The parent array type this kernel handles.
    type Parent: Matcher;

    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
