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

/// Whether a parent kernel handles a `(child, parent)` pair, as reported by
/// [`ExecuteParentKernel::applies`].
///
/// Variants are ordered by strength, so the combined answer for several kernels is their maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Applies {
    /// The kernel may return `None` depending on data only known during execution.
    Sometimes,
    /// The kernel is guaranteed to return `Some`.
    Always,
}

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

    /// Report, without executing, whether [`execute_parent`](Self::execute_parent) handles this
    /// `(child, parent)` pair.
    ///
    /// Returns `None` when the kernel will certainly decline, [`Applies::Always`] when it is
    /// guaranteed to return `Some`, and [`Applies::Sometimes`] when the outcome depends on data
    /// only known during execution. This must be cheap: inspect metadata such as dtypes and
    /// scalar function options, never buffers.
    ///
    /// Defaults to [`Applies::Sometimes`], which is always a sound answer.
    fn applies(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> Option<Applies> {
        _ = (array, parent, child_idx);
        Some(Applies::Sometimes)
    }

    /// Attempt to execute the parent array fused with the child array.
    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
