// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata-only rewrite rules for the optimizer (Layers 1 and 2 of the execution model).
//!
//! Reduce rules are the cheapest transformations in the execution pipeline: they operate
//! purely on array structure and metadata without reading any data buffers.
//!
//! There are two kinds of reduce rules:
//!
//! - [`ArrayReduceRule`] (Layer 1) -- a self-rewrite where an array simplifies itself.
//!   Example: a `FilterArray` with an all-true mask removes the filter wrapper.
//!
//! - [`ArrayParentReduceRule`] (Layer 2) -- a child-driven rewrite where a child rewrites
//!   its parent. Example: a `DictArray` child of a `ScalarFnArray` pushes the scalar function
//!   into the dictionary values.
//!
//! Reduce rules are collected into a [`ReduceRuleSet`] per encoding. Parent-reduce rules are
//! registered with the session's [`ArrayKernels`](crate::optimizer::kernels::ArrayKernels) for
//! each `(parent, child)` pair they apply to. The optimizer evaluates both in a fixpoint loop until
//! no more rules apply.

use std::fmt::Debug;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::matcher::Matcher;
use crate::trace_op;

/// A metadata-only rewrite rule that transforms an array based on its own structure (Layer 1).
///
/// These rules look only at the array's metadata and children types (not buffer contents)
/// and return a structurally simpler replacement, or `None` if the rule doesn't apply.
pub trait ArrayReduceRule<V: VTable>: Debug + Send + Sync + 'static {
    /// Attempt to rewrite this array.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce(&self, array: ArrayView<'_, V>) -> VortexResult<Option<ArrayRef>>;
}

/// A metadata-only rewrite rule where a child encoding rewrites its parent (Layer 2).
///
/// The child sees the parent's type via the associated `Parent` [`Matcher`] and can return
/// a replacement for the parent. This enables optimizations like pushing operations through
/// compression layers (e.g., pushing a scalar function into dictionary values).
pub trait ArrayParentReduceRule<V: VTable>: Debug + Send + Sync + 'static {
    /// The parent array type this rule matches against.
    type Parent: Matcher;

    /// Attempt to rewrite this child array given information about its parent.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// A collection of [`ArrayReduceRule`]s registered for a specific encoding.
///
/// During optimization, the optimizer calls [`evaluate`](Self::evaluate) which tries each rule
/// in order. The first rule that returns `Some` wins.
pub struct ReduceRuleSet<V: VTable> {
    rules: &'static [&'static dyn ArrayReduceRule<V>],
}

impl<V: VTable> ReduceRuleSet<V> {
    /// Create a new reduction rule set with the given rules.
    pub const fn new(rules: &'static [&'static dyn ArrayReduceRule<V>]) -> Self {
        Self { rules }
    }

    /// Evaluate the reduction rules on the given array.
    pub fn evaluate(&self, array: ArrayView<'_, V>) -> VortexResult<Option<ArrayRef>> {
        for rule in self.rules.iter() {
            if let Some(reduced) = rule.reduce(array)? {
                trace_op!(record_reduce_applied(array.array(), *rule, &reduced));
                return Ok(Some(reduced));
            }
            trace_op!(record_reduce_declined(array.array(), *rule));
        }
        Ok(None)
    }
}
