// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Plan nodes represent the logical structure of a pipeline.

pub mod binary_bool;
mod compare;
mod mask_future;
mod scalar_compare;

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

pub use compare::CompareOperator;
use dyn_hash::DynHash;
pub use mask_future::MaskFuture;
pub use scalar_compare::ScalarCompareOperator;
use vortex_error::VortexResult;

use crate::pipeline::Kernel;
use crate::pipeline::types::VType;
use crate::pipeline::vec::VectorId;

pub type OperatorRef = Arc<dyn Operator>;

/// An operator represents a node in a logical query plan.
pub trait Operator: Debug + DynHash + Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;

    /// The output [`VType`] of this operator.
    fn vtype(&self) -> VType;

    /// The children of this operator.
    fn children(&self) -> &[OperatorRef];

    fn with_children(&self, children: Vec<OperatorRef>) -> OperatorRef;

    /// Whether this operator works by mutating its first child in-place.
    ///
    /// If `true`, the operator is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Create a kernel for this operator
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;

    /// Operator reduction optimization examples:
    ///
    /// Step 1 - Initial pipeline:
    ///   compare(_, 12) <- for(ref=10) <- bitpacked
    ///
    /// Step 2 - reduce_children optimization:
    ///   compare_scalar(12-10=2) <- bitpacked
    ///
    /// Step 3 - Final optimized pipeline (reduce_parent):
    ///   bitpacked -> compare_scalar(2)
    ///
    /// The reduction process eliminates the FoR decoding step by adjusting
    /// the comparison constant to work directly on encoded values.
    /// Given a set of reduced children, try and reduce the current node.
    /// If Keep is returned then the children of this node as still updated.
    fn reduce_children(&self, children: &[OperatorRef]) -> Option<OperatorRef> {
        None
    }

    /// Given a reduced parent, try and reduce the current node.
    /// If `Replace` is returned then  the parent node and this node and replaced by the returned node.
    fn reduce_parent(&self, parent: OperatorRef) -> Option<OperatorRef> {
        None
    }
}

dyn_hash::hash_trait_object!(Operator);

/// The context used when binding an operator for execution.
pub trait BindContext {
    fn children(&self) -> &[VectorId];
}
