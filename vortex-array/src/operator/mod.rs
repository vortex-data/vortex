// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines a new way of modelling arrays and expressions in Vortex. To avoid naming
//! conflicts, we refer to the new model as "operators".
//!
//! Operators form a more traditional "logical plan" as might be seen in other query engines.
//! Each operator supports one primary function which is to produce a canonical representation of
//! its data, known as `canonicalization`. Operators have the option to produce this canonical
//! form using different execution models, including batch, pipelined, and GPU.
//!
//! Initial designs for this module involved passing masks down through the physical execution
//! tree as futures, allowing operators to skip computation for rows that are not needed. We
//! ultimately decided against this approach and instead introduce a `Filter` operator
//! that can be pushed down in the same way as any other operator.
//!
//! On the one hand, this means common subtree elimination is much easier, since we know the mask
//! or identity of the mask future inside the filter operator up-front. On the other hand, it
//! means that an operator no longer has a known length. In the end state, we will redefine a
//! Vortex array to be a wrapped around an operator that _does_ have a known length, amongst other
//! properties (such as non-blocking evaluation).
//!
//! We also introduce the idea of an executor that can evaluate an operator tree efficiently. It
//! supports common subtree elimination, as well as extracting sub-graphs for pipelined and GPU
//! execution. The executor is also responsible for managing memory and scheduling work across
//! different execution resources.

pub mod canonical;
pub mod compare;
mod display;
pub mod filter;
pub mod getitem;
pub mod metrics;
pub mod slice;

use std::any::{type_name, Any};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::pipeline::{BindContext, Kernel, PipelinedOperator};
use crate::Canonical;
use arcref::ArcRef;
use async_trait::async_trait;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_utils::dyn_eq::{DynEq, DynHash};

use crate::webgpu::WebGpuOperator;
pub use display::*;

pub type OperatorId = ArcRef<str>;
pub type OperatorRef = Arc<dyn Operator>;

/// An operator represents a node in a logical query plan.
pub trait Operator: 'static + Debug + DynEq + DynHash + Send + Sync {
    /// The unique identifier for this operator instance.
    fn id(&self) -> OperatorId;

    /// For downcasting.
    fn as_any(&self) -> &dyn Any;

    /// A logical hash used to represent the compute this operator performs, excluding any data
    /// such as buffer contents or statistics.
    ///
    // NOTE(ngates): it was architecturally easier (and more performant) to not have two symmetric
    //  tree structures, one to represent compute and one containing data. Instead, we use a single
    //  tree structure of `dyn Operator` nodes with a logical hash to represent compute.
    fn logical_hash(&self, _hasher: &mut dyn Hasher) {
        // Not sure if this is right..
        todo!()
    }

    /// Returns the [`DType`] of the array produced by this operator.
    fn dtype(&self) -> &DType;

    /// Returns the (min, max) length bounds of the array produced by this operator.
    fn len(&self) -> usize;

    // TODO(ngates): add StatsSet

    /// The children of this operator.
    fn children(&self) -> &[OperatorRef];

    /// The number of children of this operator.
    fn nchildren(&self) -> usize {
        self.children().len()
    }

    /// Override the default formatting of this operator.
    fn fmt_as(&self, _df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", type_name::<Self>())
    }

    /// Create a new instance of this operator with the given children.
    ///
    /// ## Panics
    ///
    /// Panics if the number or dtypes of children are incorrect.
    ///
    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef>;

    /// Attempt to optimize this node by analyzing its children.
    ///
    /// For example, if all the children are constant, this function should perform constant
    /// folding and return a constant operator.
    ///
    /// This function should typically be implemented only for self-contained optimizations based
    /// on child properties
    fn reduce_children(&self) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
    }

    /// Attempt to push down a parent operator through this node.
    ///
    /// The `child_idx` parameter indicates which child of the parent this operator occupies.
    /// For example, if the parent is a binary operator, and this operator is the left child,
    /// then `child_idx` will be 0. If this operator is the right child, then `child_idx` will be 1.
    ///
    /// The returned operator will replace the parent in the tree.
    ///
    /// This function should typically be implemented for cross-operator optimizations where the
    /// child needs to adapt to the parent's requirements
    fn reduce_parent(
        &self,
        _parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
    }

    // Which fields this operator accesses (for projection pushdown)
    // fn required_columns(&self) -> Option<Vec<FieldPath>> {
    //     None
    // }

    /// Whether this operator is aligned 1:1 with its child.
    ///
    /// Returns `None` if unknown.
    fn is_position_preserving(&self, _child_idx: usize) -> Option<bool> {
        None
    }

    /// Whether this operator preserves the nulls of the given position-preserving child.
    ///
    /// Returns `None` if unknown.
    fn is_null_preserving(&self, _child_idx: usize) -> Option<bool> {
        None
    }

    /// Returns this operator as a [`BatchOperator`] if it supports batch execution.
    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        None
    }

    /// Returns this operator as a [`PipelinedOperator`] if it supports pipelined execution.
    ///
    /// Note that operators that implement [`PipelinedOperator`] *do not need* to implement
    /// [`BatchOperator`], although they may choose to do so.
    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        None
    }

    /// Returns this operator as a [`WebGpuOperator`] if it supports GPU execution.
    fn as_webgpu(&self) -> Option<&dyn WebGpuOperator> {
        None
    }

    /// Returns this operator as a [`VulkanOperator`] if it supports Vulkan execution.
    #[cfg(feature = "vulkano")]
    fn as_vulkan(&self) -> Option<&dyn crate::vulkan::VulkanOperator> {
        None
    }
}

impl Hash for dyn Operator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        DynHash::dyn_hash(self, state)
    }
}

impl PartialEq for dyn Operator {
    fn eq(&self, other: &Self) -> bool {
        DynEq::dyn_eq(self, other.as_any())
    }
}

impl Eq for dyn Operator {}

/// The default execution mode for an operator is batch mode.
pub trait BatchOperator: Operator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef>;
}

pub trait BatchBindCtx {
    fn take_child(&mut self, idx: usize) -> VortexResult<BatchExecutionRef>;
}

/// The primary execution trait for operators.
///
/// Alternatively, or additionally, operators may choose to implement [`PipelinedOperator`].
#[async_trait]
pub trait BatchExecution: Send {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical>;
}

pub type BatchExecutionRef = Box<dyn BatchExecution>;
