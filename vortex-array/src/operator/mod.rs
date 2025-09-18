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
//! ultimately decided against this approach and instead introduce `Filter` and `Take` operators
//! that can be pushed down in the same way as any other operator.
//!
//! The initial design only supported filter workloads, meaning common fused kernels such as
//! Dict+RLE could not be effectively implemented without an entire parallel execution trait for
//! random access `take` operations. By introducing `Take` as a first-class operator, we can
//! support these fused kernels without complicating the execution model.
//!
//! We also introduce the idea of an executor that can evaluate an operator tree efficiently. It
//! supports common subtree elimination, as well as extracting sub-graphs for pipelined and GPU
//! execution. The executor is also responsible for managing memory and scheduling work across
//! different execution resources.
//!

mod executor;
mod ops;

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::pipeline::Kernel;
use crate::Canonical;
use arcref::ArcRef;
use async_trait::async_trait;
use vortex_dtype::DType;
use vortex_error::VortexResult;

pub use executor::*;
pub use ops::*;
use vortex_utils::dyn_eq::{DynEq, DynHash};

pub type OperatorId = ArcRef<str>;
pub type OperatorRef = Arc<dyn Operator>;

/// An operator represents a node in a logical query plan.
pub trait Operator: 'static + Debug + DynEq + DynHash + Send + Sync {
    /// The unique identifier for this operator instance.
    fn id(&self) -> OperatorId;

    /// For downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the [`DType`] of the array produced by this operator.
    fn dtype(&self) -> &DType;

    /// Returns the length of the array produced by this operator.
    fn len(&self) -> usize;

    /// Returns true if the array is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // TODO(ngates): add StatsSet

    /// The children of this operator.
    fn children(&self) -> &[OperatorRef];

    /// Create a new instance of this operator with the given children.
    ///
    /// ## Panics
    ///
    /// Panics if the number or dtypes of children are incorrect.
    ///
    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef>;

    /// Whether this operator works by mutating its first child in-place.
    ///
    /// If `true`, the operator is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Attempt to push down a parent operator through this node.
    ///
    /// The `child_idx` parameter indicates which child of the parent this operator occupies.
    ///
    /// For example, if the parent is a binary operator, and this operator is the left child,
    /// then `child_idx` will be 0. If this operator is the right child, then `child_idx` will be 1.
    ///
    /// The returned operator will replace the parent in the tree, therefore a no-op is to return
    /// the parent unchanged.
    fn reduce_parent(&self, parent: OperatorRef, _child_idx: usize) -> VortexResult<OperatorRef> {
        Ok(parent)
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

    /// Returns this operator as a [`GpuOperator`] if it supports GPU execution.
    fn as_gpu(&self) -> Option<&dyn GpuOperator> {
        None
    }
}

impl Hash for dyn Operator {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
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
/// Alternatively, or additionally, operators may choose to implement [`PipelinedOperator`] and
/// [`GpuOperator`] to support pipelined and GPU execution modes.
#[async_trait]
pub trait BatchExecution: Send {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical>;
}

pub type BatchExecutionRef = Box<dyn BatchExecution>;

pub trait PipelinedOperator: Operator {
    /// Whether this operator works by mutating its first child in-place.
    ///
    /// If `true`, the operator is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Bind the operator into a [`Kernel`] for pipelined execution.
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;

    /// Returns the child indices of this operator that are passed to the kernel as input vectors.
    fn vector_children(&self) -> Vec<usize>;

    /// Returns the child indices of this operator that are passed to the kernel as batch inputs.
    fn batch_children(&self) -> Vec<usize>;
}

pub trait GpuOperator: Operator {
    // TODO(ngates): no idea what this API looks like.
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
}

/// The ID of the vector to use.
pub type VectorId = usize;

/// The ID of the batch input to use.
pub type BatchId = usize;

/// The context used when binding an operator for execution.
pub trait BindContext {
    fn children(&self) -> &[VectorId];

    fn batch_inputs(&self) -> &[BatchId];
}
