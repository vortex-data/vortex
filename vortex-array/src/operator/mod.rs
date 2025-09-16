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

use crate::pipeline::operators::BindContext;
use crate::pipeline::Kernel;
use crate::Canonical;
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;

/// We're hoping to move towards a world in which arrays only have to implement a canonicalize
/// method. All other operators are performed by wrapping up arrays in new operators and applying
/// an optimization pass.
///
/// This trait will help with the migration to that world.
pub trait ArrayOperator: 'static + Send + Sync {
    /// The unique identifier for this operator instance.
    fn id(&self) -> Arc<str>;

    /// For downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the [`DType`] of the array produced by this operator.
    fn dtype(&self) -> &DType;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns true if the array is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The children of this operator in the DAG.
    fn children(&self) -> &[Arc<dyn ArrayOperator>];

    /// Create a new instance of this operator with the given children.
    ///
    /// ## Panics
    ///
    /// Panics if the number or dtypes of children is incorrect.
    fn with_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ArrayOperator>>,
    ) -> VortexResult<Arc<dyn ArrayOperator>>;

    /// Returns the metadata for this operator.
    fn metadata(&self) -> &dyn OperatorMetadata {
        &()
    }

    /// Called by the `Canonicalizer` to allow each operator to optimize itself when called by
    /// a given parent operator.
    ///
    /// The `child_idx` parameter indicates which child of the parent this operator occupies.
    ///
    /// For example, if the parent is a binary operator, and this operator is the left child,
    /// then `child_idx` will be 0. If this operator is the right child, then `child_idx` will be 1.
    fn optimize(
        &self,
        parent: Arc<dyn ArrayOperator>,
        _child_idx: usize,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
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

pub trait OperatorMetadata {
    fn as_any(&self) -> &dyn Any;
}

impl OperatorMetadata for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The default execution mode for an operator is batch mode.
pub trait BatchOperator: ArrayOperator {
    fn bind(&self, ctx: &dyn BatchBindCtx) -> VortexResult<Box<dyn BatchExecution>>;
}

pub trait BatchBindCtx {
    fn child(&self, idx: usize) -> VortexResult<Box<dyn BatchExecution>>;
}

/// The primary execution trait for operators.
///
/// Alternatively, or additionally, operators may choose to implement [`PipelinedOperator`] and
/// [`GpuOperator`] to support pipelined and GPU execution modes.
#[async_trait]
pub trait BatchExecution: Send {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical>;
}

pub trait PipelinedOperator: ArrayOperator {
    /// Whether this operator works by mutating its first child in-place.
    ///
    /// If `true`, the operator is invoked with the first child's input data passed via the
    /// mutable output view. The node is expected to mutate this data in-place.
    fn in_place(&self) -> bool {
        false
    }

    /// Bind the operator into a [`Kernel`] for pipelined execution.
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
}

pub trait GpuOperator: ArrayOperator {
    // TODO(ngates): no idea what this API looks like.
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
}
