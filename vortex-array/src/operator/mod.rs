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

pub mod bool_runs;
pub mod canonical;
pub mod cast;
mod display;
pub mod getitem;
mod hash;
mod mask;
pub mod metrics;
mod optimize;
pub mod slice;

use std::any::{type_name, Any};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arcref::ArcRef;
use async_trait::async_trait;
pub use display::*;
pub use hash::*;
pub use mask::*;
use termtree::Tree;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::pipeline::PipelinedOperator;
use crate::Canonical;

pub type OperatorId = ArcRef<str>;
pub type OperatorRef = Arc<dyn Operator>;

/// An operator represents a node in a logical query plan.
pub trait Operator: 'static + Send + Sync + Debug + DynOperatorHash + DynOperatorEq {
    /// The unique identifier for this operator instance.
    fn id(&self) -> OperatorId;

    /// For downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the [`DType`] of the array produced by this operator.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows this operator holds.
    fn len(&self) -> usize;

    /// Returns if this operator is known to be empty (i.e. max bound is 0).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The children of this operator.
    fn children(&self) -> &[OperatorRef];

    /// The number of children of this operator.
    fn nchildren(&self) -> usize {
        self.children().len()
    }

    /// Override the default formatting of this operator.
    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", type_name::<Self>())
    }

    fn fmt_all(&self) -> String {
        let node_name = TreeNodeDisplay(self).to_string();
        let child_trees: Vec<_> = self
            .children()
            .iter()
            .map(|child| child.fmt_all())
            .collect();
        Tree::new(node_name)
            .with_leaves(child_trees)
            .with_multiline(true)
            .to_string()
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
}

/// The default execution mode for an operator is batch mode.
pub trait BatchOperator: Operator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef>;

    // TODO(ngates): add reduce(&self) function here also
}

/// We provide bind functions on the context to perform common sub-expression elimination and
/// re-use the execution results from identical operators.
pub trait BatchBindCtx {
    /// Bind an operator into a projection execution.
    ///
    /// The caller should decide whether they want to pass a mask when binding the given operator.
    /// If passed, the mask operator will be used to selection true rows.
    ///
    /// The bind context ensures that multiple calls to this function with the same operators
    /// will bind to the same internal shared execution instance, avoiding duplicate work.
    fn bind_project(
        &mut self,
        operator: &OperatorRef,
        mask: Option<&OperatorRef>,
    ) -> VortexResult<BatchExecutionRef>;
}

impl dyn BatchBindCtx + '_ {
    /// Utility function for binding a [`MaskExecution`] from an [`OperatorRef`].
    ///
    /// This function provides access to the shared mask execution node, while also
    /// short-circuiting compute if the operator can be exported to a mask more efficiently, for
    /// example a constant array.
    pub fn bind_mask(&mut self, operator: &OperatorRef) -> VortexResult<MaskExecution> {
        MaskExecution::bind(operator, self)
    }
}

/// The primary execution trait for operators.
///
// TODO(ngates): this is basically just BoxFuture<'static, VortexResult<Canonical>>...
#[async_trait]
pub trait BatchExecution: Send {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical>;
}

pub type BatchExecutionRef = Box<dyn BatchExecution>;
