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
mod hash;
pub mod metrics;
mod optimize;
pub mod slice;

use std::any::{Any, type_name};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::BitAnd;
use std::sync::Arc;

use arcref::ArcRef;
use async_trait::async_trait;
pub use display::*;
pub use hash::*;
use termtree::Tree;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::pipeline::PipelinedOperator;

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

    /// Returns the bounds on the number of rows produced by this operator.
    fn bounds(&self) -> LengthBounds;

    /// Returns the exact number of rows produced by this operator, if known.
    fn len(&self) -> Option<usize> {
        self.bounds().maybe_len()
    }

    /// Returns if this operator is known to be empty (i.e. max bound is 0).
    fn is_empty(&self) -> bool {
        self.bounds().max == 0
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

    /// Return `true` if the given child is considered to be a selection target.
    ///
    /// The definition of this is such that pushing a selection operator down to all selection
    /// targets will result in the same output as a selection on this operator.
    ///
    /// For example, `select(Op, mask) == Op(select(child, mask), ...)` for all children that are
    /// selection targets.
    ///
    /// If any child index returns `None`, then selection pushdown is not possible.
    /// If all children return `Some(false)`, then selection pushdown is not possible.
    fn is_selection_target(&self, _child_idx: usize) -> Option<bool> {
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
}

/// Represents the known row count bounds of an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LengthBounds {
    pub min: usize,
    pub max: usize,
}

impl LengthBounds {
    pub fn maybe_len(&self) -> Option<usize> {
        (self.min == self.max).then_some(self.min)
    }

    pub fn contains(&self, len: usize) -> bool {
        self.min <= len && len <= self.max
    }

    pub fn intersect_all<I: IntoIterator<Item = LengthBounds>>(iters: I) -> Self {
        let mut min = 0;
        let mut max = 0;
        for bounds in iters {
            min = min.max(bounds.min);
            max = max.min(bounds.max);
        }
        Self { min, max }
    }
}

impl BitAnd for LengthBounds {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min.max(rhs.min),
            max: self.max.min(rhs.max),
        }
    }
}

impl From<usize> for LengthBounds {
    fn from(value: usize) -> Self {
        Self {
            min: value,
            max: value,
        }
    }
}

/// The default execution mode for an operator is batch mode.
pub trait BatchOperator: Operator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef>;
}

pub trait BatchBindCtx {
    /// Returns the execution for the child at the given index, consuming it from the context.
    /// Each child may be consumed only once.
    fn child(&mut self, idx: usize) -> VortexResult<BatchExecutionRef>;
}

/// The primary execution trait for operators.
///
/// Alternatively, or additionally, operators may choose to implement [`PipelinedOperator`].
#[async_trait]
pub trait BatchExecution: Send {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical>;
}

pub type BatchExecutionRef = Box<dyn BatchExecution>;
