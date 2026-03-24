// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::segments::SegmentId;
use crate::v2::layout::ChildRelationship;
use crate::v2::scan::plan::Plan;

pub type SplitPlannerRef = Arc<dyn SplitPlanner>;

pub trait SplitPlanner: Send + Sync {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId>;
}

#[derive(Clone, Copy, Debug)]
pub struct NodeId(usize);

impl NodeId {
    pub(crate) fn new(idx: usize) -> Self {
        Self(idx)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

/// Builds a DAG of compute nodes for a scan plan.
///
/// The builder tracks positional context as layouts recurse into their children via
/// [`step_into`](Self::step_into). This context is used to translate local row coordinates
/// to global coordinates for lifetime assignment.
///
/// Internally backs onto a shared [`Plan`] so that child builders created via `step_into`
/// all contribute to the same DAG.
pub struct PlanBuilder<'a> {
    /// Accumulated row offset from the root of the layout tree.
    base_offset: u64,
    /// The lifetime scope for nodes in the current subtree. `None` means use the split's
    /// row range translated by `base_offset`.
    ///
    /// Set to `Some` when stepping into an [`Auxiliary`](ChildRelationship::Auxiliary) child,
    /// where the lifetime is the parent's row range rather than the child's own coordinates.
    lifetime_scope: Option<Range<u64>>,
    /// The shared backing plan that accumulates nodes from all builders in the tree.
    plan: &'a mut Plan,
}

impl<'a> PlanBuilder<'a> {
    /// Creates a new root-level plan builder.
    pub(crate) fn new(plan: &'a mut Plan) -> Self {
        Self {
            base_offset: 0,
            lifetime_scope: None,
            plan,
        }
    }

    /// Creates a child builder by stepping into a child layout with the given relationship.
    ///
    /// - [`RowOffset(n)`](ChildRelationship::RowOffset): shifts coordinates by `n`.
    /// - [`FieldName(_)`](ChildRelationship::FieldName): same row space, no change.
    /// - [`Auxiliary(range)`](ChildRelationship::Auxiliary): enters a separate row space where
    ///   the node lifetime is fixed to the parent's row range.
    pub fn step_into(&mut self, relationship: &ChildRelationship) -> PlanBuilder<'a> {
        match relationship {
            ChildRelationship::RowOffset(offset) => PlanBuilder {
                base_offset: self.base_offset + offset,
                lifetime_scope: self.lifetime_scope.clone(),
                plan: self.plan,
            },
            ChildRelationship::FieldName(_) => PlanBuilder {
                base_offset: self.base_offset,
                lifetime_scope: self.lifetime_scope.clone(),
                plan: self.plan,
            },
            ChildRelationship::Auxiliary(parent_range) => PlanBuilder {
                base_offset: 0,
                lifetime_scope: Some(
                    parent_range.start + self.base_offset..parent_range.end + self.base_offset,
                ),
                plan: self.plan,
            },
        }
    }

    /// Returns the appropriate [`Lifetime`] for a node processing the given local row range.
    ///
    /// In a normal (row-offset or field) subtree, translates the local range to global
    /// coordinates. In an auxiliary subtree, returns the fixed lifetime scope from the
    /// parent's row range.
    pub fn row_range_lifetime(&self, local_range: Range<u64>) -> Lifetime {
        match &self.lifetime_scope {
            Some(scope) => Lifetime::RowRange(scope.clone()),
            None => Lifetime::RowRange(
                local_range.start + self.base_offset..local_range.end + self.base_offset,
            ),
        }
    }

    /// Construct a node that runs compute over its inputs.
    ///
    /// Takes `options` by value so that the `FnOnce` compute closure can be moved into the plan.
    pub fn create_node<F>(&mut self, options: NodeOpts<'_, F>) -> VortexResult<NodeId>
    where
        F: FnOnce(Vec<NodeInput>) -> VortexResult<ArrayRef> + Send + 'static,
    {
        let compute: ComputeFn = Box::new(options.compute);
        let id = self.plan.borrow_mut().add_node(
            options.inputs,
            options.segments,
            compute,
            options.lifetime,
        );
        Ok(id)
    }

    /// Construct a node with a resolved value and no input dependencies.
    pub fn create_node_resolved(&mut self, array: ArrayRef) -> NodeId {
        self.plan
            .borrow_mut()
            .add_node(&[], &[], Box::new(move |_| Ok(array)), Lifetime::Scan)
    }
}

pub struct NodeOpts<'a, F> {
    /// Wait for these nodes to complete before running.
    pub inputs: &'a [NodeId],
    /// Fetch these segments before running.
    pub segments: &'a [SegmentId],
    pub lifetime: Lifetime,
    pub compute: F,
}

/// A function to produce an array from node inputs.
pub type ComputeFn = Box<dyn FnOnce(Vec<NodeInput>) -> VortexResult<ArrayRef> + Send + 'static>;

pub enum NodeInput {
    Buffer(ByteBuffer),
    Array(ArrayRef),
    // Mask(Mask),
}

impl NodeInput {
    pub fn into_buffer(self) -> ByteBuffer {
        match self {
            NodeInput::Buffer(buffer) => buffer,
            NodeInput::Array(_) => vortex_panic!("Input is not a buffer"),
        }
    }

    pub fn into_array(self) -> ArrayRef {
        match self {
            NodeInput::Buffer(_) => vortex_panic!("Input is not a buffer"),
            NodeInput::Array(array) => array,
        }
    }
}

/// Describes the lifetime of a plan node.
pub enum Lifetime {
    /// The duration of the scan. Never evict.
    Scan,
    /// Alive for a specific row range.
    RowRange(Range<u64>),
    /// Alive until the dynamic "generation" ticks over. e.g. for dynamic expressions.
    Dynamic(Arc<AtomicUsize>),
    /// Unknown lifetime
    Unknown,
}

impl Lifetime {
    pub fn covers(&self, _row_range: &Range<u64>) -> bool {
        unimplemented!()
    }
}
