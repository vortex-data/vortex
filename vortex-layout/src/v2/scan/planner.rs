// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::FieldName;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::v2::layout::ChildRelationship;
use crate::v2::scan::plan::Plan;
use crate::v2::scan::plan::SegmentRequest;

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
    pub fn new(idx: usize) -> Self {
        Self(idx)
    }

    pub fn as_usize(self) -> usize {
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
    /// The shared backing plan that accumulates nodes from all builders in the tree.
    plan: &'a mut Plan,

    /// Accumulated row offset from the root of the layout tree.
    base_offset: u64,
    /// The lifetime scope for nodes in the current subtree. `None` means use the split's
    /// row range translated by `base_offset`.
    ///
    /// Set to `Some` when stepping into an [`Auxiliary`](ChildRelationship::Auxiliary) child,
    /// where the lifetime is the parent's row range rather than the child's own coordinates.
    lifetime_scope: Option<Range<u64>>,

    /// Layout tree path, used as part of node identity for deduplication.
    position: LayoutPosition,
}

impl<'a> PlanBuilder<'a> {
    /// Creates a new root-level plan builder.
    pub(crate) fn new(plan: &'a mut Plan) -> Self {
        Self {
            plan,
            base_offset: 0,
            lifetime_scope: None,
            position: LayoutPosition::root(),
        }
    }

    /// Creates a child builder by stepping into a child layout with the given relationship.
    ///
    /// - [`RowOffset(n)`](ChildRelationship::RowOffset): shifts coordinates by `n`.
    /// - [`FieldName(_)`](ChildRelationship::FieldName): same row space, no change.
    /// - [`Auxiliary(range)`](ChildRelationship::Auxiliary): enters a separate row space where
    ///   the node lifetime is fixed to the parent's row range.
    pub fn step_into(&mut self, relationship: &ChildRelationship) -> PlanBuilder<'_> {
        let child_position = self.position.step(relationship);
        match relationship {
            ChildRelationship::RowOffset(offset) => PlanBuilder {
                plan: self.plan,
                base_offset: self.base_offset + offset,
                lifetime_scope: self.lifetime_scope.clone(),
                position: child_position,
            },
            ChildRelationship::FieldName(_) => PlanBuilder {
                plan: self.plan,
                base_offset: self.base_offset,
                lifetime_scope: self.lifetime_scope.clone(),
                position: child_position,
            },
            ChildRelationship::Auxiliary(parent_range) => PlanBuilder {
                plan: self.plan,
                base_offset: 0,
                lifetime_scope: Some(
                    parent_range.start + self.base_offset..parent_range.end + self.base_offset,
                ),
                position: child_position,
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
            Some(scope) => Lifetime(scope.clone()),
            None => {
                Lifetime(local_range.start + self.base_offset..local_range.end + self.base_offset)
            }
        }
    }

    /// Construct a node that runs compute over its inputs.
    ///
    /// Takes `options` by value so that the `FnOnce` compute closure can be moved into the plan.
    pub fn create_node<F>(&mut self, options: NodeOpts<'_, F>) -> VortexResult<NodeId>
    where
        F: FnOnce(ComputeArgs) -> VortexResult<ArrayRef> + Send + 'static,
    {
        let key = (
            self.position.clone(),
            options.label,
            options.lifetime.0.start,
            options.lifetime.0.end,
        );
        if let Some(&existing) = self.plan.dedup.get(&key) {
            return Ok(existing);
        }
        let compute: ComputeFn = Box::new(options.compute);
        let id = self.plan.add_node(
            options.label,
            options.inputs,
            options.segments,
            compute,
            options.lifetime,
        );
        self.plan.dedup.insert(key, id);
        Ok(id)
    }

    /// Construct a node with a resolved value and no input dependencies.
    ///
    /// Internally creates a zero-dep compute node so it flows through the normal
    /// Ready → Compute → Complete → propagate path.
    pub fn create_node_resolved(&mut self, array: ArrayRef, row_range: Range<u64>) -> NodeId {
        let lifetime = self.row_range_lifetime(row_range);
        let key = (
            self.position.clone(),
            "Resolved",
            lifetime.0.start,
            lifetime.0.end,
        );
        if let Some(&existing) = self.plan.dedup.get(&key) {
            return existing;
        }
        let id = self.plan.add_node(
            "Resolved",
            &[],
            Vec::new(),
            Box::new(move |_| Ok(array)),
            lifetime,
        );
        self.plan.dedup.insert(key, id);
        id
    }
}

pub struct NodeOpts<'a, F> {
    /// Human-readable label for this node (used in DAG visualization).
    pub label: &'static str,
    /// Wait for these nodes to complete before running.
    pub inputs: &'a [NodeId],
    /// Fetch these segments before running.
    pub segments: Vec<SegmentRequest>,
    pub lifetime: Lifetime,
    pub compute: F,
}

/// A function to produce an array from resolved segment buffers and upstream node outputs.
pub type ComputeFn = Box<dyn FnOnce(ComputeArgs) -> VortexResult<ArrayRef> + Send + 'static>;

/// Arguments passed into the compute function.
pub struct ComputeArgs {
    pub segments: Vec<ByteBuffer>,
    pub inputs: Vec<ArrayRef>,
    pub ctx: ExecutionCtx,
}

/// The row-range lifetime of a plan node, in global coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lifetime(pub Range<u64>);

/// A step in the layout tree path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PositionStep {
    RowOffset(u64),
    FieldName(FieldName),
    Auxiliary(u64, u64),
}

/// The path from the root of the layout tree to the current position.
///
/// Used as part of the dedup key to uniquely identify nodes across splits.
/// Shared via `Arc` so cloning is cheap.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayoutPosition(Arc<Vec<PositionStep>>);

impl LayoutPosition {
    fn root() -> Self {
        Self(Arc::new(Vec::new()))
    }

    fn step(&self, relationship: &ChildRelationship) -> Self {
        let mut path = (*self.0).clone();
        path.push(match relationship {
            ChildRelationship::RowOffset(n) => PositionStep::RowOffset(*n),
            ChildRelationship::FieldName(name) => PositionStep::FieldName(name.clone()),
            ChildRelationship::Auxiliary(range) => PositionStep::Auxiliary(range.start, range.end),
        });
        Self(Arc::new(path))
    }
}
