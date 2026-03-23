// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::segments::SegmentId;
use crate::v2::layout::ChildRelationship;

pub type SplitPlannerRef = Arc<dyn SplitPlanner>;

pub trait SplitPlanner: Send + Sync {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId>;
}

#[derive(Clone, Copy, Debug)]
pub struct NodeId(usize);

/// Builds a DAG of compute nodes for a scan plan.
///
/// The builder tracks positional context as layouts recurse into their children via
/// [`step_into`](Self::step_into). This context is used to translate local row coordinates
/// to global coordinates for lifetime assignment.
#[derive(Default)]
pub struct PlanBuilder {
    /// Accumulated row offset from the root of the layout tree.
    base_offset: u64,
    /// The lifetime scope for nodes in the current subtree. `None` means use the split's
    /// row range translated by `base_offset`.
    ///
    /// Set to `Some` when stepping into an [`Auxiliary`](ChildRelationship::Auxiliary) child,
    /// where the lifetime is the parent's row range rather than the child's own coordinates.
    lifetime_scope: Option<Range<u64>>,
}

impl PlanBuilder {
    /// Creates a new root-level plan builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a child builder by stepping into a child layout with the given relationship.
    ///
    /// - [`RowOffset(n)`](ChildRelationship::RowOffset): shifts coordinates by `n`.
    /// - [`FieldName(_)`](ChildRelationship::FieldName): same row space, no change.
    /// - [`Auxiliary(range)`](ChildRelationship::Auxiliary): enters a separate row space where
    ///   the node lifetime is fixed to the parent's row range.
    pub fn step_into(&self, relationship: &ChildRelationship) -> PlanBuilder {
        match relationship {
            ChildRelationship::RowOffset(offset) => PlanBuilder {
                base_offset: self.base_offset + offset,
                lifetime_scope: self.lifetime_scope.clone(),
            },
            ChildRelationship::FieldName(_) => PlanBuilder {
                base_offset: self.base_offset,
                lifetime_scope: self.lifetime_scope.clone(),
            },
            ChildRelationship::Auxiliary(parent_range) => PlanBuilder {
                base_offset: 0,
                lifetime_scope: Some(
                    parent_range.start + self.base_offset..parent_range.end + self.base_offset,
                ),
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
    pub fn create_node<F>(&mut self, _options: &NodeOpts<'_, F>) -> VortexResult<NodeId>
    where
        F: FnOnce(Vec<NodeInput>) -> VortexResult<ArrayRef> + Send + 'static,
    {
        todo!()
    }
}

pub struct NodeOpts<'a, F> {
    /// Wait for these nodes to complete before running.
    pub inputs: &'a [NodeId],
    /// Fetch these segments before running.
    pub segments: &'a [SegmentId], // Can we make refine this read somehow?
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

/// A handle to the filter mask of the current split.
///
/// This handle provides a view over the "latest" filter mask, useful for pruning during planning,
/// as well as a NodeId that can be referenced to create a hard dependency in the DAG.
pub struct SplitSelection {}

impl SplitSelection {
    pub fn node_id(&self) -> NodeId {
        todo!()
    }

    /// Returns the latest selection mask for this split.
    pub fn latest(&self) -> Mask {
        todo!()
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
