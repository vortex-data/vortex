// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::segments::SegmentId;
use crate::v2::plan::Lifetime;

pub type SplitPlannerRef = Arc<dyn SplitPlanner>;

pub trait SplitPlanner: Send + Sync {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId>;
}

pub struct NodeId(usize);

pub struct PlanBuilder {}

impl PlanBuilder {
    /// Construct a node that runs compute over its inputs.
    pub fn create_node<F>(&mut self, options: &NodeOpts<'_, F>) -> VortexResult<NodeId>
    where
        F: FnOnce(&[NodeInput]) -> VortexResult<ArrayRef> + Send + 'static,
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
pub type ComputeFn = Box<dyn FnOnce(&[NodeInput]) -> VortexResult<ArrayRef> + Send + 'static>;

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
