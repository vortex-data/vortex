// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::scan::planner::ComputeFn;
use crate::v2::scan::planner::Lifetime;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeInput;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeState {
    Waiting,
    Ready,
    Complete,
}

#[derive(Debug, Clone)]
pub struct SegmentRequest {
    source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
}

struct PlanNode {
    input_nodes: Vec<NodeId>,
    segments: Vec<SegmentRequest>,
    compute: Option<ComputeFn>,
    #[allow(dead_code)]
    lifetime: Lifetime,
    state: NodeState,
    pending_deps: usize,
    /// Resolved inputs laid out as `[segments..., input_nodes...]`.
    resolved_inputs: Vec<Option<NodeInput>>,
    output: Option<ArrayRef>,
}

/// The layout scan plan DAG.
pub struct Plan {
    nodes: Vec<PlanNode>,
    root_node: Option<NodeId>,
}

// Debug is required by `Rc::try_unwrap().expect()` in `PlanBuilder::take_plan`.
impl std::fmt::Debug for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitPlan")
            .field("num_nodes", &self.nodes.len())
            .field("root_node", &self.root_node)
            .finish()
    }
}

impl Plan {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_node: None,
        }
    }

    /// Adds a compute node to the plan, returning its [`NodeId`].
    ///
    /// If the node has no dependencies it is immediately marked `Ready`.
    pub(crate) fn add_node(
        &mut self,
        inputs: &[NodeId],
        segments: Vec<SegmentRequest>,
        compute: ComputeFn,
        lifetime: Lifetime,
    ) -> NodeId {
        let total_slots = segments.len() + inputs.len();
        let state = if total_slots == 0 {
            NodeState::Ready
        } else {
            NodeState::Waiting
        };

        let node = PlanNode {
            input_nodes: inputs.to_vec(),
            segments,
            compute: Some(compute),
            lifetime,
            state,
            pending_deps: total_slots,
            resolved_inputs: (0..total_slots).map(|_| None).collect(),
            output: None,
        };

        let id = NodeId::new(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}
