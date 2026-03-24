// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::scan::planner::ComputeFn;
use crate::v2::scan::planner::Lifetime;
use crate::v2::scan::planner::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeState {
    /// Node is waiting for one or more dependencies to be resolved.
    Waiting,
    /// All dependencies are resolved; the node is ready to be dispatched.
    Ready,
    /// The node's work has been dispatched to the driver but not yet completed.
    Dispatched,
    /// The node's output is available.
    Complete,
}

/// A request to read a segment from a source.
#[derive(Clone)]
pub struct SegmentRequest {
    pub(crate) source: Arc<dyn SegmentSource>,
    pub(crate) segment_id: SegmentId,
}

impl std::fmt::Debug for SegmentRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentRequest")
            .field("segment_id", &self.segment_id)
            .finish()
    }
}

pub(crate) struct PlanNode {
    input_nodes: Vec<NodeId>,
    segments: Vec<SegmentRequest>,
    compute: Option<ComputeFn>,
    #[allow(dead_code)]
    lifetime: Lifetime,
    state: NodeState,
    pending_deps: usize,
    /// Resolved segment buffers, one per segment request.
    resolved_segments: Vec<Option<ByteBuffer>>,
    /// Resolved upstream node outputs, one per input node.
    resolved_inputs: Vec<Option<ArrayRef>>,
    output: Option<ArrayRef>,
}

/// The layout scan plan DAG.
pub struct Plan {
    nodes: Vec<PlanNode>,
    root_node: Option<NodeId>,
    /// Reverse dependency index: `dependents[node_id]` lists `(downstream_node, input_slot)`.
    dependents: Vec<Vec<(NodeId, usize)>>,
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
            dependents: Vec::new(),
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
        let total_deps = segments.len() + inputs.len();
        let state = if total_deps == 0 {
            NodeState::Ready
        } else {
            NodeState::Waiting
        };

        let id = NodeId::new(self.nodes.len());

        // Build reverse dependency edges for input_nodes.
        for (i, &input_id) in inputs.iter().enumerate() {
            if input_id.as_usize() < self.dependents.len() {
                self.dependents[input_id.as_usize()].push((id, i));
            }
        }

        let num_segments = segments.len();
        let num_inputs = inputs.len();
        let node = PlanNode {
            input_nodes: inputs.to_vec(),
            segments,
            compute: Some(compute),
            lifetime,
            state,
            pending_deps: total_deps,
            resolved_segments: (0..num_segments).map(|_| None).collect(),
            resolved_inputs: (0..num_inputs).map(|_| None).collect(),
            output: None,
        };

        self.nodes.push(node);
        self.dependents.push(Vec::new());
        id
    }

    /// Returns the number of nodes in the plan.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the state of a node.
    pub(crate) fn node_state(&self, node_id: NodeId) -> NodeState {
        self.nodes[node_id.as_usize()].state
    }

    /// Returns the output of a completed node, if available.
    pub(crate) fn node_output(&self, node_id: NodeId) -> Option<&ArrayRef> {
        self.nodes[node_id.as_usize()].output.as_ref()
    }

    /// Returns the number of segments for a node.
    pub(crate) fn node_segment_count(&self, node_id: NodeId) -> usize {
        self.nodes[node_id.as_usize()].segments.len()
    }

    /// Returns whether a node has a compute function.
    pub(crate) fn node_has_compute(&self, node_id: NodeId) -> bool {
        self.nodes[node_id.as_usize()].compute.is_some()
    }

    /// Returns the pending dependency count for a node.
    pub(crate) fn node_pending_deps(&self, node_id: NodeId) -> usize {
        self.nodes[node_id.as_usize()].pending_deps
    }

    /// Returns the downstream dependents of a node.
    pub(crate) fn dependents_of(&self, node_id: NodeId) -> &[(NodeId, usize)] {
        &self.dependents[node_id.as_usize()]
    }

    /// Iterates over nodes in `Ready` state within the given range.
    pub(crate) fn ready_nodes_in_range(
        &self,
        range: std::ops::Range<usize>,
    ) -> impl Iterator<Item = NodeId> + '_ {
        range
            .filter(move |&i| self.nodes[i].state == NodeState::Ready)
            .map(NodeId::new)
    }

    /// Takes the segment requests from a node, transitioning it to `Dispatched`.
    ///
    /// This is used for nodes that only have segment dependencies and no compute function,
    /// or as the first phase of dispatching a node with both segments and compute.
    pub(crate) fn take_segment_requests(
        &mut self,
        node_id: NodeId,
    ) -> VortexResult<Vec<SegmentRequest>> {
        let node = &mut self.nodes[node_id.as_usize()];
        if node.state != NodeState::Ready && node.state != NodeState::Waiting {
            vortex_bail!(
                "Cannot take segment requests from node {:?} in state {:?}",
                node_id,
                node.state,
            );
        }
        let segments = std::mem::take(&mut node.segments);
        node.state = NodeState::Dispatched;
        Ok(segments)
    }

    /// Resolves a segment slot for a node by providing the read buffer.
    ///
    /// Decrements `pending_deps` and transitions to `Ready` if all deps are resolved.
    pub(crate) fn resolve_segment(&mut self, node_id: NodeId, slot: usize, buffer: ByteBuffer) {
        let node = &mut self.nodes[node_id.as_usize()];
        debug_assert!(node.resolved_segments[slot].is_none());
        node.resolved_segments[slot] = Some(buffer);
        node.pending_deps -= 1;
        if node.pending_deps == 0 && node.state == NodeState::Dispatched {
            // All segment reads are done; the node can now be computed.
            node.state = NodeState::Ready;
        }
    }

    /// Resolves an input-node slot for a node.
    ///
    /// Decrements `pending_deps` and transitions to `Ready` if all deps are resolved.
    pub(crate) fn resolve_input(&mut self, node_id: NodeId, slot: usize, input: ArrayRef) {
        let node = &mut self.nodes[node_id.as_usize()];
        debug_assert!(node.resolved_inputs[slot].is_none());
        node.resolved_inputs[slot] = Some(input);
        node.pending_deps -= 1;
        if node.pending_deps == 0
            && (node.state == NodeState::Waiting || node.state == NodeState::Dispatched)
        {
            node.state = NodeState::Ready;
        }
    }

    /// Takes the compute function and all resolved inputs from a `Ready` node,
    /// transitioning it to `Dispatched`.
    pub(crate) fn take_compute(
        &mut self,
        node_id: NodeId,
    ) -> VortexResult<(ComputeFn, Vec<ByteBuffer>, Vec<ArrayRef>)> {
        let node = &mut self.nodes[node_id.as_usize()];
        if node.state != NodeState::Ready {
            vortex_bail!(
                "Cannot take compute from node {:?} in state {:?}",
                node_id,
                node.state,
            );
        }
        let compute = node
            .compute
            .take()
            .expect("Ready node must have a compute function");
        let segments: Vec<ByteBuffer> = node
            .resolved_segments
            .iter_mut()
            .map(|slot| slot.take().expect("All segments must be resolved"))
            .collect();
        let inputs: Vec<ArrayRef> = node
            .resolved_inputs
            .iter_mut()
            .map(|slot| slot.take().expect("All inputs must be resolved"))
            .collect();
        node.state = NodeState::Dispatched;
        Ok((compute, segments, inputs))
    }

    /// Marks a node as `Complete` with the given output.
    pub(crate) fn complete_node(&mut self, node_id: NodeId, output: ArrayRef) {
        let node = &mut self.nodes[node_id.as_usize()];
        debug_assert_eq!(node.state, NodeState::Dispatched);
        node.output = Some(output);
        node.state = NodeState::Complete;
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;

    use super::*;
    use crate::test::SESSION;
    use crate::v2::scan::planner::ComputeArgs;

    fn add_resolved_node(plan: &mut Plan, array: ArrayRef) -> NodeId {
        plan.add_node(
            &[],
            Vec::new(),
            Box::new(move |_args| Ok(array)),
            Lifetime::Scan,
        )
    }

    #[test]
    fn test_zero_dep_node_is_ready() {
        let mut plan = Plan::new();
        let node_id = plan.add_node(
            &[],
            Vec::new(),
            Box::new(|_args| Ok(PrimitiveArray::from_iter([1i32]).into_array())),
            Lifetime::Scan,
        );
        assert_eq!(plan.node_state(node_id), NodeState::Ready);
    }

    #[test]
    fn test_two_node_chain() -> VortexResult<()> {
        let mut plan = Plan::new();

        // Node A: no deps, immediately Ready.
        let a = add_resolved_node(&mut plan, PrimitiveArray::from_iter([10i32]).into_array());

        // Node B: depends on A.
        let b = plan.add_node(
            &[a],
            Vec::new(),
            Box::new(|_args| Ok(PrimitiveArray::from_iter([20i32]).into_array())),
            Lifetime::Scan,
        );

        // B starts Waiting since it has 1 input dep.
        assert_eq!(plan.node_state(b), NodeState::Waiting);

        // Execute A and propagate its output to B.
        let (compute_a, segs_a, inputs_a) = plan.take_compute(a)?;
        let a_output = compute_a(ComputeArgs {
            segments: segs_a,
            inputs: inputs_a,
            ctx: ExecutionCtx::new(SESSION.clone()),
        })?;
        plan.complete_node(a, a_output.clone());
        plan.resolve_input(b, 0, a_output);

        // B should now be Ready.
        assert_eq!(plan.node_state(b), NodeState::Ready);

        // Take compute.
        let (compute, segments, inputs) = plan.take_compute(b)?;
        assert_eq!(plan.node_state(b), NodeState::Dispatched);

        // Execute and complete.
        let result = compute(ComputeArgs {
            segments,
            inputs,
            ctx: ExecutionCtx::new(SESSION.clone()),
        })?;
        plan.complete_node(b, result);
        assert_eq!(plan.node_state(b), NodeState::Complete);

        Ok(())
    }

    #[test]
    fn test_dependents_reverse_index() {
        let mut plan = Plan::new();

        let a = add_resolved_node(&mut plan, PrimitiveArray::from_iter([1i32]).into_array());

        let b = plan.add_node(
            &[a],
            Vec::new(),
            Box::new(|_args| Ok(PrimitiveArray::from_iter([2i32]).into_array())),
            Lifetime::Scan,
        );

        let deps = plan.dependents_of(a);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].0.as_usize(), b.as_usize());
        assert_eq!(deps[0].1, 0); // input slot 0
    }

    #[test]
    fn test_ready_nodes_in_range() {
        let mut plan = Plan::new();

        // Both nodes have no deps, so both are Ready.
        let _a = add_resolved_node(&mut plan, PrimitiveArray::from_iter([1i32]).into_array());
        let _b = plan.add_node(
            &[],
            Vec::new(),
            Box::new(|_args| Ok(PrimitiveArray::from_iter([2i32]).into_array())),
            Lifetime::Scan,
        );

        let ready: Vec<_> = plan.ready_nodes_in_range(0..plan.len()).collect();
        assert_eq!(ready.len(), 2);
    }
}
