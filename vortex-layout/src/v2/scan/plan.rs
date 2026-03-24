// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;

use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::segments::SegmentId;
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

struct PlanNode {
    input_nodes: Vec<NodeId>,
    segments: Vec<SegmentId>,
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
    /// Reverse index: for each node, the `(dependent_node, slot)` pairs that depend on it.
    node_dependents: Vec<Vec<(NodeId, usize)>>,
    /// Reverse index: segment ID → `[(node, slot)]` for all nodes needing that segment.
    segment_dependents: BTreeMap<SegmentId, Vec<(NodeId, usize)>>,
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
            node_dependents: Vec::new(),
            segment_dependents: BTreeMap::new(),
        }
    }

    /// Adds a compute node to the plan, returning its [`NodeId`].
    ///
    /// If the node has no dependencies it is immediately marked `Ready`.
    pub(crate) fn add_node(
        &mut self,
        inputs: &[NodeId],
        segments: &[SegmentId],
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
            segments: segments.to_vec(),
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

    /// Sets the root output node of the plan.
    pub(crate) fn set_root(&mut self, node_id: NodeId) {
        self.root_node = Some(node_id);
    }

    /// Builds reverse dependency indices. Must be called after all nodes are added.
    pub(crate) fn finalize(&mut self) {
        self.node_dependents = vec![Vec::new(); self.nodes.len()];
        self.segment_dependents.clear();

        for (node_idx, node) in self.nodes.iter().enumerate() {
            let dependent = NodeId::new(node_idx);

            for (slot, seg_id) in node.segments.iter().enumerate() {
                self.segment_dependents
                    .entry(*seg_id)
                    .or_default()
                    .push((dependent, slot));
            }

            let seg_count = node.segments.len();
            for (i, input_id) in node.input_nodes.iter().enumerate() {
                self.node_dependents[input_id.as_usize()].push((dependent, seg_count + i));
            }
        }
    }

    /// Returns all distinct segment IDs still needed by waiting nodes.
    pub(crate) fn pending_segment_ids(&self) -> Vec<SegmentId> {
        let mut seen = BTreeMap::new();
        for node in &self.nodes {
            if node.state == NodeState::Waiting {
                for (slot, seg_id) in node.segments.iter().enumerate() {
                    if node.resolved_inputs[slot].is_none() {
                        seen.entry(*seg_id).or_insert(());
                    }
                }
            }
        }
        seen.into_keys().collect()
    }

    /// Delivers a fetched segment buffer to all nodes that need it.
    ///
    /// Returns the IDs of any nodes that became ready as a result.
    pub(crate) fn complete_segment(&mut self, id: SegmentId, buffer: ByteBuffer) -> Vec<NodeId> {
        let mut newly_ready = Vec::new();
        let dependents = self
            .segment_dependents
            .get(&id)
            .cloned()
            .unwrap_or_default();
        for (node_id, slot) in dependents {
            let node = &mut self.nodes[node_id.as_usize()];
            if node.resolved_inputs[slot].is_none() {
                node.resolved_inputs[slot] = Some(NodeInput::Buffer(buffer.clone()));
                node.pending_deps -= 1;
                if node.pending_deps == 0 && node.state == NodeState::Waiting {
                    node.state = NodeState::Ready;
                    newly_ready.push(node_id);
                }
            }
        }
        newly_ready
    }

    /// Returns all nodes currently in the `Ready` state.
    pub(crate) fn ready_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.state == NodeState::Ready)
            .map(|(i, _)| NodeId::new(i))
            .collect()
    }

    /// Executes a ready node's compute function and propagates its output to dependents.
    ///
    /// Returns the IDs of any nodes that became ready as a result.
    pub(crate) fn execute_node(&mut self, node_id: NodeId) -> VortexResult<Vec<NodeId>> {
        let idx = node_id.as_usize();
        if self.nodes[idx].state != NodeState::Ready {
            vortex_bail!("Node {} is not ready for execution", idx);
        }

        let compute = self.nodes[idx]
            .compute
            .take()
            .ok_or_else(|| vortex_err!("Node {} compute fn already consumed", idx))?;

        let inputs: Vec<NodeInput> = self.nodes[idx]
            .resolved_inputs
            .iter_mut()
            .enumerate()
            .map(|(i, slot)| {
                slot.take().unwrap_or_else(|| {
                    vortex_panic!(
                        "Node {} resolved input slot {} is None for a ready node",
                        idx,
                        i
                    )
                })
            })
            .collect();

        let output = compute(inputs)?;
        self.nodes[idx].output = Some(output.clone());
        self.nodes[idx].state = NodeState::Complete;

        let mut newly_ready = Vec::new();
        let dependents = self.node_dependents[idx].clone();
        for (dep_id, slot) in dependents {
            let dep = &mut self.nodes[dep_id.as_usize()];
            if dep.resolved_inputs[slot].is_none() {
                dep.resolved_inputs[slot] = Some(NodeInput::Array(output.clone()));
                dep.pending_deps -= 1;
                if dep.pending_deps == 0 && dep.state == NodeState::Waiting {
                    dep.state = NodeState::Ready;
                    newly_ready.push(dep_id);
                }
            }
        }

        Ok(newly_ready)
    }

    /// Returns true if the root node has completed execution.
    pub(crate) fn is_complete(&self) -> bool {
        self.root_node
            .map(|id| self.nodes[id.as_usize()].state == NodeState::Complete)
            .unwrap_or(false)
    }

    /// Takes the output array from the completed root node.
    pub(crate) fn take_output(&mut self) -> Option<ArrayRef> {
        self.root_node
            .and_then(|id| self.nodes[id.as_usize()].output.take())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn test_split_plan_execution() -> VortexResult<()> {
        let mut plan = Plan::new();

        // Node 0: no deps, produces [1, 2, 3]
        let n0 = plan.add_node(
            &[],
            &[],
            Box::new(|_| Ok(PrimitiveArray::from_iter([1i32, 2, 3]).into_array())),
            Lifetime::Unknown,
        );

        // Node 1: no deps, produces [4, 5, 6]
        let n1 = plan.add_node(
            &[],
            &[],
            Box::new(|_| Ok(PrimitiveArray::from_iter([4i32, 5, 6]).into_array())),
            Lifetime::Unknown,
        );

        // Node 2: depends on n0 and n1, returns first input
        let n2 = plan.add_node(
            &[n0, n1],
            &[],
            Box::new(|inputs| Ok(inputs.into_iter().next().unwrap().into_array())),
            Lifetime::Unknown,
        );

        plan.set_root(n2);
        plan.finalize();

        // n0 and n1 should be ready (no deps)
        let ready = plan.ready_nodes();
        assert_eq!(ready.len(), 2);

        // Execute n0 — n2 still waiting for n1
        let newly_ready = plan.execute_node(n0)?;
        assert!(newly_ready.is_empty());

        // Execute n1 — n2 now ready
        let newly_ready = plan.execute_node(n1)?;
        assert_eq!(newly_ready.len(), 1);
        assert_eq!(newly_ready[0].as_usize(), n2.as_usize());

        // Execute n2
        let newly_ready = plan.execute_node(n2)?;
        assert!(newly_ready.is_empty());

        assert!(plan.is_complete());
        assert!(plan.take_output().is_some());
        Ok(())
    }

    #[test]
    fn test_split_plan_with_segments() -> VortexResult<()> {
        let seg0 = SegmentId::from(0u32);
        let seg1 = SegmentId::from(1u32);

        let mut plan = Plan::new();

        let n0 = plan.add_node(
            &[],
            &[seg0, seg1],
            Box::new(|inputs| {
                assert_eq!(inputs.len(), 2);
                Ok(PrimitiveArray::from_iter([42i32]).into_array())
            }),
            Lifetime::Unknown,
        );

        plan.set_root(n0);
        plan.finalize();

        assert!(plan.ready_nodes().is_empty());
        assert_eq!(plan.pending_segment_ids().len(), 2);

        // Complete first segment — node still waiting
        let ready = plan.complete_segment(seg0, ByteBufferMut::empty().freeze());
        assert!(ready.is_empty());

        // Complete second segment — node now ready
        let ready = plan.complete_segment(seg1, ByteBufferMut::empty().freeze());
        assert_eq!(ready.len(), 1);

        plan.execute_node(n0)?;
        assert!(plan.is_complete());
        assert!(plan.take_output().is_some());
        Ok(())
    }
}
