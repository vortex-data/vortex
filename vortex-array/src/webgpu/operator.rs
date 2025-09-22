// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, DisplayFormat, Operator,
    OperatorId, OperatorRef,
};
use crate::webgpu::input::WebGpuInputOperator;
use crate::webgpu::{BatchId, GpuBindContext, GpuBufferId, GpuExecutionContext, GpuKernel};
use crate::Canonical;
use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use std::any::Any;
use std::fmt::Formatter;
use std::hash::BuildHasher;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::{HashMap, RandomState};

/// An operator that collapses a subgraph of WebGpu-capable operators into a single WebGpu operator
/// for batch execution.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct WebGpuSubgraphOperator {
    root: NodeId,
    dag: Vec<GpuNode>,
    batch_inputs: Vec<OperatorRef>,
}

type NodeId = usize;

#[derive(Debug, Clone, Hash)]
struct GpuNode {
    /// The operator at this node.
    operator: OperatorRef,
    /// The indices of the child nodes in the `dag` vector.
    children: Vec<NodeId>,
    /// The indices of this node's parents in the `dag` vector.
    parents: Vec<NodeId>,
    /// The IDs of the batch inputs that feed into this node.
    batch_inputs: Vec<BatchId>,
}

impl PartialEq for GpuNode {
    fn eq(&self, other: &Self) -> bool {
        self.operator.eq(&other.operator)
            && self.children == other.children
            && self.batch_inputs == other.batch_inputs
    }
}
impl Eq for GpuNode {}

impl WebGpuSubgraphOperator {
    /// From the given operator, constructs a `WebGpuOperator` as large as possible by
    /// traversing children that also support WebGpu execution.
    pub fn new(operator: OperatorRef) -> Option<Self> {
        operator.as_webgpu()?;

        let mut dag = vec![];
        let mut batch = vec![];
        let mut hash_to_id: HashMap<u64, NodeId> = HashMap::new();

        fn visit_node(
            node: OperatorRef,
            dag: &mut Vec<GpuNode>,
            batch: &mut Vec<OperatorRef>,
            hash_to_id: &mut HashMap<u64, NodeId>,
            random_state: &RandomState,
        ) -> NodeId {
            // Compute the hash for this subtree.
            let subtree_hash = random_state.hash_one(&node);

            // Check if we've seen this subtree before (sub-expression elimination)
            if let Some(&existing_index) = hash_to_id.get(&subtree_hash) {
                // Reuse existing node
                return existing_index;
            }

            // Process children first (post-order traversal)
            let mut child_indices: Vec<NodeId> = vec![];
            let mut batch_indices: Vec<BatchId> = vec![];

            let node_children = node.children();
            let webgpu = node.as_webgpu().vortex_expect("must support webgpu");

            // Prepare the GPU children
            for child_idx in webgpu.gpu_children() {
                let mut child_op = node_children[child_idx].clone();

                if child_op.as_webgpu().is_none() {
                    // If the child does not support WebGpu, we wrap it in an operator that
                    // loads the batch input and exposes it as a GPU input array.
                    child_op = Arc::new(WebGpuInputOperator::new(child_op));
                }

                let child_node_id = visit_node(child_op, dag, batch, hash_to_id, random_state);
                child_indices.push(child_node_id);
            }

            // And the batch input children
            for child_idx in webgpu.batch_children() {
                let child = node_children[child_idx].clone();
                let batch_id = batch.len();
                batch.push(child);
                batch_indices.push(batch_id);
            }

            // Create new DAG node
            let node_id: NodeId = dag.len();
            let dag_node = GpuNode {
                operator: node,
                children: child_indices,
                parents: vec![], // Will be filled in later
                batch_inputs: batch_indices,
            };

            dag.push(dag_node);
            hash_to_id.insert(subtree_hash, node_id);

            node_id
        }

        // Build the DAG
        let random_state = RandomState::default();
        let root_index = visit_node(
            operator,
            &mut dag,
            &mut batch,
            &mut hash_to_id,
            &random_state,
        );

        // Fill in parent relationships
        for i in 0..dag.len() {
            let children = dag[i].children.clone();
            for &child_idx in &children {
                dag[child_idx].parents.push(i);
            }
        }

        Some(WebGpuSubgraphOperator {
            root: root_index,
            dag,
            batch_inputs: batch,
        })
    }

    fn root_operator(&self) -> &OperatorRef {
        &self.dag[self.root].operator
    }
}

impl Operator for WebGpuSubgraphOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.webgpu")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.root_operator().dtype()
    }

    fn len(&self) -> usize {
        self.root_operator().len()
    }

    fn children(&self) -> &[OperatorRef] {
        &self.batch_inputs
    }

    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f, "WebGpuOperator wrapping:")?;
        write!(f, "{}", self.root_operator().display_tree())
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let mut this = self.as_ref().clone();
        this.batch_inputs = children;
        Ok(Arc::new(this))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for WebGpuSubgraphOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        // Compute the topological sort of the DAG
        let exec_order = topological_sort(&self.dag)?;

        // TODO: Compute an allocation plan for GPU buffers
        let allocation_plan = allocate_gpu_buffers(&self.dag, &exec_order)?;

        // Bind each node in the DAG to create its GPU kernel
        let kernels = bind_gpu_kernels(&self.dag, &allocation_plan)?;

        // Bind the batch input operators
        let batch_inputs: Vec<_> = (0..self.batch_inputs.len())
            .map(|i| ctx.take_child(i))
            .try_collect()?;

        Ok(Box::new(WebGpuExecution {
            len: self.len(),
            dtype: self.dtype().clone(),
            batch_inputs,
            kernels,
            exec_order,
            allocation_plan,
        }))
    }
}

struct WebGpuExecution {
    len: usize,
    dtype: DType,
    batch_inputs: Vec<BatchExecutionRef>,
    kernels: Vec<Box<dyn GpuKernel>>,
    exec_order: Vec<NodeId>,
    allocation_plan: GpuAllocationPlan,
}

#[async_trait]
impl BatchExecution for WebGpuExecution {
    async fn execute(mut self: Box<Self>) -> VortexResult<Canonical> {
        // Execute all batch input operators concurrently.
        let batch_inputs =
            try_join_all(self.batch_inputs.into_iter().map(|exec| exec.execute())).await?;

        // Create a GPU execution context with the batch inputs.
        let ctx = GpuExecutionContext { batch_inputs };

        // TODO: Initialize WebGpu resources (device, queue, etc.)

        // Execute kernels in topological order
        for &node_idx in &self.exec_order {
            let kernel = &mut self.kernels[node_idx];
            kernel.execute(&ctx)?;
        }

        // TODO: Read back the final result from GPU and convert to Canonical array

        vortex_bail!("WebGpu execution not yet implemented")
    }
}

/// Placeholder for GPU buffer allocation planning
struct GpuAllocationPlan {
    // TODO: Add fields for buffer allocation information
}

/// Topological sort of the GPU DAG nodes.
fn topological_sort(dag: &[GpuNode]) -> VortexResult<Vec<NodeId>> {
    // TODO: Implement proper topological sort (can reuse logic from pipeline::operator::toposort)
    // For now, return a simple sequential order
    Ok((0..dag.len()).collect())
}

/// Allocate GPU buffers for the execution plan.
fn allocate_gpu_buffers(
    _dag: &[GpuNode],
    _exec_order: &[NodeId],
) -> VortexResult<GpuAllocationPlan> {
    // TODO: Implement GPU buffer allocation planning
    Ok(GpuAllocationPlan {})
}

/// Bind GPU kernels for each node in the DAG.
fn bind_gpu_kernels(
    dag: &[GpuNode],
    _allocation_plan: &GpuAllocationPlan,
) -> VortexResult<Vec<Box<dyn GpuKernel>>> {
    let mut kernels = Vec::with_capacity(dag.len());
    for node in dag {
        // TODO: Create proper bind context with buffer IDs
        let bind_context = GpuBindContextImpl {
            children: vec![],
            batch_inputs: &node.batch_inputs,
        };

        let webgpu = node.operator.as_webgpu().ok_or_else(|| {
            vortex_error::vortex_err!("Operator does not support WebGpu execution")
        })?;
        kernels.push(webgpu.bind_gpu(&bind_context)?);
    }
    Ok(kernels)
}

struct GpuBindContextImpl<'a> {
    children: Vec<GpuBufferId>,
    batch_inputs: &'a [BatchId],
}

impl GpuBindContext for GpuBindContextImpl<'_> {
    fn children(&self) -> &[GpuBufferId] {
        &self.children
    }

    fn batch_inputs(&self) -> &[BatchId] {
        self.batch_inputs
    }
}
