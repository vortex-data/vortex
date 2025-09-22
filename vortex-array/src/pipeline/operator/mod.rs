// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bind;
pub mod buffers;
mod input;
mod toposort;

use crate::arrays::{BoolArray, PrimitiveArray};
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, DisplayFormat, Operator,
    OperatorId, OperatorRef,
};
use crate::pipeline::operator::bind::bind_kernels;
use crate::pipeline::operator::buffers::{allocate_vectors, OutputTarget};
use crate::pipeline::operator::input::PipelineInputOperator;
use crate::pipeline::operator::toposort::topological_sort;
use crate::pipeline::vec::Vector;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{BatchId, Element, Kernel, KernelContext, N};
use crate::validity::Validity;
use crate::Canonical;
use arrow_buffer::BooleanBuffer;
use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use std::any::Any;
use std::cell::RefCell;
use std::fmt::Formatter;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::sync::Arc;
use vortex_buffer::{Alignment, BufferMut, ByteBuffer};
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::{HashMap, RandomState};

/// An operator node used during execution planning to represent a pipelined execution.
///
/// This operator builds up a DAG of operators that can be executed in a pipelined fashion, as well
/// as any batch input operators that provide batch data to the pipeline.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct PipelineOperator {
    root: NodeId,
    dag: Vec<PipelineNode>,
    batch_inputs: Vec<OperatorRef>,
}

type NodeId = usize;

#[derive(Debug, Clone, Hash)]
struct PipelineNode {
    // The operator at this node.
    operator: OperatorRef,
    // The indices of the child nodes in the `nodes` vector.
    children: Vec<NodeId>,
    // The indices of this node's parents in the `nodes` vector.
    parents: Vec<NodeId>,
    // The IDs of the batch inputs that feed into this node.
    batch_inputs: Vec<BatchId>,
}

impl PartialEq for PipelineNode {
    fn eq(&self, other: &Self) -> bool {
        self.operator.eq(&other.operator)
            && self.children == other.children
            && self.batch_inputs == other.batch_inputs
    }
}
impl Eq for PipelineNode {}

impl PipelineOperator {
    /// From the given operator, constructs a `PipelineOperator` as large as possible by
    /// traversing children that also support pipelined execution.
    pub fn new(operator: OperatorRef) -> Option<Self> {
        operator.as_pipelined()?;

        let mut dag = vec![];
        let mut batch = vec![];
        let mut hash_to_id: HashMap<u64, NodeId> = HashMap::new();

        fn visit_node(
            node: OperatorRef,
            dag: &mut Vec<PipelineNode>,
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
            let pipelined = node.as_pipelined().vortex_expect("must be pipelined");

            // Prepare the pipelined vector children
            for child_idx in pipelined.vector_children() {
                let mut child_op = node_children[child_idx].clone();

                if child_op.as_pipelined().is_none() {
                    // If the child does not support pipelining, we wrap it in an operator that
                    // loads the batch input and exposes it as a pipelined kernel over the
                    // resulting canonical array.
                    child_op = Arc::new(PipelineInputOperator::new(child_op));
                }

                let child_node_id = visit_node(child_op, dag, batch, hash_to_id, random_state);
                child_indices.push(child_node_id);
            }

            // And the batch input children
            for child_idx in pipelined.batch_children() {
                let child = node_children[child_idx].clone();
                let batch_id = batch.len();
                batch.push(child);
                batch_indices.push(batch_id);
            }

            // Create new DAG node
            let node_id: NodeId = dag.len();
            let dag_node = PipelineNode {
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

        Some(PipelineOperator {
            root: root_index,
            dag,
            batch_inputs: batch,
        })
    }

    fn root_operator(&self) -> &OperatorRef {
        &self.dag[self.root].operator
    }
}

impl Operator for PipelineOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.pipeline")
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
        write!(f, "PipelineOperator wrapping:\n")?;
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

impl BatchOperator for PipelineOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        // Compute the toposort of the DAG
        let exec_order = topological_sort(&self.dag)?;

        // Compute an allocation plan for intermediate vectors
        let allocation_plan = allocate_vectors(&self.dag, &exec_order)?;

        // Bind each node in the DAG to create its kernel
        let kernels = bind_kernels(&self.dag, &allocation_plan)?;

        // Bind the batch input operators
        let batch_inputs: Vec<_> = (0..self.batch_inputs.len())
            .map(|i| ctx.take_child(i))
            .try_collect()?;

        let vectors = allocation_plan.vectors;
        let pipeline = Pipeline {
            kernels,
            exec_order,
            output_targets: allocation_plan.output_targets,
        };

        match self.dtype() {
            DType::Bool(_) => Ok(Box::new(BoolPipelineExecution {
                len: self.len(),
                batch_inputs,
                vectors,
                pipeline,
            })),
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |T| {
                    Ok(Box::new(PrimitivePipelineExecution {
                        len: self.len(),
                        batch_inputs,
                        vectors,
                        pipeline,
                        phantom_data: PhantomData::<T>,
                    }))
                })
            }
            _ => vortex_bail!(
                "PipelineOperator currently only supports primitive output types {}",
                self.dtype()
            ),
        }
    }
}

struct BoolPipelineExecution {
    len: usize,
    batch_inputs: Vec<BatchExecutionRef>,
    vectors: Vec<RefCell<Vector>>,
    pipeline: Pipeline,
}

#[async_trait]
impl BatchExecution for BoolPipelineExecution {
    async fn execute(mut self: Box<Self>) -> VortexResult<Canonical> {
        // Execute all batch input operators concurrently.
        let batch_inputs =
            try_join_all(self.batch_inputs.into_iter().map(|exec| exec.execute())).await?;

        // Create a kernel context with the batch inputs.
        let ctx = KernelContext {
            vectors: self.vectors,
            batch_inputs,
        };

        // Allocate the output vector and validity.
        let capacity = self.len.next_multiple_of(N) + N;
        let mut elements = BufferMut::<bool>::with_capacity(capacity);
        unsafe { elements.set_len(capacity) };

        // Run the pipeline to completion.
        let mut output_len = 0;
        while output_len < self.len {
            let mut elements_view = ViewMut::new(&mut elements[output_len..][..N], None);
            self.pipeline.step(&ctx, &mut elements_view)?;
            output_len += elements_view.len;
            // TODO(ngates): we should call Handle::yield every X iterations to avoid
            //  starving other tasks in async contexts.
        }
        unsafe { elements.set_len(output_len) };

        let buffer = ByteBuffer::from_arrow_buffer(
            BooleanBuffer::from(elements.as_ref()).into_inner(),
            Alignment::of::<u64>(),
        );

        Ok(Canonical::Bool(BoolArray::try_new(
            buffer,
            0,
            output_len,
            Validity::NonNullable,
        )?))
    }
}

struct PrimitivePipelineExecution<T> {
    len: usize,
    batch_inputs: Vec<BatchExecutionRef>,
    vectors: Vec<RefCell<Vector>>,
    pipeline: Pipeline,
    phantom_data: PhantomData<T>,
}

#[async_trait]
impl<T: Element + NativePType> BatchExecution for PrimitivePipelineExecution<T> {
    async fn execute(mut self: Box<Self>) -> VortexResult<Canonical> {
        // Execute all batch input operators concurrently.
        let batch_inputs =
            try_join_all(self.batch_inputs.into_iter().map(|exec| exec.execute())).await?;

        // Create a kernel context with the batch inputs.
        let ctx = KernelContext {
            vectors: self.vectors,
            batch_inputs,
        };

        // Allocate the output vector and validity.
        let capacity = self.len.next_multiple_of(N) + N;
        let mut elements = BufferMut::<T>::with_capacity(capacity);
        unsafe { elements.set_len(capacity) };

        // Run the pipeline to completion.
        let mut output_len = 0;
        while output_len < self.len {
            let mut elements_view = ViewMut::new(&mut elements[output_len..][..N], None);
            self.pipeline.step(&ctx, &mut elements_view)?;
            output_len += elements_view.len;
        }
        unsafe { elements.set_len(output_len) };

        Ok(Canonical::Primitive(PrimitiveArray::new(
            elements.freeze(),
            Validity::NonNullable,
        )))
    }
}

struct Pipeline {
    kernels: Vec<Box<dyn Kernel>>,
    exec_order: Vec<NodeId>,
    output_targets: Vec<OutputTarget>,
}

impl Kernel for Pipeline {
    fn step(&mut self, ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        for &node_idx in self.exec_order.iter() {
            let kernel = self.kernels[node_idx].as_mut();

            match &self.output_targets[node_idx] {
                OutputTarget::ExternalOutput => kernel.step(ctx, out)?,
                OutputTarget::IntermediateVector(vector_idx) => {
                    let mut vector_ref = ctx.vectors[*vector_idx].borrow_mut();
                    let len = {
                        let mut view = vector_ref.as_view_mut();
                        kernel.step(ctx, &mut view)?;
                        view.len
                    };
                    // Propagate the length set by the kernel to the vector
                    vector_ref.set_len(len);
                }
            };
        }

        Ok(())
    }
}
