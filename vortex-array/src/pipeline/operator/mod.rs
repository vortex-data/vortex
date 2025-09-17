// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bind;
pub mod buffers;
mod toposort;

use crate::arrays::PrimitiveArray;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchId, BatchOperator, Operator, OperatorId,
    OperatorRef,
};
use crate::pipeline::bits::{BitVector, BitView};
use crate::pipeline::operator::bind::bind_kernels;
use crate::pipeline::operator::buffers::{allocate_vectors, OutputTarget};
use crate::pipeline::operator::toposort::topological_sort;
use crate::pipeline::vec::Vector;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N};
use crate::validity::Validity;
use crate::Canonical;
use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use std::any::Any;
use std::cell::RefCell;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::ops::DerefMut;
use std::sync::Arc;
use vortex_buffer::BufferMut;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::VortexResult;
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
        if operator.as_pipelined().is_none() {
            return None;
        }

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
            for child in node.children() {
                if child.as_pipelined().is_some() {
                    let child_id = visit_node(child.clone(), dag, batch, hash_to_id, random_state);
                    child_indices.push(child_id)
                } else {
                    // Otherwise, it's a batch input operator.
                    let batch_id = batch.len();
                    batch.push(child.clone());
                    batch_indices.push(batch_id);
                }
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
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |T| {
                    Ok(Box::new(PrimitivePipelineExecution {
                        len: self.len(),
                        batch_inputs,
                        vectors,
                        pipeline,
                        phantom_data: PhantomData::<T>::default(),
                    }))
                })
            }
            _ => todo!("PipelineOperator currently only supports primitive output types"),
        }
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
        let len = self.len;
        let capacity = len.next_multiple_of(N) + N;

        let mut elements = BufferMut::<T>::with_capacity(capacity);
        unsafe { elements.set_len(capacity) };

        let mut remaining = len;
        while remaining >= N {
            let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
            self.pipeline
                .step(&ctx, BitView::all_true(), &mut elements_view)?;
            remaining -= N;
        }

        if remaining > 0 {
            let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
            let mask = BitVector::true_until(remaining);
            self.pipeline
                .step(&ctx, mask.as_view(), &mut elements_view)?;
        }

        unsafe { elements.set_len(len) };

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
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.kernels
            .iter_mut()
            .try_for_each(|op| op.seek(chunk_idx))
    }

    fn step(
        &mut self,
        ctx: &KernelContext,
        _selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        for &node_idx in self.exec_order.iter() {
            let kernel = self.kernels[node_idx].as_mut();

            match &self.output_targets[node_idx] {
                OutputTarget::ExternalOutput => kernel.step(&ctx, _selected, out)?,
                OutputTarget::IntermediateVector(vector_idx) => {
                    let mut vector_ref = ctx.vectors[*vector_idx].borrow_mut();
                    let result = {
                        let mut view = vector_ref.as_view_mut();
                        kernel.step(&ctx, _selected, &mut view)
                    };
                    vector_ref.deref_mut().set_len(_selected.true_count());
                    result?
                }
            }
        }
        Ok(())
    }
}
