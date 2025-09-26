// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bind;
pub mod buffers;
mod input;
mod output;
mod toposort;

use std::any::Any;
use std::cell::RefCell;
use std::fmt::Formatter;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use output::{BoolOutput, PipelineOutput};
use termtree::Tree;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::{HashMap, RandomState};

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, DisplayFormat, LengthBounds,
    MaskExecution, Operator, OperatorEq, OperatorHash, OperatorId, OperatorKey, OperatorRef,
};
use crate::pipeline::bits::{BitVector, BitView, BitViewMut};
use crate::pipeline::operator::bind::bind_kernels;
use crate::pipeline::operator::buffers::{allocate_vectors, OutputTarget};
use crate::pipeline::operator::input::PipelineInputOperator;
use crate::pipeline::operator::output::PrimitiveOutput;
use crate::pipeline::operator::toposort::topological_sort;
use crate::pipeline::vec::Vector;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{BatchId, Element, Kernel, KernelContext, N, N_WORDS};
use crate::Canonical;

/// An operator node used during execution planning to represent a pipelined execution.
///
/// This operator builds up a DAG of operators that can be executed in a pipelined fashion, as well
/// as any batch input operators that provide batch data to the operator.
#[derive(Clone, Debug)]
pub(crate) struct PipelineOperator {
    root: NodeId,
    dag: Vec<PipelineNode>,
    /// The set of operators that provide canonicalized batch inputs to the pipeline.
    batch_inputs: Vec<OperatorRef>,
    /// The set of the batch inputs that should have the mask propagated to them.
    domain_inputs: Vec<BatchId>,
}

impl OperatorHash for PipelineOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.root.hash(state);
        for node in &self.dag {
            node.operator_hash(state);
        }
        for input in &self.batch_inputs {
            input.operator_hash(state);
        }
    }
}

impl OperatorEq for PipelineOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        if self.root != other.root || self.dag.len() != other.dag.len() {
            return false;
        }
        for (node_a, node_b) in self.dag.iter().zip(other.dag.iter()) {
            if !node_a.operator_eq(node_b) {
                return false;
            }
        }
        if self.batch_inputs.len() != other.batch_inputs.len() {
            return false;
        }
        for (input_a, input_b) in self.batch_inputs.iter().zip(other.batch_inputs.iter()) {
            if !input_a.operator_eq(input_b) {
                return false;
            }
        }
        true
    }
}

type NodeId = usize;

#[derive(Debug, Clone)]
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

impl OperatorHash for PipelineNode {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.operator.operator_hash(state);
        self.children.hash(state);
        self.batch_inputs.hash(state);
    }
}

impl OperatorEq for PipelineNode {
    fn operator_eq(&self, other: &Self) -> bool {
        self.operator.operator_eq(&other.operator)
            && self.children == other.children
            && self.batch_inputs == other.batch_inputs
    }
}

impl PipelineOperator {
    /// From the given operator, constructs a `PipelineOperator` as large as possible by
    /// traversing children that also support pipelined execution.
    pub fn new(operator: OperatorRef) -> Option<Self> {
        operator.as_pipelined()?;

        fn visit_node(
            node: OperatorRef,
            dag: &mut Vec<PipelineNode>,
            batch: &mut Vec<OperatorRef>,
            domain_inputs: &mut Vec<BatchId>,
            hash_to_id: &mut HashMap<u64, NodeId>,
            random_state: &RandomState,
        ) -> NodeId {
            // Compute the hash for this subtree.
            let subtree_hash = random_state.hash_one(OperatorKey(node.clone()));

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
                let mut is_domain_input = false;

                if child_op.as_pipelined().is_none() {
                    // If the child supports pipelining and has the same row selection, we can
                    // include it in our pipeline. Otherwise, we need to stop the pipeline here and
                    // treat this child as a batch input.
                    child_op = Arc::new(PipelineInputOperator::new(child_op));

                    // If the child is marked as a vector child, but it doesn't itself support
                    // pipelined compute, then we know it has the same domain of rows as the
                    // pipeline. We track it so we can use its row count as the domain size at
                    // execution time.
                    is_domain_input = true;
                }

                let child_node_id = visit_node(
                    child_op,
                    dag,
                    batch,
                    domain_inputs,
                    hash_to_id,
                    random_state,
                );
                child_indices.push(child_node_id);

                if is_domain_input {
                    let domain_batch = &dag[child_node_id].batch_inputs;
                    assert_eq!(domain_batch.len(), 1);
                    domain_inputs.push(domain_batch[0]);
                }
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
        let mut dag = vec![];
        let mut batch = vec![];
        let mut domain_inputs = vec![];
        let mut hash_to_id: HashMap<u64, NodeId> = HashMap::new();
        let random_state = RandomState::default();
        let root_index = visit_node(
            operator,
            &mut dag,
            &mut batch,
            &mut domain_inputs,
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
            domain_inputs,
        })
    }

    fn root_operator(&self) -> &OperatorRef {
        &self.dag[self.root].operator
    }
}

impl Operator for PipelineOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.operator")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.root_operator().dtype()
    }

    fn bounds(&self) -> LengthBounds {
        self.root_operator().bounds()
    }

    fn children(&self) -> &[OperatorRef] {
        &self.batch_inputs
    }

    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f, "PipelineOperator")?;
        write!(f, "{}", self.root_operator().display_tree(),)
    }

    fn fmt_all(&self) -> String {
        let node_name = "PipelineOperator".to_string();

        let child_trees: Vec<_> = iter::once(self.root_operator().fmt_all())
            .chain(self.children().iter().map(|child| child.fmt_all()))
            .collect();
        Tree::new(node_name)
            .with_leaves(child_trees)
            .with_multiline(true)
            .to_string()
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
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        // Compute the toposort of the DAG
        let exec_order = topological_sort(&self.dag)?;

        // Compute an allocation plan for intermediate vectors
        let allocation_plan = allocate_vectors(&self.dag, &exec_order)?;

        // Bind each node in the DAG to create its kernel
        let kernels = bind_kernels(&self.dag, &allocation_plan)?;

        // Bind the batch input operators
        let batch_inputs: Vec<_> = self
            .batch_inputs
            .iter()
            .enumerate()
            // TODO(ngates): some batch inputs need to be masked!
            // TODO(ngates): pipeline inputs are different from batch inputs. We probably want
            //  batch execution to support returning a MaskedCanonical array for deferred masking.
            .map(|(i, operator)| {
                if self.domain_inputs.contains(&i) {
                    ctx.bind_project(operator, Some(mask))
                } else {
                    ctx.bind_project(operator, None)
                }
            })
            .try_collect()?;

        let vectors = allocation_plan.vectors;
        let pipeline = Pipeline {
            kernels,
            exec_order,
            output_targets: allocation_plan.output_targets,
        };

        let len = self
            .root_operator()
            .bounds()
            .maybe_len()
            .vortex_expect("Must have length");
        let mask = ctx.bind_mask(mask)?;

        match self.dtype() {
            DType::Bool(Nullability::NonNullable) => Ok(Box::new(
                PipelineExecution::<BoolOutput>::new(len, mask, batch_inputs, vectors, pipeline),
            )),
            DType::Primitive(ptype, Nullability::NonNullable) => {
                match_each_native_ptype!(ptype, |T| {
                    Ok(Box::new(PipelineExecution::<PrimitiveOutput<T>>::new(
                        len,
                        mask,
                        batch_inputs,
                        vectors,
                        pipeline,
                    )))
                })
            }
            _ => vortex_bail!(
                "PipelineOperator currently only supports non-nullable bool or primitive output types {}",
                self.dtype()
            ),
        }
    }
}

struct PipelineExecution<O> {
    len: usize,
    mask: MaskExecution,
    // The children store the batch inputs to the pipeline. If the LenProvider indicates that we
    // are running over a masked domain of rows, the final child will be the mask operator.
    children: Vec<BatchExecutionRef>,
    vectors: Vec<RefCell<Vector>>,
    pipeline: Pipeline,
    _element: PhantomData<O>,
}

impl<O> PipelineExecution<O> {
    fn new(
        len: usize,
        mask: MaskExecution,
        batch_inputs: Vec<BatchExecutionRef>,
        vectors: Vec<RefCell<Vector>>,
        pipeline: Pipeline,
    ) -> Self {
        PipelineExecution {
            len,
            mask,
            children: batch_inputs,
            vectors,
            pipeline,
            _element: PhantomData,
        }
    }
}

#[async_trait]
impl<O: PipelineOutput> BatchExecution for PipelineExecution<O> {
    async fn execute(mut self: Box<Self>) -> VortexResult<Canonical> {
        // Execute all child operators concurrently with the row selection.
        let children = try_join_all(self.children.into_iter().map(|exec| exec.execute())).await?;

        // Extract the selection mask
        let mask = self.mask.await?;
        assert_eq!(self.len, mask.len(), "Incorrect mask length");
        let len = self.len;

        // Create a kernel context with the batch inputs.
        let ctx = KernelContext {
            vectors: self.vectors,
            batch_inputs: children,
        };

        // Allocate the output vector and validity.
        let capacity = len.next_multiple_of(N) + N;
        let mut output = O::allocate(capacity);

        if mask.all_true() {
            // Run the operator to completion with all rows selected.
            let nchunks = len.div_ceil(N);
            let mut position = 0;
            for chunk_idx in 0..nchunks {
                let mask_len = (len - position).min(N);
                let selection_vec = (mask_len < N).then(|| BitVector::true_until(mask_len));
                let selection = selection_vec.as_ref().unwrap_or_else(|| BitVector::full());

                let mut elements_view = output.view_mut(position);
                self.pipeline
                    .step(&ctx, chunk_idx, &selection.as_view(), &mut elements_view)?;

                // Flatten the elements view such that the selected elements are at the front.
                elements_view.flatten::<O::Element>(&selection.as_view());

                // Advance the position by the number of true bits in the selection
                position += selection.true_count();

                // TODO(ngates): we should call Handle::yield every X iterations to avoid
                //  starving other tasks in async contexts.
            }
            assert_eq!(position, len);
        } else {
            // Step the pipeline over each chunk of the mask.
            let mask_buffer = mask.to_boolean_buffer();
            let mut mask_iter = mask_buffer.bit_chunks().iter_padded();

            let mut selection_words = [0usize; N_WORDS];
            let mut selection_view_mut = BitViewMut::new(&mut selection_words);

            let nchunks = (self.len + N - 1) / N;

            let mut position = 0;
            for chunk_idx in 0..nchunks {
                // Populate the mask for this chunk
                selection_view_mut.clear();
                selection_view_mut.fill_with_words(&mut mask_iter);

                let mut elements_view = output.view_mut(position);
                self.pipeline.step(
                    &ctx,
                    chunk_idx,
                    &selection_view_mut.as_view(),
                    &mut elements_view,
                )?;

                // Flatten the elements view such that the selected elements are at the front.
                elements_view.flatten::<O::Element>(&selection_view_mut.as_view());

                // Advance the position by the number of true bits in the selection
                position += selection_view_mut.true_count();
            }
            assert_eq!(position, mask.true_count());
        }

        output.into_canonical(mask.true_count())
    }
}

struct Pipeline {
    kernels: Vec<Box<dyn Kernel>>,
    exec_order: Vec<NodeId>,
    output_targets: Vec<OutputTarget>,
}

impl Kernel for Pipeline {
    fn step(
        &self,
        ctx: &KernelContext,
        chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        for &node_idx in self.exec_order.iter() {
            let kernel = &self.kernels[node_idx];

            match &self.output_targets[node_idx] {
                OutputTarget::ExternalOutput => kernel.step(ctx, chunk_idx, selection, out)?,
                OutputTarget::IntermediateVector(vector_idx) => {
                    let mut vector_ref = ctx.vectors[*vector_idx].borrow_mut();
                    let selection = {
                        let mut view = vector_ref.as_view_mut();
                        kernel.step(ctx, chunk_idx, selection, &mut view)?;
                        view.selection
                    };
                    // Propagate the selection set by the kernel to the stored vector
                    vector_ref.set_selection(selection);
                }
            };
        }

        Ok(())
    }
}
