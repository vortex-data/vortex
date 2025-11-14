// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod allocation;
mod bind;
mod input;
mod toposort;

use std::hash::{BuildHasher, Hash, Hasher};

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_ensure};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::{HashMap, RandomState};
use vortex_vector::{Vector, VectorMut, VectorMutOps};

use crate::pipeline::bit_view::{BitView, BitViewExt};
use crate::pipeline::driver::allocation::{OutputTarget, allocate_vectors};
use crate::pipeline::driver::bind::bind_kernels;
use crate::pipeline::driver::toposort::topological_sort;
use crate::pipeline::{Kernel, KernelCtx, N, PipelineInputs};
use crate::{Array, ArrayEq, ArrayHash, ArrayOperator, ArrayRef, ArrayVisitor, Precision};

/// A pipeline driver takes a Vortex array and executes it into a canonical vector.
///
/// The driver builds up a DAG of pipeline nodes from the array tree up to the edges of this
/// pipeline. The edge of a pipeline is defined as an array node that has zero pipelined children.
/// In other words, a pipeline encompasses the execution of a single "domain" of rows, where each
/// node has the same understanding of what a single "row" is. For example, for a DictArray the
/// codes child is pipelined and therefore the pipeline DAG continues, but the values child is not
/// pipelined and will be executed via a separate pipeline driver.
///
/// Once constructed, the pipeline driver can be executed to produce a canonical vector.
#[derive(Clone, Debug)]
pub(crate) struct PipelineDriver {
    /// The pipeline stored as a DAG where all `NodeId`s index into the dag vec.
    dag: Vec<Node>,
    root: NodeId,

    /// The set of _all_ non-pipelined children from _all_ nodes of the pipeline.
    batch_inputs: Vec<ArrayRef>,
}

type NodeId = usize;
type BatchId = usize;

#[derive(Debug, Clone)]
struct Node {
    // This node's underlying array.
    array: ArrayRef,
    /// The type of pipeline node.
    #[allow(dead_code)] // TODO(ngates): pipeline execute does not yet use this
    kind: NodeKind,
    // The indices of the pipelined children nodes in the `nodes` vector.
    children: Vec<NodeId>,
    // The indices of this node's parents in the `nodes` vector.
    parents: Vec<NodeId>,
    // The IDs of the batch inputs that feed into this node.
    batch_inputs: Vec<BatchId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    /// An input node feeds a batch vector into the pipeline chunk-by-chunk.
    Input,
    /// A source node provides input to the pipeline by writing into mutable output vectors one
    /// batch at a time.
    Source,
    /// A transform node takes pipelined inputs from its children and produces output vectors
    Transform,
}

impl PipelineDriver {
    /// Construct a pipeline driver from the given array.
    ///
    /// The constructor will traverse the array tree, walking the edges where the child is
    /// reported to be a "pipelined" input.
    pub fn new(array: ArrayRef) -> PipelineDriver {
        fn visit_node(
            array: ArrayRef,
            dag: &mut Vec<Node>,
            batch: &mut Vec<ArrayRef>,
            hash_to_id: &mut HashMap<u64, NodeId>,
            random_state: &RandomState,
        ) -> NodeId {
            // Compute the hash for this subtree.
            let subtree_hash = random_state.hash_one(ArrayKey(array.clone()));

            // Check if we've seen this subtree before (sub-expression elimination)
            if let Some(&existing_index) = hash_to_id.get(&subtree_hash) {
                // Reuse existing node
                return existing_index;
            }

            let node = match array.as_pipelined() {
                None => {
                    // If the array cannot be executed as a pipeline, then it becomes a view node.
                    let batch_id = batch.len();
                    batch.push(array.clone());

                    Node {
                        array,
                        kind: NodeKind::Input,
                        children: vec![],
                        parents: vec![],
                        batch_inputs: vec![batch_id],
                    }
                }
                Some(pipelined) => match pipelined.inputs() {
                    PipelineInputs::Source => {
                        // All inputs of a source node are batch inputs.
                        let children = array.children();
                        let mut batch_inputs = Vec::with_capacity(children.len());
                        for child in children {
                            batch_inputs.push(batch.len());
                            batch.push(child);
                        }

                        Node {
                            array,
                            kind: NodeKind::Source,
                            children: vec![],
                            parents: vec![],
                            batch_inputs,
                        }
                    }
                    PipelineInputs::Transform { pipelined_inputs } => {
                        // Only one child is the pipelined input
                        let children = array.children();
                        let mut batch_inputs = Vec::with_capacity(children.len());
                        let mut pipeline_inputs = Vec::with_capacity(1);

                        for (child_idx, child) in children.into_iter().enumerate() {
                            if pipelined_inputs.contains(&child_idx) {
                                pipeline_inputs.push(visit_node(
                                    child.clone(),
                                    dag,
                                    batch,
                                    hash_to_id,
                                    random_state,
                                ));
                            } else {
                                let batch_id = batch.len();
                                batch.push(child);
                                batch_inputs.push(batch_id);
                            }
                        }

                        Node {
                            array,
                            kind: NodeKind::Transform,
                            children: pipeline_inputs,
                            parents: vec![],
                            batch_inputs,
                        }
                    }
                },
            };

            let node_id = dag.len();
            dag.push(node);
            hash_to_id.insert(subtree_hash, node_id);

            node_id
        }

        // Build the DAG
        let mut dag = vec![];
        let mut batch = vec![];
        let mut hash_to_id: HashMap<u64, NodeId> = HashMap::new();
        let random_state = RandomState::default();
        let root_index = visit_node(array, &mut dag, &mut batch, &mut hash_to_id, &random_state);

        // Fill in parent relationships
        for i in 0..dag.len() {
            let children = dag[i].children.clone();
            for &child_idx in &children {
                dag[child_idx].parents.push(i);
            }
        }

        PipelineDriver {
            root: root_index,
            dag,
            batch_inputs: batch,
        }
    }

    fn root_array(&self) -> &ArrayRef {
        &self.dag[self.root].array
    }

    /// Execute the pipeline after first executing all batch inputs.
    pub fn execute(self, selection: &Mask) -> VortexResult<Vector> {
        let dtype = self.root_array().dtype().clone();

        // Execute the batch inputs of the pipeline.
        let batch_inputs: Vec<_> = self
            .batch_inputs
            .into_iter()
            .map(|array| array.execute().map(Some))
            .try_collect()?;

        // Compute the toposort of the DAG
        let exec_order = topological_sort(&self.dag)?;

        // Compute an allocation plan for intermediate vectors
        let allocation_plan = allocate_vectors(&self.dag, &exec_order)?;

        // Bind each node in the DAG to create its kernel
        let kernels = bind_kernels(self.dag, &allocation_plan, batch_inputs)?;

        // Construct the kernel execution context
        let ctx = KernelCtx::new(allocation_plan.vectors);

        Pipeline {
            dtype,
            ctx,
            kernels,
            exec_order,
            output_targets: allocation_plan.output_targets,
        }
        .execute(selection)
    }
}

struct Pipeline {
    dtype: DType,
    ctx: KernelCtx,
    kernels: Vec<Box<dyn Kernel>>,
    exec_order: Vec<NodeId>,
    output_targets: Vec<OutputTarget>,
}

impl Pipeline {
    fn execute(&mut self, selection: &Mask) -> VortexResult<Vector> {
        // Start by allocating the output vector.
        let capacity = selection.true_count().next_multiple_of(N);
        let mut output = VectorMut::with_capacity(&self.dtype, capacity);

        match selection {
            Mask::AllFalse(_) => {}
            Mask::AllTrue(_) => {
                // Run the operator to completion with all rows selected.
                // The number of _full_ chunks we need to process.
                let nchunks = selection.len() / N;
                for _ in 0..nchunks {
                    self.step(&BitView::all_true(), &mut output)?;
                }

                // Now process the final partial chunk, if any.
                let remaining = selection.len() % N;
                if remaining > 0 {
                    let selection_view = BitView::with_prefix(remaining);
                    self.step(&selection_view, &mut output)?;
                }
            }
            Mask::Values(mask_values) => {
                // Loop over each chunk of N elements in the mask as a bit view.
                let selection_bits = mask_values.bit_buffer();
                for selection_view in selection_bits.iter_bit_views() {
                    self.step(&selection_view, &mut output)?;
                }
            }
        }

        Ok(output.freeze())
    }

    /// Perform a single step of the pipeline.
    fn step(&mut self, selection: &BitView, output: &mut VectorMut) -> VortexResult<()> {
        // Loop over the kernels in toposorted execution order.
        for &node_idx in self.exec_order.iter() {
            let kernel = &mut self.kernels[node_idx];

            // Depending on the output target, either write directly to the pipeline output, or
            // take the intermediate vector and write into that.
            match &self.output_targets[node_idx] {
                OutputTarget::ExternalOutput => {
                    // We split off the next N elements of capacity from the external output vector.
                    let mut tail = output.split_off(output.len());
                    assert!(tail.is_empty());

                    kernel.step(&self.ctx, selection, &mut tail)?;

                    let len = tail.len();
                    vortex_ensure!(
                        len == N || len == selection.true_count(),
                        "Kernel produced incorrect number of output elements, \
                            expected either {N} or {}, got {len}",
                        selection.true_count(),
                    );

                    // Since we are writing to the final vector, there are no other kernels who we
                    // can delegate filtering the selection mask out to, so check if we need to do
                    // a final filter before we return.
                    if selection.true_count() < N && len == N {
                        // tail.filter(selection_mask)
                        todo!("Filter via a bit mask")
                    }

                    // Now we join the produced output back to the main output vector.
                    output.unsplit(tail);
                }
                OutputTarget::IntermediateVector(vector_id) => {
                    let mut out_vector = self.ctx.take_output(vector_id);
                    out_vector.clear();
                    debug_assert!(out_vector.is_empty());

                    kernel.step(&self.ctx, selection, &mut out_vector)?;

                    let len = out_vector.len();
                    vortex_ensure!(
                        len == N || len == selection.true_count(),
                        "Kernel produced incorrect number of output elements, \
                            expected either {N} or {}, got {len}",
                        selection.true_count(),
                    );

                    // If the kernel added N elements, the output is in-place.
                    self.ctx.replace_output(vector_id, out_vector);
                }
            };
        }

        Ok(())
    }
}

/// A hashable array compared with [`Precision::Ptr`].
struct ArrayKey(ArrayRef);
impl Hash for ArrayKey {
    fn hash<H: Hasher>(&self, mut state: &mut H) {
        self.0.array_hash(&mut state, Precision::Ptr)
    }
}
impl PartialEq for ArrayKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.array_eq(&other.0, Precision::Ptr)
    }
}
impl Eq for ArrayKey {}
