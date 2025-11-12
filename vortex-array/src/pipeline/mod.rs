// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bit_view;
pub mod driver;

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::{Vector, VectorMut};

use crate::pipeline::bit_view::BitView;

/// The number of elements in each step of a Vortex evaluation operator.
pub const N: usize = 1024;

/// Number of bytes needed to store N bits
pub const N_BYTES: usize = N / 8;

/// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

/// A pipeline node is a trait that enables an array to participate in pipelined execution.
pub trait PipelinedNode {
    /// Returns information about the children of this node and how the node should participate
    /// in pipelined execution.
    fn inputs(&self) -> PipelineInputs;

    /// Bind the node into a [`Kernel`] for pipelined execution.
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
}

/// Describes the type of pipeline node and its input information.
pub enum PipelineInputs {
    /// This node acts as a pipeline source.
    ///
    /// All array inputs will be available as pre-computed batch inputs in the [`BindContext`].
    Source,

    /// This node acts as a transform node.
    ///
    /// Each listed index indicates a child that should be provided as a pipelined input. Each
    /// pipelined input should be bound to a [`VectorId`] via the [`BindContext`] and then
    /// accessed within the kernel by passing the [`VectorId`] to the [`KernelCtx`].
    ///
    /// All other children will be available as pre-computed batch inputs in the [`BindContext`].
    Transform { pipelined_inputs: Vec<usize> },
    // TODO(ngates): we may want a Chain variant in the future to support pipelining chunked arrays
}

/// The context used when binding an operator for execution.
pub trait BindContext {
    /// Returns the [`VectorId`] for the given child that can be passed to the
    /// [`KernelCtx`] within each step to access the given input.
    ///
    /// Note that this child index references the pipelined inputs only, not all children of the
    /// array.
    fn pipelined_input(&self, pipelined_child_idx: usize) -> VectorId;

    /// Returns the batch input vector for the given child.
    ///
    /// Note that this child index references the batch inputs only, not all children of the
    /// array.
    fn batch_input(&mut self, batch_child_idx: usize) -> Vector;
}

/// A pipeline kernel is a stateful object that performs steps of a pipeline.
///
/// Each step of the kernel processes zero or more input vectors, and writes output to a
/// pre-allocated mutable output vector.
///
/// Input vectors will either have length [`N`], indicating that all elements from the step are
/// present. Or they will have length equal to the [`BitView::true_count`] of the selection mask,
/// in which case only the selected elements are present.
///
/// Output vectors will always be passed with length zero.
///
/// Kernels may choose to output either all `N` elements in their original positions, or output
/// only the selected elements to the first `true_count` positions of the output vector. When
/// emitting `N` elements in-place, the kernel may omit expensive computations over the unselected
/// elements, provided that the output elements in those positions are still valid (i.e. typically
/// zeroed, rather than undefined).
///
/// The pipeline driver will verify these conditions before and after each step.
pub trait Kernel: Send {
    /// Perform a single step of the kernel.
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()>;
}

/// The context provided to kernels during execution to access input vectors.
pub struct KernelCtx {
    vectors: Vec<Option<VectorMut>>,
}

impl KernelCtx {
    fn new(vectors: Vec<VectorMut>) -> Self {
        Self {
            vectors: vectors.into_iter().map(Some).collect(),
        }
    }

    /// Returns the input vector at the given index.
    ///
    /// Note that a [`VectorMut`] is returned here, indicating that this is the only instance of
    /// the data. It does not imply that the caller is able to mutate the data (it is returned
    /// as an immutable reference).
    ///
    /// # Panics
    ///
    /// If the input vector at the given index is not available (typically because the vector
    /// happens to be currently borrowed as an output vector!).
    pub fn input(&mut self, id: VectorId) -> &VectorMut {
        self.vectors[id.0]
            .as_ref()
            .vortex_expect("Input vector at index is not available")
    }

    #[inline]
    fn take_output(&mut self, id: &VectorId) -> VectorMut {
        self.vectors[id.0]
            .take()
            .vortex_expect("Output vector at index is not available")
    }

    #[inline]
    fn replace_output(&mut self, id: &VectorId, vec: VectorMut) {
        self.vectors[id.0] = Some(vec);
    }
}

/// A unique identifier for a vector in the pipeline execution context.
#[derive(Debug, Clone, Copy)]
pub struct VectorId(usize);
impl VectorId {
    // Non-public constructor to keep the type opaque to end users.
    fn new(idx: usize) -> Self {
        VectorId(idx)
    }
}
