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
    fn batch_input(&self, batch_child_idx: usize) -> Vector;
}

/// A pipeline kernel is a stateful object that performs steps of a pipeline.
///
/// Each step of the kernel takes and returns vectors (depending on the type of kernel) of `N`
/// elements. Input vectors are provided via the [`KernelCtx`] and indicate the position of their
/// elements as either [`PipelineVector::Sparse`] or [`PipelineVector::Compact`] based on whether
/// the selected elements are in their original positions or compacted at the start of the vector
/// respectively.
///
/// The provided mutable output vector is guaranteed to have at least `N` elements of capacity.
/// The kernel **must** return a vector of exactly `N` elements in each step, and indicate with
/// the returned [`ElementPosition`] enum whether the output elements are in their original
/// positions or compacted at the start of the output vector.
pub trait Kernel: Send {
    /// Perform a single step of the kernel.
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<ElementPosition>;
}

/// Defines where the elements produced by a kernel are written in the output vector.
pub enum ElementPosition {
    /// Elements are written to the output vector in their original selected positions.
    Sparse,
    /// Elements are compacted at the start of the output vector.
    Compact,
}

/// The context provided to kernels during execution to access input vectors.
pub struct KernelCtx {
    vectors: Vec<Option<PipelineVector>>,
}

impl KernelCtx {
    fn new(vectors: Vec<PipelineVector>) -> Self {
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
    pub fn input(&mut self, id: VectorId) -> &PipelineVector {
        self.vectors[id.0]
            .as_ref()
            .vortex_expect("Input vector at index is not available")
    }

    #[inline]
    fn take_output(&mut self, id: &VectorId) -> PipelineVector {
        self.vectors[id.0]
            .take()
            .vortex_expect("Output vector at index is not available")
    }

    #[inline]
    fn replace_output(&mut self, id: &VectorId, vec: PipelineVector) {
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

/// A pipeline vector passed into and out of pipeline kernels.
#[derive(Debug)]
pub enum PipelineVector {
    /// Sparse indicates that the elements indicated by the selection mask are in their original
    /// sparse positions within the vector.
    Sparse(VectorMut),
    /// Compact indicates that the selected elements are compacted at the start of the vector in
    /// positions `0..true_count`.
    Compact(VectorMut),
}

impl PipelineVector {
    pub fn from_position(position: ElementPosition, vec: VectorMut) -> PipelineVector {
        match position {
            ElementPosition::Sparse => PipelineVector::Sparse(vec),
            ElementPosition::Compact => PipelineVector::Compact(vec),
        }
    }
}

impl From<PipelineVector> for VectorMut {
    fn from(value: PipelineVector) -> Self {
        match value {
            PipelineVector::Sparse(vec) => vec,
            PipelineVector::Compact(vec) => vec,
        }
    }
}
