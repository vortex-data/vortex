// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bit_view;
pub mod source_driver;

use std::ops::Deref;

use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMut, VectorMutOps};

use crate::Array;

/// The number of elements in each step of a Vortex evaluation operator.
pub const N: usize = 1024;

/// Number of bytes needed to store N bits
pub const N_BYTES: usize = N / 8;

/// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

/// A pipeline source node has zero pipelined inputs and produces data to feed into a pipeline.
///
/// All children of the array are considered to be batch inputs and will be fully computed before
/// pipelined execution begins.
pub trait PipelineSource: Deref<Target = dyn Array> {
    /// Bind the operator into a [`SourceKernel`] for pipelined execution.
    ///
    /// The provided [`BindContext`] can be used to obtain vector IDs for pipelined children and
    /// batch IDs for batch children. Each child can only be bound once.
    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn SourceKernel>>;
}

/// Indicates that an array supports acting as a transformation node in a pipelined execution.
///
/// Transform nodes have exactly one pipelined input, with zero or more batch inputs.
pub trait PipelineTransform: Deref<Target = dyn Array> {
    // Whether this operator works by mutating its first child in-place.
    //
    // If `true`, the operator is invoked with the first child's input data passed via the
    // mutable output view. The node is expected to mutate this data in-place.
    // TODO(ngates): enable this
    // fn in_place(&self) -> bool {
    //     false
    // }

    /// Returns the index of the array child that should be passed as a pipelined input
    fn pipelined_child(&self) -> usize;

    /// Bind the operator into a [`TransformKernel`] for pipelined execution.
    ///
    /// The provided [`BindContext`] can be used to obtain vector IDs for pipelined children and
    /// batch IDs for batch children. Each child can only be bound once.
    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn TransformKernel>>;
}

/// Indicates that an array supports acting as a transformation node in a pipelined execution
/// with multiple pipelined inputs and zero or more batch inputs.
pub trait PipelineZipTransform: Deref<Target = dyn Array> {
    /// Returns the index of the array child that should be passed as a pipelined input
    fn is_pipelined_child(&self, child_idx: usize) -> bool;

    /// Bind the operator into a [`TransformKernel`] for pipelined execution.
    ///
    /// The provided [`BindContext`] can be used to obtain vector IDs for pipelined children and
    /// batch IDs for batch children. Each child can only be bound once.
    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn TransformKernel>>;
}

/// The context used when binding an operator for execution.
pub trait BindContext {
    /// Returns the batch input vector for the given child.
    ///
    /// # Panics
    ///
    /// If the child index requested here was marked as a pipelined child in
    /// [`PipelineTransform::is_pipelined_child`].
    fn batch_input(&self, child_idx: usize) -> Vector;
}

/// A source kernel produces data to feed into pipelined execution.
///
/// The kernel is provided with a mutable output vector that is guaranteed to have capacity for at
/// least `2 * N` elements. Each invocation of the kernel is expected to append between `N` and
/// `2 * N` elements to the output vector, except when the end of the data is reached.
///
/// Vectors of `N` elements will be propagated throughout the pipeline, and any remaining elements
/// will be passed back to the kernel on the next iteration and appear at the start of the output
/// vector.
///
/// This kerfuffle allows kernels that are optimized for 1024-element chunks to operate efficiently,
/// while avoiding passing very sparsely selected vectors throughout the pipeline.
pub trait SourceKernel: Send {
    /// Perform a single step of the kernel.
    fn step(&mut self, out: &mut VectorMut) -> VortexResult<()>;
}

/// A transform kernel processes one or more input vectors and produces an output vector.
///
/// Besides the final chunk of data, each invocation of the kernel will be passed vectors of
/// exactly `N` elements. The kernel **must** append exactly the same number of elements to its
/// output vector.
///
/// The output vector is guaranteed to have at least `N` elements of capacity.
pub trait TransformKernel: Send {
    /// Perform a single step of the kernel.
    fn step(&mut self, input: &Vector, out: &mut VectorMut) -> VortexResult<()>;
}

/// A transform kernel that takes multiple input vectors and produces an output vector.
///
/// The pipeline driver will ensure that each invocation of the kernel is passed vectors of equal
/// length.
///
/// The output vector is guaranteed to have at least `N` elements of capacity.
pub trait ZipTransformKernel: Send {
    /// Perform a single step of the kernel.
    fn step(&mut self, inputs: &[Vector], out: &mut VectorMut) -> VortexResult<()>;
}

/// A general implementation of a source kernel that produces all null values.
pub struct AllNullSourceKernel {
    remaining: usize,
}

impl SourceKernel for AllNullSourceKernel {
    fn step(&mut self, out: &mut VectorMut) -> VortexResult<()> {
        let to_produce = self.remaining.min(N);
        self.remaining -= to_produce;
        out.append_nulls(to_produce);
        Ok(())
    }
}
