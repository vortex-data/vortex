// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex crate containing vectorized operator processing.
//!
//! This module contains experiments into pipelined data processing within Vortex.
//!
//! Arrays (and eventually Layouts) will be convertible into a [`Kernel`] that can then be
//! exported into a [`ViewMut`] one chunk of [`N`] elements at a time. This allows us to keep
//! compute largely within the L1 cache, as well as to write out canonical data into externally
//! provided buffers.
//!
//! Each chunk is represented in a canonical physical form, as determined by the logical
//! [`vortex_dtype::DType`] of the array. This provides a predicate base on which to perform
//! compute. Unlike DuckDB and other vectorized systems, we force a single canonical representation
//! instead of supporting multiple encodings because compute push-down is applied a priori to the
//! logical representation.
//!
//! It is a work-in-progress and is not yet used in production.

pub mod bits;
pub mod source_driver;

use crate::Array;
use bits::BitView;
use std::ops::Deref;
use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMut};

/// The number of elements in each step of a Vortex evaluation operator.
pub const N: usize = 1024;

/// Number of bytes needed to store N bits
pub const N_BYTES: usize = N / 8;

/// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

/// Returned by an array to indicate that it can be executed in a pipelined fashion.
pub trait PipelinedOperator: Array {
    // Whether this operator works by mutating its first child in-place.
    //
    // If `true`, the operator is invoked with the first child's input data passed via the
    // mutable output view. The node is expected to mutate this data in-place.
    // TODO(ngates): enable this
    // fn in_place(&self) -> bool {
    //     false
    // }

    /// Returns whether the nth child of this array should be passed to the kernel as a pipelined
    /// input vector, 1024 elements at a time.
    ///
    /// Any child that reports `false` will be treated as a batch input, and the full vector will be
    /// computed before pipelined execution begins.
    fn is_pipelined_child(&self, child_idx: usize) -> bool;

    /// Bind the operator into a [`Kernel`] for pipelined execution.
    ///
    /// The provided [`BindContext`] can be used to obtain vector IDs for pipelined children and
    /// batch IDs for batch children. Each child can only be bound once.
    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn OperatorKernel>>;
}

pub trait PipelinedSource: Deref<Target = dyn Array> {
    /// Bind the operator into a [`Kernel`] for pipelined execution.
    ///
    /// The provided [`BindContext`] can be used to obtain vector IDs for pipelined children and
    /// batch IDs for batch children. Each child can only be bound once.
    fn bind_source(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn SourceKernel>>;
}

/// The context used when binding an operator for execution.
pub trait BindContext {
    /// Returns a [`VectorId`] that can be passed to the [`KernelContext`] within the body of
    /// the [`Kernel`] to access the given child as a pipelined input vector.
    ///
    /// # Panics
    ///
    /// If the child index requested here was not listed in [`Pipelined::pipelined_children`].
    fn pipelined_input(&self, child_idx: usize) -> VectorId;

    /// Returns the batch input vector for the given child.
    ///
    /// # Panics
    ///
    /// If the child index requested here was listed in [`Pipelined::pipelined_children`].
    fn batch_input(&self, child_idx: usize) -> Vector;
}

/// The ID of the vector to use.
pub type VectorId = usize;

/// A kernel implements the physical compute required for pipelined execution. It is driven in a
/// push-based way, typically as part of a larger pipeline of kernels.
///
/// By passing multiple vector computations through the same operator, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.), and to make better
/// use of CPU caches by performing all operations while the data is hot.
///
/// The [`SourceKernel::step`] method will be invoked repeatedly to process chunks of data, [`N`] elements
/// at a time. Each invocation is passed a selection mask indicating which elements of the chunk
/// should be written to the start of the output vector.
///
/// The mutable output vector is **guaranteed** to have a capacity of at least [`N`] elements, and
/// its length will initially be set to zero. It is therefore safe to invoke unchecked writes up to
/// `N` elements.
///
/// The pipeline may invoke the `SourceKernel::skip` method to skip over some number of chunks of data.
/// The kernel should mutate any internal state as necessary to account for the skipped data.
pub trait SourceKernel: Send {
    /// Skip over the given number of chunks of data.
    ///
    /// For example, if `n` is 3, then the kernel should skip over `3 * N` elements of input data.
    fn skip(&mut self, n: usize);

    /// Attempts to perform a single step of the operator, writing data to the output vector.
    fn step(
        &mut self,
        ctx: &KernelContext,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()>;
}

pub trait OperatorKernel: Send {
    /// Attempts to perform a single step of the operator, writing data to the output vector.
    ///
    /// The output vector has length equal to the number of valid elements in the input vectors.
    /// This number of values should be written to the output vector.
    fn step(&self, ctx: &KernelContext, out: &mut VectorMut) -> VortexResult<()>;
}

/// Context passed to kernels during execution, providing access to vectors.
pub struct KernelContext {
    /// The allocated vectors for intermediate results.
    pub(crate) vectors: Vec<Vector>,
}

impl KernelContext {
    pub fn empty() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    /// Get a vector by its ID.
    pub fn vector(&self, vector_id: VectorId) -> &Vector {
        &self.vectors[vector_id]
    }
}
