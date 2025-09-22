// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex crate containing vectorized pipeline processing.
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
pub(crate) mod operator;
mod types;
pub mod vec;
pub mod view;

/// The number of elements in each step of a Vortex evaluation pipeline.
pub const N: usize = 1024;

// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

use std::cell::RefCell;

pub use types::*;
use vec::VectorRef;
use vortex_error::VortexResult;

use self::vec::Vector;
use self::view::ViewMut;
use crate::Canonical;
use crate::operator::Operator;

pub trait PipelinedOperator: Operator {
    // Whether this operator works by mutating its first child in-place.
    //
    // If `true`, the operator is invoked with the first child's input data passed via the
    // mutable output view. The node is expected to mutate this data in-place.
    // TODO(ngates): enable this
    // fn in_place(&self) -> bool {
    //     false
    // }

    /// Bind the operator into a [`Kernel`] for pipelined execution.
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>>;

    /// Returns the child indices of this operator that are passed to the kernel as input vectors.
    fn vector_children(&self) -> Vec<usize>;

    /// Returns the child indices of this operator that are passed to the kernel as batch inputs.
    fn batch_children(&self) -> Vec<usize>;
}

/// The context used when binding an operator for execution.
pub trait BindContext {
    fn children(&self) -> &[VectorId];

    fn batch_inputs(&self) -> &[BatchId];
}

/// The ID of the vector to use.
pub type VectorId = usize;
/// The ID of the batch input to use.
pub type BatchId = usize;

/// A pipeline provides a push-based way to emit a stream of canonical data.
///
/// By passing multiple vector computations through the same pipeline, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.), and to make better
/// use of CPU caches by performing all operations while the data is hot.
pub trait Kernel: Send {
    /// Attempts to perform a single step of the pipeline, writing data to the output vector.
    ///
    /// The output vector is guaranteed to have space for at least `N` elements. The kernel
    /// may write up to `N` elements to the output vector, and must update the length of the
    /// output vector to reflect the number of elements written.
    ///
    /// TODO(ngates): alternatively, we allow the kernel to write sparse output vectors using a
    ///  Selection enum of Prefix(n), Masked(Mask), or All. This would allow parent kernels to
    ///  decide when to flatten the vector. The problem is it becomes ambiguous who is responsible
    ///  for compacting very sparse vectors.
    fn step(&mut self, ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()>;
}

/// Context passed to kernels during execution, providing access to vectors.
pub struct KernelContext {
    /// The allocated vectors for intermediate results.
    pub(crate) vectors: Vec<RefCell<Vector>>,
    /// The computed batch inputs.
    pub(crate) batch_inputs: Vec<Canonical>,
}

impl KernelContext {
    /// Get a vector by its ID.
    pub fn vector(&self, vector_id: VectorId) -> VectorRef<'_> {
        VectorRef::new(self.vectors[vector_id].borrow())
    }

    /// Get a batch input by its ID.
    pub fn batch_input(&self, batch_id: BatchId) -> &Canonical {
        &self.batch_inputs[batch_id]
    }
}
