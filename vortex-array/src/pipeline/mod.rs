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
pub(crate) mod operator;
pub mod row_selection;
mod types;
pub mod vec;
pub mod view;

use std::cell::RefCell;

pub use row_selection::*;
pub use types::*;
use vec::VectorRef;
use vortex_error::VortexResult;

use self::vec::Vector;
use self::view::ViewMut;
use crate::Canonical;
use crate::operator::Operator;
use crate::pipeline::bits::BitView;

/// The number of elements in each step of a Vortex evaluation operator.
pub const N: usize = 1024;

// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

pub trait PipelinedOperator: Operator {
    /// Defines the row selection of this pipeline operator.
    fn row_selection(&self) -> RowSelection;

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

/// A operator provides a push-based way to emit a stream of canonical data.
///
/// By passing multiple vector computations through the same operator, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.), and to make better
/// use of CPU caches by performing all operations while the data is hot.
pub trait Kernel: Send {
    /// Attempts to perform a single step of the operator, writing data to the output vector.
    ///
    /// The kernel step should be stateless and is passed the chunk index as well as the selection
    /// mask for this chunk.
    ///
    /// Input and output vectors have a `Selection` enum indicating which elements of the vector
    /// are valid for processing. This is one of:
    /// * Full - all N elements are valid.
    /// * Prefix - the first n elements are valid, where n is the true count of the selection mask.
    /// * Mask - only the elements indicated by the selection mask are valid.
    ///
    /// Kernel should inspect the selection enum of the input and iterate the values accordingly.
    /// They may choose to write the output vector in any selection mode, but should choose the most
    /// efficient mode possible - not forgetting to update the output vector's selection enum.
    fn step(
        &self,
        ctx: &KernelContext,
        chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()>;
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
