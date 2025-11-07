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
mod types;
pub mod vec;
pub mod view;

use std::cell::RefCell;

use self::vec::Vector;
use crate::operator::Operator;
use crate::pipeline::bits::BitView;
use crate::Canonical;
pub use types::*;
use vec::VectorRef;
use vortex_error::VortexResult;
use vortex_vector::VectorMut;

/// The number of elements in each step of a Vortex evaluation operator.
pub const N: usize = 1024;

// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

/// Returned by an array to indicate that it can be executed in a pipelined fashion.
pub trait Pipelined {
    // Whether this operator works by mutating its first child in-place.
    //
    // If `true`, the operator is invoked with the first child's input data passed via the
    // mutable output view. The node is expected to mutate this data in-place.
    // TODO(ngates): enable this
    // fn in_place(&self) -> bool {
    //     false
    // }

    /// Returns the indices of the children of this array that should be passed to the kernel as
    /// pipelined input vectors, 1024 elements at a time.
    ///
    /// Any child not listed here will be treated as a batch input, and the full vector will be
    /// computed before pipelined execution begins.
    fn pipelined_children(&self) -> Vec<usize>;

    /// Bind the operator into a [`Kernel`] for pipelined execution.
    ///
    /// The provided [`BindContext`] can be used to obtain vector IDs for pipelined children and
    /// batch IDs for batch children. Each child can only be bound once.
    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn Kernel>>;
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
/// The [`Kernel::step`] method will be invoked repeatedly to process chunks of data, [`N`] elements
/// at a time. Each invocation is passed a selection mask indicating which elements of the chunk
/// should be written to the start of the output vector.
///
/// The mutable output vector is **guaranteed** to have a capacity of at least [`N`] elements, and
/// its length will initially be set to zero. It is therefore safe to invoke unchecked writes up to
/// `N` elements.
///
/// The pipeline may invoke the `Kernel::skip` method to skip over some number of chunks of data.
/// The kernel should mutate any internal state as necessary to account for the skipped data.
pub trait Kernel: Send {
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
}
