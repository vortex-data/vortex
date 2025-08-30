// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_variables)]
#![cfg_attr(vortex_nightly, feature(portable_simd))]
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
mod canonical;
pub mod operators;
pub mod query;
mod types;
pub mod vec;
pub mod view;

/// The number of elements in each step of a Vortex evaluation pipeline.
pub const N: usize = 1024;

// Number of usize words needed to store N bits
pub const N_WORDS: usize = N / usize::BITS as usize;

use std::cell::RefCell;

pub use canonical::*;
pub use operators::{Operator, OperatorRef};
pub use types::*;
use vec::{VectorId, VectorRef};
use vortex_error::VortexResult;

use self::bits::BitView;
use self::vec::Vector;
use self::view::ViewMut;

/// A pipeline provides a push-based way to emit a stream of canonical data.
///
/// By passing multiple vector computations through the same pipeline, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.), and to make better
/// use of CPU caches by performing all operations while the data is hot.
///
/// By passing a mask into the `step` function, we give encodings visibility into the data that
/// will be read by their parents. Some encodings may choose to decode all `N` elements, and then
/// set the given selection mask on the output vector. Other encodings may choose to only unpack
/// the selected elements.
///
/// We are considering further adding a `defined` parameter that indicates which elements are
/// defined and will be interpreted by the parent. This differs from masking, in that undefined
/// elements should still live in the correct location, it just doesn't matter what their value
/// is. This will allow, e.g. a validity encoding to tell its children that the values in certain
/// positions are going to be masked out anyway, so don't bother doing any expensive compute.
pub trait Kernel {
    /// Seek the kernel to a specific chunk offset.
    ///
    /// Note this will be called on all kernels in a pipeline.
    ///
    /// i.e. the resulting row offset is `idx * N`, where `N` is the number of elements in a chunk.
    ///
    /// The reason for a separate seek function (vs passing an offset directly to `step`) is that
    /// it allows the pipeline to optimize for sequential access patterns, which is common in
    /// many encodings. For example, a run-length encoding can efficiently seek to the start of a
    /// chunk without needing to perform a full binary search of the ends in each step.
    // TODO(ngates): should this be `skip(n)` instead? Depends if we want to support going
    //  backwards?
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    /// Attempts to perform a single step of the pipeline, writing data to the output vector.
    /// Returns `Poll::Done` if the pipeline is complete, or `Poll::Pending` if buffers are
    /// required to continue.
    ///
    /// The `selected` parameter defines which elements of the chunk should be exported, where
    /// `None` indicates that all elements are selected.
    ///
    // TODO(ngates): we could introduce a `defined` parameter to indicate which elements are
    //  defined and will be interpreted by the parent. This would allow us to skip writing
    //  elements that are not defined, for example if the parent is a dense null validity encoding.
    fn step(
        &mut self,
        ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()>;
}

/// Context passed to kernels during execution, providing access to vectors.
#[derive(Default)]
pub struct KernelContext {
    /// Optional allocation plan for resolving vector IDs
    pub(crate) vectors: Vec<RefCell<Vector>>,
}

impl KernelContext {
    pub fn new(allocation_plan: Vec<RefCell<Vector>>) -> Self {
        Self {
            vectors: allocation_plan,
        }
    }

    /// Get a vector by its ID.
    pub fn vector(&self, vector_id: VectorId) -> VectorRef<'_> {
        VectorRef::new(self.vectors[*vector_id].borrow())
    }
}

use crate::vtable::{NotSupported, VTable};

pub trait PipelineVTable<V: VTable> {
    /// Convert the current array into a [`Operator`].
    /// Returns `None` if the array cannot be converted to an operator.
    fn to_operator(array: &V::Array) -> VortexResult<Option<OperatorRef>>;
}

impl<V: VTable> PipelineVTable<V> for NotSupported {
    fn to_operator(_array: &V::Array) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
    }
}
