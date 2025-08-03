// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bitpacked;
pub mod primitive;
pub mod validity;

use crate::experiment::mask::BitMask;
use crate::experiment::vector::{BitVector, Vector};
use std::ops::Deref;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, Poll};
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

pub trait Encoding {
    /// [`DType`] and length of the node are passed down in the bind context.
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>>;
}

/// Context required for binding a node.
///
/// During the bind phase, context is passed down from parent nodes to child nodes.
pub struct BindContext<'a> {
    pub len: usize,
    pub dtype: &'a DType,
    pub stats: Option<&'a StatsSet>,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufferId(usize);

impl BufferId {
    /// Creates a new `BufferId` with a unique identifier.
    pub fn new() -> Self {
        BufferId(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

impl Deref for BufferId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// An instantiated evaluation of a pipeline.
pub trait Evaluation {
    /// Seek the evaluation to a specific chunk offset.
    /// The resulting row offset should be `idx * N`, where `N` is the number of elements in
    /// a chunk.
    ///
    // NOTE(ngates): we have a separate seek function since it can often be more efficient for
    //  arrays to assume they will be evaluated in order, e.g. run-length would have to do a full
    //  binary search of the ends in each step if we passed an offset that way.
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()>;

    /// Attempts to perform a single step of the evaluation, writing data to the output vector.
    /// Returns `Poll::Done` if the evaluation is complete, or `Poll::Pending` if buffers are
    /// required to continue.
    ///
    /// The `selected` parameter defines which elements of the chunk should be exported, where
    /// `None` indicates that all elements are selected.
    ///
    /// The `defined` parameter indicates which elements are defined, meaning which elements the
    /// exporter "cares about", where `None` indicates that all elements are defined. For example,
    /// if the parent knows all but one element is null, we must still "select" all elements
    /// in order to make sure they are correctly ordered within the vector, but it's useful to
    /// know that only one of the element is useful to the parent. The others can be ignored.
    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &BitMask,
        defined: &BitMask,
        out: &mut Vector,
    ) -> Poll<VortexResult<()>>;
}

pub trait EvaluationContext {
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>>;
}

impl EvaluationContext for HashMap<BufferId, ByteBuffer> {
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        match self.get(&buffer_id) {
            Some(buffer) => Poll::Ready(Ok(buffer.clone())),
            None => Poll::Pending,
        }
    }
}
