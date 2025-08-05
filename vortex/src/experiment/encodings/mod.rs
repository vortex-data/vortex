// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod bitpacked;
// mod compare;
// pub mod primitive;
// pub mod validity;

use crate::experiment::buffers::BufferId;
use crate::experiment::view::{BitView, ViewMut};
use bitvec::view::BitViewSized;
use std::ops::{Deref, Range};
use std::sync::atomic::AtomicUsize;
use std::task::Poll;
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};
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
    // TODO(ngates): we could introduce a `defined` parameter to indicate which elements are
    //  defined and will be interpreted by the parent. This would allow us to skip writing
    //  elements that are not defined, for example if the parent is a dense null validity encoding.
    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>>;
}

pub trait EvaluationContext {
    /// Get a buffer by its ID.
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>>;

    /// Pre-fetch buffers for future use (non-blocking hint).
    fn prefetch(&self, buffer_ids: &[BufferId]) {
        for &buffer_id in buffer_ids {
            let _ = self.buffer(buffer_id);
        }
    }

    /// Request a range of data from a buffer (for partial reads).
    fn buffer_range(
        &self,
        buffer_id: BufferId,
        range: Range<usize>,
    ) -> Poll<VortexResult<ByteBuffer>> {
        match self.buffer(buffer_id) {
            Poll::Ready(Ok(buffer)) => {
                let start = range.start;
                let end = range.end;
                if start < end && end <= buffer.len() {
                    Poll::Ready(Ok(buffer.slice(start..end)))
                } else {
                    Poll::Ready(Err(vortex_err!(
                        "Invalid range for buffer: {}..{}",
                        start,
                        end
                    )))
                }
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl EvaluationContext for () {
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        Poll::Ready(Err(vortex_err!(
            "EvaluationContext is not implemented for ()"
        )))
    }
}
