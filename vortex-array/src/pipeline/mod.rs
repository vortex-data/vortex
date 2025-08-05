// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This crate contains experiments into vectorized data processing within Vortex.
//!
//! Vectors are fixed-size chunks of data in Vortex represented in a canonical form. The size
//! is a compile-time constant [`N`], which is set to 1024 elements by default.

pub mod bits;
pub mod buffers;
pub mod common;
// pub mod export;
pub mod selection;
pub mod vector;
pub mod view;

/// The number of elements in each step of a Vortex evaluation pipeline.
pub const N: usize = 1024;

use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferId;
use crate::pipeline::view::ViewMut;
use std::ops::Range;
use std::task::Poll;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};

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
pub trait Pipeline {
    /// Seek the pipeline to a specific chunk offset.
    ///
    /// i.e. the resulting row offset is `idx * N`, where `N` is the number of elements in a chunk.
    ///
    /// The reason for a separate seek function (vs passing an offset directly to `step`) is that
    /// it allows the pipeline to optimize for sequential access patterns, which is common in
    /// many encodings. For example, a run-length encoding can efficiently seek to the start of a
    /// chunk without needing to perform a full binary search of the ends in each step.
    ///
    // TODO(ngates): should this be `skip(n)` instead? Depends if we want to support going
    //  backwards?
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()>;

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
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>>;
}

pub trait ToPipeline {
    /// Create a pipeline.
    fn to_pipeline(&self) -> Box<dyn Pipeline>;
}

pub trait PipelineContext {
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

impl PipelineContext for () {
    fn buffer(&self, _buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        Poll::Ready(Err(vortex_err!(
            "EvaluationContext is not implemented for ()"
        )))
    }
}
