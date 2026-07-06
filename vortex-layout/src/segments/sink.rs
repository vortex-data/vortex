// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::sequence::SequenceId;

/// Shared writer-side segment sink.
pub type SegmentSinkRef = Arc<dyn SegmentSink>;

#[async_trait]
/// Assigns segment ids and writes segment buffers during layout writing.
///
/// Segment sinks are responsible for preserving any ordering guarantees required by the storage
/// backend. The [`SequenceId`] argument lets sinks serialize id assignment while still allowing
/// upstream layout strategies to do work concurrently.
pub trait SegmentSink: Send + Sync {
    /// Write the given data into a segment, ordered based on the provided sequence identifier.
    ///
    /// Implementations of this trait should call [`SequenceId::collapse`] on the provided
    /// `sequence_id` if they need to ensure that the segment IDs are monotonically increasing.
    /// While they hold onto the returned `SequenceId`, they can be sure that no other subsequent
    /// calls to [`SequenceId::collapse`] will complete.
    ///
    /// If they do not require ordered segment IDs, for example if segments are stored in
    /// random-access key/values storage, then the sequence ID can be dropped and the segment
    /// written immediately.
    async fn write(
        &self,
        sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId>;
}
