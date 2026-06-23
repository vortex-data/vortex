// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayContext;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

// [layout writer]
/// Writes an ordered array stream into a layout tree and segment sink.
///
/// Layout strategies are writer-side extension points. Strategies may repartition, buffer,
/// collect columns, compute statistics, compress arrays, or delegate to child strategies before
/// finally emitting segments. They must preserve the logical row order represented by the
/// [`SequencePointer`]s in the input stream.
#[async_trait]
pub trait LayoutStrategy: 'static + Send + Sync {
    /// Asynchronously process an ordered stream of array chunks, emitting them into a sink and
    /// returning the [`Layout`][crate::Layout] instance that can be parsed to retrieve the data
    /// from rest.
    ///
    /// This trait uses the `#[async_trait]` attribute to denote that trait objects of this type
    /// can be `Box`ed or `Arc`ed and shared around. Commonly, these strategies are composed to
    /// form a operator of operations, each of which modifies the chunk stream in some way before
    /// passing the data on to a downstream writer.
    ///
    /// # Sequencing and EOF
    ///
    /// The `stream` parameter is a stream of ordered array chunks, each of which is associated
    /// with a sequence pointer that indicates its position in the overall array. By passing
    /// around these pointers (essentially vector clocks), the writer can support concurrent
    /// and parallel processing while maintaining a deterministic order of data in the file.
    ///
    /// The `eof` parameter is a guaranteed to be greater than all sequence pointers in the stream.
    ///
    /// Because child strategies can write to the end-of-file pointer, it is very important that
    /// **all strategies must await all children concurrently**. Otherwise it is possible to
    /// deadlock if one child is waiting to write to EOF while your strategy is preventing the
    /// stream from progressing to completion.
    ///
    /// # Blocking operations
    ///
    /// This is an async trait method, which will return a `BoxFuture` that you can await from
    /// any runtime. Implementations should avoid directly performing blocking work within the
    /// `write_stream`, and should instead spawn it onto an appropriate runtime or threadpool
    /// dedicated to such work.
    ///
    /// Such operations are common, and include things like compression and parsing large blobs
    /// of data, or serializing very large messages to flatbuffers.
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef>;

    /// Returns the number of bytes currently buffered by this strategy and any child strategies.
    ///
    /// This method allows tracking of data that has been processed by the strategy but not yet
    /// written to the underlying sink, providing more accurate estimates of final file size
    /// during write operations.
    fn buffered_bytes(&self) -> u64 {
        0
    }
}

#[async_trait]
impl LayoutStrategy for std::sync::Arc<dyn LayoutStrategy> {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        (**self)
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        (**self).buffered_bytes()
    }
}
// [layout writer]
