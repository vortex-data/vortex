// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayContext;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::LayoutRef;
use crate::segments::SegmentSink;
use crate::sequence::{SendableSequentialStream, SequencePointer};

// [layout writer]
#[async_trait]
pub trait LayoutStrategy: Send + Sync {
    /// Asynchronously process an ordered stream of array chunks, emitting them into a sink and
    /// returning the [`Layout`][crate::Layout] instance that can be parsed to retrieve the data
    /// from rest.
    ///
    /// This trait uses the `#[async_trait]` attribute to denote that trait objects of this type
    /// can be `Box`ed or `Arc`ed and shared around. Commonly, these strategies are composed to
    /// form a pipeline of operations, each of which modifies the chunk stream in some way before
    /// passing the data on to a downstream writer.
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
    ///
    /// Consider accepting a [`TaskExecutor`][crate::TaskExecutor] as an input to your strategy
    /// to support spawning this work in the background.
    async fn write_stream<'a>(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        stream: SendableSequentialStream<'a>,
        eof: SequencePointer,
        handle: Handle<'a>,
    ) -> VortexResult<LayoutRef>;
}
// [layout writer]
