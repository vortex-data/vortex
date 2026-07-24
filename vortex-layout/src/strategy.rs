// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// State shared by every strategy participating in a single layout write.
///
/// Clones share the buffered-byte counter while retaining the array serialization context. Passing
/// this context through the strategy tree keeps writer-scoped state independent of the strategy
/// instances, which may be shared by multiple leaves or writers.
#[derive(Clone)]
pub struct LayoutWriterContext {
    array_ctx: ArrayContext,
    buffered_bytes: Arc<AtomicU64>,
}

impl LayoutWriterContext {
    /// Creates a context for a layout write.
    pub fn new(array_ctx: ArrayContext) -> Self {
        Self {
            array_ctx,
            buffered_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the array serialization context.
    pub fn array_ctx(&self) -> &ArrayContext {
        &self.array_ctx
    }

    /// Returns the number of bytes currently retained by layout strategies.
    pub fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn add_buffered_bytes(&self, bytes: u64) {
        self.buffered_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn remove_buffered_bytes(&self, bytes: u64) {
        self.buffered_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }
}

impl From<ArrayContext> for LayoutWriterContext {
    fn from(array_ctx: ArrayContext) -> Self {
        Self::new(array_ctx)
    }
}

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
    /// The `ctx` parameter carries both array serialization state and writer-scoped accounting
    /// through every child strategy.
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
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef>;
}

/// A layout strategy wrapper that rejects arrays containing encodings outside an allow-list.
///
/// Canonical encodings are always permitted. Every chunk is recursively validated before it is
/// passed to the wrapped strategy.
#[derive(Clone)]
pub struct LayoutStrategyEncodingValidator {
    child: Arc<dyn LayoutStrategy>,
    allowed_encodings: Arc<HashSet<ArrayId>>,
}

impl LayoutStrategyEncodingValidator {
    /// Creates a validator around `child` using the supplied encoding allow-list.
    pub fn new<S: LayoutStrategy>(child: S, allowed_encodings: HashSet<ArrayId>) -> Self {
        Self {
            child: Arc::new(child),
            allowed_encodings: Arc::new(allowed_encodings),
        }
    }
}

#[async_trait]
impl LayoutStrategy for LayoutStrategyEncodingValidator {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let allowed_encodings = Arc::clone(&self.allowed_encodings);
        let stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let chunk = chunk.normalize(&mut NormalizeOptions {
                allowed: &allowed_encodings,
                operation: Operation::Error,
            })?;
            Ok((sequence_id, chunk))
        });

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                session,
            )
            .await
    }
}

#[async_trait]
impl LayoutStrategy for Arc<dyn LayoutStrategy> {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        (**self)
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await
    }
}
// [layout writer]
