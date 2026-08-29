// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutContext;
use crate::LayoutRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A shared counter of the bytes that layout strategies are holding but have not yet emitted.
///
/// Clones share the same counter, so a tracker can be handed to a writer before the write begins
/// and polled while it runs. Strategies report their own retained bytes with
/// [`Self::reserve`], which releases the reservation on drop.
#[derive(Clone, Debug, Default)]
pub struct BufferedBytesTracker(Arc<AtomicU64>);

impl BufferedBytesTracker {
    /// Creates a tracker with a zeroed counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of bytes currently retained by layout strategies.
    pub fn buffered_bytes(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// Records `bytes` as buffered until the returned reservation is dropped.
    pub fn reserve(&self, bytes: u64) -> BufferedBytesReservation {
        self.0.fetch_add(bytes, Ordering::Relaxed);
        BufferedBytesReservation {
            tracker: self.clone(),
            bytes,
        }
    }
}

/// An outstanding claim on a [`BufferedBytesTracker`], released when dropped.
#[derive(Debug)]
pub struct BufferedBytesReservation {
    tracker: BufferedBytesTracker,
    bytes: u64,
}

impl BufferedBytesReservation {
    /// Returns the number of bytes held by this reservation.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for BufferedBytesReservation {
    fn drop(&mut self) {
        self.tracker.0.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// State shared by every strategy participating in a single layout write.
///
/// Clones share the [`BufferedBytesTracker`] while retaining the array serialization context.
/// Passing this context through the strategy tree keeps writer-scoped state independent of the
/// strategy instances, which may be shared by multiple leaves or writers.
#[derive(Clone)]
pub struct LayoutWriterContext {
    array_ctx: ArrayContext,
    layout_ctx: LayoutContext,
    allowed_aggregates: Option<Arc<HashSet<AggregateFnId>>>,
    buffered_bytes: BufferedBytesTracker,
}

impl LayoutWriterContext {
    /// Creates a context for a layout write with a fresh buffered bytes tracker.
    pub fn new(array_ctx: ArrayContext) -> Self {
        Self {
            array_ctx,
            layout_ctx: LayoutContext::default(),
            allowed_aggregates: None,
            buffered_bytes: BufferedBytesTracker::new(),
        }
    }

    /// Restrict the aggregate functions this write may record, e.g. in a zone map.
    ///
    /// A write that would record an aggregate outside `allowed` fails, matching the array and
    /// layout contexts: a silently thinner zone map is a file that prunes worse than the
    /// caller asked for, with nothing in the output saying so. The id set is a plain set of
    /// ids — callers that source it from editions resolve it themselves.
    pub fn with_allowed_aggregates(mut self, allowed: HashSet<AggregateFnId>) -> Self {
        self.allowed_aggregates = Some(Arc::new(allowed));
        self
    }

    /// Returns whether `aggregate` may be recorded by this write. Unrestricted contexts
    /// permit every aggregate.
    pub fn allows_aggregate(&self, aggregate: &AggregateFnId) -> bool {
        self.allowed_aggregates
            .as_ref()
            .is_none_or(|allowed| allowed.contains(aggregate))
    }

    /// Replaces the buffered bytes tracker, so callers can observe the counter from outside the
    /// strategy tree.
    pub fn with_buffered_bytes_tracker(mut self, tracker: BufferedBytesTracker) -> Self {
        self.buffered_bytes = tracker;
        self
    }

    /// Returns the array serialization context.
    /// Intern layout encoding ids for any layout this write serializes itself, such as a page.
    ///
    /// Share the context the footer serializes through, so a nested layout flatbuffer indexes into
    /// the file's one dictionary rather than carrying a copy.
    pub fn with_layout_context(mut self, layout_ctx: LayoutContext) -> Self {
        self.layout_ctx = layout_ctx;
        self
    }

    /// Returns the layout encoding context shared with the footer.
    pub fn layout_ctx(&self) -> &LayoutContext {
        &self.layout_ctx
    }

    pub fn array_ctx(&self) -> &ArrayContext {
        &self.array_ctx
    }

    /// Returns the tracker that accounts for bytes retained by layout strategies.
    pub fn buffered_bytes_tracker(&self) -> &BufferedBytesTracker {
        &self.buffered_bytes
    }

    /// Returns the number of bytes currently retained by layout strategies.
    pub fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes.buffered_bytes()
    }

    /// Records `bytes` as retained by this write until the returned reservation is dropped.
    pub fn reserve_buffered_bytes(&self, bytes: u64) -> BufferedBytesReservation {
        self.buffered_bytes.reserve(bytes)
    }
}

impl From<ArrayContext> for LayoutWriterContext {
    fn from(array_ctx: ArrayContext) -> Self {
        Self::new(array_ctx)
    }
}

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

#[cfg(test)]
mod tests {
    use crate::strategy::BufferedBytesTracker;

    #[test]
    fn reservations_accumulate_and_release() {
        let tracker = BufferedBytesTracker::new();
        assert_eq!(tracker.buffered_bytes(), 0);

        let first = tracker.reserve(16);
        let second = tracker.reserve(32);
        assert_eq!(tracker.buffered_bytes(), 48);
        assert_eq!(first.bytes(), 16);

        drop(first);
        assert_eq!(tracker.buffered_bytes(), 32);

        drop(second);
        assert_eq!(tracker.buffered_bytes(), 0);
    }

    #[test]
    fn clones_share_the_same_counter() {
        let tracker = BufferedBytesTracker::new();
        let observer = tracker.clone();

        let reservation = tracker.reserve(8);
        assert_eq!(observer.buffered_bytes(), 8);

        drop(reservation);
        assert_eq!(observer.buffered_bytes(), 0);
    }
}
