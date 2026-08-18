// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::dtype::DType;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;

/// A shared counter of the logical byte size of arrays retained by layout strategies.
///
/// Clones share the same counter, so a tracker can be handed to a writer before the write begins
/// and polled while it runs. This includes arrays queued for asynchronous strategy work, but not
/// allocator overhead, statistics-builder state, or buffering performed by the output sink.
/// Strategies report their own retained bytes with [`Self::reserve`], which releases the
/// reservation on drop.
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
    allowed_aggregates: Option<Arc<HashSet<AggregateFnId>>>,
    buffered_bytes: BufferedBytesTracker,
}

impl LayoutWriterContext {
    /// Creates a context for a layout write with a fresh buffered bytes tracker.
    pub fn new(array_ctx: ArrayContext) -> Self {
        Self {
            array_ctx,
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

/// Creates a stateful writer node in a layout writer tree.
///
/// Layout strategies are writer-side extension points. Strategies may repartition, buffer,
/// collect columns, compute statistics, compress arrays, or delegate to child strategies before
/// finally emitting segments. Each node receives arrays in logical row order.
#[async_trait]
pub trait LayoutStrategy: 'static + Send + Sync {
    /// Construct a writer for one dtype.
    ///
    /// The `ctx` parameter carries both array serialization state and writer-scoped accounting
    /// through every child strategy. Expensive work and independent children may run concurrently;
    /// [`SequenceId`] preserves the logical segment order for sinks that require it.
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>>;

    /// Drive this strategy from an existing stream.
    ///
    /// This compatibility adapter is useful at stream-owning API boundaries and in tests. Layout
    /// strategies compose by constructing child writers and calling [`LayoutWriter::write`]
    /// directly.
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        eof: crate::sequence::SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let mut writer = self.new_writer(ctx, segment_sink, stream.dtype().clone(), session)?;
        while let Some(chunk) = stream.next().await {
            let (sequence_id, chunk) = chunk?;
            writer.write(sequence_id, chunk).await?;
        }
        drop(stream);
        writer.finish(eof.downgrade()).await?;
        writer.close().await
    }
}

/// A stateful node in a push-based layout writer tree.
#[async_trait]
pub trait LayoutWriter: Send {
    /// Push one ordered array into this node.
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()>;

    /// Establish the terminal input barrier and drain retained arrays and asynchronous work.
    ///
    /// Composite nodes call this on every child before closing any child. This lets strategies
    /// such as zoned writers commit all primary data across sibling columns before emitting
    /// derived metadata. This is called exactly once, and no arrays may be written afterward.
    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()>;

    /// Consume this already-finished node and return the completed layout.
    async fn close(self: Box<Self>) -> VortexResult<LayoutRef>;
}

enum ActorMessage {
    Write(SequenceId, ArrayRef, BufferedBytesReservation),
    Finish(SequenceId),
}

const CHILD_WRITER_QUEUE_CAPACITY: usize = 1;

/// Drives one independent child writer on the runtime. Its bounded mailbox lets backpressure
/// from the segment sink propagate up to the public writer while retaining enough slack for
/// sibling writers to make progress independently. Queued arrays are accounted by
/// [`BufferedBytesTracker`].
pub struct LayoutWriterActor {
    sender: Option<kanal::AsyncSender<ActorMessage>>,
    task: Option<Task<VortexResult<LayoutRef>>>,
    layout: Option<LayoutRef>,
    buffered_bytes: BufferedBytesTracker,
}

impl LayoutWriterActor {
    /// Spawn a writer with a bounded input mailbox on `handle`.
    pub fn spawn(
        mut writer: Box<dyn LayoutWriter>,
        buffered_bytes: BufferedBytesTracker,
        handle: &Handle,
    ) -> Self {
        let (sender, receiver) = kanal::bounded_async::<ActorMessage>(CHILD_WRITER_QUEUE_CAPACITY);
        let task = handle.spawn(async move {
            loop {
                match receiver.recv().await {
                    Ok(ActorMessage::Write(sequence_id, chunk, reservation)) => {
                        writer.write(sequence_id, chunk).await?;
                        drop(reservation);
                    }
                    Ok(ActorMessage::Finish(sequence_id)) => {
                        writer.finish(sequence_id).await?;
                        return writer.close().await;
                    }
                    Err(_) => return Err(vortex_err!("layout child sender dropped before finish")),
                }
            }
        });
        Self {
            sender: Some(sender),
            task: Some(task),
            layout: None,
            buffered_bytes,
        }
    }

    /// Push an array into the child, waiting when its mailbox is full.
    pub async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let reservation = self.buffered_bytes.reserve(chunk.nbytes());
        self.sender
            .as_ref()
            .ok_or_else(|| vortex_err!("layout child is already finished"))?
            .send(ActorMessage::Write(sequence_id, chunk, reservation))
            .await
            .map_err(|_| vortex_err!("layout child finished before all chunks were pushed"))
    }

    /// Finish the child and wait for its layout to be produced.
    pub async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        if self.layout.is_some() {
            return Ok(());
        }
        self.sender
            .take()
            .ok_or_else(|| vortex_err!("layout child sender is missing"))?
            .send(ActorMessage::Finish(sequence_id))
            .await
            .map_err(|_| vortex_err!("layout child finished before its terminal barrier"))?;
        let task = self
            .task
            .take()
            .ok_or_else(|| vortex_err!("layout child task is missing"))?;
        self.layout = Some(task.await?);
        Ok(())
    }

    /// Take the completed layout after [`Self::finish`].
    pub fn take_layout(&mut self) -> VortexResult<LayoutRef> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("layout child was not finished"))
    }
}

/// Drive a push-based layout writer from an existing sequential stream.
///
/// This is an adapter for callers and tests that already own streams; strategies themselves are
/// composed exclusively through [`LayoutWriter::write`].
pub async fn write_stream(
    strategy: &dyn LayoutStrategy,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    mut stream: SendableSequentialStream,
    eof: crate::sequence::SequencePointer,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    let mut writer = strategy.new_writer(ctx, segment_sink, stream.dtype().clone(), session)?;
    while let Some(chunk) = stream.next().await {
        let (sequence_id, chunk) = chunk?;
        writer.write(sequence_id, chunk).await?;
    }
    drop(stream);
    writer.finish(eof.downgrade()).await?;
    writer.close().await
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

impl LayoutStrategy for LayoutStrategyEncodingValidator {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(Box::new(EncodingValidatorWriter {
            child: self.child.new_writer(ctx, segment_sink, dtype, session)?,
            allowed_encodings: Arc::clone(&self.allowed_encodings),
        }))
    }
}

impl LayoutStrategy for Arc<dyn LayoutStrategy> {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        (**self).new_writer(ctx, segment_sink, dtype, session)
    }
}

#[async_trait]
impl LayoutWriter for EncodingValidatorWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let chunk = chunk.normalize(&mut NormalizeOptions {
            allowed: &self.allowed_encodings,
            operation: Operation::Error,
        })?;
        self.child.write(sequence_id, chunk).await
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        self.child.finish(sequence_id).await
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        self.child.close().await
    }
}

struct EncodingValidatorWriter {
    child: Box<dyn LayoutWriter>,
    allowed_encodings: Arc<HashSet<ArrayId>>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::FutureExt;
    use tokio::sync::Semaphore;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_io::session::RuntimeSessionExt;

    use crate::LayoutRef;
    use crate::LayoutWriter;
    use crate::children::OwnedLayoutChildren;
    use crate::layouts::chunked::ChunkedLayout;
    use crate::sequence::SequenceId;
    use crate::strategy::BufferedBytesTracker;
    use crate::strategy::CHILD_WRITER_QUEUE_CAPACITY;
    use crate::strategy::LayoutWriterActor;
    use crate::test::new_session;

    struct BlockingWriter {
        permits: Arc<Semaphore>,
    }

    #[async_trait]
    impl LayoutWriter for BlockingWriter {
        async fn write(&mut self, _sequence_id: SequenceId, _chunk: ArrayRef) -> VortexResult<()> {
            self.permits
                .acquire()
                .await
                .map_err(|_| vortex_err!("test semaphore closed"))?
                .forget();
            Ok(())
        }

        async fn finish(&mut self, _sequence_id: SequenceId) -> VortexResult<()> {
            Ok(())
        }

        async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
            Ok(ChunkedLayout::new(
                0,
                DType::Bool(Nullability::NonNullable),
                OwnedLayoutChildren::layout_children(vec![]),
            )
            .into_layout())
        }
    }

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

    #[tokio::test]
    async fn child_writer_mailbox_applies_backpressure_when_full() -> VortexResult<()> {
        let permits = Arc::new(Semaphore::new(0));
        let tracker = BufferedBytesTracker::new();
        let session = new_session().with_tokio();
        let mut actor = LayoutWriterActor::spawn(
            Box::new(BlockingWriter {
                permits: Arc::clone(&permits),
            }),
            tracker.clone(),
            &session.handle(),
        );
        let chunk = buffer![1u64].into_array();
        let chunk_bytes = chunk.nbytes();
        let mut sequence = SequenceId::root();

        for _ in 0..=CHILD_WRITER_QUEUE_CAPACITY {
            actor.write(sequence.advance(), chunk.clone()).await?;
        }
        assert_eq!(
            tracker.buffered_bytes(),
            (CHILD_WRITER_QUEUE_CAPACITY as u64 + 1) * chunk_bytes
        );

        assert!(
            actor
                .write(sequence.advance(), chunk)
                .now_or_never()
                .is_none(),
            "a full mailbox must block the producer"
        );

        permits.add_permits(CHILD_WRITER_QUEUE_CAPACITY + 1);
        actor.finish(sequence.advance()).await?;
        actor.take_layout()?;
        assert_eq!(tracker.buffered_bytes(), 0);
        Ok(())
    }
}
