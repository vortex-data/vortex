// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutId;
use crate::LayoutRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

/// A shared counter of the bytes that a write accepted but did not yet send to the file.
///
/// A [`LayoutWriterContext`] has two of these counters:
///
/// - [`LayoutWriterContext::buffered_bytes_tracker`] counts compressed bytes. These are the
///   bytes that layout strategies hold after compression, and the bytes of the segments that the
///   layouts emitted, but that wait for earlier segments. Segments reach the file in sequence
///   order, so a layout that holds an early sequence ID can hold back the later segments.
/// - The [`SizeScope`] of [`LayoutWriterContext::size_scope`] counts the bytes that the write did
///   not compress yet, such as the data that a strategy coalesces before compression, in one
///   counter for each scope.
///
/// Each strategy reports its own retained bytes with [`Self::reserve`], which releases the
/// reservation on drop. The bytes written to the file, plus the compressed count, plus the
/// [estimated compressed size](SizeScope::estimated_compressed_bytes) of the uncompressed bytes,
/// are an estimate of the final file size.
///
/// Clones share the same counter, so a tracker can be handed to a writer before the write begins
/// and polled while it runs.
#[derive(Clone, Debug, Default)]
pub struct BufferedBytesTracker(Arc<AtomicU64>);

impl BufferedBytesTracker {
    /// Creates a tracker with a zeroed counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of bytes that the write accepted but did not yet send to the file.
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

/// The compression ratio to use before a write measures one.
///
/// A ratio of one assumes that the data does not compress. Thus, a size estimate made before the
/// first measurement does not underestimate the file.
const DEFAULT_COMPRESSION_RATIO: f64 = 1.0;

/// A shared measurement of a compression ratio: the compressed bytes out divided by the
/// uncompressed bytes in, over all the recorded samples.
#[derive(Clone, Debug, Default)]
struct CompressionRatio(Arc<CompressionRatioCounts>);

#[derive(Debug, Default)]
struct CompressionRatioCounts {
    uncompressed: AtomicU64,
    compressed: AtomicU64,
}

impl CompressionRatio {
    fn record(&self, uncompressed: u64, compressed: u64) {
        self.0
            .uncompressed
            .fetch_add(uncompressed, Ordering::Relaxed);
        self.0.compressed.fetch_add(compressed, Ordering::Relaxed);
    }

    /// Returns the ratio, or `None` if there is no sample.
    fn measured(&self) -> Option<f64> {
        let uncompressed = self.0.uncompressed.load(Ordering::Relaxed);
        if uncompressed == 0 {
            return None;
        }

        let compressed = self.0.compressed.load(Ordering::Relaxed);
        Some(compressed as f64 / uncompressed as f64)
    }
}

/// One scope of a [`SizeScope`] tree.
#[derive(Debug, Default)]
struct SizeScopeNode {
    bytes: BufferedBytesTracker,
    ratio: CompressionRatio,
    parent: Option<Arc<SizeScopeNode>>,
}

impl SizeScopeNode {
    /// Returns the ratio of this scope, or of the nearest ancestor that has a sample.
    fn ratio(&self) -> Option<f64> {
        self.ratio
            .measured()
            .or_else(|| self.parent.as_ref().and_then(|parent| parent.ratio()))
    }
}

/// The scopes of one write, and the ratio over all the samples of the write.
#[derive(Debug, Default)]
struct SizeScopeRegistry {
    scopes: Mutex<Vec<Arc<SizeScopeNode>>>,
    overall: CompressionRatio,
}

/// The accounting of the bytes that a write accepted but did not compress yet.
///
/// Data changes its form as it goes through the strategy tree. For example, a dictionary strategy
/// replaces the values of a column with codes, and the codes compress at a different ratio than
/// the values. Columns also compress at very different ratios. So one ratio for all the
/// uncompressed bytes of a write gives a bad estimate.
///
/// Thus the bytes are counted in a tree of scopes. Each scope has its own byte count and its own
/// compression ratio, and a strategy opens a child scope with [`Self::child`] for the data that
/// it holds in a new form. A strategy that holds uncompressed data reserves it on its scope with
/// [`Self::reserve`]. A strategy that compresses data records the result on the scope of the
/// data with [`Self::record`] or [`Self::record_probe`].
///
/// The estimated compressed size of the uncompressed bytes is the sum over all the scopes of the
/// bytes multiplied by the ratio of the scope. A scope without a sample uses the ratio of its
/// nearest ancestor with a sample, then the ratio over all the samples of the write, then one.
///
/// Clones refer to the same scope, and all the scopes of a tree share the same totals, so a scope
/// can be handed to a writer before the write begins and polled while it runs.
#[derive(Clone, Debug)]
pub struct SizeScope {
    node: Arc<SizeScopeNode>,
    registry: Arc<SizeScopeRegistry>,
}

impl Default for SizeScope {
    fn default() -> Self {
        Self::new()
    }
}

impl SizeScope {
    /// Creates the root scope of a new tree.
    pub fn new() -> Self {
        let node = Arc::new(SizeScopeNode::default());
        let registry = SizeScopeRegistry::default();
        registry.scopes.lock().push(Arc::clone(&node));
        Self {
            node,
            registry: Arc::new(registry),
        }
    }

    /// Creates a scope for data in a new form, which inherits the ratio of this scope until it
    /// has a sample of its own.
    pub fn child(&self) -> Self {
        let node = Arc::new(SizeScopeNode {
            parent: Some(Arc::clone(&self.node)),
            ..SizeScopeNode::default()
        });

        let mut scopes = self.registry.scopes.lock();
        // Remove the scopes that no strategy uses and that hold no bytes.
        scopes.retain(|scope| Arc::strong_count(scope) > 1 || scope.bytes.buffered_bytes() > 0);
        scopes.push(Arc::clone(&node));
        drop(scopes);

        Self {
            node,
            registry: Arc::clone(&self.registry),
        }
    }

    /// Records `bytes` of uncompressed data in this scope until the returned reservation is
    /// dropped.
    pub fn reserve(&self, bytes: u64) -> BufferedBytesReservation {
        self.node.bytes.reserve(bytes)
    }

    /// Records that `uncompressed` bytes of a chunk in this scope compressed to `compressed`
    /// bytes.
    pub fn record(&self, uncompressed: u64, compressed: u64) {
        self.node.ratio.record(uncompressed, compressed);
        self.registry.overall.record(uncompressed, compressed);
    }

    /// Records the result of a probe that compressed `uncompressed` bytes of the data in this
    /// scope to `compressed` bytes.
    ///
    /// A strategy compresses a probe early in the write, for example to choose an encoding. The
    /// probe gives a ratio before the first full chunk is compressed. A probe counts the same as a
    /// chunk, and the chunks soon outweigh it.
    pub fn record_probe(&self, uncompressed: u64, compressed: u64) {
        self.record(uncompressed, compressed);
    }

    /// Returns the uncompressed bytes in all the scopes of the tree.
    pub fn uncompressed_bytes(&self) -> u64 {
        self.registry
            .scopes
            .lock()
            .iter()
            .map(|scope| scope.bytes.buffered_bytes())
            .sum()
    }

    /// Returns the compressed bytes divided by the uncompressed bytes over all the samples of the
    /// tree, or one if there is no sample.
    pub fn compression_ratio(&self) -> f64 {
        self.registry
            .overall
            .measured()
            .unwrap_or(DEFAULT_COMPRESSION_RATIO)
    }

    /// Returns the estimated compressed size of the uncompressed bytes in all the scopes of the
    /// tree.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the ratios are not negative, and the cast saturates at u64::MAX"
    )]
    pub fn estimated_compressed_bytes(&self) -> u64 {
        let fallback = self.compression_ratio();
        let estimate: f64 = self
            .registry
            .scopes
            .lock()
            .iter()
            .map(|scope| {
                let bytes = scope.bytes.buffered_bytes();
                if bytes == 0 {
                    return 0.0;
                }
                bytes as f64 * scope.ratio().unwrap_or(fallback)
            })
            .sum();
        estimate.round() as u64
    }
}

/// State shared by every strategy participating in a single layout write.
///
/// Clones share the [`BufferedBytesTracker`] and the [`SizeScope`] while retaining the array
/// serialization context.
/// Passing this context through the strategy tree keeps writer-scoped state independent of the
/// strategy instances, which may be shared by multiple leaves or writers.
#[derive(Clone)]
pub struct LayoutWriterContext {
    array_ctx: ArrayContext,
    allowed_aggregates: Option<Arc<HashSet<AggregateFnId>>>,
    allowed_layouts: Option<Arc<HashSet<LayoutId>>>,
    nullable_ancestor: bool,
    buffered_bytes: BufferedBytesTracker,
    size_scope: SizeScope,
}

impl LayoutWriterContext {
    /// Creates a context for a layout write with a fresh buffered bytes tracker and size scope.
    pub fn new(array_ctx: ArrayContext) -> Self {
        Self {
            array_ctx,
            allowed_aggregates: None,
            allowed_layouts: None,
            nullable_ancestor: false,
            buffered_bytes: BufferedBytesTracker::new(),
            size_scope: SizeScope::new(),
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

    /// Restrict the layouts this write may produce.
    ///
    /// Strategies that can write more than one layout use this to select a layout that the
    /// readers of the file know. The footer serializer still rejects a layout that is not
    /// permitted.
    pub fn with_allowed_layouts(mut self, allowed: HashSet<LayoutId>) -> Self {
        self.allowed_layouts = Some(Arc::new(allowed));
        self
    }

    /// Returns whether this write may produce `layout`. Unrestricted contexts permit every
    /// layout.
    pub fn allows_layout(&self, layout: &LayoutId) -> bool {
        self.allowed_layouts
            .as_ref()
            .is_none_or(|allowed| allowed.contains(layout))
    }

    /// Marks this write as nested under a nullable container, such as a field of a nullable
    /// struct.
    ///
    /// Container writers store their children unmasked: a field holds a value even in rows where
    /// the containing struct is null, and the struct's validity is written separately. Statistics
    /// computed over such a child therefore see values the logical column does not contain, so
    /// writers that record statistics must omit them rather than record inexact values.
    pub fn with_nullable_ancestor(mut self) -> Self {
        self.nullable_ancestor = true;
        self
    }

    /// Returns whether a nullable container above this write may hide some of its values.
    pub fn has_nullable_ancestor(&self) -> bool {
        self.nullable_ancestor
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
    ///
    /// Strategies that hold compressed data reserve it here. Strategies that hold data that is
    /// not compressed yet reserve it on [`Self::size_scope`].
    pub fn buffered_bytes_tracker(&self) -> &BufferedBytesTracker {
        &self.buffered_bytes
    }

    /// Replaces the scope of the bytes that the write accepted but did not compress yet, so
    /// callers can observe the counts from outside the strategy tree.
    pub fn with_size_scope(mut self, scope: SizeScope) -> Self {
        self.size_scope = scope;
        self
    }

    /// Returns a context in which the uncompressed bytes count in a new child of the current
    /// [`SizeScope`].
    ///
    /// A strategy calls this for the data that it holds or emits in a new form, so that the data
    /// gets its own compression ratio.
    pub fn with_child_size_scope(mut self) -> Self {
        self.size_scope = self.size_scope.child();
        self
    }

    /// Returns the scope of the bytes that the write accepted but did not compress yet.
    ///
    /// Strategies that hold data before compression, such as the repartition strategy and the
    /// input of the compressing strategy, reserve their bytes here. Strategies that compress data
    /// record the compression ratio here. See [`SizeScope`].
    pub fn size_scope(&self) -> &SizeScope {
        &self.size_scope
    }

    /// Returns the number of bytes that the write accepted but did not yet send to the file.
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
    ///
    /// # Writer context
    ///
    /// The `ctx` parameter carries array serialization state and writer-scoped accounting, and
    /// also facts about the position of this write in the layout tree, such as the aggregates the
    /// write may record and whether a nullable container above it hides some of its values. Those
    /// facts only reach a child because its parent passed them on, so **strategies must hand the
    /// `ctx` they received, or a clone of it, to every child they write**. Building a fresh
    /// [`LayoutWriterContext`] inside a strategy silently discards them.
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
    use crate::strategy::SizeScope;

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
    fn size_scope_defaults_to_a_ratio_of_one() {
        let scope = SizeScope::new();
        let reservation = scope.reserve(100);
        assert_eq!(scope.compression_ratio(), 1.0);
        assert_eq!(scope.uncompressed_bytes(), 100);
        assert_eq!(scope.estimated_compressed_bytes(), 100);

        drop(reservation);
        assert_eq!(scope.estimated_compressed_bytes(), 0);
    }

    #[test]
    fn size_scopes_use_their_own_ratio_or_the_nearest_sample() {
        let root = SizeScope::new();
        let values = root.child();
        let codes = values.child();
        let other = root.child();

        values.record_probe(100, 10);
        let _values_bytes = values.reserve(1000);
        let _codes_bytes = codes.reserve(1000);
        let _other_bytes = other.reserve(1000);

        // The codes inherit the ratio of the values, and the other scope uses the ratio over all
        // the samples.
        assert_eq!(root.estimated_compressed_bytes(), 300);

        codes.record(100, 50);
        // The ratio over all the samples is 60 / 200.
        assert_eq!(root.estimated_compressed_bytes(), 100 + 500 + 300);
        assert_eq!(root.compression_ratio(), 0.3);
        assert_eq!(root.uncompressed_bytes(), 3000);
    }

    #[test]
    fn unused_size_scopes_are_removed() {
        let root = SizeScope::new();
        let child = root.child();
        let reservation = child.reserve(10);
        drop(child);

        // The bytes of a dropped scope still count until the reservation is released.
        let _other = root.child();
        assert_eq!(root.uncompressed_bytes(), 10);

        drop(reservation);
        let _another = root.child();
        assert_eq!(root.uncompressed_bytes(), 0);
        assert_eq!(root.registry.scopes.lock().len(), 3);
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
