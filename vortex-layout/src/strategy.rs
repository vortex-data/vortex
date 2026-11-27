// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ArrayContext;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_error::VortexResult;
use vortex_io::Backpressure;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutId;
use crate::LayoutRef;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

/// State shared by every strategy participating in a single layout write.
///
/// Clones share the [`Backpressure`] while retaining the array serialization context.
/// Passing this context through the strategy tree keeps writer-scoped state independent of the
/// strategy instances, which may be shared by multiple leaves or writers.
#[derive(Clone)]
pub struct LayoutWriterContext {
    array_ctx: ArrayContext,
    allowed_aggregates: Option<Arc<HashSet<AggregateFnId>>>,
    allowed_layouts: Option<Arc<HashSet<LayoutId>>>,
    nullable_ancestor: bool,
    backpressure: Backpressure,
}

impl LayoutWriterContext {
    /// Creates a context for a layout write with a fresh [`Backpressure`].
    pub fn new(array_ctx: ArrayContext) -> Self {
        Self {
            array_ctx,
            allowed_aggregates: None,
            allowed_layouts: None,
            nullable_ancestor: false,
            backpressure: Backpressure::new(),
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

    /// Returns the array serialization context.
    pub fn array_ctx(&self) -> &ArrayContext {
        &self.array_ctx
    }

    /// Replaces the byte accounting of this write, so callers can observe the counts from outside
    /// the strategy tree.
    pub fn with_backpressure(mut self, backpressure: Backpressure) -> Self {
        self.backpressure = backpressure;
        self
    }

    /// Returns a context in which the pending bytes count in a new child scope of the current
    /// [`Backpressure`].
    ///
    /// A strategy calls this for the data that it holds or emits in a new form, so that the data
    /// gets its own compression ratio.
    pub fn with_child_scope(mut self) -> Self {
        self.backpressure = self.backpressure.child();
        self
    }

    /// Returns the byte accounting of this write, in the scope of this context.
    ///
    /// A strategy reserves here the data that it holds: pending before compression, buffered
    /// after it. A strategy that compresses data records the compression ratio here.
    pub fn backpressure(&self) -> &Backpressure {
        &self.backpressure
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
