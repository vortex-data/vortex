// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::array_tree::ArrayTreeLayout;
use crate::layouts::array_tree::flat::ArrayTreeFlatLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentId;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;

/// Side channel for shipping `(segment_id, compact_tree)` pairs from leaf strategies to the
/// collector strategy.
///
/// Each leaf pushes after `segment_sink.write` resolves (so the leaf's `SequenceId` has been
/// dropped before we touch the sink). The collector drains the sink only after the entire
/// data subtree has completed, which means every leaf has already pushed.
type Sink = Arc<Mutex<Vec<(SegmentId, ByteBuffer)>>>;

/// Creates a cooperating pair of strategies for array tree collection.
///
/// Returns `(collector, leaf)` where:
/// - `leaf` replaces [`FlatLayoutStrategy`] in the data pipeline — it serializes chunks,
///   produces compact flatbuffers, and pushes them onto the shared sink.
/// - `collector` wraps the data pipeline — after data is written, it drains the sink and
///   writes the collected pairs as a struct array (`{segment_id, compact_tree}`) via the
///   configured `array_trees_strategy`.
pub fn writer(
    flat: FlatLayoutStrategy,
    array_trees_strategy: Arc<dyn LayoutStrategy>,
) -> (ArrayTreeCollectorStrategy, ArrayTreeFlatStrategy) {
    let sink: Sink = Arc::new(Mutex::new(Vec::new()));
    let leaf = ArrayTreeFlatStrategy {
        flat,
        sink: Arc::clone(&sink),
    };
    let collector = ArrayTreeCollectorStrategy {
        child: None,
        array_trees_strategy,
        sink,
    };
    (collector, leaf)
}

/// Leaf strategy (TX) that replaces [`FlatLayoutStrategy`].
///
/// For each chunk, it produces both the compact flatbuffer (encoding tree + buffer
/// descriptors, no stats) and the full data segment, then pushes `(segment_id, compact_tree)`
/// onto the shared sink for the collector to consume.
#[derive(Clone)]
pub struct ArrayTreeFlatStrategy {
    flat: FlatLayoutStrategy,
    sink: Sink,
}

#[async_trait]
impl LayoutStrategy for ArrayTreeFlatStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        _eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let ctx = ctx.clone();
        let Some(chunk) = stream.next().await else {
            vortex_bail!("array tree flat layout needs a single chunk");
        };
        let (sequence_id, chunk) = chunk?;

        let row_count = chunk.len() as u64;

        // Normalize if needed (delegate to flat strategy's normalization).
        let chunk = if let Some(allowed) = &self.flat.allowed_encodings {
            use vortex_array::normalize::NormalizeOptions;
            use vortex_array::normalize::Operation;
            chunk.normalize(&mut NormalizeOptions {
                allowed,
                operation: Operation::Error,
            })?
        } else {
            chunk
        };

        // Produce the compact flatbuffer (no stats, with buffer descriptors).
        let compact_tree = chunk.serialize_array_tree(
            &ctx,
            session,
            &SerializeOptions {
                offset: 0,
                include_padding: self.flat.include_padding,
            },
        )?;

        // Full serialization for the data segment.
        let buffers = chunk.serialize(
            &ctx,
            session,
            &SerializeOptions {
                offset: 0,
                include_padding: self.flat.include_padding,
            },
        )?;
        assert!(buffers.len() >= 2);

        // IMPORTANT ORDERING CONSTRAINT: write the segment first, then push to the sink.
        //
        // `segment_sink.write` consumes our `SequenceId` and only drops it on return. Pushing
        // to the sink before that point would risk holding the sink mutex while later leaves
        // are blocked on `SequenceId::collapse`, creating a dependency from "later leaf is
        // ready to write" → "earlier leaf must drop its SequenceId" → "earlier leaf must
        // finish its sink push" → mutex contention with the later leaf. Doing the push after
        // `await?` resolves means our SequenceId is already gone before we touch the sink.
        let segment_id = segment_sink.write(sequence_id, buffers).await?;
        self.sink.lock().push((segment_id, compact_tree));

        let None = stream.next().await else {
            vortex_bail!("array tree flat layout received stream with more than a single chunk");
        };

        Ok(ArrayTreeFlatLayout::new(FlatLayout::new(
            row_count,
            stream.dtype().clone(),
            segment_id,
            ReadContext::new(ctx.to_ids()),
        ))
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        0
    }
}

/// Collector strategy (RX) that wraps the data pipeline.
///
/// After the data child completes, drains the shared sink and writes the collected
/// `(segment_id, compact_tree)` pairs as a struct array via the configured
/// `array_trees_strategy`.
pub struct ArrayTreeCollectorStrategy {
    child: Option<Arc<dyn LayoutStrategy>>,
    array_trees_strategy: Arc<dyn LayoutStrategy>,
    sink: Sink,
}

impl ArrayTreeCollectorStrategy {
    /// Sets the data child pipeline that this collector wraps.
    pub fn wrap(mut self, child: impl LayoutStrategy) -> Self {
        self.child = Some(Arc::new(child));
        self
    }
}

#[async_trait]
impl LayoutStrategy for ArrayTreeCollectorStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let Some(child) = self.child.as_ref() else {
            vortex_bail!("ArrayTreeCollectorStrategy must have a child set via wrap()")
        };

        // Data segments get earlier sequence IDs than array tree segments.
        let data_eof = eof.split_off();

        let data_layout = child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        // By the time the data subtree future resolves, every leaf has finished its
        // `segment_sink.write().await?` and pushed onto the sink. Drain it now.
        let mut entries = std::mem::take(&mut *self.sink.lock());

        // Sort by segment ID so the on-disk order matches segment-write order — this gives
        // good locality and predictable lookup-table layout.
        entries.sort_by_key(|(seg, _)| *seg);

        // Build a struct array of {segment_id: u32, compact_tree: bytes}.
        let nrows = entries.len();
        let segment_ids: Buffer<u32> = entries.iter().map(|(seg, _)| **seg).collect();
        let segment_ids_array =
            PrimitiveArray::new(segment_ids, Validity::NonNullable).into_array();

        let mut tree_builder =
            VarBinViewBuilder::with_capacity(DType::Binary(Nullability::NonNullable), nrows);
        for (_, tree) in &entries {
            tree_builder.append_value(tree.as_slice());
        }
        let trees_array = tree_builder.finish().into_array();

        let array_trees_array = StructArray::try_new(
            vec![
                FieldName::from("segment_id"),
                FieldName::from("compact_tree"),
            ]
            .into(),
            vec![segment_ids_array, trees_array],
            nrows,
            Validity::NonNullable,
        )?
        .into_array();

        // Write the struct array via the array_trees strategy.
        let trees_stream = array_trees_array
            .to_array_stream()
            .sequenced(eof.split_off());
        let array_trees_layout = self
            .array_trees_strategy
            .write_stream(ctx, segment_sink, trees_stream, eof, session)
            .await?;

        Ok(ArrayTreeLayout::new(data_layout, array_trees_layout).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.as_ref().map(|c| c.buffered_bytes()).unwrap_or(0)
            + self.array_trees_strategy.buffered_bytes()
    }
}
