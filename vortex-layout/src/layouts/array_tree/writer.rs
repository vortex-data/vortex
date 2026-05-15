// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
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
use crate::layouts::array_tree::flat::ArrayTreeFlat;
use crate::layouts::array_tree::flat::ArrayTreeFlatLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentId;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;

/// Creates a cooperating pair of strategies for array tree collection.
///
/// Returns `(collector, leaf)` where:
/// - `leaf` replaces [`FlatLayoutStrategy`] in the data pipeline — it serializes chunks and
///   produces compact flatbuffers attached to [`ArrayTreeFlatLayout`].
/// - `collector` wraps the data pipeline — after data is written, it walks the layout tree to
///   collect compact flatbuffers from all [`ArrayTreeFlatLayout`] leaves and writes them as a
///   struct array (`{segment_id, compact_tree}`) via the configured `array_trees_strategy`.
pub fn writer(
    flat: FlatLayoutStrategy,
    array_trees_strategy: Arc<dyn LayoutStrategy>,
) -> (ArrayTreeCollectorStrategy, ArrayTreeFlatStrategy) {
    let leaf = ArrayTreeFlatStrategy { flat };
    let collector = ArrayTreeCollectorStrategy {
        child: None,
        array_trees_strategy,
    };
    (collector, leaf)
}

/// Leaf strategy (TX) that replaces [`FlatLayoutStrategy`].
///
/// For each chunk, it produces both the compact flatbuffer (encoding tree + buffer
/// descriptors, no stats) and the full data segment, and returns an [`ArrayTreeFlatLayout`]
/// with the compact tree attached for later collection.
#[derive(Clone)]
pub struct ArrayTreeFlatStrategy {
    flat: FlatLayoutStrategy,
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
        let segment_id = segment_sink.write(sequence_id, buffers).await?;

        let None = stream.next().await else {
            vortex_bail!("array tree flat layout received stream with more than a single chunk");
        };

        Ok(ArrayTreeFlatLayout::new(
            FlatLayout::new(
                row_count,
                stream.dtype().clone(),
                segment_id,
                ReadContext::new(ctx.to_ids()),
            ),
            compact_tree,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        0
    }
}

/// Collector strategy (RX) that wraps the data pipeline.
///
/// After the data child completes, walks the returned layout tree to extract compact
/// flatbuffers and segment IDs from all [`ArrayTreeFlatLayout`] leaves, builds a struct
/// array of `{segment_id, compact_tree}`, and writes it as an auxiliary child via the
/// configured `array_trees_strategy`.
pub struct ArrayTreeCollectorStrategy {
    child: Option<Arc<dyn LayoutStrategy>>,
    array_trees_strategy: Arc<dyn LayoutStrategy>,
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

        // Walk the layout tree to collect (segment_id, compact_tree) pairs from
        // ArrayTreeFlatLayout leaves.
        let mut entries: Vec<(SegmentId, ByteBuffer)> = Vec::new();
        for layout_ref in data_layout.depth_first_traversal() {
            let layout_ref = layout_ref?;
            if let Some(atf) = layout_ref.as_opt::<ArrayTreeFlat>()
                && let Some(tree) = atf.compact_tree()
            {
                entries.push((atf.inner().segment_id(), tree.clone()));
            }
        }

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
