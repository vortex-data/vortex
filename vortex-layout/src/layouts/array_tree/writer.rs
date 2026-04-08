// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::SerializeOptions;
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
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;

/// Creates a cooperating pair of strategies for array tree collection.
///
/// Returns `(collector, leaf)` where:
/// - `leaf` replaces [`FlatLayoutStrategy`] in the data pipeline — it serializes chunks and
///   produces compact flatbuffers.
/// - `collector` wraps the data pipeline — after data is written, it collects compact flatbuffers
///   from the layout tree and writes them as a VarBin array.
pub fn writer(
    flat: FlatLayoutStrategy,
    array_trees_strategy: Arc<dyn LayoutStrategy>,
) -> (ArrayTreeCollectorStrategy, ArrayTreeFlatStrategy) {
    let chunk_counter = Arc::new(AtomicUsize::new(0));
    let leaf = ArrayTreeFlatStrategy {
        flat,
        chunk_counter,
    };
    let collector = ArrayTreeCollectorStrategy {
        child: None,
        array_trees_strategy,
    };
    (collector, leaf)
}

/// Leaf strategy (TX) that replaces [`FlatLayoutStrategy`].
///
/// For each chunk, it delegates serialization to the inner [`FlatLayoutStrategy`], also produces
/// a compact flatbuffer (encoding tree + buffer descriptors, no stats), and returns an
/// [`ArrayTreeFlatLayout`] with the compact tree attached.
#[derive(Clone)]
pub struct ArrayTreeFlatStrategy {
    flat: FlatLayoutStrategy,
    chunk_counter: Arc<AtomicUsize>,
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

        // Full serialization (with stats) for the data segment.
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

        let chunk_idx = self.chunk_counter.fetch_add(1, Ordering::Relaxed);

        Ok(ArrayTreeFlatLayout::new(
            FlatLayout::new(
                row_count,
                stream.dtype().clone(),
                segment_id,
                ReadContext::new(ctx.to_ids()),
            ),
            chunk_idx,
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
/// After the data child completes, walks the returned layout tree to extract compact flatbuffers
/// from all [`ArrayTreeFlatLayout`] leaves, builds a VarBin array, and writes it as an
/// auxiliary child.
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

        // Walk the layout tree to collect compact flatbuffers from ArrayTreeFlatLayout leaves.
        let mut compact_trees: Vec<(usize, ByteBuffer)> = Vec::new();
        for layout_ref in data_layout.depth_first_traversal() {
            let layout_ref = layout_ref?;
            if let Some(atf) = layout_ref.as_opt::<ArrayTreeFlat>()
                && let Some(tree) = atf.compact_tree()
            {
                compact_trees.push((atf.chunk_idx(), tree.clone()));
            }
        }

        // Sort by chunk index to ensure deterministic order.
        compact_trees.sort_by_key(|(idx, _)| *idx);

        // Build a VarBin array of compact flatbuffers.
        let dtype = DType::Binary(Nullability::NonNullable);
        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), compact_trees.len());
        for (_, tree) in &compact_trees {
            builder.append_value(tree.as_slice());
        }
        let array_trees_array = builder.finish().into_array();

        // Write the VarBin array via the array_trees strategy.
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
