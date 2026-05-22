// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::list::ListArray;
use vortex_array::dtype::FieldName;
use vortex_array::serde::ColumnarArrayTree;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::serialize_to_columnar_tree;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
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

/// Returns a `(collector, leaf)` pair of cooperating strategies for array-tree collection.
///
/// The leaf strategy replaces [`FlatLayoutStrategy`] in the data pipeline and attaches each
/// chunk's [`ColumnarArrayTree`] to the resulting [`ArrayTreeFlatLayout`]. The collector
/// wraps the data pipeline and, after it completes, walks the data subtree to extract those
/// attached trees and writes them as a consolidated columnar struct array via the
/// `array_trees_strategy` (typically the same compress-then-flat strategy used for data).
///
/// Why attach trees to the leaf layout rather than push to a shared sink: column writers in
/// [`crate::layouts::table::TableStrategy`] run concurrently, and a shared sink would mix
/// leaves across collector invocations. The leaf-attached design keeps every collector
/// scoped to its own subtree.
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
/// Walks the chunk's array tree once via [`serialize_to_columnar_tree`] to produce data
/// buffers and a [`ColumnarArrayTree`], writes the buffers as a data-only segment (no
/// trailing flatbuffer), then attaches the tree to the returned layout for the collector to
/// extract.
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

        // Normalize if the flat strategy restricts encodings.
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

        // Single walk: data buffers + ColumnarArrayTree. No trailing flatbuffer in the
        // segment — the consolidated array_trees segment is the sole source of encoding
        // metadata for this leaf.
        let (buffers, tree) = serialize_to_columnar_tree(
            &chunk,
            &ctx,
            session,
            &SerializeOptions {
                offset: 0,
                include_padding: self.flat.include_padding,
            },
        )?;

        // IMPORTANT ORDERING CONSTRAINT: write the segment first, then advance past the
        // sequence id. `segment_sink.write` consumes the SequenceId and only drops it on
        // return; doing more work while holding it would let later leaves wait on
        // SequenceId::collapse.
        let segment_id = segment_sink.write(sequence_id, buffers).await?;

        let None = stream.next().await else {
            vortex_bail!("array tree flat layout received stream with more than a single chunk");
        };

        Ok(ArrayTreeFlatLayout::with_tree(
            FlatLayout::new(
                row_count,
                stream.dtype().clone(),
                segment_id,
                ReadContext::new(ctx.to_ids()),
            ),
            tree,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        0
    }
}

/// Collector strategy (RX) that wraps the data pipeline.
///
/// After the data child completes, walks the resulting subtree to extract each
/// [`ArrayTreeFlatLayout`]'s attached [`ColumnarArrayTree`], builds the consolidated
/// `{segment_id, nodes, buffers}` struct array (see [`ArrayTreeLayout::array_trees_dtype`]),
/// and writes it via the configured `array_trees_strategy`.
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

        // Data segments get earlier sequence IDs than the consolidated array_trees segment.
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

        // Walk the data subtree to extract per-leaf trees. Each ArrayTreeFlatLayout leaf
        // carries its tree attached as transient write-time state (not serialized to disk).
        let mut entries: Vec<(SegmentId, ColumnarArrayTree)> = Vec::new();
        for layout_ref in data_layout.depth_first_traversal() {
            let layout_ref = layout_ref?;
            if let Some(atf) = layout_ref.as_opt::<ArrayTreeFlat>()
                && let Some(tree) = atf.take_tree()
            {
                entries.push((atf.inner().segment_id(), tree));
            }
        }

        // Sort by segment ID so on-disk row order matches segment-write order.
        entries.sort_by_key(|(seg, _)| *seg);

        let array_trees_array = build_consolidated_array(&entries)?;

        // Write the consolidated array via the array_trees strategy.
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

/// Build the consolidated `{segment_id, nodes, buffers}` struct array from a sorted list of
/// per-chunk entries. Each List<>'s elements are the concatenated per-chunk nodes (resp.
/// buffers) struct arrays, with offsets recorded per row.
fn build_consolidated_array(entries: &[(SegmentId, ColumnarArrayTree)]) -> VortexResult<ArrayRef> {
    let nrows = entries.len();

    let segment_ids: Buffer<u32> = entries.iter().map(|(seg, _)| **seg).collect();
    let segment_ids_array = PrimitiveArray::new(segment_ids, Validity::NonNullable).into_array();

    // Build the nodes list: concatenate every chunk's `tree.nodes` and record per-row offsets.
    let mut nodes_offsets: Vec<i32> = Vec::with_capacity(nrows + 1);
    nodes_offsets.push(0);
    let mut cum: i32 = 0;
    for (_, tree) in entries {
        cum += i32::try_from(tree.nodes.as_ref().len())
            .map_err(|_| vortex_err!("array tree node count overflows i32 offsets"))?;
        nodes_offsets.push(cum);
    }
    let nodes_inner = StructArray::try_concat(entries.iter().map(|(_, t)| &t.nodes))?.into_array();
    let nodes_list = ListArray::try_new(
        nodes_inner,
        PrimitiveArray::new(Buffer::from(nodes_offsets), Validity::NonNullable).into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    // Build the buffers list: same pattern.
    let mut buffers_offsets: Vec<i32> = Vec::with_capacity(nrows + 1);
    buffers_offsets.push(0);
    let mut cum: i32 = 0;
    for (_, tree) in entries {
        cum += i32::try_from(tree.buffers.as_ref().len())
            .map_err(|_| vortex_err!("array tree buffer count overflows i32 offsets"))?;
        buffers_offsets.push(cum);
    }
    let buffers_inner =
        StructArray::try_concat(entries.iter().map(|(_, t)| &t.buffers))?.into_array();
    let buffers_list = ListArray::try_new(
        buffers_inner,
        PrimitiveArray::new(Buffer::from(buffers_offsets), Validity::NonNullable).into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    Ok(StructArray::try_new(
        vec![
            FieldName::from("segment_id"),
            FieldName::from("nodes"),
            FieldName::from("buffers"),
        ]
        .into(),
        vec![segment_ids_array, nodes_list, buffers_list],
        nrows,
        Validity::NonNullable,
    )?
    .into_array())
}
