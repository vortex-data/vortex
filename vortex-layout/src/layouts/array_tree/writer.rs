// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::list::ListArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::ColumnarChunkData;
use vortex_array::serde::SegmentMode;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::serialize_with_columnar_chunk;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
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

/// Creates a cooperating pair of strategies for array tree collection.
///
/// Returns `(collector, leaf)` where the leaf replaces [`FlatLayoutStrategy`] in the data
/// pipeline and attaches per-chunk [`ColumnarChunkData`] to each [`ArrayTreeFlatLayout`] it
/// produces. The collector wraps the data pipeline, walks the resulting data subtree to
/// extract those attached chunks, and writes them as a columnar struct array via the
/// configured `array_trees_strategy`. Attaching chunk data to the leaf layout (rather than
/// using a shared sink) keeps every collector invocation scoped to its own subtree, which
/// matters because [`crate::layouts::table::TableStrategy`] writes columns concurrently and
/// a shared sink would mix leaves across collectors.
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
/// Walks each chunk's array tree once via [`serialize_with_columnar_chunk`] to produce both
/// the data-segment buffers (no inline trailing flatbuffer — see [`SegmentMode::DataOnly`])
/// and a [`ColumnarChunkData`] which is attached to the returned [`ArrayTreeFlatLayout`] as
/// transient write-time state for the collector to extract.
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

        // Single walk: produce data-only segment buffers + columnar chunk data for the
        // collector to consume. No trailing flatbuffer in the segment — the consolidated
        // columnar array_trees segment is the sole source of encoding metadata for this
        // leaf.
        let (buffers, columnar_chunk) = serialize_with_columnar_chunk(
            &chunk,
            &ctx,
            session,
            &SerializeOptions {
                offset: 0,
                include_padding: self.flat.include_padding,
            },
            SegmentMode::DataOnly,
        )?;

        // IMPORTANT ORDERING CONSTRAINT: write the segment first, then push to the sink.
        // `segment_sink.write` consumes our `SequenceId` and only drops it on return; pushing
        // before that would risk holding the sink mutex while later leaves wait on
        // `SequenceId::collapse`.
        let segment_id = segment_sink.write(sequence_id, buffers).await?;

        let None = stream.next().await else {
            vortex_bail!("array tree flat layout received stream with more than a single chunk");
        };

        Ok(ArrayTreeFlatLayout::with_chunk(
            FlatLayout::new(
                row_count,
                stream.dtype().clone(),
                segment_id,
                ReadContext::new(ctx.to_ids()),
            ),
            columnar_chunk,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        0
    }
}

/// Collector strategy (RX) that wraps the data pipeline.
///
/// After the data child completes, walks the resulting data subtree to extract each
/// [`ArrayTreeFlatLayout`] leaf's attached [`ColumnarChunkData`], then serializes the
/// per-chunk data into one row-per-chunk struct array matching
/// [`ArrayTreeLayout::array_trees_dtype`] and writes it via the configured
/// `array_trees_strategy`.
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

        // Walk the data subtree to extract per-leaf columnar chunk data. Each
        // `ArrayTreeFlatLayout` leaf carries its chunk attached as transient write-time
        // state (not serialized to disk). This per-column walk keeps the collector's view
        // scoped to its own leaves, even when columns write concurrently — unlike a
        // shared sink which would mix leaves across collector invocations.
        let mut entries: Vec<(SegmentId, ColumnarChunkData)> = Vec::new();
        for layout_ref in data_layout.depth_first_traversal() {
            let layout_ref = layout_ref?;
            if let Some(atf) = layout_ref.as_opt::<ArrayTreeFlat>()
                && let Some(chunk) = atf.take_chunk()
            {
                entries.push((atf.inner().segment_id(), chunk));
            }
        }

        // Sort by segment ID so the on-disk row order matches segment-write order.
        entries.sort_by_key(|(seg, _)| *seg);

        let array_trees_array = build_consolidated_struct(&entries)?;

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

/// Assemble the consolidated columnar struct array from a sorted list of per-chunk entries.
///
/// One row per chunk. The `nodes` and `buffers` List<Struct> columns are built by
/// concatenating each chunk's per-node / per-buffer values and recording offsets per row.
///
/// **Stats are intentionally written as all-null in this initial implementation** — the
/// columnar schema has nullable stat columns ready to receive stats, but populating them
/// requires bridging the existing `StatsSet`/`ScalarValue` serialization to typed columns.
/// That's a focused follow-up; for now the consolidated carries tree shape + metadata +
/// buffer descriptors, which is sufficient for the new reader path to decode every chunk.
fn build_consolidated_struct(entries: &[(SegmentId, ColumnarChunkData)]) -> VortexResult<ArrayRef> {
    let nrows = entries.len();
    let nn = Nullability::NonNullable;

    // segment_id column.
    let segment_ids: Buffer<u32> = entries.iter().map(|(seg, _)| **seg).collect();
    let segment_ids_array = PrimitiveArray::new(segment_ids, Validity::NonNullable).into_array();

    let total_nodes: usize = entries.iter().map(|(_, c)| c.nnodes()).sum();
    let total_buffers: usize = entries.iter().map(|(_, c)| c.buffer_padding.len()).sum();

    // ---- Per-node columns ----

    let mut encoding_ids: Vec<u16> = Vec::with_capacity(total_nodes);
    let mut child_counts: Vec<u8> = Vec::with_capacity(total_nodes);
    let mut buffers_per_node: Vec<u16> = Vec::with_capacity(total_nodes);
    let mut metadata_builder = VarBinViewBuilder::with_capacity(DType::Binary(nn), total_nodes);

    let mut nodes_offsets: Vec<i32> = Vec::with_capacity(nrows + 1);
    nodes_offsets.push(0);
    let mut nodes_cumulative: i32 = 0;

    for (_, chunk) in entries {
        for i in 0..chunk.nnodes() {
            encoding_ids.push(chunk.encoding_ids[i]);
            child_counts.push(chunk.child_counts[i]);
            buffers_per_node.push(chunk.buffers_per_node[i]);
            metadata_builder.append_value(chunk.node_metadata[i].as_slice());
        }
        nodes_cumulative += i32::try_from(chunk.nnodes())
            .map_err(|_| vortex_err!("array tree node count overflows i32 offsets"))?;
        nodes_offsets.push(nodes_cumulative);
    }

    let encoding_id_arr =
        PrimitiveArray::new(Buffer::from(encoding_ids), Validity::NonNullable).into_array();
    let child_count_arr =
        PrimitiveArray::new(Buffer::from(child_counts), Validity::NonNullable).into_array();
    let buffers_per_node_arr =
        PrimitiveArray::new(Buffer::from(buffers_per_node), Validity::NonNullable).into_array();
    let metadata_arr = metadata_builder.finish().into_array();

    // All-null stat columns. Placeholder values per row to satisfy the typed-column shape.
    let stat_binary = || -> VortexResult<ArrayRef> { all_null_binary(total_nodes) };
    let stat_u8 = || -> VortexResult<ArrayRef> { all_null_primitive::<u8>(total_nodes) };
    let stat_u64 = || -> VortexResult<ArrayRef> { all_null_primitive::<u64>(total_nodes) };
    let stat_bool = || -> VortexResult<ArrayRef> { all_null_bool(total_nodes) };

    let node_names: Vec<&str> = vec![
        "encoding_id",
        "child_count",
        "metadata",
        "buffers_per_node",
        "stat_min",
        "stat_min_precision",
        "stat_max",
        "stat_max_precision",
        "stat_sum",
        "stat_null_count",
        "stat_nan_count",
        "stat_uncompressed_size_in_bytes",
        "stat_is_constant",
        "stat_is_sorted",
        "stat_is_strict_sorted",
    ];
    let node_inner_struct = StructArray::try_new(
        node_names.into(),
        vec![
            encoding_id_arr,
            child_count_arr,
            metadata_arr,
            buffers_per_node_arr,
            stat_binary()?,
            stat_u8()?,
            stat_binary()?,
            stat_u8()?,
            stat_binary()?,
            stat_u64()?,
            stat_u64()?,
            stat_u64()?,
            stat_bool()?,
            stat_bool()?,
            stat_bool()?,
        ],
        total_nodes,
        Validity::NonNullable,
    )?
    .into_array();

    let nodes_offsets_arr =
        PrimitiveArray::new(Buffer::from(nodes_offsets), Validity::NonNullable).into_array();
    let nodes_list =
        ListArray::try_new(node_inner_struct, nodes_offsets_arr, Validity::NonNullable)?
            .into_array();

    // ---- Per-buffer columns ----

    let mut buffer_padding: Vec<u16> = Vec::with_capacity(total_buffers);
    let mut buffer_alignment_exp: Vec<u8> = Vec::with_capacity(total_buffers);
    let mut buffer_length: Vec<u32> = Vec::with_capacity(total_buffers);

    let mut buffers_offsets: Vec<i32> = Vec::with_capacity(nrows + 1);
    buffers_offsets.push(0);
    let mut buffers_cumulative: i32 = 0;

    for (_, chunk) in entries {
        buffer_padding.extend_from_slice(&chunk.buffer_padding);
        buffer_alignment_exp.extend_from_slice(&chunk.buffer_alignment_exponent);
        buffer_length.extend_from_slice(&chunk.buffer_length);
        buffers_cumulative += i32::try_from(chunk.buffer_padding.len())
            .map_err(|_| vortex_err!("array tree buffer count overflows i32 offsets"))?;
        buffers_offsets.push(buffers_cumulative);
    }

    let padding_arr =
        PrimitiveArray::new(Buffer::from(buffer_padding), Validity::NonNullable).into_array();
    let alignment_arr =
        PrimitiveArray::new(Buffer::from(buffer_alignment_exp), Validity::NonNullable).into_array();
    let length_arr =
        PrimitiveArray::new(Buffer::from(buffer_length), Validity::NonNullable).into_array();

    let buffer_names: Vec<&str> = vec!["padding", "alignment_exponent", "length"];
    let buffer_inner_struct = StructArray::try_new(
        buffer_names.into(),
        vec![padding_arr, alignment_arr, length_arr],
        total_buffers,
        Validity::NonNullable,
    )?
    .into_array();

    let buffers_offsets_arr =
        PrimitiveArray::new(Buffer::from(buffers_offsets), Validity::NonNullable).into_array();
    let buffers_list = ListArray::try_new(
        buffer_inner_struct,
        buffers_offsets_arr,
        Validity::NonNullable,
    )?
    .into_array();

    // ---- Outer struct (one row per chunk) ----

    let outer_names: Vec<&str> = vec!["segment_id", "nodes", "buffers"];
    let outer = StructArray::try_new(
        outer_names.into(),
        vec![segment_ids_array, nodes_list, buffers_list],
        nrows,
        Validity::NonNullable,
    )?
    .into_array();

    Ok(outer)
}

/// Build an all-null primitive column of the given length with the right typed dtype.
fn all_null_primitive<T: vortex_array::dtype::NativePType + Default + Copy>(
    n: usize,
) -> VortexResult<ArrayRef> {
    let values: Vec<T> = vec![T::default(); n];
    let mut validity = BitBufferMut::with_capacity(n);
    for _ in 0..n {
        validity.append(false);
    }
    let validity_arr = BoolArray::new(validity.freeze(), Validity::NonNullable).into_array();
    Ok(PrimitiveArray::new(Buffer::from(values), Validity::Array(validity_arr)).into_array())
}

fn all_null_bool(n: usize) -> VortexResult<ArrayRef> {
    let mut bits = BitBufferMut::with_capacity(n);
    let mut validity = BitBufferMut::with_capacity(n);
    for _ in 0..n {
        bits.append(false);
        validity.append(false);
    }
    let validity_arr = BoolArray::new(validity.freeze(), Validity::NonNullable).into_array();
    Ok(BoolArray::new(bits.freeze(), Validity::Array(validity_arr)).into_array())
}

fn all_null_binary(n: usize) -> VortexResult<ArrayRef> {
    let mut builder = VarBinViewBuilder::with_capacity(DType::Binary(Nullability::Nullable), n);
    for _ in 0..n {
        builder.append_null();
    }
    Ok(builder.finish().into_array())
}
