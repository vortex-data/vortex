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
use vortex_array::serde::RawNodeStats;
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
/// Stats are hydrated from each node's `RawNodeStats` into typed nullable columns mirroring
/// the schema on [`ArrayTreeLayout::array_trees_dtype`].
fn build_consolidated_struct(entries: &[(SegmentId, ColumnarChunkData)]) -> VortexResult<ArrayRef> {
    let nrows = entries.len();
    let nn = Nullability::NonNullable;
    let nullable = Nullability::Nullable;

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

    // Nullable stat column accumulators. For each binary stat we use a nullable
    // `VarBinViewBuilder`; for primitive/bool stats we accumulate (values, validity)
    // separately and assemble at finish time.
    let mut min_builder = VarBinViewBuilder::with_capacity(DType::Binary(nullable), total_nodes);
    let mut max_builder = VarBinViewBuilder::with_capacity(DType::Binary(nullable), total_nodes);
    let mut sum_builder = VarBinViewBuilder::with_capacity(DType::Binary(nullable), total_nodes);
    let mut min_prec = NullableValues::<u8>::with_capacity(total_nodes);
    let mut max_prec = NullableValues::<u8>::with_capacity(total_nodes);
    let mut null_count = NullableValues::<u64>::with_capacity(total_nodes);
    let mut nan_count = NullableValues::<u64>::with_capacity(total_nodes);
    let mut uncompressed_size = NullableValues::<u64>::with_capacity(total_nodes);
    let mut is_constant = NullableBools::with_capacity(total_nodes);
    let mut is_sorted = NullableBools::with_capacity(total_nodes);
    let mut is_strict_sorted = NullableBools::with_capacity(total_nodes);

    let mut nodes_offsets: Vec<i32> = Vec::with_capacity(nrows + 1);
    nodes_offsets.push(0);
    let mut nodes_cumulative: i32 = 0;

    for (_, chunk) in entries {
        for i in 0..chunk.nnodes() {
            encoding_ids.push(chunk.encoding_ids[i]);
            child_counts.push(chunk.child_counts[i]);
            buffers_per_node.push(chunk.buffers_per_node[i]);
            metadata_builder.append_value(chunk.node_metadata[i].as_slice());

            let raw: Option<&RawNodeStats> = chunk.stats[i].as_ref();
            append_stat_columns(
                raw,
                &mut min_builder,
                &mut max_builder,
                &mut sum_builder,
                &mut min_prec,
                &mut max_prec,
                &mut null_count,
                &mut nan_count,
                &mut uncompressed_size,
                &mut is_constant,
                &mut is_sorted,
                &mut is_strict_sorted,
            );
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
            min_builder.finish().into_array(),
            min_prec.finish(),
            max_builder.finish().into_array(),
            max_prec.finish(),
            sum_builder.finish().into_array(),
            null_count.finish(),
            nan_count.finish(),
            uncompressed_size.finish(),
            is_constant.finish(),
            is_sorted.finish(),
            is_strict_sorted.finish(),
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

/// Mapping of `RawStatValue::exact` onto the u8 used in our columnar schema. Stable on disk
/// — old readers will assume `0 = Exact, 1 = Inexact`.
const PRECISION_EXACT: u8 = 0;
const PRECISION_INEXACT: u8 = 1;

/// Push one node's stats onto every per-stat column. Nulls are pushed wherever the
/// `RawNodeStats` slot is `None` (or `raw` itself is `None`).
#[allow(clippy::too_many_arguments)]
fn append_stat_columns(
    raw: Option<&RawNodeStats>,
    min: &mut VarBinViewBuilder,
    max: &mut VarBinViewBuilder,
    sum: &mut VarBinViewBuilder,
    min_prec: &mut NullableValues<u8>,
    max_prec: &mut NullableValues<u8>,
    null_count: &mut NullableValues<u64>,
    nan_count: &mut NullableValues<u64>,
    uncompressed_size: &mut NullableValues<u64>,
    is_constant: &mut NullableBools,
    is_sorted: &mut NullableBools,
    is_strict_sorted: &mut NullableBools,
) {
    match raw {
        Some(raw) => {
            match &raw.min {
                Some(rv) => {
                    min.append_value(rv.bytes.as_slice());
                    min_prec.push(if rv.exact {
                        PRECISION_EXACT
                    } else {
                        PRECISION_INEXACT
                    });
                }
                None => {
                    min.append_null();
                    min_prec.push_null();
                }
            }
            match &raw.max {
                Some(rv) => {
                    max.append_value(rv.bytes.as_slice());
                    max_prec.push(if rv.exact {
                        PRECISION_EXACT
                    } else {
                        PRECISION_INEXACT
                    });
                }
                None => {
                    max.append_null();
                    max_prec.push_null();
                }
            }
            match &raw.sum {
                Some(b) => sum.append_value(b.as_slice()),
                None => sum.append_null(),
            }
            null_count.push_opt(raw.null_count);
            nan_count.push_opt(raw.nan_count);
            uncompressed_size.push_opt(raw.uncompressed_size_in_bytes);
            is_constant.push_opt(raw.is_constant);
            is_sorted.push_opt(raw.is_sorted);
            is_strict_sorted.push_opt(raw.is_strict_sorted);
        }
        None => {
            min.append_null();
            max.append_null();
            sum.append_null();
            min_prec.push_null();
            max_prec.push_null();
            null_count.push_null();
            nan_count.push_null();
            uncompressed_size.push_null();
            is_constant.push_null();
            is_sorted.push_null();
            is_strict_sorted.push_null();
        }
    }
}

/// Accumulator for a nullable primitive column.
struct NullableValues<T: vortex_array::dtype::NativePType + Default + Copy> {
    values: Vec<T>,
    validity: BitBufferMut,
}

impl<T: vortex_array::dtype::NativePType + Default + Copy> NullableValues<T> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            values: Vec::with_capacity(cap),
            validity: BitBufferMut::with_capacity(cap),
        }
    }
    fn push(&mut self, v: T) {
        self.values.push(v);
        self.validity.append(true);
    }
    fn push_null(&mut self) {
        self.values.push(T::default());
        self.validity.append(false);
    }
    fn push_opt(&mut self, v: Option<T>) {
        match v {
            Some(v) => self.push(v),
            None => self.push_null(),
        }
    }
    fn finish(self) -> ArrayRef {
        let validity_arr =
            BoolArray::new(self.validity.freeze(), Validity::NonNullable).into_array();
        PrimitiveArray::new(Buffer::from(self.values), Validity::Array(validity_arr)).into_array()
    }
}

/// Accumulator for a nullable bool column.
struct NullableBools {
    bits: BitBufferMut,
    validity: BitBufferMut,
}

impl NullableBools {
    fn with_capacity(cap: usize) -> Self {
        Self {
            bits: BitBufferMut::with_capacity(cap),
            validity: BitBufferMut::with_capacity(cap),
        }
    }
    fn push(&mut self, v: bool) {
        self.bits.append(v);
        self.validity.append(true);
    }
    fn push_null(&mut self) {
        self.bits.append(false);
        self.validity.append(false);
    }
    fn push_opt(&mut self, v: Option<bool>) {
        match v {
            Some(v) => self.push(v),
            None => self.push_null(),
        }
    }
    fn finish(self) -> ArrayRef {
        let validity_arr =
            BoolArray::new(self.validity.freeze(), Validity::NonNullable).into_array();
        BoolArray::new(self.bits.freeze(), Validity::Array(validity_arr)).into_array()
    }
}
