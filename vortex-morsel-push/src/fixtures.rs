// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout fixtures for the correctness suites and the comparison harness.
//!
//! Layouts are assembled by hand rather than through a writer strategy, because the strategies
//! chunk every column on the same boundaries and the interesting cases are the misaligned ones —
//! a morsel whose range cuts column `a` mid-chunk and column `b` on a boundary.

use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::segments::ReadAtNowait;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_session::VortexSession;

/// One column of a fixture: a name and the chunks it is stored in.
pub struct Column {
    /// The field name.
    pub name: FieldName,
    /// The chunks, in row order. Chunk boundaries need not agree with any other column's.
    pub chunks: Vec<ArrayRef>,
}

/// An immutable in-memory segment source used after fixture writing completes.
struct MemorySegments {
    buffers: Arc<[ByteBuffer]>,
}

impl SegmentSource for MemorySegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        future::ready(
            self.buffers
                .get(*id as usize)
                .cloned()
                .map(BufferHandle::new_host)
                .ok_or_else(|| vortex_err!("Segment not found")),
        )
        .boxed()
    }

    fn request_nowait(&self, id: SegmentId) -> VortexResult<ReadAtNowait> {
        self.buffers
            .get(*id as usize)
            .cloned()
            .map(BufferHandle::new_host)
            .map(ReadAtNowait::Ready)
            .ok_or_else(|| vortex_err!("Segment not found"))
    }
}

impl Column {
    /// Build a column from a name and its chunks.
    pub fn new(name: impl Into<FieldName>, chunks: Vec<ArrayRef>) -> Self {
        Self {
            name: name.into(),
            chunks,
        }
    }
}

/// A written fixture: the segments holding it, the layout over them, and the whole table as one
/// in-memory array for oracle comparisons.
pub struct Fixture {
    /// The segment source the layout reads from.
    pub segments: Arc<dyn SegmentSource>,
    /// The exact encoded segment buffers, in segment-id order.
    pub segment_buffers: Vec<ByteBuffer>,
    /// The struct-of-chunked-flat layout.
    pub layout: LayoutRef,
    /// The complete table, unchunked. `None` when the caller asked not to retain it.
    pub table: Option<ArrayRef>,
    /// The number of rows.
    pub row_count: u64,
}

/// Write a struct-of-chunked-flat fixture with per-column chunking, uncompressed.
///
/// Every column must cover the same total number of rows; their chunk boundaries need not agree.
pub async fn write_fixture(columns: Vec<Column>, session: &VortexSession) -> VortexResult<Fixture> {
    write_fixture_with(columns, Arc::new(FlatLayoutStrategy::default()), session).await
}

/// Write a fixture, running each column's chunks through `strategy`.
///
/// Passing a compressing strategy is what makes decode cost real: the leaves carry btrblocks
/// encodings rather than raw buffers, so the decode work both executors share is the work a real
/// file imposes.
pub async fn write_fixture_with(
    columns: Vec<Column>,
    strategy: Arc<dyn LayoutStrategy>,
    session: &VortexSession,
) -> VortexResult<Fixture> {
    write_fixture_inner(columns, strategy, session, true, false).await
}

/// Write a fixture as whole-column streams without retaining an in-memory table copy.
///
/// This matches how the file writer drives a strategy: repartitioning and buffering see across
/// incoming array boundaries. The omitted copy exists only so a caller can compare against the
/// source data directly; when V1 is the oracle it is dead weight.
pub async fn write_streaming_fixture_no_table(
    columns: Vec<Column>,
    strategy: Arc<dyn LayoutStrategy>,
    session: &VortexSession,
) -> VortexResult<Fixture> {
    write_fixture_inner(columns, strategy, session, false, true).await
}

async fn write_fixture_inner(
    columns: Vec<Column>,
    strategy: Arc<dyn LayoutStrategy>,
    session: &VortexSession,
    keep_table: bool,
    stream_whole_column: bool,
) -> VortexResult<Fixture> {
    let segments = Arc::new(TestSegments::default());
    let ctx = vortex_array::ArrayContext::empty();

    let mut row_count = None;
    let mut field_layouts = Vec::with_capacity(columns.len());
    let mut field_names = Vec::with_capacity(columns.len());
    let mut field_dtypes = Vec::with_capacity(columns.len());
    let mut table_fields: Vec<ArrayRef> = Vec::with_capacity(columns.len());

    for column in &columns {
        let dtype = column
            .chunks
            .first()
            .map(|chunk| chunk.dtype().clone())
            .ok_or_else(|| vortex_err!("a column needs at least one chunk"))?;

        let rows: u64 = column.chunks.iter().map(|chunk| chunk.len() as u64).sum();
        match row_count {
            None => row_count = Some(rows),
            Some(expected) if expected == rows => {}
            Some(expected) => {
                vortex_bail!("columns must have equal row counts: {expected} vs {rows}")
            }
        }

        let column_layout = if stream_whole_column {
            // The TPC-H column goes through one strategy invocation, exactly as the real file
            // writer drives it. Repartitioning and buffering therefore see across incoming batch
            // boundaries instead of treating each generated batch as end-of-file.
            let (ptr, eof) = SequenceId::root().split();
            let chunks = column.chunks.clone().into_iter().map(VortexResult::Ok);
            strategy
                .write_stream(
                    ctx.clone().into(),
                    Arc::<TestSegments>::clone(&segments),
                    ArrayStreamAdapter::new(dtype.clone(), stream::iter(chunks)).sequenced(ptr),
                    eof,
                    session,
                )
                .await?
        } else {
            // Small correctness fixtures intentionally preserve caller-provided per-column chunk
            // boundaries and do not require a runtime-backed chunked writer.
            let mut chunk_layouts = Vec::with_capacity(column.chunks.len());
            for chunk in &column.chunks {
                let (ptr, eof) = SequenceId::root().split();
                chunk_layouts.push(
                    strategy
                        .write_stream(
                            ctx.clone().into(),
                            Arc::<TestSegments>::clone(&segments),
                            chunk.clone().to_array_stream().sequenced(ptr),
                            eof,
                            session,
                        )
                        .await?,
                );
            }
            if chunk_layouts.len() == 1 {
                chunk_layouts
                    .pop()
                    .ok_or_else(|| vortex_err!("a column needs at least one chunk"))?
            } else {
                ChunkedLayout::new(rows, dtype.clone(), layout_children(chunk_layouts))
                    .into_layout()
            }
        };
        field_layouts.push(column_layout);
        field_names.push(column.name.clone());
        field_dtypes.push(dtype);

        if keep_table {
            // The oracle copy of the column, concatenated.
            table_fields.push(concat_chunks(&column.chunks)?);
        }
    }

    let rows = row_count.unwrap_or(0);
    let struct_dtype = DType::Struct(
        StructFields::new(field_names.clone().into(), field_dtypes),
        Nullability::NonNullable,
    );
    let layout = StructLayout::new(rows, struct_dtype, field_layouts).into_layout();

    let table = if keep_table {
        Some(
            StructArray::try_new(
                field_names.into(),
                table_fields,
                usize::try_from(rows).map_err(|_| vortex_err!("row count exceeds usize"))?,
                vortex_array::validity::Validity::NonNullable,
            )?
            .into_array(),
        )
    } else {
        None
    };

    let segment_buffers = segments.buffers();
    let frozen_segments: Arc<dyn SegmentSource> = Arc::new(MemorySegments {
        buffers: segment_buffers.clone().into(),
    });

    Ok(Fixture {
        segment_buffers,
        segments: frozen_segments,
        layout,
        table,
        row_count: rows,
    })
}

fn concat_chunks(chunks: &[ArrayRef]) -> VortexResult<ArrayRef> {
    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }
    let dtype = chunks[0].dtype().clone();
    Ok(vortex_array::arrays::ChunkedArray::try_new(chunks.to_vec(), dtype)?.into_array())
}
