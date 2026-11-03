// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::AllOr;
use vortex_session::VortexSession;
use vortex_sparse::Sparse as SparseArray;

use super::INDICES_DTYPE;
use super::SparseBlocks;
use super::SparseLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// The default number of rows in each block of a sparse layout's metadata.
pub const DEFAULT_BLOCK_LEN: u64 = 8192;

/// The default number of present rows buffered before a column dense so far is handed to the
/// fallback without waiting for the rest of it.
pub const DEFAULT_MAX_BUFFERED_ROWS: u64 = 1 << 20;

/// Writes a nullable column as a [`SparseLayout`] when enough of its rows are null.
///
/// The column is read to its end, keeping only its present rows, and then written as a sparse
/// layout when at least `min_null_fraction` of its rows are null. Otherwise, the column is
/// rebuilt from its present rows and handed to `fallback`, as is a non-nullable column.
///
/// The present rows are buffered until the column ends, unless more than `max_buffered_rows`
/// are present while the column is still too dense for a sparse layout, in which case it falls
/// back at once.
#[derive(Clone)]
pub struct SparseLayoutStrategy {
    indices: Arc<dyn LayoutStrategy>,
    values: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    min_null_fraction: f64,
    block_len: u64,
    max_buffered_rows: u64,
}

impl SparseLayoutStrategy {
    /// Creates a strategy writing `indices` and `values` through the given strategies, and dense
    /// columns through `fallback`.
    pub fn new(
        indices: Arc<dyn LayoutStrategy>,
        values: Arc<dyn LayoutStrategy>,
        fallback: Arc<dyn LayoutStrategy>,
        min_null_fraction: f64,
    ) -> Self {
        Self {
            indices,
            values,
            fallback,
            min_null_fraction,
            block_len: DEFAULT_BLOCK_LEN,
            max_buffered_rows: DEFAULT_MAX_BUFFERED_ROWS,
        }
    }

    /// Sets the number of rows in each block of the layout's metadata.
    pub fn with_block_len(mut self, block_len: u64) -> Self {
        self.block_len = block_len;
        self
    }

    /// Sets how many present rows of a column dense so far are buffered before falling back.
    pub fn with_max_buffered_rows(mut self, max_buffered_rows: u64) -> Self {
        self.max_buffered_rows = max_buffered_rows;
        self
    }

    fn is_sparse_enough(&self, present: u64, rows: u64) -> bool {
        present > 0 && (rows - present) as f64 >= self.min_null_fraction * rows as f64
    }
}

/// The present rows of the column read so far.
struct Buffered {
    dtype: DType,
    indices: Vec<u64>,
    values: Vec<ArrayRef>,
    chunk_lens: Vec<usize>,
    rows: u64,
}

impl Buffered {
    fn present(&self) -> u64 {
        self.indices.len() as u64
    }

    fn indices_array(&self) -> ArrayRef {
        PrimitiveArray::new(Buffer::copy_from(&self.indices), Validity::NonNullable)
            .into_array()
    }

    fn values_array(&self) -> VortexResult<ArrayRef> {
        Ok(match self.values.len() {
            1 => self.values[0].clone(),
            _ => ChunkedArray::try_new(self.values.clone(), self.dtype.clone())?.into_array(),
        })
    }

    /// The buffered rows rebuilt with their nulls, in the chunks they arrived in.
    fn rebuild(&self, session: &VortexSession) -> VortexResult<Vec<ArrayRef>> {
        let len = usize::try_from(self.rows)?;
        let whole = if self.indices.is_empty() {
            ConstantArray::new(Scalar::null(self.dtype.clone()), len).into_array()
        } else {
            let mut ctx = session.create_execution_ctx();
            SparseArray::try_new(
                self.indices_array(),
                self.values_array()?,
                len,
                Scalar::null(self.dtype.clone()),
                &mut ctx,
            )?
            .into_array()
        };
        let mut start = 0;
        self.chunk_lens
            .iter()
            .map(|&chunk_len| {
                let chunk = whole.slice(start..start + chunk_len)?;
                start += chunk_len;
                Ok(chunk)
            })
            .collect()
    }
}

#[async_trait]
impl LayoutStrategy for SparseLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_nullable() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let mut exec_ctx = session.create_execution_ctx();
        let mut buffered = Buffered {
            dtype: dtype.clone(),
            indices: Vec::new(),
            values: Vec::new(),
            chunk_lens: Vec::new(),
            rows: 0,
        };
        // A held sequence ID blocks every writer ordered after it, which can block the producer
        // of this stream, so each is dropped once its chunk is read. The one chunk whose ID is
        // kept is the one that switches the column to its fallback while the stream continues.
        let mut switched_at = None;
        while let Some(chunk) = stream.next().await {
            let (sequence_id, chunk) = chunk?;
            let len = chunk.len();
            let mask = chunk.validity()?.execute_mask(len, &mut exec_ctx)?;
            match mask.indices() {
                AllOr::None => {}
                AllOr::All => {
                    buffered.indices.extend(buffered.rows..buffered.rows + len as u64);
                    buffered.values.push(chunk);
                }
                AllOr::Some(positions) => {
                    buffered
                        .indices
                        .extend(positions.iter().map(|&pos| buffered.rows + pos as u64));
                    let present = chunk
                        .filter(mask.clone())?
                        .execute::<Canonical>(&mut exec_ctx)?
                        .into_array();
                    buffered.values.push(present);
                }
            }
            buffered.chunk_lens.push(len);
            buffered.rows += len as u64;

            if buffered.present() > self.max_buffered_rows
                && !self.is_sparse_enough(buffered.present(), buffered.rows)
            {
                switched_at = Some(sequence_id);
                break;
            }
        }

        if buffered.rows == 0 {
            // No rows: the fallback writes the empty column.
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        if switched_at.is_some() || !self.is_sparse_enough(buffered.present(), buffered.rows) {
            // The buffered rows go first: beneath the switching chunk's ID they come before the
            // rest of the stream, and once the stream has ended they come before its end.
            // The pointer numbering them is dropped before the fallback writes: a live pointer
            // holds back every writer ordered after it, the fallback included.
            let rebuilt: Vec<_> = {
                let mut pointer = match switched_at {
                    Some(sequence_id) => sequence_id.descend(),
                    None => eof.split_off(),
                };
                buffered
                    .rebuild(session)?
                    .into_iter()
                    .map(|chunk| Ok((pointer.advance(), chunk)))
                    .collect()
            };
            let replay = SequentialStreamAdapter::new(dtype, stream::iter(rebuilt).chain(stream))
                .sendable();
            return self
                .fallback
                .write_stream(ctx, segment_sink, replay, eof, session)
                .await;
        }

        let rows = buffered.rows;
        let blocks = block_counts(&buffered.indices, rows, self.block_len)?;
        // Each child chunk is ordered beneath its own child's end, after the input stream ended.
        let mut indices_eof = eof.split_off();
        let mut values_eof = eof.split_off();
        let indices = single_chunk_stream(
            INDICES_DTYPE,
            indices_eof.split_off().advance(),
            buffered.indices_array(),
        );
        let values = single_chunk_stream(
            dtype.clone(),
            values_eof.split_off().advance(),
            buffered.values_array()?,
        );
        drop(buffered);

        let handle = session.handle();
        let write_child = |strategy: Arc<dyn LayoutStrategy>,
                           child: SendableSequentialStream,
                           child_eof: SequencePointer| {
            let ctx = ctx.clone();
            let segment_sink = Arc::clone(&segment_sink);
            let session = session.clone();
            handle.spawn_nested(move |_| async move {
                strategy
                    .write_stream(ctx, segment_sink, child, child_eof, &session)
                    .await
            })
        };
        let indices = write_child(Arc::clone(&self.indices), indices, indices_eof);
        let values = write_child(Arc::clone(&self.values), values, values_eof);
        let (indices, values) = try_join(indices, values).await?;

        Ok(SparseLayout::try_new(dtype, rows, blocks, indices, values)?.into_layout())
    }
}

fn single_chunk_stream(
    dtype: DType,
    sequence_id: SequenceId,
    chunk: ArrayRef,
) -> SendableSequentialStream {
    SequentialStreamAdapter::new(dtype, stream::iter([Ok((sequence_id, chunk))])).sendable()
}

/// Counts the sorted present-row `indices` falling in each block of `block_len` rows.
fn block_counts(indices: &[u64], rows: u64, block_len: u64) -> VortexResult<SparseBlocks> {
    let blocks = usize::try_from(rows.div_ceil(block_len))?;
    let mut counts = vec![0u64; blocks];
    for &index in indices {
        counts[usize::try_from(index / block_len)?] += 1;
    }
    SparseBlocks::try_new(block_len, counts, rows)
}
