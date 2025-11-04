// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryStreamExt};
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_expr::Expression;

use crate::gpu::children::LazyGpuReaderChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSource;
use crate::{GpuArrayFuture, GpuLayoutReader, GpuLayoutReaderRef};

pub struct GpuChunkedLayoutReader {
    layout: ChunkedLayout,
    name: Arc<str>,
    ctx: Arc<CudaContext>,
    lazy_children: LazyGpuReaderChildren,
    /// Row offset for each chunk
    chunk_offsets: Vec<u64>,
}

impl GpuChunkedLayoutReader {
    pub fn new(
        layout: ChunkedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: Arc<CudaContext>,
    ) -> Self {
        let nchildren = layout.nchildren();

        let mut chunk_offsets = vec![0; nchildren + 1];
        for i in 1..nchildren {
            chunk_offsets[i] = chunk_offsets[i - 1] + layout.children().child_row_count(i - 1);
        }
        chunk_offsets[nchildren] = layout.row_count();

        let lazy_children = LazyGpuReaderChildren::new(layout.children().clone(), segment_source);

        Self {
            layout,
            name,
            ctx,
            lazy_children,
            chunk_offsets,
        }
    }

    /// Return the [`LayoutReader`] for the given chunk.
    fn chunk_reader(&self, idx: usize) -> VortexResult<&GpuLayoutReaderRef> {
        self.lazy_children.get(
            idx,
            self.layout.dtype(),
            &format!("{}.[{}]", self.name, idx).into(),
            &self.ctx,
        )
    }

    fn chunk_offset(&self, idx: usize) -> u64 {
        self.chunk_offsets.get(idx).copied().unwrap_or_else(|| {
            vortex_panic!(
                "Internal error: Chunk offset {idx} out of bounds (num_children: {}, num_offsets: {}). \
                This indicates a bug in ChunkedReader initialization or chunk_range calculation.",
                self.layout.nchildren(),
                self.chunk_offsets.len()
            )
        })
    }

    fn chunk_range(&self, row_range: &Range<u64>) -> Range<usize> {
        let start_chunk = self
            .chunk_offsets
            .binary_search(&row_range.start)
            .unwrap_or_else(|_| vortex_panic!("GpuChunkedLayoutReader can only read full chunks"));
        let end_chunk = self
            .chunk_offsets
            .binary_search(&row_range.end)
            .unwrap_or_else(|_| vortex_panic!("GpuChunkedLayoutReader can only read full chunks"));
        start_chunk..end_chunk
    }

    fn ranges(&self, row_range: &Range<u64>) -> impl Iterator<Item = (usize, Range<u64>)> {
        self.chunk_range(row_range).map(move |chunk_idx| {
            let chunk_row_range = self.chunk_offset(chunk_idx)..self.chunk_offset(chunk_idx + 1);

            let chunk_len = chunk_row_range
                .end
                .checked_sub(chunk_row_range.start)
                .vortex_expect("Invalid row range");

            let chunk_range = 0..chunk_len;

            (chunk_idx, chunk_range)
        })
    }
}

impl GpuLayoutReader for GpuChunkedLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let mut offset = row_offset;
        for i in 0..self.layout.nchildren() {
            let child = self.chunk_reader(i)?;
            child.register_splits(field_mask, offset, splits)?;
            offset += self.layout.child(i)?.row_count();
            splits.insert(offset);
        }
        Ok(())
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
    ) -> VortexResult<GpuArrayFuture> {
        let mut chunk_evals = FuturesOrdered::new();

        for (chunk_idx, chunk_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval = chunk_reader.projection_evaluation(&chunk_range, expr)?;
            chunk_evals.push_back(chunk_eval);
        }

        Ok(async move {
            let chunks: Vec<_> = chunk_evals.try_collect().await?;
            Ok(chunks
                .into_iter()
                .map(|mut c| {
                    assert_eq!(c.len(), 1);
                    c.pop().vortex_expect("must have one chunk")
                })
                .collect())
        }
        .boxed())
    }
}
