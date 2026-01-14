// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

pub struct ChunkedReader2 {
    row_count: u64,
    dtype: DType,
    chunks: Vec<ReaderRef>,
}

impl Reader for ChunkedReader2 {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let mut remaining_start = row_range.start;
        let mut remaining_end = row_range.end;
        let mut streams = Vec::new();

        for chunk in &self.chunks {
            let chunk_row_count = chunk.row_count();

            if remaining_start >= chunk_row_count {
                // This chunk is before the requested range
                remaining_start -= chunk_row_count;
                remaining_end -= chunk_row_count;
                continue;
            }

            let start_in_chunk = remaining_start;
            let end_in_chunk = if remaining_end <= chunk_row_count {
                remaining_end
            } else {
                chunk_row_count
            };

            streams.push(chunk.execute(start_in_chunk..end_in_chunk)?);

            remaining_start = 0;
            if remaining_end <= chunk_row_count {
                break;
            } else {
                remaining_end -= chunk_row_count;
            }
        }

        Ok(Box::new(ChunkedReaderStream {
            dtype: self.dtype.clone(),
            chunks: streams,
        }))
    }
}

struct ChunkedReaderStream {
    dtype: DType,
    chunks: Vec<ReaderStreamRef>,
}

impl ReaderStream for ChunkedReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        self.chunks
            .iter()
            .map(|s| s.next_chunk_len())
            .find(|len| len.is_some())
            .flatten()
    }

    fn next_chunk(
        &mut self,
        selection: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        // Remove any chunks that are already exhausted
        loop {
            if self.chunks.is_empty() {
                vortex_bail!("Early termination of chunked layout");
            }
            if self.chunks[0].next_chunk_len().is_none() {
                self.chunks.remove(0);
            } else {
                break;
            }
        }

        // Get the length of the next chunk
        let mut next_len = self.chunks[0]
            .next_chunk_len()
            .ok_or_else(|| vortex_err!("Early termination of chunked layout"))?;

        if selection.len() <= next_len {
            // The selection is smaller than the next chunk length, therefore we only need one chunk
            return self.chunks[0].next_chunk(selection);
        }

        // Otherwise, we need to gather from multiple chunks
        let mut selection = selection.clone();
        let mut futs = vec![];
        while !selection.is_empty() {
            if self.chunks.is_empty() {
                vortex_bail!("Early termination of chunked layout");
            }

            // Slice off the right amount of selection for this chunk
            let next_sel = selection.slice(..next_len);
            selection = selection.slice(next_len..);

            let fut = self.chunks[0].next_chunk(&next_sel)?;
            futs.push(fut);

            // Remove any chunks that are already exhausted
            loop {
                if self.chunks[0].next_chunk_len().is_none() {
                    self.chunks.remove(0);
                } else {
                    break;
                }
            }
        }

        let dtype = self.dtype.clone();
        Ok(async move {
            let arrays = try_join_all(futs).await?;
            Ok(ChunkedArray::try_new(arrays, dtype)?.into_array())
        }
        .boxed())
    }
}
