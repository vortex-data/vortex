use std::ops::Range;

use futures_util::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::{ChunkedArray, PrimitiveArray};
use vortex_array::compute::unary::{subtract_scalar, try_cast};
use vortex_array::compute::{search_sorted, slice, take, SearchSortedSide, TakeOptions};
use vortex_array::stats::ArrayStatistics;
use vortex_array::stream::{ArrayStream, ArrayStreamExt};
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_dtype::PType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_io::{VortexBufReader, VortexReadAt};
use vortex_ipc::stream_reader::StreamArrayReader;
use vortex_scalar::Scalar;

use crate::chunked_reader::ChunkedArrayReader;

impl<R: VortexReadAt> ChunkedArrayReader<R> {
    pub async fn take_rows(&mut self, indices: &ArrayData) -> VortexResult<ArrayData> {
        // Figure out if the row indices are sorted / unique. If not, we need to sort them.
        if indices
            .statistics()
            .compute_is_strict_sorted()
            .unwrap_or(false)
        {
            // With strict-sorted indices, we can take the rows directly.
            return self.take_rows_strict_sorted(indices).await;
        }

        //         // Figure out which chunks are relevant to the read operation using the row_offsets array.
        //         // Depending on whether there are more indices than chunks, we may wish to perform this
        //         // join differently.
        //
        //         // Coalesce the chunks we care about by some metric.
        //
        //         // TODO(ngates): we could support read_into for array builders since we know the size
        //         //  of the result.
        //         // Read the relevant chunks.
        // Reshuffle the result as per the original sort order.
        unimplemented!("Unsorted 'take' operation is not supported yet")
    }

    /// Take rows from a chunked array given strict sorted indices.
    ///
    /// The strategy for doing this depends on the quantity and distribution of the indices...
    ///
    /// For now, we will find the relevant chunks, coalesce them, and read.
    async fn take_rows_strict_sorted(&mut self, indices: &ArrayData) -> VortexResult<ArrayData> {
        // Figure out which chunks are relevant.
        let chunk_idxs = find_chunks(&self.row_offsets, indices)?;
        // Coalesce the chunks that we're going to read from.
        let coalesced_chunks = self.coalesce_chunks(chunk_idxs.as_ref());

        let mut start_chunks: Vec<u32> = Vec::with_capacity(coalesced_chunks.len());
        let mut stop_chunks: Vec<u32> = Vec::with_capacity(coalesced_chunks.len());
        for (i, chunks) in coalesced_chunks.iter().enumerate() {
            start_chunks.push(
                chunks
                    .first()
                    .ok_or_else(|| vortex_err!("Coalesced chunk {i} cannot be empty"))?
                    .chunk_idx,
            );
            stop_chunks.push(
                chunks
                    .last()
                    .ok_or_else(|| vortex_err!("Coalesced chunk {i} cannot be empty"))?
                    .chunk_idx
                    + 1,
            );
        }

        // Grab the row and byte offsets for each chunk range.
        let start_chunks = PrimitiveArray::from(start_chunks).into_array();
        let start_rows =
            take(&self.row_offsets, &start_chunks, TakeOptions::default())?.into_primitive()?;
        let start_bytes =
            take(&self.byte_offsets, &start_chunks, TakeOptions::default())?.into_primitive()?;

        let stop_chunks = PrimitiveArray::from(stop_chunks).into_array();
        let stop_rows =
            take(&self.row_offsets, &stop_chunks, TakeOptions::default())?.into_primitive()?;
        let stop_bytes =
            take(&self.byte_offsets, &stop_chunks, TakeOptions::default())?.into_primitive()?;

        // For each chunk-range, read the data as an ArrayStream and call take on it.
        let chunks = stream::iter(0..coalesced_chunks.len())
            .map(|chunk_idx| {
                let (start_byte, stop_byte) = (
                    start_bytes.get_as_cast::<u64>(chunk_idx),
                    stop_bytes.get_as_cast::<u64>(chunk_idx),
                );
                let (start_row, stop_row) = (
                    start_rows.get_as_cast::<u64>(chunk_idx),
                    stop_rows.get_as_cast::<u64>(chunk_idx),
                );
                self.take_from_chunk(indices, start_byte..stop_byte, start_row..stop_row)
            })
            .buffered(10)
            .try_flatten()
            .try_collect()
            .await?;

        Ok(ChunkedArray::try_new(chunks, (*self.dtype).clone())?.into_array())
    }

    /// Coalesce reads for the given chunks.
    ///
    /// This depends on a few factors:
    /// * The number of bytes between adjacent selected chunks.
    /// * The latency of the underlying storage.
    /// * The throughput of the underlying storage.
    fn coalesce_chunks(&self, chunk_idxs: &[ChunkIndices]) -> Vec<Vec<ChunkIndices>> {
        let _hint = self.read.performance_hint();
        chunk_idxs
            .iter()
            .map(|chunk_idx| vec![chunk_idx.clone()])
            .collect_vec()
    }

    async fn take_from_chunk(
        &self,
        indices: &ArrayData,
        byte_range: Range<u64>,
        row_range: Range<u64>,
    ) -> VortexResult<impl ArrayStream> {
        let range_byte_len = byte_range.end - byte_range.start;

        // Relativize the indices to these chunks
        let indices_start =
            search_sorted(indices, row_range.start, SearchSortedSide::Left)?.to_index();
        let indices_stop =
            search_sorted(indices, row_range.end, SearchSortedSide::Right)?.to_index();
        let relative_indices = slice(indices, indices_start, indices_stop)?;
        let row_start_scalar = Scalar::from(row_range.start).cast(relative_indices.dtype())?;
        let relative_indices = subtract_scalar(&relative_indices, &row_start_scalar)?;

        // TODO(ngates): instead of reading the whole range into a buffer, we should stream
        //  the byte range (e.g. if its coming from an HTTP endpoint) and wrap that with an
        //  MesssageReader.
        let buffer = self
            .read
            .read_byte_range(byte_range.start, range_byte_len)
            .await?;

        let buf_reader = VortexBufReader::new(buffer);

        let reader = StreamArrayReader::try_new(buf_reader, self.context.clone())
            .await?
            .with_dtype(self.dtype.clone());

        // Take the indices from the stream.
        reader.into_array_stream().take_rows(relative_indices)
    }
}

/// Find the chunks that are relevant to the read operation.
/// Both the row_offsets and indices arrays must be strict-sorted.
fn find_chunks(row_offsets: &ArrayData, indices: &ArrayData) -> VortexResult<Vec<ChunkIndices>> {
    // TODO(ngates): lots of optimizations to be had here, potentially lots of push-down.
    //  For now, we just flatten everything into primitive arrays and iterate.
    let row_offsets = try_cast(row_offsets, PType::U64.into())?.into_primitive()?;
    let _rows = format!("{:?}", row_offsets.maybe_null_slice::<u64>());
    let indices = try_cast(indices, PType::U64.into())?.into_primitive()?;
    let _indices = format!("{:?}", indices.maybe_null_slice::<u64>());

    if let (Some(last_idx), Some(num_rows)) = (
        indices.maybe_null_slice::<u64>().last(),
        row_offsets.maybe_null_slice::<u64>().last(),
    ) {
        if last_idx >= num_rows {
            vortex_bail!("Index {} out of bounds {}", last_idx, num_rows);
        }
    }

    let mut chunks = HashMap::new();

    for (pos, idx) in indices.maybe_null_slice::<u64>().iter().enumerate() {
        let chunk_idx = search_sorted(row_offsets.as_ref(), *idx, SearchSortedSide::Right)?
            .to_ends_index(row_offsets.len())
            .saturating_sub(1);
        chunks
            .entry(chunk_idx as u32)
            .and_modify(|chunk_indices: &mut ChunkIndices| {
                chunk_indices.indices_stop = (pos + 1) as u64;
            })
            .or_insert(ChunkIndices {
                chunk_idx: chunk_idx as u32,
                indices_start: pos as u64,
                indices_stop: (pos + 1) as u64,
            });
    }

    Ok(chunks
        .keys()
        .sorted()
        .map(|k| &chunks[k])
        .cloned()
        .collect_vec())
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ChunkIndices {
    chunk_idx: u32,
    // The position into the indices array that is covered by this chunk.
    indices_start: u64,
    indices_stop: u64,
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod test {
    use std::sync::Arc;

    use futures_executor::block_on;
    use itertools::Itertools;
    use vortex_array::array::{ChunkedArray, PrimitiveArray};
    use vortex_array::{ArrayLen, Context, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::Buffer;
    use vortex_dtype::PType;
    use vortex_error::VortexResult;
    use vortex_io::VortexBufReader;
    use vortex_ipc::messages::reader::MessageReader;
    use vortex_ipc::stream_writer::StreamArrayWriter;

    use crate::chunked_reader::ChunkedArrayReader;

    fn chunked_array() -> VortexResult<StreamArrayWriter<Vec<u8>>> {
        let c = ChunkedArray::try_new(
            vec![PrimitiveArray::from((0i32..1000).collect_vec()).into_array(); 10],
            PType::I32.into(),
        )?
        .into_array();

        block_on(async { StreamArrayWriter::new(vec![]).write_array(c).await })
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_take_rows() -> VortexResult<()> {
        let writer = chunked_array()?;

        let array_layout = writer.array_layouts()[0].clone();
        let byte_offsets = PrimitiveArray::from(array_layout.chunks.byte_offsets.clone());
        let row_offsets = PrimitiveArray::from(array_layout.chunks.row_offsets);

        let buffer = Buffer::from(writer.into_inner());

        let mut msgs =
            block_on(async { MessageReader::try_new(VortexBufReader::new(buffer.clone())).await })?;
        let dtype = Arc::new(block_on(async { msgs.read_dtype().await })?);

        let mut reader = ChunkedArrayReader::try_new(
            buffer,
            Arc::new(Context::default()),
            dtype,
            byte_offsets.into_array(),
            row_offsets.into_array(),
        )
        .unwrap();

        let result = block_on(async {
            reader
                .take_rows(&PrimitiveArray::from(vec![0u64, 10, 10_000 - 1]).into_array())
                .await
        })?
        .into_primitive()?;

        assert_eq!(result.len(), 3);
        assert_eq!(result.maybe_null_slice::<i32>(), &[0, 10, 999]);
        Ok(())
    }
}
