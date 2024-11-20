use std::fmt::{Display, Formatter};
use std::ops::Range;

use futures_util::{Stream, TryStreamExt};
use vortex_array::array::{ChunkedArray, ChunkedEncoding};
use vortex_array::encoding::EncodingVTable;
use vortex_array::stream::ArrayStream;
use vortex_array::ArrayData;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexWrite;

use crate::messages::writer::MessageWriter;

#[cfg(test)]
mod tests;

pub struct StreamArrayWriter<W: VortexWrite> {
    msgs: MessageWriter<W>,

    array_layouts: Vec<ArrayLayout>,
    page_ranges: Vec<ByteRange>,
}

impl<W: VortexWrite> StreamArrayWriter<W> {
    pub fn new(write: W) -> Self {
        Self {
            msgs: MessageWriter::new(write),
            array_layouts: vec![],
            page_ranges: vec![],
        }
    }

    pub fn array_layouts(&self) -> &[ArrayLayout] {
        &self.array_layouts
    }

    pub fn page_ranges(&self) -> &[ByteRange] {
        &self.page_ranges
    }

    pub fn into_inner(self) -> W {
        self.msgs.into_inner()
    }

    async fn write_dtype(&mut self, dtype: &DType) -> VortexResult<ByteRange> {
        let begin = self.msgs.tell();
        self.msgs.write_dtype(dtype).await?;
        let end = self.msgs.tell();
        Ok(ByteRange { begin, end })
    }

    async fn write_array_chunks<S>(&mut self, mut stream: S) -> VortexResult<ChunkOffsets>
    where
        S: Stream<Item = VortexResult<ArrayData>> + Unpin,
    {
        let mut byte_offsets = vec![self.msgs.tell()];
        let mut row_offsets = vec![0];
        let mut row_offset = 0;

        while let Some(chunk) = stream.try_next().await? {
            row_offset += chunk.len() as u64;
            row_offsets.push(row_offset);
            self.msgs.write_batch(chunk).await?;
            byte_offsets.push(self.msgs.tell());
        }

        Ok(ChunkOffsets::new(byte_offsets, row_offsets))
    }

    pub async fn write_array_stream<S: ArrayStream + Unpin>(
        mut self,
        mut array_stream: S,
    ) -> VortexResult<Self> {
        let dtype_pos = self.write_dtype(array_stream.dtype()).await?;
        let chunk_pos = self.write_array_chunks(&mut array_stream).await?;
        self.array_layouts.push(ArrayLayout {
            dtype: dtype_pos,
            chunks: chunk_pos,
        });
        Ok(self)
    }

    pub async fn write_array(self, array: ArrayData) -> VortexResult<Self> {
        if array.is_encoding(ChunkedEncoding.id()) {
            self.write_array_stream(ChunkedArray::try_from(array)?.array_stream())
                .await
        } else {
            self.write_array_stream(array.into_array_stream()).await
        }
    }

    pub async fn write_page(mut self, buffer: Buffer) -> VortexResult<Self> {
        let begin = self.msgs.tell();
        self.msgs.write_page(buffer).await?;
        let end = self.msgs.tell();
        self.page_ranges.push(ByteRange { begin, end });
        Ok(self)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ByteRange {
    pub begin: u64,
    pub end: u64,
}

impl Display for ByteRange {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.begin, self.end)
    }
}

impl ByteRange {
    pub fn new(begin: u64, end: u64) -> Self {
        assert!(end > begin, "Buffer end must be after its beginning");
        Self { begin, end }
    }

    pub fn len(&self) -> u64 {
        self.end - self.begin
    }

    pub fn is_empty(&self) -> bool {
        self.begin == self.end
    }

    pub fn to_range(&self) -> Range<usize> {
        self.begin as usize..self.end as usize
    }
}

#[derive(Clone, Debug)]
pub struct ArrayLayout {
    pub dtype: ByteRange,
    pub chunks: ChunkOffsets,
}

#[derive(Clone, Debug)]
pub struct ChunkOffsets {
    pub byte_offsets: Vec<u64>,
    pub row_offsets: Vec<u64>,
}

impl ChunkOffsets {
    pub fn new(byte_offsets: Vec<u64>, row_offsets: Vec<u64>) -> Self {
        Self {
            byte_offsets,
            row_offsets,
        }
    }
}
