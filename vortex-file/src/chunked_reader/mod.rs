use std::sync::Arc;

use vortex_array::compute::unary::scalar_at;
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayData, Context};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_io::{VortexBufReader, VortexReadAt};
use vortex_ipc::stream_reader::StreamArrayReader;

mod take_rows;

/// A reader for a chunked array.
pub struct ChunkedArrayReader<R: VortexReadAt> {
    read: R,
    context: Arc<Context>,
    dtype: Arc<DType>,

    // One row per chunk + 1 row for the end of the last chunk.
    byte_offsets: ArrayData,
    row_offsets: ArrayData,
}

impl<R: VortexReadAt> ChunkedArrayReader<R> {
    pub fn try_new(
        read: R,
        context: Arc<Context>,
        dtype: Arc<DType>,
        byte_offsets: ArrayData,
        row_offsets: ArrayData,
    ) -> VortexResult<Self> {
        Self::validate(&byte_offsets, &row_offsets)?;
        Ok(Self {
            read,
            context,
            dtype,
            byte_offsets,
            row_offsets,
        })
    }

    pub fn nchunks(&self) -> usize {
        self.byte_offsets.len()
    }

    fn validate(byte_offsets: &ArrayData, row_offsets: &ArrayData) -> VortexResult<()> {
        if byte_offsets.len() != row_offsets.len() {
            vortex_bail!("byte_offsets and row_offsets must have the same length");
        }
        Ok(())
    }

    // Making a new ArrayStream requires us to clone the reader to make
    // multiple streams that can each use the reader.
    pub async fn array_stream(&mut self) -> impl ArrayStream + '_ {
        let byte_offset = scalar_at(&self.byte_offsets, 0)
            .and_then(|s| u64::try_from(&s))
            .vortex_expect("Failed to convert byte_offset to u64");

        let mut buf_reader = VortexBufReader::new(self.read.clone());
        buf_reader.set_position(byte_offset);

        StreamArrayReader::try_new(buf_reader, self.context.clone())
            .await
            .vortex_expect("Failed to create stream array reader")
            .with_dtype(self.dtype.clone())
            .into_array_stream()
    }
}
