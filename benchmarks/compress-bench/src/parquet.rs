// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use vortex::array::Array;
use vortex::array::arrays::ChunkedVTable;
use vortex_bench::Format;

use crate::bench::Compressor;
use crate::chunked_to_vec_record_batch;

/// Compressor implementation for Parquet format with ZSTD compression.
pub struct ParquetCompressor {
    compression: Compression,
}

impl ParquetCompressor {
    pub fn new() -> Self {
        Self {
            compression: Compression::ZSTD(ZstdLevel::default()),
        }
    }

    pub fn with_compression(compression: Compression) -> Self {
        Self { compression }
    }
}

impl Default for ParquetCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for ParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn compress(&self, array: &dyn Array) -> anyhow::Result<(Bytes, Duration)> {
        let chunked = array.as_::<ChunkedVTable>().clone();
        let (batches, schema) = chunked_to_vec_record_batch(chunked)?;

        let mut buf = Vec::new();
        let start = Instant::now();
        parquet_compress_write(batches, schema, self.compression, &mut buf)?;
        let elapsed = start.elapsed();
        Ok((Bytes::from(buf), elapsed))
    }

    async fn decompress(&self, data: Bytes) -> anyhow::Result<usize> {
        parquet_decompress_read(data)
    }
}

#[inline(never)]
pub fn parquet_compress_write(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    compression: Compression,
    buf: &mut Vec<u8>,
) -> anyhow::Result<usize> {
    let mut buf = Cursor::new(buf);
    let writer_properties = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(writer_properties))?;
    for batch in batches {
        writer.write(&batch)?;
    }
    writer.flush()?;
    let n_bytes = writer.bytes_written();
    writer.close()?;
    Ok(n_bytes)
}

#[inline(never)]
pub fn parquet_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;
    let reader = builder.build()?;
    let mut nbytes = 0;
    for batch in reader {
        nbytes += batch?.get_array_memory_size()
    }

    Ok(nbytes)
}
