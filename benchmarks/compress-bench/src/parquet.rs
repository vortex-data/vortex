// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::bail;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use vortex_bench::Format;
use vortex_bench::compress::Compressed;
use vortex_bench::compress::CompressedData;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::Uncompressed;
use vortex_bench::compress::read_projection;

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

/// Return the Arrow memory size after decoding the input Parquet file.
pub fn arrow_uncompressed_size(parquet_path: &Path) -> anyhow::Result<u64> {
    let file = File::open(parquet_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut total = 0u64;

    for batch in reader {
        let batch = batch?;
        let batch_size = u64::try_from(batch.get_array_memory_size())?;
        total = total
            .checked_add(batch_size)
            .ok_or_else(|| anyhow::anyhow!("Arrow memory size exceeds u64"))?;
    }

    Ok(total)
}

#[async_trait]
impl Compressor for ParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn load(&self, parquet_path: &Path) -> anyhow::Result<Uncompressed> {
        Uncompressed::read_arrow(parquet_path)
    }

    async fn compress(&self, input: &Uncompressed) -> anyhow::Result<Compressed> {
        let (schema, batches) = input.arrow()?;

        // Compress with our compression settings
        let mut buf = Vec::new();
        let start = Instant::now();
        let size = parquet_compress_write(batches, Arc::clone(schema), self.compression, &mut buf)?;
        let elapsed = start.elapsed();

        Ok(Compressed {
            data: CompressedData::Bytes(Bytes::from(buf)),
            size: size as u64,
            elapsed,
        })
    }

    async fn decompress(&self, compressed: &Compressed) -> anyhow::Result<Duration> {
        let CompressedData::Bytes(buf) = &compressed.data else {
            bail!("Parquet decompression expects in-memory bytes");
        };

        let timer = Instant::now();
        parquet_decompress_read(buf.clone())?;
        Ok(timer.elapsed())
    }
}

#[inline(never)]
pub fn parquet_compress_write(
    batches: &[RecordBatch],
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
        writer.write(batch)?;
    }
    writer.flush()?;
    let n_bytes = writer.bytes_written();
    writer.close()?;
    Ok(n_bytes)
}

#[inline(never)]
pub fn parquet_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(buf)?;
    if let Some(cols) = read_projection(builder.schema().fields().len()) {
        // Project the given top-level (root) columns.
        let mask = ProjectionMask::roots(builder.parquet_schema(), cols.iter().copied());
        builder = builder.with_projection(mask);
    }
    let reader = builder.build()?;
    let mut nbytes = 0;
    for batch in reader {
        nbytes += batch?.get_array_memory_size()
    }

    Ok(nbytes)
}
