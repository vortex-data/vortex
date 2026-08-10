// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

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
use vortex_bench::compress::Compressor;
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

#[async_trait]
impl Compressor for ParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn compress(&self, parquet_path: &Path) -> anyhow::Result<(u64, Duration)> {
        // Read the input parquet file
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let reader = builder.build()?;
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

        // Compress with our compression settings
        let mut buf = Vec::new();
        let start = Instant::now();
        let size = parquet_compress_write(batches, schema, self.compression, &mut buf)?;
        let elapsed = start.elapsed();
        Ok((size as u64, elapsed))
    }

    async fn decompress(&self, parquet_path: &Path) -> anyhow::Result<Duration> {
        // First compress to get the bytes we'll decompress
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let reader = builder.build()?;
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

        let mut buf = Vec::new();
        parquet_compress_write(batches, schema, self.compression, &mut buf)?;

        let buf = Bytes::from(buf);

        // Now decompress
        let timer = Instant::now();
        parquet_decompress_read(buf)?;
        Ok(timer.elapsed())
    }

    async fn compress_runs(
        &self,
        parquet_path: &Path,
        iterations: usize,
    ) -> anyhow::Result<(u64, Vec<Duration>)> {
        // Read the input once. `RecordBatch` is `Arc`-backed, so cloning the batch list per
        // run shares the same buffers instead of re-reading the file.
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let batches: Vec<RecordBatch> = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        let mut compressed_size = 0;
        let mut runs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let mut buf = Vec::new();
            let start = Instant::now();
            let size = parquet_compress_write(
                batches.clone(),
                Arc::clone(&schema),
                self.compression,
                &mut buf,
            )?;
            runs.push(start.elapsed());
            compressed_size = size as u64;
        }
        Ok((compressed_size, runs))
    }

    async fn decompress_runs(
        &self,
        parquet_path: &Path,
        iterations: usize,
    ) -> anyhow::Result<Vec<Duration>> {
        // Read and compress once; only the read back is timed.
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let batches: Vec<RecordBatch> = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        let mut buf = Vec::new();
        parquet_compress_write(batches, schema, self.compression, &mut buf)?;
        let buf = Bytes::from(buf);

        let mut runs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let timer = Instant::now();
            parquet_decompress_read(buf.clone())?;
            runs.push(timer.elapsed());
        }
        Ok(runs)
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
