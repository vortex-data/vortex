// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_bench::Format;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::read_projection;

/// Uncompressed Arrow IPC file baseline.
pub struct ArrowCompressor;

#[async_trait]
impl Compressor for ArrowCompressor {
    fn format(&self) -> Format {
        Format::Arrow
    }

    async fn compress(&self, parquet_path: &Path) -> anyhow::Result<(u64, Duration)> {
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        let mut buf = Vec::new();
        let start = Instant::now();
        arrow_file_write(&mut buf, &schema, &batches)?;
        let elapsed = start.elapsed();
        Ok((buf.len() as u64, elapsed))
    }

    async fn decompress(&self, parquet_path: &Path) -> anyhow::Result<Duration> {
        let file = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = Arc::clone(builder.schema());
        let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        let mut buf = Vec::new();
        arrow_file_write(&mut buf, &schema, &batches)?;

        let start = Instant::now();
        arrow_file_read(Bytes::from(buf), schema.fields().len())?;
        Ok(start.elapsed())
    }
}

#[inline(never)]
fn arrow_file_write(
    buf: &mut Vec<u8>,
    schema: &Schema,
    batches: &[RecordBatch],
) -> anyhow::Result<()> {
    let mut writer = FileWriter::try_new(buf, schema)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

#[inline(never)]
fn arrow_file_read(buf: Bytes, root_columns: usize) -> anyhow::Result<usize> {
    let cursor = Cursor::new(buf);
    let projection = read_projection(root_columns).map(<[usize]>::to_vec);
    let reader = FileReader::try_new(cursor, projection)?;

    let mut nbytes = 0;
    for batch in reader {
        nbytes += batch?.get_array_memory_size();
    }
    Ok(nbytes)
}
