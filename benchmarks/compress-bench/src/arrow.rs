// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use anyhow::bail;
use arrow_array::RecordBatch;
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use vortex_bench::Format;
use vortex_bench::compress::Compressed;
use vortex_bench::compress::CompressedData;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::Uncompressed;
use vortex_bench::compress::read_projection;

/// Uncompressed Arrow IPC file baseline.
pub struct ArrowIpcCompressor;

#[async_trait]
impl Compressor for ArrowIpcCompressor {
    fn format(&self) -> Format {
        Format::ArrowIpc
    }

    async fn load(&self, parquet_path: &Path) -> anyhow::Result<Uncompressed> {
        Uncompressed::read_arrow(parquet_path)
    }

    async fn compress(&self, input: &Uncompressed) -> anyhow::Result<Compressed> {
        let (schema, batches) = input.arrow()?;

        let mut buf = Vec::new();
        let start = Instant::now();
        arrow_file_write(&mut buf, schema, batches)?;
        let elapsed = start.elapsed();

        Ok(Compressed {
            size: buf.len() as u64,
            data: CompressedData::Bytes(Bytes::from(buf)),
            elapsed,
        })
    }

    async fn decompress(&self, compressed: &Compressed) -> anyhow::Result<Duration> {
        let CompressedData::Bytes(buf) = &compressed.data else {
            bail!("Arrow IPC decompression expects in-memory bytes");
        };

        // The projection needs the column count; read the footer before the clock starts.
        let root_columns = FileReader::try_new(Cursor::new(buf.clone()), None)?
            .schema()
            .fields()
            .len();

        let start = Instant::now();
        arrow_file_read(buf.clone(), root_columns)?;
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
