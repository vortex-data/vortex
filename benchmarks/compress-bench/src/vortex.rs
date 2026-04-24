// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::pin_mut;
use vortex::array::IntoArray;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::conversions::parquet_to_vortex_chunks;

/// Compressor implementation for Vortex format.
pub struct VortexCompressor;

#[async_trait]
impl Compressor for VortexCompressor {
    fn format(&self) -> Format {
        Format::OnDiskVortex
    }

    async fn compress(&self, parquet_path: &Path) -> Result<(u64, Duration)> {
        // Read the parquet file as an array stream
        let uncompressed = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;

        let mut buf = Vec::new();
        let start = Instant::now();
        let mut cursor = Cursor::new(&mut buf);
        SESSION
            .write_options()
            .write(&mut cursor, uncompressed.into_array().to_array_stream())
            .await?;
        let elapsed = start.elapsed();

        Ok((buf.len() as u64, elapsed))
    }

    async fn decompress(&self, parquet_path: &Path) -> Result<Duration> {
        // First compress to get the bytes we'll decompress
        let uncompressed = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        SESSION
            .write_options()
            .write(&mut cursor, uncompressed.into_array().to_array_stream())
            .await?;

        // Now decompress
        let start = Instant::now();
        let data = Bytes::from(buf);
        let scan = SESSION.open_options().open_buffer(data)?.scan()?;
        let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

        let stream = scan.into_record_batch_stream(schema)?;
        pin_mut!(stream);

        while let Some(batch) = stream.next().await {
            let _batch = batch?;
        }
        Ok(start.elapsed())
    }
}
