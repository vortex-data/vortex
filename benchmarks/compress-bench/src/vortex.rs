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
use vortex::dtype::FieldNames;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex_arrow::ToArrowType;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::read_projection;
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
        let mut scan = SESSION.open_options().open_buffer(data)?.scan()?;
        let source_dtype = scan.dtype()?;
        let root_columns = source_dtype
            .as_struct_fields_opt()
            .map_or(0, |fields| fields.nfields());
        if let Some(cols) = read_projection(root_columns) {
            // Columns are named "0".."num_columns-1"; project the given subset.
            let names: FieldNames = cols.iter().map(|i| i.to_string()).collect();
            let projection = select(names, root())
                .optimize_recursive(&source_dtype)?
                .bind(&source_dtype)?;
            scan = scan.with_projection(projection);
        }
        let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

        let stream = scan.into_record_batch_stream(schema)?;
        pin_mut!(stream);

        while let Some(batch) = stream.next().await {
            let _batch = batch?;
        }
        Ok(start.elapsed())
    }

    async fn compress_runs(
        &self,
        parquet_path: &Path,
        iterations: usize,
    ) -> Result<(u64, Vec<Duration>)> {
        // Decode once. `ArrayRef` is `Arc`-backed, so each run re-streams the same arrays
        // rather than re-reading and re-canonicalizing the Parquet.
        let uncompressed = parquet_to_vortex_chunks(parquet_path.to_path_buf())
            .await?
            .into_array();

        let mut compressed_size = 0;
        let mut runs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let mut buf = Vec::new();
            let start = Instant::now();
            let mut cursor = Cursor::new(&mut buf);
            SESSION
                .write_options()
                .write(&mut cursor, uncompressed.clone().to_array_stream())
                .await?;
            runs.push(start.elapsed());
            compressed_size = buf.len() as u64;
        }
        Ok((compressed_size, runs))
    }

    async fn decompress_runs(
        &self,
        parquet_path: &Path,
        iterations: usize,
    ) -> Result<Vec<Duration>> {
        // Decode and compress once; only the read back is timed, so repeating either per
        // iteration would be invisible in the results and pure cost.
        let uncompressed = parquet_to_vortex_chunks(parquet_path.to_path_buf()).await?;
        let mut buf = Vec::new();
        let mut cursor = Cursor::new(&mut buf);
        SESSION
            .write_options()
            .write(&mut cursor, uncompressed.into_array().to_array_stream())
            .await?;
        let data = Bytes::from(buf);

        let mut runs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let mut scan = SESSION.open_options().open_buffer(data.clone())?.scan()?;
            let source_dtype = scan.dtype()?;
            let root_columns = source_dtype
                .as_struct_fields_opt()
                .map_or(0, |fields| fields.nfields());
            if let Some(cols) = read_projection(root_columns) {
                let names: FieldNames = cols.iter().map(|i| i.to_string()).collect();
                let projection = select(names, root())
                    .optimize_recursive(&source_dtype)?
                    .bind(&source_dtype)?;
                scan = scan.with_projection(projection);
            }
            let schema = Arc::new(scan.dtype()?.to_arrow_schema()?);

            let stream = scan.into_record_batch_stream(schema)?;
            pin_mut!(stream);
            while let Some(batch) = stream.next().await {
                let _batch = batch?;
            }
            runs.push(start.elapsed());
        }
        Ok(runs)
    }
}
