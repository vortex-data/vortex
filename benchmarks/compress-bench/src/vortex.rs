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
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::dtype::FieldNames;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::utils::parallelism::get_available_parallelism;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::read_projection;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_morsel::MorselScan;
use vortex_morsel::build_plan;
use vortex_morsel::morsels;
use vortex_morsel::nodes::ConjunctMode;

const MORSEL_ROWS: u64 = 131_072;

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
        let file = SESSION.open_options().open_buffer(data)?;
        let source_dtype = file.dtype().clone();
        let root_columns = source_dtype
            .as_struct_fields_opt()
            .map_or(0, |fields| fields.nfields());
        let projection = if let Some(cols) = read_projection(root_columns) {
            // Columns are named "0".."num_columns-1"; project the given subset.
            let names: FieldNames = cols.iter().map(|i| i.to_string()).collect();
            select(names, root())
        } else {
            root()
        };
        let plan = Arc::new(build_plan(
            file.footer().layout(),
            &projection,
            None,
            ConjunctMode::Cascade,
        )?);
        let cut = morsels(&plan, MORSEL_ROWS);
        let threads = get_available_parallelism().unwrap_or(1);
        let (batches, _) = MorselScan::new(plan, file.segment_source(), SESSION.clone())
            .with_threads(threads)
            .with_morsels(cut)
            .run()?;

        let mut ctx = SESSION.create_execution_ctx();
        for batch in batches {
            let _canonical = batch.execute::<Canonical>(&mut ctx)?;
        }
        Ok(start.elapsed())
    }
}
