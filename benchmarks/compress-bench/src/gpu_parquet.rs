// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU Parquet decompression backend, timed through cuDF.
//!
//! cuDF's `read_parquet` performs the whole read on the device — page header decode,
//! codec decompression, dictionary/RLE/plain decoding and column assembly — which makes it
//! the like-for-like opponent for the Vortex GPU backend, which likewise decodes all the way
//! to canonical arrays on device.
//!
//! cuDF is reached through its prebuilt `cudf-cu12` wheel rather than by linking libcudf, so
//! it stays a runtime dependency of this benchmark and never enters the Rust build. The
//! measurement is taken inside [`CUDF_SCRIPT`], so interpreter start, `import cudf` and CUDA
//! context creation are excluded; only the reads themselves are timed.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use tempfile::NamedTempFile;
use vortex_bench::Format;
use vortex_bench::compress::Compressor;

use crate::gpu_writer::GpuCodec;
use crate::gpu_writer::gpu_writer_properties;

/// Repo-relative path of the script that performs and times the cuDF read.
const CUDF_SCRIPT: &str = "scripts/cudf-parquet-read.py";

/// Parquet compressor whose decompression measurement is a full cuDF GPU read.
pub struct GpuParquetCompressor {
    codec: GpuCodec,
    verify: bool,
}

/// What the cuDF script reports back.
#[derive(Debug, Deserialize)]
struct CudfReadReport {
    /// Fastest timed read, in nanoseconds.
    min_ns: u64,
    rows: u64,
    columns: u64,
}

impl GpuParquetCompressor {
    /// Create a backend that writes pages with `codec` and times cuDF reading them back.
    ///
    /// When `verify` is set, the GPU read is cross-checked against a CPU Parquet read of the
    /// same file before the measurement is reported.
    pub fn new(codec: GpuCodec, verify: bool) -> Self {
        Self { codec, verify }
    }

    /// Rewrite the source Parquet file with GPU-friendly writer settings.
    fn write_gpu_parquet(&self, parquet_path: &Path) -> Result<(NamedTempFile, u64)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(parquet_path)?)?;
        let schema = Arc::clone(builder.schema());
        let batches: Vec<RecordBatch> = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        let output = NamedTempFile::new()?;
        let mut writer = ArrowWriter::try_new(
            output.reopen()?,
            schema,
            Some(gpu_writer_properties(self.codec)),
        )?;
        for batch in batches {
            writer.write(&batch)?;
        }
        writer.flush()?;
        let size = writer.bytes_written() as u64;
        writer.close()?;
        Ok((output, size))
    }
}

#[async_trait]
impl Compressor for GpuParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn compress(&self, parquet_path: &Path) -> Result<(u64, Duration)> {
        let start = Instant::now();
        let (_file, size) = self.write_gpu_parquet(parquet_path)?;
        Ok((size, start.elapsed()))
    }

    async fn decompress(&self, parquet_path: &Path) -> Result<Duration> {
        let (gpu_file, _) = self.write_gpu_parquet(parquet_path)?;
        let report = run_cudf_read(gpu_file.path(), self.verify)?;

        ensure!(
            report.rows > 0 && report.columns > 0,
            "cuDF read {} rows and {} columns, expected a non-empty table",
            report.rows,
            report.columns
        );

        Ok(Duration::from_nanos(report.min_ns))
    }
}

/// Runs the cuDF read script and returns the timing it measured.
fn run_cudf_read(path: &Path, verify: bool) -> Result<CudfReadReport> {
    let mut command = Command::new("python3");
    command.arg(CUDF_SCRIPT).arg(path);
    if verify {
        command.arg("--verify");
    }

    let output = command.output().with_context(|| {
        format!("failed to run {CUDF_SCRIPT}; is cudf-cu12 installed on this host?")
    })?;

    if !output.status.success() {
        bail!(
            "{CUDF_SCRIPT} exited with {}:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "could not parse the report from {CUDF_SCRIPT}: {}",
            String::from_utf8_lossy(&output.stdout).trim()
        )
    })
}
