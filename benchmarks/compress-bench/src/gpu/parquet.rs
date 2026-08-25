// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU Parquet decompression backend, timed through cuDF.
//!
//! cuDF's `read_parquet` performs the whole read on the device — page header decode,
//! codec decompression, dictionary/RLE/plain decoding and column assembly — which makes it
//! the like-for-like opponent for the Vortex GPU backend, which likewise decodes all the way
//! to canonical arrays on device.
//!
//! cuDF has no Rust binding, so this backend drives it out of process: it spawns `python3`
//! running [`CUDF_SCRIPT`], which imports cuDF from the prebuilt `cudf-cu12` Python package
//! and prints its timings back as JSON on stdout. Nothing here links against libcudf, so cuDF
//! is a runtime requirement of the benchmark rather than a Rust build dependency — see the
//! README for the install.
//!
//! The clock lives inside that script, not around the subprocess, so process spawn,
//! interpreter start, `import cudf` and CUDA context creation are all excluded; only the reads
//! themselves are timed.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

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

use crate::gpu::writer::GpuCodec;
use crate::gpu::writer::gpu_writer_properties;

/// Repo-relative path of the script that performs and times the cuDF read.
const CUDF_SCRIPT: &str = "scripts/cudf-parquet-read.py";

/// Parquet compressor whose decompression measurement is a whole-file cuDF GPU read.
///
/// "Whole-file" means every row and every column: no projection and no filter is pushed into
/// the read, so the measurement covers decoding the entire table. That matches what the Vortex
/// GPU backend does, which decodes every field of every batch it scans.
pub struct GpuParquetCompressor {
    codec: GpuCodec,
    /// Cross-check the GPU read against a CPU Parquet read of the same file.
    ///
    /// cuDF has no verification of its own — a `read_parquet` either succeeds or raises — so
    /// [`CUDF_SCRIPT`] does the checking, comparing the frame it read on the device against
    /// one pandas read on the host. Off by default: it is a correctness pass, not a benchmark.
    verify: bool,
}

/// Timed reads to ask [`CUDF_SCRIPT`] for per invocation.
///
/// One read is a noisy sample, and repeating inside the script is nearly free — the cost that
/// dominates a cuDF read is process spawn and `import cudf`, which is paid once either way.
const TIMED_READS: usize = 3;

/// What the cuDF script reports back.
#[derive(Debug, Deserialize)]
struct CudfReadReport {
    /// Fastest of the script's [`TIMED_READS`] reads, in nanoseconds.
    ///
    /// The outer harness calls `decompress` once per `--iterations` and takes its own minimum,
    /// so the published number is the fastest read across every process the run spawned.
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

    /// Unsupported: GPU mode measures decompression only.
    ///
    /// `--gpu-decompress` restricts the suite to [`CompressOp::Decompress`], so nothing calls
    /// this. It used to time [`Self::write_gpu_parquet`], but that measures the host Parquet
    /// writer rather than anything on the device, and the result was never rendered — so it was
    /// a number nobody could read and nobody should have compared. The Vortex GPU backend
    /// refuses the same way.
    ///
    /// [`CompressOp::Decompress`]: vortex_bench::compress::CompressOp::Decompress
    async fn compress(&self, _parquet_path: &Path) -> Result<(u64, Duration)> {
        bail!("GPU compress-bench only supports decompression measurements")
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
    let output = Command::new("python3")
        .arg(CUDF_SCRIPT)
        .arg(path)
        .arg("--iterations")
        .arg(TIMED_READS.to_string())
        .args(verify.then_some("--verify"))
        .output()
        .with_context(|| format!("failed to spawn python3 to run {CUDF_SCRIPT}"))?;

    // A missing cuDF surfaces here rather than above: python3 starts fine and then fails on
    // `import cudf`, so the traceback is on stderr and the install hint belongs with it.
    if !output.status.success() {
        bail!(
            "{CUDF_SCRIPT} exited with {}; is cudf-cu12 installed on this host?\n{}",
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
