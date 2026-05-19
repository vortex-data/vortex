// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-data Vortex FSST GPU decompression baseline.
//!
//! Mirrors `onpair_real_data.rs` / `nvcomp_zstd_real_data.rs` but runs
//! each string column through Vortex's FSST encoding + its CUDA decode
//! path. Apples-to-apples comparison: same parquet inputs, same ≥100 KB
//! min-bytes filter, same kernel-only timing via the bench's
//! TimedLaunchStrategy.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(unused_qualifications)]
#![expect(clippy::let_underscore_must_use)]
#![expect(let_underscore_drop)]

mod timed_launch_strategy;

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use arrow_array::Array as ArrowArray;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use futures::executor::block_on;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::IntoArray;
use vortex::array::arrays::VarBinArray;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::encodings::fsst::FSSTArray;
use vortex::encodings::fsst::fsst_compress;
use vortex::encodings::fsst::fsst_train_compressor;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;

use crate::timed_launch_strategy::TimedLaunchStrategy;

const VARBIN_BYTE_CAP: u64 = 3_500_000_000;

fn load_parquet(path: &PathBuf) -> anyhow::Result<Vec<arrow_array::RecordBatch>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for b in reader {
        out.push(b?);
    }
    Ok(out)
}

fn measure_utf8_column(
    batches: &[arrow_array::RecordBatch],
    col_idx: usize,
) -> Option<(usize, usize)> {
    let mut raw_bytes = 0usize;
    let mut rows = 0usize;
    for batch in batches {
        let col = batch.column(col_idx);
        if let Some(s) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
            rows += s.len();
            for i in 0..s.len() {
                raw_bytes += s.value(i).len();
            }
        } else if let Some(s) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            rows += s.len();
            for i in 0..s.len() {
                raw_bytes += s.value(i).len();
            }
        } else if let Some(s) = col.as_any().downcast_ref::<arrow_array::StringViewArray>() {
            rows += s.len();
            for i in 0..s.len() {
                raw_bytes += s.value(i).len();
            }
        } else {
            return None;
        }
    }
    Some((raw_bytes, rows))
}

fn find_row_cap(
    batches: &[arrow_array::RecordBatch],
    col_idx: usize,
) -> (usize, usize) {
    let mut bytes: u64 = 0;
    let mut rows: usize = 0;
    for b in batches {
        let col = b.column(col_idx);
        macro_rules! handle {
            ($t:ty) => {{
                let s = col.as_any().downcast_ref::<$t>().unwrap();
                for i in 0..s.len() {
                    let l = s.value(i).len() as u64;
                    if bytes + l > VARBIN_BYTE_CAP {
                        return (rows, bytes as usize);
                    }
                    bytes += l;
                    rows += 1;
                }
            }};
        }
        if col.as_any().is::<arrow_array::StringArray>() {
            handle!(arrow_array::StringArray);
        } else if col.as_any().is::<arrow_array::LargeStringArray>() {
            handle!(arrow_array::LargeStringArray);
        } else if col.as_any().is::<arrow_array::StringViewArray>() {
            handle!(arrow_array::StringViewArray);
        }
    }
    (rows, bytes as usize)
}

fn build_varbin(
    batches: &[arrow_array::RecordBatch],
    col_idx: usize,
    row_cap: usize,
) -> Option<VarBinArray> {
    let first = batches.first()?.column(col_idx);
    let dtype = DType::Utf8(Nullability::NonNullable);
    if first.as_any().is::<arrow_array::StringArray>() {
        Some(VarBinArray::from_iter(
            batches.iter().flat_map(|b| {
                let s = b.column(col_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
                (0..s.len()).map(move |i| Some(s.value(i).as_bytes()))
            }).take(row_cap),
            dtype,
        ))
    } else if first.as_any().is::<arrow_array::LargeStringArray>() {
        Some(VarBinArray::from_iter(
            batches.iter().flat_map(|b| {
                let s = b.column(col_idx).as_any().downcast_ref::<arrow_array::LargeStringArray>().unwrap();
                (0..s.len()).map(move |i| Some(s.value(i).as_bytes()))
            }).take(row_cap),
            dtype,
        ))
    } else if first.as_any().is::<arrow_array::StringViewArray>() {
        Some(VarBinArray::from_iter(
            batches.iter().flat_map(|b| {
                let s = b.column(col_idx).as_any().downcast_ref::<arrow_array::StringViewArray>().unwrap();
                (0..s.len()).map(move |i| Some(s.value(i).as_bytes()))
            }).take(row_cap),
            dtype,
        ))
    } else {
        None
    }
}

#[derive(Debug, Clone)]
struct ColResult {
    name: String,
    rows: usize,
    raw_bytes: usize,
    compressed_bytes: usize,
    ratio: f64,
    kernel_time_ms: f64,
    throughput_raw_gib_s: f64,
    throughput_cmp_gib_s: f64,
}

fn bench_column(
    name: &str,
    raw_bytes: usize,
    rows: usize,
    vbn: VarBinArray,
    iters: u64,
) -> anyhow::Result<ColResult> {
    let dtype = DType::Utf8(Nullability::NonNullable);
    let mut setup_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
    let compressor = fsst_train_compressor(&vbn);
    let fsst: FSSTArray = fsst_compress(vbn, rows, &dtype, &compressor, setup_ctx.execution_ctx());

    // FSST stores symbols + symbol_lengths + compressed-codes bytes.
    let symbols_bytes = fsst.symbols().len() * core::mem::size_of::<u64>();
    let symbol_lens_bytes = fsst.symbol_lengths().len();
    let codes_bytes = fsst.codes_bytes().len();
    let compressed_bytes = symbols_bytes + symbol_lens_bytes + codes_bytes;
    let ratio = raw_bytes as f64 / compressed_bytes.max(1) as f64;
    let fsst_array = fsst.into_array();

    // Warm-up
    for _ in 0..2 {
        let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let _ = block_on(fsst_array.clone().execute_cuda(&mut ctx))?;
    }

    let timed = TimedLaunchStrategy::default();
    let timer = timed.timer();
    let mut bench_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
        .with_launch_strategy(Arc::new(timed));
    timer.store(0, Ordering::Relaxed);
    for _ in 0..iters {
        let _ = block_on(fsst_array.clone().execute_cuda(&mut bench_ctx))?;
    }
    let kernel_time_ms = (timer.load(Ordering::Relaxed) as f64) / 1_000_000.0 / iters as f64;
    let _ = Duration::from_secs(0); // keep Duration import used

    let throughput_raw_gib_s =
        (raw_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
    let throughput_cmp_gib_s =
        (compressed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);

    Ok(ColResult {
        name: format!("{name} [vortex-fsst]"),
        rows,
        raw_bytes,
        compressed_bytes,
        ratio,
        kernel_time_ms,
        throughput_raw_gib_s,
        throughput_cmp_gib_s,
    })
}

fn print_results(label: &str, results: &[ColResult]) {
    println!();
    println!("# {label}");
    println!();
    println!(
        "| Column | Rows | Raw MB | Cmp MB | Ratio | Decode ms | GiB/s [raw] | GiB/s [cmp] |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    let mut total_raw = 0usize;
    let mut total_cmp = 0usize;
    let mut total_time_ms = 0.0;
    for r in results {
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x | {:.3} | **{:.1}** | {:.1} |",
            r.name,
            r.rows,
            r.raw_bytes as f64 / 1_048_576.0,
            r.compressed_bytes as f64 / 1_048_576.0,
            r.ratio,
            r.kernel_time_ms,
            r.throughput_raw_gib_s,
            r.throughput_cmp_gib_s,
        );
        total_raw += r.raw_bytes;
        total_cmp += r.compressed_bytes;
        total_time_ms += r.kernel_time_ms;
    }
    println!();
    println!(
        "**Aggregate:** {:.2} GB raw, {:.2} GB compressed ({:.2}x ratio), {:.2} ms total kernel time, {:.1} GiB/s [raw], {:.1} GiB/s [cmp].",
        total_raw as f64 / 1_000_000_000.0,
        total_cmp as f64 / 1_000_000_000.0,
        total_raw as f64 / total_cmp.max(1) as f64,
        total_time_ms,
        (total_raw as f64 / (1024.0 * 1024.0 * 1024.0)) / (total_time_ms / 1000.0),
        (total_cmp as f64 / (1024.0 * 1024.0 * 1024.0)) / (total_time_ms / 1000.0),
    );
    println!();
}

fn run_dataset(path: PathBuf) -> anyhow::Result<()> {
    let label = path.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset").to_string();
    println!("[vortex-fsst-real-data] loading {}", path.display());
    let batches = load_parquet(&path)?;
    if batches.is_empty() {
        anyhow::bail!("no batches read");
    }
    let schema = batches[0].schema();
    let n_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "[vortex-fsst-real-data] {} batches, {} rows, {} columns",
        batches.len(),
        n_rows,
        schema.fields().len()
    );

    let min_bytes = env::var("ONPAIR_MIN_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000);

    let mut results = Vec::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let dt = field.data_type();
        let is_str = matches!(
            dt,
            arrow_schema::DataType::Utf8
                | arrow_schema::DataType::LargeUtf8
                | arrow_schema::DataType::Utf8View
        );
        if !is_str {
            continue;
        }
        let Some((total_raw, total_rows)) = measure_utf8_column(&batches, col_idx) else {
            continue;
        };
        if total_rows < 100_000 || total_raw < min_bytes {
            continue;
        }
        let (row_cap, raw_bytes) = find_row_cap(&batches, col_idx);
        let capped = row_cap < total_rows;
        println!(
            "[vortex-fsst-real-data] column {col_idx}: {} (rows={}{}, raw={:.1} MB{})",
            field.name(),
            row_cap,
            if capped {
                format!(" of {total_rows}")
            } else {
                String::new()
            },
            raw_bytes as f64 / 1_048_576.0,
            if capped { " capped" } else { "" }
        );
        let Some(vbn) = build_varbin(&batches, col_idx, row_cap) else {
            continue;
        };
        let iters: u64 = if raw_bytes < 10_000_000 {
            50
        } else if raw_bytes < 100_000_000 {
            20
        } else {
            5
        };
        match bench_column(field.name(), raw_bytes, row_cap, vbn, iters) {
            Ok(r) => results.push(r),
            Err(e) => println!(
                "[vortex-fsst-real-data]   bench failed for {}: {e}",
                field.name()
            ),
        }
    }
    print_results(&label, &results);
    Ok(())
}

fn bench(_c: &mut Criterion) {
    let path_env = env::var("VORTEX_FSST_DATA_PATH").or_else(|_| env::var("ONPAIR_DATA_PATH"));
    let Ok(paths) = path_env else {
        println!("[vortex-fsst-real-data] set ONPAIR_DATA_PATH (colon-separated parquet paths)");
        return;
    };
    for p in paths.split(':').filter(|s| !s.is_empty()) {
        if let Err(e) = run_dataset(PathBuf::from(p)) {
            println!("[vortex-fsst-real-data] dataset failed: {p}: {e}");
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
