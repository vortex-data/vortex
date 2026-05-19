// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-data nvCOMP zstd decompression baseline.
//!
//! Mirrors `onpair_real_data.rs` but runs each string column through
//! nvCOMP's batched zstd kernel instead of the OnPair shmem kernels.
//! Output table matches the OnPair bench's format so the two can be
//! diffed cleanly. Compressed bytes = sum of zstd frame sizes.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::too_many_lines)]

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use arrow_array::Array as ArrowArray;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::vtable::child_to_validity;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdArray;
use vortex::encodings::zstd::ZstdDataParts;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::ZstdKernelPrep;
use vortex_cuda::nvcomp::zstd as nvcomp_zstd;
use vortex_cuda::zstd_kernel_prepare;

/// nvCOMP zstd uses fixed-size frames; this matches the synthetic
/// `zstd_cuda` bench's chunk size for a fair compression-side
/// comparison.
const ZSTD_VALUES_PER_FRAME: usize = 2048;
/// Compression level. Negative levels are "fast"; -10 matches the
/// synthetic bench and is what someone choosing zstd for throughput
/// would actually pick.
const ZSTD_LEVEL: i32 = -10;
/// VarBin view offsets are i32, so cap per-column bytes well below 4 GiB.
const VARBIN_BYTE_CAP: u64 = 3_500_000_000;

#[derive(Debug, Clone)]
struct ColResult {
    name: String,
    rows: usize,
    raw_bytes: usize,
    compressed_bytes: usize,
    ratio: f64,
    num_frames: usize,
    kernel_time_ms: f64,
    throughput_raw_gib_s: f64,
    throughput_cmp_gib_s: f64,
}

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
        let n = col.len();
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
        } else {
            let _ = n;
        }
    }
    (rows, bytes as usize)
}

fn build_varbin_view(
    batches: &[arrow_array::RecordBatch],
    col_idx: usize,
    row_cap: usize,
) -> Option<VarBinViewArray> {
    let first = batches.first()?.column(col_idx);
    if first.as_any().is::<arrow_array::StringArray>() {
        let it = batches.iter().flat_map(|b| {
            let s = b.column(col_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
            (0..s.len()).map(move |i| s.value(i).to_string())
        }).take(row_cap);
        let v: Vec<String> = it.collect();
        Some(VarBinViewArray::from_iter_str(v.iter().map(|s| s.as_str())))
    } else if first.as_any().is::<arrow_array::LargeStringArray>() {
        let it = batches.iter().flat_map(|b| {
            let s = b.column(col_idx).as_any().downcast_ref::<arrow_array::LargeStringArray>().unwrap();
            (0..s.len()).map(move |i| s.value(i).to_string())
        }).take(row_cap);
        let v: Vec<String> = it.collect();
        Some(VarBinViewArray::from_iter_str(v.iter().map(|s| s.as_str())))
    } else if first.as_any().is::<arrow_array::StringViewArray>() {
        let it = batches.iter().flat_map(|b| {
            let s = b.column(col_idx).as_any().downcast_ref::<arrow_array::StringViewArray>().unwrap();
            (0..s.len()).map(move |i| s.value(i).to_string())
        }).take(row_cap);
        let v: Vec<String> = it.collect();
        Some(VarBinViewArray::from_iter_str(v.iter().map(|s| s.as_str())))
    } else {
        None
    }
}

/// Time a single nvcomp_zstd decompression launch via CUDA events.
async fn execute_zstd_kernel(
    mut exec: ZstdKernelPrep,
    cuda_ctx: &mut vortex_cuda::CudaExecutionCtx,
) -> VortexResult<Duration> {
    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("Failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("Failed to record start event: {:?}", e))?;

    let (device_actual_sizes_ptr, record_actual_sizes) =
        exec.device_actual_sizes.device_ptr_mut(stream);
    let (nvcomp_temp_buffer_ptr, record_temp) = exec.nvcomp_temp_buffer.device_ptr_mut(stream);
    let (device_statuses_ptr, record_statuses) = exec.device_statuses.device_ptr_mut(stream);

    unsafe {
        nvcomp_zstd::decompress_async(
            exec.frame_ptrs_ptr as _,
            exec.frame_sizes_ptr as _,
            exec.output_sizes_ptr as _,
            device_actual_sizes_ptr as _,
            exec.num_frames,
            nvcomp_temp_buffer_ptr as _,
            exec.nvcomp_temp_buffer_size,
            exec.output_ptrs_ptr as _,
            device_statuses_ptr as _,
            stream.cu_stream().cast(),
        )
        .map_err(|e| vortex_err!("nvcomp decompress_async failed: {}", e))?;
    }
    drop((record_actual_sizes, record_temp, record_statuses));

    let end_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("Failed to create end event: {:?}", e))?;
    end_event
        .record(stream)
        .map_err(|e| vortex_err!("Failed to record end event: {:?}", e))?;

    let elapsed_ms = start_event
        .elapsed_ms(&end_event)
        .map_err(|e| vortex_err!("Failed to get elapsed time: {:?}", e))?;
    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

fn bench_column(
    name: &str,
    raw_bytes: usize,
    rows: usize,
    vbv: VarBinViewArray,
    iters: u64,
) -> anyhow::Result<ColResult> {
    let mut setup_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
    let zstd_array: ZstdArray = Zstd::from_var_bin_view_without_dict(
        &vbv,
        ZSTD_LEVEL,
        ZSTD_VALUES_PER_FRAME,
        setup_ctx.execution_ctx(),
    )?;

    // Extract frame stats via one throwaway into_parts. (Parts are
    // consumed by zstd_kernel_prepare; we re-clone for each iter.)
    let (compressed_bytes, num_frames) = {
        let validity = child_to_validity(
            zstd_array.as_ref().slots()[0].as_ref(),
            zstd_array.dtype().nullability(),
        );
        let parts: ZstdDataParts = zstd_array.clone().into_data().into_parts(validity);
        let bytes: usize = parts.frames.iter().map(|f| f.len()).sum();
        let n = parts.frames.len();
        (bytes, n)
    };
    let ratio = raw_bytes as f64 / compressed_bytes.max(1) as f64;

    // Warm-up: 2 untimed launches.
    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
    for _ in 0..2 {
        let validity = child_to_validity(
            zstd_array.as_ref().slots()[0].as_ref(),
            zstd_array.dtype().nullability(),
        );
        let parts: ZstdDataParts = zstd_array.clone().into_data().into_parts(validity);
        let ZstdDataParts { frames, metadata, .. } = parts;
        let exec = block_on(zstd_kernel_prepare(frames, &metadata, &mut cuda_ctx))?;
        let _ = block_on(execute_zstd_kernel(exec, &mut cuda_ctx))?;
    }

    let mut total_time = Duration::ZERO;
    for _ in 0..iters {
        let validity = child_to_validity(
            zstd_array.as_ref().slots()[0].as_ref(),
            zstd_array.dtype().nullability(),
        );
        let parts: ZstdDataParts = zstd_array.clone().into_data().into_parts(validity);
        let ZstdDataParts { frames, metadata, .. } = parts;
        let exec = block_on(zstd_kernel_prepare(frames, &metadata, &mut cuda_ctx))?;
        let dur = block_on(execute_zstd_kernel(exec, &mut cuda_ctx))?;
        total_time += dur;
    }
    let kernel_time_ms = (total_time.as_secs_f64() * 1000.0) / iters as f64;
    let throughput_raw_gib_s =
        (raw_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
    let throughput_cmp_gib_s =
        (compressed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);

    Ok(ColResult {
        name: format!("{name} [nvcomp-zstd]"),
        rows,
        raw_bytes,
        compressed_bytes,
        ratio,
        num_frames,
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
        "| Column | Rows | Raw MB | Cmp MB | Ratio | Frames | Decode ms | GiB/s [raw] | GiB/s [cmp] |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut total_raw = 0usize;
    let mut total_cmp = 0usize;
    let mut total_time_ms = 0.0;
    for r in results {
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x | {} | {:.3} | **{:.1}** | {:.1} |",
            r.name,
            r.rows,
            r.raw_bytes as f64 / 1_048_576.0,
            r.compressed_bytes as f64 / 1_048_576.0,
            r.ratio,
            r.num_frames,
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
    println!("[nvcomp-zstd-real-data] loading {}", path.display());
    let batches = load_parquet(&path)?;
    if batches.is_empty() {
        anyhow::bail!("no batches read");
    }
    let schema = batches[0].schema();
    let n_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "[nvcomp-zstd-real-data] {} batches, {} rows, {} columns",
        batches.len(),
        n_rows,
        schema.fields().len()
    );

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
        // Filter: skip narrow / tiny columns. Threshold matches the
        // onpair bench so the two are comparable.
        let min_bytes = env::var("ONPAIR_MIN_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(100_000);
        if total_rows < 100_000 || total_raw < min_bytes {
            continue;
        }
        let (row_cap, raw_bytes) = find_row_cap(&batches, col_idx);
        let capped = row_cap < total_rows;
        println!(
            "[nvcomp-zstd-real-data] column {col_idx}: {} (rows={}{}, raw={:.1} MB{})",
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
        let Some(vbv) = build_varbin_view(&batches, col_idx, row_cap) else {
            continue;
        };
        let iters: u64 = if raw_bytes < 10_000_000 {
            50
        } else if raw_bytes < 100_000_000 {
            20
        } else {
            5
        };
        match bench_column(field.name(), raw_bytes, row_cap, vbv, iters) {
            Ok(r) => results.push(r),
            Err(e) => println!("[nvcomp-zstd-real-data]   bench failed for {}: {e}", field.name()),
        }
    }
    print_results(&label, &results);
    Ok(())
}

fn bench(_c: &mut Criterion) {
    let path_env = env::var("NVCOMP_DATA_PATH").or_else(|_| env::var("ONPAIR_DATA_PATH"));
    let Ok(paths) = path_env else {
        println!("[nvcomp-zstd-real-data] set NVCOMP_DATA_PATH or ONPAIR_DATA_PATH (colon-separated parquet paths)");
        return;
    };
    for p in paths.split(':').filter(|s| !s.is_empty()) {
        if let Err(e) = run_dataset(PathBuf::from(p)) {
            println!("[nvcomp-zstd-real-data] dataset failed: {p}: {e}");
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
