// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-data benchmark for `onpair_shmem`.
//!
//! Reads parquet path(s) from `ONPAIR_DATA_PATH` (colon-separated for
//! multiple). For each, iterates every Utf8 / Utf8View string column,
//! OnPair-compresses it with `DEFAULT_DICT12_CONFIG`, stages on GPU,
//! and times the kernel. Prints a markdown table per dataset.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::too_many_lines)]

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
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use futures::executor::block_on;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaSession;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::onpair_compress;

use crate::timed_launch_strategy::TimedLaunchStrategy;

/// Warps per block for the shmem family kernels. Tunable via
/// `ONPAIR_WARPS_PER_BLOCK` for architecture sweeps. Capped at 32
/// (kernel-side `WARPS_PER_BLOCK_MAX`). Default 8 matches the original
/// A100-tuned configuration; on Hopper (GH200) 16 typically wins.
fn warps_per_block() -> u32 {
    match env::var("ONPAIR_WARPS_PER_BLOCK").ok().and_then(|s| s.parse::<u32>().ok()) {
        Some(w) if (1..=32).contains(&w) => w,
        _ => 8,
    }
}

#[derive(Debug, Clone)]
struct ColResult {
    name: String,
    rows: usize,
    raw_bytes: usize,
    #[expect(dead_code, reason = "kept for debug-format output via Debug derive")]
    compressed_bytes: usize,
    ratio: f64,
    tokens: usize,
    dict_entries: usize,
    avg_token_len: f64,
    kernel_time_ms: f64,
    throughput_gib_s: f64,
}

/// Load a parquet file and concatenate all batches into one Vec<RecordBatch>.
fn load_parquet(
    path: &PathBuf,
) -> anyhow::Result<Vec<arrow_array::RecordBatch>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out = Vec::new();
    for b in reader {
        out.push(b?);
    }
    Ok(out)
}

/// Compute (raw_bytes_total, row_count) for a string column across batches,
/// without allocating per-row.
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
        } else if let Some(s) = col
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
        {
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

/// VarBinArray uses u32 offsets — keep total bytes per column under ~4 GiB.
const VARBIN_BYTE_CAP: u64 = 3_500_000_000;

/// Find the largest row prefix of this column whose total bytes is ≤
/// VARBIN_BYTE_CAP. Returns (row_cap, byte_total_within_cap).
fn find_row_cap(
    batches: &[arrow_array::RecordBatch],
    col_idx: usize,
) -> (usize, usize) {
    let mut bytes: u64 = 0;
    let mut rows: usize = 0;
    for b in batches {
        let col = b.column(col_idx);
        if let Some(s) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
            for i in 0..s.len() {
                let l = s.value(i).len() as u64;
                if bytes + l > VARBIN_BYTE_CAP {
                    return (rows, bytes as usize);
                }
                bytes += l;
                rows += 1;
            }
        } else if let Some(s) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            for i in 0..s.len() {
                let l = s.value(i).len() as u64;
                if bytes + l > VARBIN_BYTE_CAP {
                    return (rows, bytes as usize);
                }
                bytes += l;
                rows += 1;
            }
        } else if let Some(s) = col
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
        {
            for i in 0..s.len() {
                let l = s.value(i).len() as u64;
                if bytes + l > VARBIN_BYTE_CAP {
                    return (rows, bytes as usize);
                }
                bytes += l;
                rows += 1;
            }
        }
    }
    (rows, bytes as usize)
}

/// Build a VarBinArray over the first `row_cap` rows of `col_idx`.
fn build_varbin(
    batches: &[arrow_array::RecordBatch],
    col_idx: usize,
    row_cap: usize,
) -> Option<VarBinArray> {
    let first = batches.first()?.column(col_idx);
    let dtype = DType::Utf8(Nullability::NonNullable);
    if first.as_any().is::<arrow_array::StringArray>() {
        Some(VarBinArray::from_iter(
            batches
                .iter()
                .flat_map(|b| {
                    let s = b
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .unwrap();
                    (0..s.len()).map(move |i| Some(s.value(i).as_bytes()))
                })
                .take(row_cap),
            dtype,
        ))
    } else if first.as_any().is::<arrow_array::LargeStringArray>() {
        Some(VarBinArray::from_iter(
            batches
                .iter()
                .flat_map(|b| {
                    let s = b
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .unwrap();
                    (0..s.len()).map(move |i| Some(s.value(i).as_bytes()))
                })
                .take(row_cap),
            dtype,
        ))
    } else if first.as_any().is::<arrow_array::StringViewArray>() {
        Some(VarBinArray::from_iter(
            batches
                .iter()
                .flat_map(|b| {
                    let s = b
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<arrow_array::StringViewArray>()
                        .unwrap();
                    (0..s.len()).map(move |i| Some(s.value(i).as_bytes()))
                })
                .take(row_cap),
            dtype,
        ))
    } else {
        None
    }
}

fn bench_column(
    name: &str,
    raw_bytes: usize,
    rows: usize,
    varbin: VarBinArray,
    iters: u64,
) -> anyhow::Result<Vec<ColResult>> {
    let dtype = DType::Utf8(Nullability::NonNullable);
    let onpair = onpair_compress(&varbin, rows, &dtype, DEFAULT_DICT12_CONFIG)?;
    drop(varbin);

    let mut setup_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

    let codes_arr = onpair
        .codes()
        .clone()
        .execute::<PrimitiveArray>(setup_ctx.execution_ctx())?;
    let codes_offsets_arr = onpair
        .codes_offsets()
        .clone()
        .execute::<PrimitiveArray>(setup_ctx.execution_ctx())?;
    let dict_offsets_arr = onpair
        .dict_offsets()
        .clone()
        .execute::<PrimitiveArray>(setup_ctx.execution_ctx())?;
    let lens_arr = onpair
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(setup_ctx.execution_ctx())?;

    let codes_u16: Vec<u16> = match_each_integer_ptype!(codes_arr.ptype(), |P| {
        codes_arr.as_slice::<P>().iter().map(|&v| v as u16).collect()
    });

    let dict_bytes_host: &[u8] = onpair.dict_bytes().as_slice();
    let (dict_padded, lens_table) = match_each_integer_ptype!(dict_offsets_arr.ptype(), |P| {
        let s = dict_offsets_arr.as_slice::<P>();
        let dict_size = s.len().saturating_sub(1);
        let mut padded = vec![0u8; dict_size * vortex_onpair::MAX_TOKEN_SIZE];
        let mut lens = vec![0u8; dict_size];
        for i in 0..dict_size {
            let start: usize = s[i].try_into().unwrap_or(0);
            let end: usize = s[i + 1].try_into().unwrap_or(0);
            let len = end - start;
            padded[i * vortex_onpair::MAX_TOKEN_SIZE..i * vortex_onpair::MAX_TOKEN_SIZE + len]
                .copy_from_slice(&dict_bytes_host[start..end]);
            lens[i] = len as u8;
        }
        (padded, lens)
    });

    let dict_entries = lens_table.len();
    let dict_max_len = *lens_table.iter().max().unwrap_or(&0);
    let dict_mean_len = if dict_entries > 0 {
        lens_table.iter().map(|&v| v as u64).sum::<u64>() as f64 / dict_entries as f64
    } else {
        0.0
    };
    let mut sorted_lens: Vec<u8> = lens_table.clone();
    sorted_lens.sort_unstable();
    let pct = |p: f64| -> u8 {
        if sorted_lens.is_empty() {
            0
        } else {
            let idx = ((sorted_lens.len() as f64 - 1.0) * p).round() as usize;
            sorted_lens[idx]
        }
    };
    let dict_p50 = pct(0.50);
    let dict_p95 = pct(0.95);
    let pad_to_8 = dict_max_len <= 8;
    let pad_to_4 = dict_max_len <= 4;
    println!(
        "[onpair-real-data]   dict: {dict_entries} entries, max_len={dict_max_len}, mean={dict_mean_len:.2}, p50={dict_p50}, p95={dict_p95} | stride-fit: {}",
        if pad_to_4 {
            "4"
        } else if pad_to_8 {
            "8"
        } else {
            "16"
        }
    );

    let total_size: u64 = match_each_integer_ptype!(lens_arr.ptype(), |P| {
        lens_arr.as_slice::<P>().iter().map(|&v| v as u64).sum()
    });

    let total_tokens = codes_u16.len();
    let total_chunks = total_tokens.div_ceil(32);
    let mut chunk_offsets: Vec<u64> = Vec::with_capacity(total_chunks + 1);
    chunk_offsets.push(0u64);
    let mut chunk_acc: u64 = 0;
    for c in 0..total_chunks {
        let start = c * 32;
        let end = (start + 32).min(total_tokens);
        for i in start..end {
            chunk_acc += lens_table[codes_u16[i] as usize] as u64;
        }
        chunk_offsets.push(chunk_acc);
    }
    assert_eq!(chunk_acc, total_size);

    let bits = onpair.bits() as usize;
    let codes_bits = total_tokens * bits;
    let compressed_bytes = (codes_bits + 7) / 8 + dict_bytes_host.len() + 8;

    let total_tokens_u64 = total_tokens as u64;
    let avg_token_len = total_size as f64 / total_tokens.max(1) as f64;
    let ratio = raw_bytes as f64 / compressed_bytes.max(1) as f64;

    // Build inputs for the reference `onpair` (thread-per-row) kernel.
    // ABI: codes + codes_offsets (per row) + dict_table (u64=off<<16|len) +
    // dict_bytes + output_offsets (per row, u64) + validity_bits + num_rows.
    let dict_table: Vec<u64> = match_each_integer_ptype!(dict_offsets_arr.ptype(), |P| {
        let s = dict_offsets_arr.as_slice::<P>();
        (0..s.len().saturating_sub(1))
            .map(|i| {
                let off: u64 = s[i].try_into().unwrap_or(0);
                let len: u64 = (s[i + 1] - s[i]).try_into().unwrap_or(0);
                (off << 16) | len
            })
            .collect()
    });
    let dict_bytes_with_pad: Vec<u8> = {
        let mut v = Vec::with_capacity(dict_bytes_host.len() + 16);
        v.extend_from_slice(dict_bytes_host);
        v.extend(std::iter::repeat_n(0u8, 16));
        v
    };
    let mut output_offsets: Vec<u64> = Vec::with_capacity(rows + 1);
    output_offsets.push(0);
    let mut acc = 0u64;
    match_each_integer_ptype!(lens_arr.ptype(), |P| {
        for &l in lens_arr.as_slice::<P>() {
            acc += u64::try_from(l).unwrap_or(0);
            output_offsets.push(acc);
        }
    });
    let validity_bits: Vec<u8> = vec![0xFFu8; rows.div_ceil(8)];

    // Build stride-4 / stride-8 dicts on host (cheap re-pack from `dict_padded`).
    let mut dict_s8: Vec<u8> = vec![0u8; dict_entries * 8];
    let mut dict_s4: Vec<u8> = vec![0u8; dict_entries * 4];
    for i in 0..dict_entries {
        let src_off = i * vortex_onpair::MAX_TOKEN_SIZE;
        let len = lens_table[i] as usize;
        let n8 = len.min(8);
        dict_s8[i * 8..i * 8 + n8].copy_from_slice(&dict_padded[src_off..src_off + n8]);
        let n4 = len.min(4);
        dict_s4[i * 4..i * 4 + n4].copy_from_slice(&dict_padded[src_off..src_off + n4]);
    }
    let _ = pad_to_4; // s4 (shared-mem dict) variant was removed — regressed on A100
    let codes_device = block_on(setup_ctx.copy_to_device(codes_u16)?)?;
    let dict_padded_device = block_on(setup_ctx.copy_to_device(dict_padded)?)?;
    let dict_s8_device = block_on(setup_ctx.copy_to_device(dict_s8)?)?;
    let dict_s4_device = block_on(setup_ctx.copy_to_device(dict_s4)?)?;
    let dict_table_device = block_on(setup_ctx.copy_to_device(dict_table)?)?;
    let dict_bytes_device = block_on(setup_ctx.copy_to_device(dict_bytes_with_pad)?)?;
    let output_offsets_device = block_on(setup_ctx.copy_to_device(output_offsets)?)?;
    let validity_device = block_on(setup_ctx.copy_to_device(validity_bits)?)?;
    let lens_table_device = block_on(setup_ctx.copy_to_device(lens_table.clone())?)?;
    let chunk_offsets_device = block_on(setup_ctx.copy_to_device(chunk_offsets)?)?;
    let device_output = block_on(
        setup_ctx.copy_to_device(vec![0u8; total_size as usize + 16])?,
    )?;

    let codes_v = codes_device.cuda_view::<u16>().unwrap();
    let chunk_offsets_v = chunk_offsets_device.cuda_view::<u64>().unwrap();
    let dict_padded_v = dict_padded_device.cuda_view::<u8>().unwrap();
    let dict_s8_v = dict_s8_device.cuda_view::<u8>().unwrap();
    let dict_s4_v = dict_s4_device.cuda_view::<u8>().unwrap();
    let dict_table_v = dict_table_device.cuda_view::<u64>().unwrap();
    let dict_bytes_v = dict_bytes_device.cuda_view::<u8>().unwrap();
    let output_offsets_v = output_offsets_device.cuda_view::<u64>().unwrap();
    let validity_v = validity_device.cuda_view::<u8>().unwrap();
    let lens_v = lens_table_device.cuda_view::<u8>().unwrap();
    let output_v = device_output.cuda_view::<u8>().unwrap();

    let warps = warps_per_block();
    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks.div_ceil(warps as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut results: Vec<ColResult> = Vec::new();
    let to_gib_s = |kernel_time_ms: f64| -> f64 {
        (total_size as f64 / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0)
    };

    // ref: thread-per-row reference kernel (`onpair_<offt>`). Same logic as
    // the CPU decoder. Run as a floor / sanity check. We always cast
    // codes_offsets to u64 and launch the u64-suffixed variant, since the
    // input dataset row count can exceed u32 for ClickBench-scale columns.
    {
        let num_rows_u64 = rows as u64;
        let codes_off_u64: Vec<u64> = match_each_integer_ptype!(codes_offsets_arr.ptype(), |P| {
            codes_offsets_arr.as_slice::<P>().iter().map(|&v| v as u64).collect()
        });
        let codes_off_device = block_on(setup_ctx.copy_to_device(codes_off_u64)?)?;
        let codes_off_v = codes_off_device.cuda_view::<u64>().unwrap();
        let timed = TimedLaunchStrategy::default();
        let timer = timed.timer();
        let mut bench_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
            .with_launch_strategy(Arc::new(timed));
        let function = bench_ctx.load_function("onpair", &[u64::PTYPE])?;
        for _ in 0..2 {
            bench_ctx.launch_kernel(&function, rows, |args| {
                args.arg(&codes_v)
                    .arg(&codes_off_v)
                    .arg(&dict_table_v)
                    .arg(&dict_bytes_v)
                    .arg(&output_offsets_v)
                    .arg(&validity_v)
                    .arg(&output_v)
                    .arg(&num_rows_u64);
            })?;
        }
        timer.store(0, Ordering::Relaxed);
        for _ in 0..iters {
            bench_ctx.launch_kernel(&function, rows, |args| {
                args.arg(&codes_v)
                    .arg(&codes_off_v)
                    .arg(&dict_table_v)
                    .arg(&dict_bytes_v)
                    .arg(&output_offsets_v)
                    .arg(&validity_v)
                    .arg(&output_v)
                    .arg(&num_rows_u64);
            })?;
        }
        let kernel_time_ms =
            (timer.load(Ordering::Relaxed) as f64) / 1_000_000.0 / iters as f64;
        results.push(ColResult {
            name: format!("{name} [ref]"),
            rows,
            raw_bytes,
            compressed_bytes,
            ratio,
            tokens: total_tokens,
            dict_entries,
            avg_token_len,
            kernel_time_ms,
            throughput_gib_s: to_gib_s(kernel_time_ms),
        });
    }

    // s16: baseline `onpair_shmem`.
    {
        let timed = TimedLaunchStrategy::default();
        let timer = timed.timer();
        let mut bench_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
            .with_launch_strategy(Arc::new(timed));
        let function = bench_ctx.load_function("onpair_shmem", &[])?;
        for _ in 0..2 {
            bench_ctx.launch_kernel_config(&function, cfg, total_tokens, |args| {
                args.arg(&codes_v)
                    .arg(&chunk_offsets_v)
                    .arg(&dict_padded_v)
                    .arg(&lens_v)
                    .arg(&output_v)
                    .arg(&total_tokens_u64);
            })?;
        }
        timer.store(0, Ordering::Relaxed);
        for _ in 0..iters {
            bench_ctx.launch_kernel_config(&function, cfg, total_tokens, |args| {
                args.arg(&codes_v)
                    .arg(&chunk_offsets_v)
                    .arg(&dict_padded_v)
                    .arg(&lens_v)
                    .arg(&output_v)
                    .arg(&total_tokens_u64);
            })?;
        }
        let kernel_time_ms = (timer.load(Ordering::Relaxed) as f64) / 1_000_000.0 / iters as f64;
        results.push(ColResult {
            name: format!("{name} [s16]"),
            rows,
            raw_bytes,
            compressed_bytes,
            ratio,
            tokens: total_tokens,
            dict_entries,
            avg_token_len,
            kernel_time_ms,
            throughput_gib_s: to_gib_s(kernel_time_ms),
        });
    }

    // s8: stride-8 kernel, only if dict max_len ≤ 8.
    if pad_to_8 {
        let timed = TimedLaunchStrategy::default();
        let timer = timed.timer();
        let mut bench_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
            .with_launch_strategy(Arc::new(timed));
        let function = bench_ctx.load_function("onpair_shmem_s8", &[])?;
        for _ in 0..2 {
            bench_ctx.launch_kernel_config(&function, cfg, total_tokens, |args| {
                args.arg(&codes_v)
                    .arg(&chunk_offsets_v)
                    .arg(&dict_s8_v)
                    .arg(&lens_v)
                    .arg(&output_v)
                    .arg(&total_tokens_u64);
            })?;
        }
        timer.store(0, Ordering::Relaxed);
        for _ in 0..iters {
            bench_ctx.launch_kernel_config(&function, cfg, total_tokens, |args| {
                args.arg(&codes_v)
                    .arg(&chunk_offsets_v)
                    .arg(&dict_s8_v)
                    .arg(&lens_v)
                    .arg(&output_v)
                    .arg(&total_tokens_u64);
            })?;
        }
        let kernel_time_ms = (timer.load(Ordering::Relaxed) as f64) / 1_000_000.0 / iters as f64;
        results.push(ColResult {
            name: format!("{name} [s8]"),
            rows,
            raw_bytes,
            compressed_bytes,
            ratio,
            tokens: total_tokens,
            dict_entries,
            avg_token_len,
            kernel_time_ms,
            throughput_gib_s: to_gib_s(kernel_time_ms),
        });
    }

    // s4l1: stride-4 kernel, dict in L1 (no shared cache, no __syncthreads).
    if pad_to_4 {
        let timed = TimedLaunchStrategy::default();
        let timer = timed.timer();
        let mut bench_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
            .with_launch_strategy(Arc::new(timed));
        let function = bench_ctx.load_function("onpair_shmem_s4l1", &[])?;
        for _ in 0..2 {
            bench_ctx.launch_kernel_config(&function, cfg, total_tokens, |args| {
                args.arg(&codes_v)
                    .arg(&chunk_offsets_v)
                    .arg(&dict_s4_v)
                    .arg(&lens_v)
                    .arg(&output_v)
                    .arg(&total_tokens_u64);
            })?;
        }
        timer.store(0, Ordering::Relaxed);
        for _ in 0..iters {
            bench_ctx.launch_kernel_config(&function, cfg, total_tokens, |args| {
                args.arg(&codes_v)
                    .arg(&chunk_offsets_v)
                    .arg(&dict_s4_v)
                    .arg(&lens_v)
                    .arg(&output_v)
                    .arg(&total_tokens_u64);
            })?;
        }
        let kernel_time_ms = (timer.load(Ordering::Relaxed) as f64) / 1_000_000.0 / iters as f64;
        results.push(ColResult {
            name: format!("{name} [s4l1]"),
            rows,
            raw_bytes,
            compressed_bytes,
            ratio,
            tokens: total_tokens,
            dict_entries,
            avg_token_len,
            kernel_time_ms,
            throughput_gib_s: to_gib_s(kernel_time_ms),
        });
    }

    // tma16: stride-16 with dict TMA-prefetched into shared at block start
    // via `cp.async.bulk` + mbarrier. Hopper-only (sm_90+). Avoids the
    // `__syncthreads` barrier that killed the cooperative-load variant on
    // both A100 and Hopper. Gated by dict fitting under 32 KB shared so we
    // stay below the default 48 KB carveout.
    {
        let dict_bytes = dict_entries * 16;
        let lens_bytes = dict_entries;
        let scratch_offset = ((dict_bytes + lens_bytes + 15) & !15) as u32;
        let shared_bytes =
            scratch_offset + (warps as u32) * 544 + 64 /* mbarrier + slack */;
        let tma_enabled = shared_bytes <= 32 * 1024;
        if tma_enabled {
            let dict_entries_u32 = dict_entries as u32;
            let tma_cfg = LaunchConfig {
                grid_dim: cfg.grid_dim,
                block_dim: cfg.block_dim,
                shared_mem_bytes: shared_bytes,
            };
            let timed = TimedLaunchStrategy::default();
            let timer = timed.timer();
            let mut bench_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
                .with_launch_strategy(Arc::new(timed));
            let function = bench_ctx.load_function("onpair_shmem_tma", &[])?;
            for _ in 0..2 {
                bench_ctx.launch_kernel_config(&function, tma_cfg, total_tokens, |args| {
                    args.arg(&codes_v)
                        .arg(&chunk_offsets_v)
                        .arg(&dict_padded_v)
                        .arg(&lens_v)
                        .arg(&output_v)
                        .arg(&total_tokens_u64)
                        .arg(&dict_entries_u32);
                })?;
            }
            timer.store(0, Ordering::Relaxed);
            for _ in 0..iters {
                bench_ctx.launch_kernel_config(&function, tma_cfg, total_tokens, |args| {
                    args.arg(&codes_v)
                        .arg(&chunk_offsets_v)
                        .arg(&dict_padded_v)
                        .arg(&lens_v)
                        .arg(&output_v)
                        .arg(&total_tokens_u64)
                        .arg(&dict_entries_u32);
                })?;
            }
            let kernel_time_ms =
                (timer.load(Ordering::Relaxed) as f64) / 1_000_000.0 / iters as f64;
            results.push(ColResult {
                name: format!("{name} [tma16]"),
                rows,
                raw_bytes,
                compressed_bytes,
                ratio,
                tokens: total_tokens,
                dict_entries,
                avg_token_len,
                kernel_time_ms,
                throughput_gib_s: to_gib_s(kernel_time_ms),
            });
        }
    }

    Ok(results)
}

fn print_results(label: &str, results: &[ColResult]) {
    println!();
    println!("# {label}");
    println!();
    // GiB/s [raw] is decoded throughput (raw_bytes / time).
    // GiB/s [cmp] is compressed-input throughput (compressed_bytes / time)
    // = how fast the kernel eats compressed input. Equal to raw / ratio.
    println!(
        "| Column | Rows | Raw MB | Cmp MB | Ratio | Tokens | Dict | Avg B/tok | Decode ms | GiB/s [raw] | GiB/s [cmp] |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let mut total_raw = 0usize;
    let mut total_cmp = 0usize;
    let mut total_time_ms = 0.0;
    for r in results {
        let cmp_throughput =
            (r.compressed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (r.kernel_time_ms / 1000.0);
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.2}x | {} | {} | {:.2} | {:.3} | **{:.1}** | {:.1} |",
            r.name,
            r.rows,
            r.raw_bytes as f64 / 1_048_576.0,
            r.compressed_bytes as f64 / 1_048_576.0,
            r.ratio,
            r.tokens,
            r.dict_entries,
            r.avg_token_len,
            r.kernel_time_ms,
            r.throughput_gib_s,
            cmp_throughput,
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
    println!("[onpair-real-data] loading {}", path.display());
    let batches = load_parquet(&path)?;
    if batches.is_empty() {
        anyhow::bail!("no batches read");
    }
    let schema = batches[0].schema();
    let n_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "[onpair-real-data] {} batches, {} rows, {} columns",
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
        if total_rows < 100_000 || total_raw < 1_000_000 {
            continue;
        }
        let (row_cap, raw_bytes) = find_row_cap(&batches, col_idx);
        let capped = row_cap < total_rows;
        println!(
            "[onpair-real-data] column {col_idx}: {} (rows={}{}, raw={:.1} MB{})",
            field.name(),
            row_cap,
            if capped {
                format!(" of {total_rows}")
            } else {
                String::new()
            },
            raw_bytes as f64 / 1_048_576.0,
            if capped {
                format!(" capped from {:.1} GB", total_raw as f64 / 1e9)
            } else {
                String::new()
            }
        );
        let Some(varbin) = build_varbin(&batches, col_idx, row_cap) else {
            eprintln!(
                "[onpair-real-data]   unsupported arrow type for {}",
                field.name()
            );
            continue;
        };
        match bench_column(field.name(), raw_bytes, row_cap, varbin, 10) {
            Ok(rs) => results.extend(rs),
            Err(e) => eprintln!(
                "[onpair-real-data]   bench failed for {}: {e}",
                field.name()
            ),
        }
    }

    let label = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    print_results(&label, &results);
    Ok(())
}

fn bench_real_data(_c: &mut Criterion) {
    let Some(path_env) = env::var("ONPAIR_DATA_PATH").ok() else {
        eprintln!("[onpair-real-data] ONPAIR_DATA_PATH not set; skipping");
        return;
    };
    let paths: Vec<PathBuf> = path_env.split(':').map(PathBuf::from).collect();
    for path in paths {
        if let Err(e) = run_dataset(path.clone()) {
            eprintln!("[onpair-real-data] {} failed: {e}", path.display());
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(1))
        .warm_up_time(Duration::from_millis(100));
    targets = bench_real_data
}
criterion_main!(benches);
