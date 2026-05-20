// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dispatch-and-time benchmark.
//!
//! Demonstrates: given a workload's `(dict_max_len, mean_bytes_per_token)`,
//! a small lookup picks an OnPair kernel variant. We then measure each
//! candidate kernel against the same workload and show that the
//! dispatched choice matches (or closely tracks) the best fixed choice.
//!
//! The "signal" we dispatch on (`mean_bytes_per_token`) is the field that
//! would be stored in `OnPairMetadata` once a metadata extension lands;
//! here we compute it directly from the dict's `lens` table to keep the
//! demo self-contained. The same picker function works in both worlds.
//!
//! Run with:
//!   cargo test -p vortex-cuda --test onpair_dispatch --release -- \
//!       --ignored --nocapture onpair_dispatch_benchmark

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaKernelEvents;
use vortex_cuda::CudaSession;
use vortex_cuda::LaunchStrategy;

const TOTAL_TOKENS: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Dispatch picker
// ---------------------------------------------------------------------------

/// A concrete kernel choice the host can launch.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct KernelChoice {
    name: &'static str,
    chunk_size: usize,
    block_warps: u32,
    /// Stride of the padded dictionary the kernel expects (4 / 8 / 16, or
    /// 0 for the const-family which doesn't use `lens`). Surfaced so the
    /// caller can pre-pack the dict to the matching layout.
    stride: usize,
}

/// Pick an OnPair-shmem kernel given the column's bulk statistics.
///
/// Rules (derived from `ncu --set full` sweep, 1M-token synthetic workload):
///   * `dict_max_len` selects the *family* (s4l1 / s8 / shmem) — the same
///     decision the bench already makes via `pad_to_4`/`pad_to_8`.
///   * `mean_bpt` (weighted by code frequency, ideally; falls back to
///     dict-mean) picks the *tpt knob* within the family. The target is
///     ~150-300 output bytes per warp-chunk so the aligned `uint4` body
///     drain saturates 10-20 lanes without paying for over-amortisation.
fn pick_kernel(dict_max_len: u8, mean_bpt: f32) -> KernelChoice {
    // Family selection. Note: callers that have already detected
    // all-len-1 / all-len-2 should dispatch to const1/const2 *before*
    // calling this picker — that decision needs `dict_min_len` which
    // isn't a parameter here.
    let (family_prefix, stride) = if dict_max_len <= 4 {
        ("onpair_shmem_s4l1", 4usize)
    } else if dict_max_len <= 8 {
        ("onpair_shmem_s8", 8)
    } else {
        ("onpair_shmem", 16)
    };

    // Within-family tpt selection. The real-data sweep across headlines,
    // sentiment140, book_reviews, fineweb, tpch_sf10 showed that on
    // dictionaries with thousands of entries the per-warp L1 footprint
    // of higher-tpt kernels is the binding cost, not the divergence
    // savings — so the family alone picks the right kernel and
    // `mean_bpt` is left unused here (kept in the signature for now
    // because it's the field we'd eventually store in metadata, and a
    // future calibration may resurrect it as a tiebreaker for small
    // dictionaries).
    let _ = mean_bpt;
    let (suffix, chunk_size, block_warps) = match stride {
        16 => ("_2tpt", 64usize, 16u32),
        8 => ("_4tpt", 128, 16),
        4 => ("_16tpt", 512, 8),
        _ => unreachable!("stride {stride} is not in {{4,8,16}}"),
    };

    // The base-tpt variants (chunk=32) are named without a suffix
    // ("onpair_shmem", "onpair_shmem_s8", "onpair_shmem_s4l1"). We never
    // pick them in this table because the ncu sweep showed at least
    // `_2tpt` always beats the base in this regime; the case where the
    // base would win is too short a column to be worth tuning.
    let name: &'static str = match (family_prefix, suffix) {
        ("onpair_shmem", "_2tpt") => "onpair_shmem_2tpt",
        ("onpair_shmem_s8", "_4tpt") => "onpair_shmem_s8_4tpt",
        ("onpair_shmem_s4l1", "_16tpt") => "onpair_shmem_s4l1_16tpt",
        _ => unreachable!(),
    };

    KernelChoice {
        name,
        chunk_size,
        block_warps,
        stride,
    }
}

// ---------------------------------------------------------------------------
// Timing strategy (mirrors the bench's `TimedLaunchStrategy`)
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct TimedLaunchStrategy {
    total_time_ns: Arc<AtomicU64>,
}

impl LaunchStrategy for TimedLaunchStrategy {
    fn event_flags(&self) -> CUevent_flags {
        CUevent_flags::CU_EVENT_BLOCKING_SYNC
    }
    fn on_complete(&self, events: &CudaKernelEvents, _len: usize) -> VortexResult<()> {
        let elapsed_ns = events.duration()?.as_nanos() as u64;
        self.total_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Workload + timing harness
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Workload {
    label: &'static str,
    dict_max_len: u8,
    /// Length per dict entry; entry count = `lens.len()`.
    lens: Vec<u8>,
    /// Padded stride (16 for shmem, 8 for s8, 4 for s4l1). We pre-pad the
    /// dict to this stride.
    stride: usize,
}

impl Workload {
    fn mean_bpt(&self, codes: &[u16]) -> f32 {
        // Code-weighted mean — this is the signal we'd store in metadata.
        let total: u64 = codes.iter().map(|&c| self.lens[c as usize] as u64).sum();
        total as f32 / codes.len() as f32
    }

    fn dict_padded(&self) -> Vec<u8> {
        let mut padded = vec![0u8; self.lens.len() * self.stride];
        for (i, &n) in self.lens.iter().enumerate() {
            for j in 0..(n as usize) {
                padded[i * self.stride + j] = (i as u32 * 17 + j as u32 * 3 + 5) as u8;
            }
        }
        padded
    }
}

fn build_codes(dict_entries: usize) -> Vec<u16> {
    (0..TOTAL_TOKENS)
        .map(|i| (i.wrapping_mul(11).wrapping_add(7) % dict_entries) as u16)
        .collect()
}

/// Time a single OnPair-shmem-style kernel `iters` times against
/// pre-uploaded device buffers; returns the mean per-launch duration in
/// microseconds.
#[allow(clippy::too_many_arguments)]
fn time_kernel_us(
    kernel_name: &str,
    chunk_size: usize,
    block_warps: u32,
    codes_dev: &BufferHandle,
    padded_dev: &BufferHandle,
    lens_dev: &BufferHandle,
    output_dev: &BufferHandle,
    total_tokens: usize,
    iters: u32,
) -> VortexResult<f64> {
    let chunk_offs =
        build_chunk_offsets(codes_dev, padded_dev, lens_dev, chunk_size, total_tokens)?;

    let timed = TimedLaunchStrategy::default();
    let timer = Arc::clone(&timed.total_time_ns);
    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?
        .with_launch_strategy(Arc::new(timed));
    let chunk_offs_dev: BufferHandle = block_on(ctx.copy_to_device(chunk_offs)?)?;

    let func = ctx.load_function(kernel_name, &[])?;
    let total_chunks = total_tokens.div_ceil(chunk_size);
    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks.div_ceil(block_warps as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (block_warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    let codes_v = codes_dev.cuda_view::<u16>()?;
    let chunk_offs_v = chunk_offs_dev.cuda_view::<u64>()?;
    let padded_v = padded_dev.cuda_view::<u8>()?;
    let lens_v = lens_dev.cuda_view::<u8>()?;
    let output_v = output_dev.cuda_view::<u8>()?;
    let total_tokens_u64 = total_tokens as u64;

    // Warm-up: 3 untimed launches.
    for _ in 0..3 {
        ctx.launch_kernel_config(&func, cfg, total_tokens, |args| {
            args.arg(&codes_v)
                .arg(&chunk_offs_v)
                .arg(&padded_v)
                .arg(&lens_v)
                .arg(&output_v)
                .arg(&total_tokens_u64);
        })?;
    }
    timer.store(0, Ordering::Relaxed);
    for _ in 0..iters {
        ctx.launch_kernel_config(&func, cfg, total_tokens, |args| {
            args.arg(&codes_v)
                .arg(&chunk_offs_v)
                .arg(&padded_v)
                .arg(&lens_v)
                .arg(&output_v)
                .arg(&total_tokens_u64);
        })?;
    }
    let total_ns = timer.load(Ordering::Relaxed) as f64;
    Ok(total_ns / iters as f64 / 1000.0) // ns → µs per launch
}

fn build_chunk_offsets(
    codes_dev: &BufferHandle,
    _padded_dev: &BufferHandle,
    lens_dev: &BufferHandle,
    chunk_size: usize,
    total_tokens: usize,
) -> VortexResult<Vec<u64>> {
    // Read codes + lens back from device — cheaper than threading them as
    // host-side parameters through every call site; the test isn't perf
    // sensitive on this path. (In production, the host has these natively.)
    let codes_bytes = codes_dev
        .as_device()
        .copy_to_host_sync(vortex::buffer::Alignment::of::<u16>())?;
    let lens_bytes = lens_dev
        .as_device()
        .copy_to_host_sync(vortex::buffer::Alignment::of::<u8>())?;
    // codes are u16; reinterpret the byte slice. Vortex's ByteBuffer aligns
    // to the requested alignment, so this is sound.
    let codes_raw: &[u8] = codes_bytes.as_ref();
    assert!(codes_raw.len() % 2 == 0);
    let codes: Vec<u16> = codes_raw
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    let lens: &[u8] = lens_bytes.as_ref();

    let num_chunks = total_tokens.div_ceil(chunk_size);
    let mut offs = Vec::with_capacity(num_chunks + 1);
    offs.push(0u64);
    let mut acc = 0u64;
    for c in 0..num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(total_tokens);
        for &code in &codes[start..end] {
            acc += lens[code as usize] as u64;
        }
        offs.push(acc);
    }
    Ok(offs)
}

// ---------------------------------------------------------------------------
// Workload definitions
// ---------------------------------------------------------------------------

fn workloads() -> Vec<Workload> {
    vec![
        Workload {
            label: "shmem-long  (lens 12-16, mean~14)",
            dict_max_len: 16,
            lens: vec![12, 13, 14, 14, 15, 15, 16, 16],
            stride: 16,
        },
        Workload {
            label: "shmem-mid   (lens 6-8, mean~7)",
            dict_max_len: 16,
            lens: vec![6, 6, 7, 7, 7, 8, 8, 8],
            stride: 16,
        },
        Workload {
            label: "shmem-short (lens 3-4, mean~3.5)",
            dict_max_len: 16,
            lens: vec![3, 3, 3, 4, 4, 4, 4, 4],
            stride: 16,
        },
        Workload {
            label: "s8 family   (lens 4-6, mean~5)",
            dict_max_len: 8,
            lens: vec![4, 4, 5, 5, 5, 6, 6, 6],
            stride: 8,
        },
        Workload {
            label: "s4l1-tiny   (lens 1-2, mean~1.5)",
            dict_max_len: 4,
            lens: vec![1, 1, 2, 2, 1, 2, 1, 2],
            stride: 4,
        },
    ]
}

/// All candidate kernels per family — for showing what each fixed choice
/// would have given us.
fn candidates_for(stride: usize) -> &'static [(&'static str, usize, u32)] {
    match stride {
        16 => &[
            ("onpair_shmem", 32, 16),
            ("onpair_shmem_2tpt", 64, 16),
            ("onpair_shmem_4tpt", 128, 16),
        ],
        8 => &[
            ("onpair_shmem_s8", 32, 16),
            ("onpair_shmem_s8_2tpt", 64, 16),
            ("onpair_shmem_s8_4tpt", 128, 16),
            ("onpair_shmem_s8_8tpt", 256, 12),
        ],
        4 => &[
            ("onpair_shmem_s4l1", 32, 16),
            ("onpair_shmem_s4l1_2tpt", 64, 16),
            ("onpair_shmem_s4l1_4tpt", 128, 16),
            ("onpair_shmem_s4l1_8tpt", 256, 12),
            ("onpair_shmem_s4l1_16tpt", 512, 8),
        ],
        _ => &[],
    }
}

#[vortex_cuda_macros::test]
#[ignore = "benchmark, run explicitly with --ignored --nocapture"]
fn onpair_dispatch_benchmark() {
    let iters: u32 = 50;
    let total_tokens = TOTAL_TOKENS;
    let ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).expect("ctx");
    let codes = build_codes(8);
    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes.clone()).unwrap()).unwrap();

    println!();
    println!(
        "OnPair dispatch benchmark — {} tokens/run, {} timed launches",
        total_tokens, iters
    );
    println!();
    println!(
        "{:<38}  {:>8}  {:>12}  {:>12}  {:>10}",
        "Workload", "Mean bpt", "Dispatch", "Best fixed", "Δ from best"
    );
    println!("{}", "-".repeat(90));

    for wl in workloads() {
        let padded = wl.dict_padded();
        let total_out: u64 = codes.iter().map(|&c| wl.lens[c as usize] as u64).sum();
        let mean_bpt = wl.mean_bpt(&codes);

        let padded_dev: BufferHandle = block_on(ctx.copy_to_device(padded).unwrap()).unwrap();
        let lens_dev: BufferHandle =
            block_on(ctx.copy_to_device(wl.lens.clone()).unwrap()).unwrap();
        let output_dev: BufferHandle = block_on(
            ctx.copy_to_device(vec![0u8; total_out as usize + 16])
                .unwrap(),
        )
        .unwrap();

        let dispatched = pick_kernel(wl.dict_max_len, mean_bpt);
        let disp_us = time_kernel_us(
            dispatched.name,
            dispatched.chunk_size,
            dispatched.block_warps,
            &codes_dev,
            &padded_dev,
            &lens_dev,
            &output_dev,
            total_tokens,
            iters,
        )
        .expect("dispatch run");

        let mut all: Vec<(String, f64)> = Vec::new();
        for &(k, csz, warps) in candidates_for(wl.stride) {
            let us = time_kernel_us(
                k,
                csz,
                warps,
                &codes_dev,
                &padded_dev,
                &lens_dev,
                &output_dev,
                total_tokens,
                iters,
            )
            .expect("candidate run");
            all.push((k.to_string(), us));
        }

        let (best_name, best_us) = all
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .cloned()
            .unwrap();
        let delta_pct = (disp_us - best_us) / best_us * 100.0;

        println!(
            "{:<38}  {:>8.2}  {:>12}  {:>12}  {:>+9.2}%",
            wl.label,
            mean_bpt,
            format!("{:.2}µs", disp_us),
            format!("{:.2}µs", best_us),
            delta_pct
        );
        println!("    dispatched: {} ({:.2}µs)", dispatched.name, disp_us);
        println!("    best fixed: {} ({:.2}µs)", best_name, best_us);
        for (name, us) in &all {
            let tag = if *name == dispatched.name { " [*]" } else { "" };
            println!("      {:<32} {:>7.2}µs{}", name, us, tag);
        }
        println!();
    }
}
