// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Profile-only targets for every OnPair shmem variant, one `#[ignore]`
//! test per kernel. The intent is to give `ncu` a stable, isolated
//! launch per kernel so we can capture metrics with `--set full`.
//!
//! Workload: 1M codes drawn deterministically from a small dictionary
//! sized to the kernel's stride family (16/8/4 B padding, or constant).
//! Numbers won't match the real-data bench in absolute terms (an 8-entry
//! dict fits entirely in L1), but the *relative* shape of the metrics
//! between variants is informative.

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use futures::executor::block_on;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaSession;

const TOTAL_TOKENS: usize = 1 << 20;
const DICT_ENTRIES: usize = 8;

/// Pseudo-random deterministic codes that touch every dict entry.
fn build_codes() -> Vec<u16> {
    (0..TOTAL_TOKENS)
        .map(|i| (i.wrapping_mul(11).wrapping_add(7) % DICT_ENTRIES) as u16)
        .collect()
}

/// Stride-padded dict for the shmem (16-B) / s8 (8-B) / s4l1 (4-B) families.
/// `lens[i]` must satisfy `lens[i] <= stride`.
fn build_stride_dict(stride: usize, lens_table: [u8; DICT_ENTRIES]) -> (Vec<u8>, Vec<u8>) {
    let mut padded = vec![0u8; DICT_ENTRIES * stride];
    let mut lens = vec![0u8; DICT_ENTRIES];
    for i in 0..DICT_ENTRIES {
        let n = lens_table[i] as usize;
        assert!(n <= stride);
        for j in 0..n {
            padded[i * stride + j] = (i as u32 * 17 + j as u32 * 3 + 5) as u8;
        }
        lens[i] = lens_table[i];
    }
    (padded, lens)
}

fn host_decode_stride(codes: &[u16], lens: &[u8]) -> usize {
    codes
        .iter()
        .map(|&c| lens[c as usize] as usize)
        .sum::<usize>()
}

/// Launch a `(codes, chunk_offsets, dict_padded, lens, output, total_tokens)`
/// kernel. Covers shmem / s8 / s4l1 families.
fn launch_stride_variant(
    kernel_name: &str,
    chunk_size: usize,
    block_warps: u32,
    stride: usize,
    lens_table: [u8; DICT_ENTRIES],
) {
    let codes = build_codes();
    let (padded, lens) = build_stride_dict(stride, lens_table);
    let total_size = host_decode_stride(&codes, &lens) as u64;

    let total_chunks = codes.len().div_ceil(chunk_size);
    let mut chunk_offs = Vec::with_capacity(total_chunks + 1);
    chunk_offs.push(0u64);
    let mut acc = 0u64;
    for c in 0..total_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(codes.len());
        for &code in &codes[start..end] {
            acc += lens[code as usize] as u64;
        }
        chunk_offs.push(acc);
    }

    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).expect("ctx");
    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes).unwrap()).unwrap();
    let padded_dev: BufferHandle = block_on(ctx.copy_to_device(padded).unwrap()).unwrap();
    let lens_dev: BufferHandle = block_on(ctx.copy_to_device(lens).unwrap()).unwrap();
    let chunk_offs_dev: BufferHandle = block_on(ctx.copy_to_device(chunk_offs).unwrap()).unwrap();
    let output_dev: BufferHandle = block_on(
        ctx.copy_to_device(vec![0u8; total_size as usize + 16])
            .unwrap(),
    )
    .unwrap();

    let total_tokens = TOTAL_TOKENS;
    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks.div_ceil(block_warps as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (block_warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = ctx
        .load_function(kernel_name, &[])
        .unwrap_or_else(|e| panic!("load `{kernel_name}`: {e}"));
    let codes_v = codes_dev.cuda_view::<u16>().unwrap();
    let chunk_offs_v = chunk_offs_dev.cuda_view::<u64>().unwrap();
    let padded_v = padded_dev.cuda_view::<u8>().unwrap();
    let lens_v = lens_dev.cuda_view::<u8>().unwrap();
    let output_v = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens_u64 = total_tokens as u64;
    ctx.launch_kernel_config(&func, cfg, total_tokens, |args| {
        args.arg(&codes_v)
            .arg(&chunk_offs_v)
            .arg(&padded_v)
            .arg(&lens_v)
            .arg(&output_v)
            .arg(&total_tokens_u64);
    })
    .expect("launch");
    // Force completion (avoid measuring teardown).
    let _flushed = output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .unwrap();
}

/// const1: `(codes, dict_const1, output, total_tokens)`. 1 B/token.
fn launch_const1() {
    let codes = build_codes();
    let dict_const1: Vec<u8> = (0..DICT_ENTRIES).map(|i| (i as u8) * 17 + 5).collect();
    let total_size = codes.len() as u64;

    let total_chunks_512 = codes.len().div_ceil(512);
    let block_warps: u32 = 16;
    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).expect("ctx");
    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes).unwrap()).unwrap();
    let dict_dev: BufferHandle = block_on(ctx.copy_to_device(dict_const1).unwrap()).unwrap();
    let output_dev: BufferHandle = block_on(
        ctx.copy_to_device(vec![0u8; total_size as usize + 16])
            .unwrap(),
    )
    .unwrap();

    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks_512.div_ceil(block_warps as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (block_warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = ctx.load_function("onpair_shmem_const1", &[]).unwrap();
    let codes_v = codes_dev.cuda_view::<u16>().unwrap();
    let dict_v = dict_dev.cuda_view::<u8>().unwrap();
    let output_v = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens_u64 = TOTAL_TOKENS as u64;
    ctx.launch_kernel_config(&func, cfg, TOTAL_TOKENS, |args| {
        args.arg(&codes_v)
            .arg(&dict_v)
            .arg(&output_v)
            .arg(&total_tokens_u64);
    })
    .unwrap();
    let _flushed = output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .unwrap();
}

/// const2: `(codes, dict_const2 as *u16, output, total_tokens)`. 2 B/token.
fn launch_const2() {
    let codes = build_codes();
    let dict_const2_u16: Vec<u16> = (0..DICT_ENTRIES)
        .map(|i| (i as u16) * 0x0101 + 0x2020)
        .collect();
    let total_size = (codes.len() * 2) as u64;

    let total_chunks_256 = codes.len().div_ceil(256);
    let block_warps: u32 = 16;
    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).expect("ctx");
    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes).unwrap()).unwrap();
    let dict_dev: BufferHandle = block_on(ctx.copy_to_device(dict_const2_u16).unwrap()).unwrap();
    let output_dev: BufferHandle = block_on(
        ctx.copy_to_device(vec![0u8; total_size as usize + 16])
            .unwrap(),
    )
    .unwrap();

    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks_256.div_ceil(block_warps as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (block_warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = ctx.load_function("onpair_shmem_const2", &[]).unwrap();
    let codes_v = codes_dev.cuda_view::<u16>().unwrap();
    let dict_v = dict_dev.cuda_view::<u16>().unwrap();
    let output_v = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens_u64 = TOTAL_TOKENS as u64;
    ctx.launch_kernel_config(&func, cfg, TOTAL_TOKENS, |args| {
        args.arg(&codes_v)
            .arg(&dict_v)
            .arg(&output_v)
            .arg(&total_tokens_u64);
    })
    .unwrap();
    let _flushed = output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .unwrap();
}

// 16-B-stride dict (full range of lengths up to 16).
const LENS_16: [u8; 8] = [1, 2, 3, 4, 8, 12, 15, 16];
// 8-B-stride dict (max_len ≤ 8).
const LENS_8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
// 4-B-stride dict (max_len ≤ 4).
const LENS_4: [u8; 8] = [1, 2, 3, 4, 1, 2, 3, 4];

// ============================================================================
// shmem family — 16-B padded dict, `lens`.
// ============================================================================

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem() {
    launch_stride_variant("onpair_shmem", 32, 16, 16, LENS_16);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_2tpt() {
    launch_stride_variant("onpair_shmem_2tpt", 64, 16, 16, LENS_16);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_4tpt() {
    launch_stride_variant("onpair_shmem_4tpt", 128, 16, 16, LENS_16);
}

// ============================================================================
// s8 family — 8-B padded dict, `lens`. Requires max_len ≤ 8.
// ============================================================================

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s8() {
    launch_stride_variant("onpair_shmem_s8", 32, 16, 8, LENS_8);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s8_2tpt() {
    launch_stride_variant("onpair_shmem_s8_2tpt", 64, 16, 8, LENS_8);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s8_4tpt() {
    launch_stride_variant("onpair_shmem_s8_4tpt", 128, 16, 8, LENS_8);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s8_8tpt() {
    // __launch_bounds__(384, 4) → max 12 warps/block
    launch_stride_variant("onpair_shmem_s8_8tpt", 256, 12, 8, LENS_8);
}

// ============================================================================
// s4l1 family — 4-B padded dict, `lens`. Requires max_len ≤ 4.
// ============================================================================

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s4l1() {
    launch_stride_variant("onpair_shmem_s4l1", 32, 16, 4, LENS_4);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s4l1_2tpt() {
    launch_stride_variant("onpair_shmem_s4l1_2tpt", 64, 16, 4, LENS_4);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s4l1_4tpt() {
    launch_stride_variant("onpair_shmem_s4l1_4tpt", 128, 16, 4, LENS_4);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s4l1_8tpt() {
    // __launch_bounds__(384, 4) → max 12 warps/block
    launch_stride_variant("onpair_shmem_s4l1_8tpt", 256, 12, 4, LENS_4);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s4l1_16tpt() {
    // __launch_bounds__(256, 4) → max 8 warps/block
    launch_stride_variant("onpair_shmem_s4l1_16tpt", 512, 8, 4, LENS_4);
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_s4l1_32tpt() {
    // No launch_bounds attribute. Bench uses 8 warps/block.
    launch_stride_variant("onpair_shmem_s4l1_32tpt", 1024, 8, 4, LENS_4);
}

// ============================================================================
// const family — different ABI (no chunk_offsets, no lens).
// ============================================================================

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_const1() {
    launch_const1();
}

#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target"]
fn profile_onpair_shmem_const2() {
    launch_const2();
}
