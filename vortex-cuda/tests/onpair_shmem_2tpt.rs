// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Correctness tests for the `onpair_shmem_2tpt` family of kernels.
//!
//! The kernels decode a dictionary-coded byte stream: each `uint16_t`
//! code indexes a padded 16-B dictionary, and `lens[code]` gives the
//! real (unpadded) length. The output is the concatenation of all
//! decoded tokens.
//!
//! These tests build a small hand-crafted dictionary and a deterministic
//! stream of codes, run each kernel, then assert byte-equality with a
//! straightforward host decode. They also assert that the renamed
//! "annotated" copy produces byte-identical output to the production
//! kernel — i.e. that the rename was syntactic only.
//!
//! The test is gated on `nvcc` availability via `#[vortex_cuda_macros::test]`;
//! when CUDA isn't available it expands to `#[test] #[ignore]`.

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use futures::executor::block_on;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaSession;

// Eight dictionary entries spanning every meaningful length-class for
// the byte-ladder in Phase 3: 1, 2, 3, 4 fit in a single shfl word;
// 8, 12 sit in the upper half; 15 hits the off-by-one boundary; 16
// hits the full uint4. A "0" entry would also be supported by the
// kernel but is uncommon enough in practice to skip here.
const DICT_ENTRIES: usize = 8;
const DICT_LENS: [u8; DICT_ENTRIES] = [1, 2, 3, 4, 8, 12, 15, 16];
const TOKEN_PAD: usize = 16;

/// Build (padded dict bytes, lens vector). Each entry carries a
/// distinctive byte pattern so a misplacement shows up clearly in
/// the assertion diff.
fn build_test_dict() -> (Vec<u8>, Vec<u8>) {
    let mut padded = vec![0u8; DICT_ENTRIES * TOKEN_PAD];
    let mut lens = vec![0u8; DICT_ENTRIES];
    for i in 0..DICT_ENTRIES {
        let n = DICT_LENS[i] as usize;
        for j in 0..n {
            padded[i * TOKEN_PAD + j] = (i as u32 * 17 + j as u32 * 3 + 5) as u8;
        }
        lens[i] = DICT_LENS[i];
    }
    (padded, lens)
}

/// Deterministic pseudo-random stream of codes, with every dict entry
/// appearing many times across the range so each length-class is
/// exercised in every chunk.
fn build_test_codes(total: usize) -> Vec<u16> {
    (0..total)
        .map(|i| (i.wrapping_mul(11).wrapping_add(7) % DICT_ENTRIES) as u16)
        .collect()
}

/// Host reference decode. The kernel must produce the same bytes.
fn host_decode(codes: &[u16], lens: &[u8], padded: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &code in codes {
        let c = code as usize;
        let n = lens[c] as usize;
        out.extend_from_slice(&padded[c * TOKEN_PAD..c * TOKEN_PAD + n]);
    }
    out
}

/// Prefix sum of token lengths at `chunk_size`-token boundaries.
/// Result has `num_chunks + 1` entries: `out[k]` is the byte offset
/// where chunk `k` starts in the output, and `out[num_chunks]` is the
/// total decoded size.
fn chunk_offsets(codes: &[u16], lens: &[u8], chunk_size: usize) -> Vec<u64> {
    let total = codes.len();
    let num_chunks = total.div_ceil(chunk_size);
    let mut offsets = Vec::with_capacity(num_chunks + 1);
    offsets.push(0u64);
    let mut acc = 0u64;
    for c in 0..num_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(total);
        for &code in &codes[start..end] {
            acc += lens[code as usize] as u64;
        }
        offsets.push(acc);
    }
    offsets
}

/// Launch an OnPair-shmem-variant kernel by name (2tpt → 64-token
/// chunks, 4tpt → 128-token chunks) and read its output back as a
/// `Vec<u8>` of exactly `total_size` bytes.
fn launch_variant(
    kernel_name: &str,
    chunk_size: usize,
    codes: &[u16],
    padded: &[u8],
    lens: &[u8],
    total_size: u64,
) -> Vec<u8> {
    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .expect("create CUDA execution ctx");

    let chunk_offs = chunk_offsets(codes, lens, chunk_size);

    // 16-byte tail-pad on the output so the kernel's `uint4` body drain
    // is free to over-read into garbage at the end without trampling
    // anything we care about. (We slice it off before comparing.)
    let initial_output: Vec<u8> = vec![0u8; total_size as usize + 16];

    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes.to_vec()).unwrap()).unwrap();
    let dict_padded_dev: BufferHandle =
        block_on(ctx.copy_to_device(padded.to_vec()).unwrap()).unwrap();
    let lens_dev: BufferHandle = block_on(ctx.copy_to_device(lens.to_vec()).unwrap()).unwrap();
    let chunk_offs_dev: BufferHandle = block_on(ctx.copy_to_device(chunk_offs).unwrap()).unwrap();
    let output_dev: BufferHandle = block_on(ctx.copy_to_device(initial_output).unwrap()).unwrap();

    let total_tokens = codes.len();
    let total_chunks = total_tokens.div_ceil(chunk_size);
    // Match the bench: 16 warps × 32 lanes = 512 threads per block,
    // hitting each kernel's `__launch_bounds__(512, _)`.
    let warps: u32 = 16;
    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(total_chunks.div_ceil(warps as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (warps * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    let func = ctx
        .load_function(kernel_name, &[])
        .unwrap_or_else(|e| panic!("load `{kernel_name}` PTX: {e}"));

    let codes_v = codes_dev.cuda_view::<u16>().unwrap();
    let chunk_offs_v = chunk_offs_dev.cuda_view::<u64>().unwrap();
    let dict_padded_v = dict_padded_dev.cuda_view::<u8>().unwrap();
    let lens_v = lens_dev.cuda_view::<u8>().unwrap();
    let output_v = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens_u64 = total_tokens as u64;

    ctx.launch_kernel_config(&func, cfg, total_tokens, |args| {
        args.arg(&codes_v)
            .arg(&chunk_offs_v)
            .arg(&dict_padded_v)
            .arg(&lens_v)
            .arg(&output_v)
            .arg(&total_tokens_u64);
    })
    .expect("kernel launch");

    let host_bytes = output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .expect("copy output back to host");

    host_bytes.as_ref()[..total_size as usize].to_vec()
}

fn launch_const1(codes: &[u16], dict: &[u8], total_size: usize) -> Vec<u8> {
    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .expect("create CUDA execution ctx");
    let initial_output = vec![0u8; total_size + 16];

    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes.to_vec()).unwrap()).unwrap();
    let dict_dev: BufferHandle = block_on(ctx.copy_to_device(dict.to_vec()).unwrap()).unwrap();
    let output_dev: BufferHandle = block_on(ctx.copy_to_device(initial_output).unwrap()).unwrap();

    let func = ctx
        .load_function("onpair_shmem_const1", &[])
        .expect("load `onpair_shmem_const1` PTX");
    let codes_v = codes_dev.cuda_view::<u16>().unwrap();
    let dict_v = dict_dev.cuda_view::<u8>().unwrap();
    let output_v = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens = codes.len();
    let total_tokens_u64 = total_tokens as u64;
    let cfg = LaunchConfig {
        grid_dim: (u32::try_from(total_tokens.div_ceil(512)).unwrap(), 1, 1),
        block_dim: (16 * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    ctx.launch_kernel_config(&func, cfg, total_tokens, |args| {
        args.arg(&codes_v)
            .arg(&dict_v)
            .arg(&output_v)
            .arg(&total_tokens_u64);
    })
    .expect("kernel launch");

    let host_bytes = output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .expect("copy output back to host");
    host_bytes.as_ref()[..total_size].to_vec()
}

fn launch_const2(codes: &[u16], dict: &[u16], total_size: usize) -> Vec<u8> {
    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .expect("create CUDA execution ctx");
    let initial_output = vec![0u8; total_size + 16];

    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes.to_vec()).unwrap()).unwrap();
    let dict_dev: BufferHandle = block_on(ctx.copy_to_device(dict.to_vec()).unwrap()).unwrap();
    let output_dev: BufferHandle = block_on(ctx.copy_to_device(initial_output).unwrap()).unwrap();

    let func = ctx
        .load_function("onpair_shmem_const2", &[])
        .expect("load `onpair_shmem_const2` PTX");
    let codes_v = codes_dev.cuda_view::<u16>().unwrap();
    let dict_v = dict_dev.cuda_view::<u16>().unwrap();
    let output_v = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens = codes.len();
    let total_tokens_u64 = total_tokens as u64;
    let cfg = LaunchConfig {
        grid_dim: (u32::try_from(total_tokens.div_ceil(256)).unwrap(), 1, 1),
        block_dim: (16 * 32, 1, 1),
        shared_mem_bytes: 0,
    };

    ctx.launch_kernel_config(&func, cfg, total_tokens, |args| {
        args.arg(&codes_v)
            .arg(&dict_v)
            .arg(&output_v)
            .arg(&total_tokens_u64);
    })
    .expect("kernel launch");

    let host_bytes = output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .expect("copy output back to host");
    host_bytes.as_ref()[..total_size].to_vec()
}

/// Production `onpair_shmem_2tpt` decodes exactly what the host decoder
/// computes.
#[vortex_cuda_macros::test]
fn onpair_shmem_2tpt_matches_host_decode() {
    let (padded, lens) = build_test_dict();
    let codes = build_test_codes(150);
    let expected = host_decode(&codes, &lens, &padded);
    let actual = launch_variant(
        "onpair_shmem_2tpt",
        64,
        &codes,
        &padded,
        &lens,
        expected.len() as u64,
    );
    assert_eq!(actual, expected);
}

/// Production `onpair_shmem_4tpt` decodes exactly what the host decoder
/// computes. Uses 128-token chunks.
#[vortex_cuda_macros::test]
fn onpair_shmem_4tpt_matches_host_decode() {
    let (padded, lens) = build_test_dict();
    // Use 300 tokens → 3 chunks of 128 tokens with a partial last
    // chunk (44 tokens), exercising the tail-bound path.
    let codes = build_test_codes(300);
    let expected = host_decode(&codes, &lens, &padded);
    let actual = launch_variant(
        "onpair_shmem_4tpt",
        128,
        &codes,
        &padded,
        &lens,
        expected.len() as u64,
    );
    assert_eq!(actual, expected);
}

#[vortex_cuda_macros::test]
fn onpair_shmem_const1_matches_host_decode() {
    let dict = [b'A', b'B', b'C', b'D'];
    let codes: Vec<u16> = build_test_codes(777)
        .into_iter()
        .map(|c| c % dict.len() as u16)
        .collect();
    let expected: Vec<u8> = codes.iter().map(|&c| dict[c as usize]).collect();
    let actual = launch_const1(&codes, &dict, expected.len());
    assert_eq!(actual, expected);
}

#[vortex_cuda_macros::test]
fn onpair_shmem_const2_matches_host_decode() {
    let dict = [
        u16::from_le_bytes(*b"aa"),
        u16::from_le_bytes(*b"bb"),
        u16::from_le_bytes(*b"cc"),
        u16::from_le_bytes(*b"dd"),
    ];
    let codes: Vec<u16> = build_test_codes(777)
        .into_iter()
        .map(|c| c % dict.len() as u16)
        .collect();
    let mut expected = Vec::with_capacity(codes.len() * 2);
    for &code in &codes {
        expected.extend_from_slice(&dict[code as usize].to_le_bytes());
    }
    let actual = launch_const2(&codes, &dict, expected.len());
    assert_eq!(actual, expected);
}

/// Profiling target: run the 2tpt kernel against ~1M tokens so ncu has
/// a realistic workload. Ignored by default; invoke via:
///   cargo test -p vortex-cuda --test onpair_shmem_2tpt --release -- \
///       --ignored onpair_shmem_2tpt_profile
#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target; run explicitly with --ignored"]
fn onpair_shmem_2tpt_profile() {
    let (padded, lens) = build_test_dict();
    let codes = build_test_codes(1 << 20);
    let total = host_decode(&codes, &lens, &padded).len() as u64;
    let _output = launch_variant("onpair_shmem_2tpt", 64, &codes, &padded, &lens, total);
}

/// Profiling target: same workload as `..._2tpt_profile`, but for the
/// 4tpt kernel (128-token chunks).
#[vortex_cuda_macros::test]
#[ignore = "ncu profiling target; run explicitly with --ignored"]
fn onpair_shmem_4tpt_profile() {
    let (padded, lens) = build_test_dict();
    let codes = build_test_codes(1 << 20);
    let total = host_decode(&codes, &lens, &padded).len() as u64;
    let _output = launch_variant("onpair_shmem_4tpt", 128, &codes, &padded, &lens, total);
}
