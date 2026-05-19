// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for OnPair decompression.
//!
//! OnPair is a variable-bit-width (`9..=16` bits/token) dictionary-based
//! short-string compressor (see `onpair-lib`). The on-disk format that this
//! module consumes (`onpair_lib::Parts`):
//!
//! * `dict_bytes`       — concatenated dictionary entry bytes
//! * `dict_offsets`     — `[dict_size + 1]` byte boundaries into `dict_bytes`
//! * `codes_packed`     — LSB-first bit-packed token stream over u64 words
//! * `codes_boundaries` — `[num_rows + 1]` per-row token-index boundaries
//! * `bits`             — bit width of each token (`9..=16`)
//!
//! Two kernels run sequentially:
//!
//! 1. `onpair_lengths_b<BITS>` writes per-row decoded byte counts to a
//!    device buffer. The host scans this into per-row u64 output offsets
//!    and copies them back to the device. The CPU baseline does the same
//!    bitstream-walk as a host pre-pass, so moving it to the GPU is the
//!    single largest end-to-end win at multi-million-row scale.
//!
//! 2. `onpair_decode_b<BITS>` writes the decoded bytes to
//!    `output_bytes[output_offsets[row]..]`. One thread per row; each
//!    token triggers an unconditional 16-byte over-copy from `dict_bytes`
//!    advanced by the token's true length. `dict_bytes` is padded with
//!    [`MAX_TOKEN_SIZE`] trailing zeros so the over-copy never reads OOB.
//!
//! Both kernels are templated over `BITS ∈ 9..=16` so every shift / mask
//! folds to a literal — same effect as the CPU `dispatch_bits!` macro in
//! `onpair-lib`.

use cudarc::driver::PushKernelArg;
use onpair_lib::MAX_TOKEN_SIZE;
use onpair_lib::Parts;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaExecutionCtx;

/// GPU-decoded OnPair column.
///
/// `bytes` is the flat decoded byte buffer (host-resident copy). `offsets`
/// has length `num_rows + 1`; row `i` occupies `bytes[offsets[i]..offsets[i + 1]]`.
pub struct OnPairGpuDecoded {
    pub bytes: Vec<u8>,
    pub offsets: Vec<u32>,
}

/// Pack the dictionary into the per-token `(byte_offset << 16) | byte_length`
/// table the kernel reads. Token lengths are bounded by [`MAX_TOKEN_SIZE`]
/// so 16 bits suffice. Mirrors `onpair_lib::column::build_dict_table`.
pub fn build_dict_table(dict_offsets: &[u32]) -> Vec<u64> {
    let n = dict_offsets.len().saturating_sub(1);
    let mut table = Vec::with_capacity(n);
    for i in 0..n {
        let off = dict_offsets[i] as u64;
        let len = (dict_offsets[i + 1] - dict_offsets[i]) as u64;
        debug_assert!(len <= MAX_TOKEN_SIZE as u64);
        table.push((off << 16) | len);
    }
    table
}

/// Pre-pad `dict_bytes` with [`MAX_TOKEN_SIZE`] trailing zeros so the
/// kernel's unconditional 16-byte over-copy from any token offset stays in
/// bounds.
fn padded_dict_bytes(dict_bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(dict_bytes.len() + MAX_TOKEN_SIZE);
    out.extend_from_slice(dict_bytes);
    out.resize(out.len() + MAX_TOKEN_SIZE, 0);
    out
}

/// Build the `[num_rows + 1]` exclusive prefix-sum of u32 lengths as u64
/// (for the decode kernel) and u32 (for the Arrow-style host return). Both
/// are produced in one pass. Errors if any cumulative position would
/// overflow the host u32 offset.
fn build_output_offsets(lens: &[u32]) -> VortexResult<(Vec<u64>, Vec<u32>)> {
    let n = lens.len();
    let mut off_u64 = Vec::with_capacity(n + 1);
    let mut off_u32 = Vec::with_capacity(n + 1);
    off_u64.push(0u64);
    off_u32.push(0u32);
    let mut acc: u64 = 0;
    for &l in lens {
        acc += l as u64;
        off_u64.push(acc);
        off_u32.push(u32::try_from(acc).map_err(|_| {
            vortex_err!("OnPair: total decoded size {acc} overflows u32 row offset")
        })?);
    }
    Ok((off_u64, off_u32))
}

/// Decompress every row of an OnPair-encoded column on the GPU.
///
/// Pipeline (one synchronous host thread driving CUDA):
///
/// 1. Host: build packed `dict_table`, pad `dict_bytes`, copy compressed
///    inputs to the device.
/// 2. GPU: `onpair_lengths_b<BITS>` — per-row decoded byte counts.
/// 3. Host: D2H `row_lengths`, exclusive scan into `(output_offsets_u64,
///    host_offsets_u32)`, H2D `output_offsets_u64`.
/// 4. GPU: `onpair_decode_b<BITS>` — per-row decode into the output buffer.
/// 5. Host: D2H the decoded bytes.
///
/// The host exclusive scan on `num_rows` u32 values is microseconds even
/// at `1e7` rows. Doing it on the GPU would require a multi-block scan
/// kernel that this crate doesn't have yet; the D2H + scan + H2D
/// round-trip on a ~4 MB / row count buffer is bandwidth-bound at <1 ms
/// over PCIe Gen4 and effectively free on NVLink-C2C.
pub async fn onpair_gpu_decode_all(
    parts: &Parts<'_>,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<OnPairGpuDecoded> {
    let num_rows = parts.num_rows;
    if num_rows == 0 {
        return Ok(OnPairGpuDecoded {
            bytes: Vec::new(),
            offsets: vec![0u32],
        });
    }
    if !(9..=16).contains(&parts.bits) {
        vortex_bail!(
            "OnPair: unsupported bit width {} (expected 9..=16)",
            parts.bits
        );
    }

    let bits_suffix = format!("b{}", parts.bits);
    let lengths_function = ctx.load_function_with_suffixes("onpair_lengths", &[&bits_suffix])?;
    let decode_function = ctx.load_function_with_suffixes("onpair_decode", &[&bits_suffix])?;

    // ── Host setup: tiny (dict_table for bits=12 is 32 KB) ──
    let dict_table = build_dict_table(parts.dict_offsets);
    let dict_bytes_padded = padded_dict_bytes(parts.dict_bytes);

    // ── H2D copies for the lengths pass (decode pass reuses the same
    //    buffers; only output_offsets is new) ──
    let codes_packed_vec = parts.codes_packed.to_vec();
    let codes_boundaries_vec = parts.codes_boundaries.to_vec();
    let (dict_table_d, dict_bytes_d, packed_d, boundaries_d) = futures::try_join!(
        ctx.copy_to_device(dict_table)?,
        ctx.copy_to_device(dict_bytes_padded)?,
        ctx.copy_to_device(codes_packed_vec)?,
        ctx.copy_to_device(codes_boundaries_vec)?,
    )?;

    let dict_table_view = dict_table_d.cuda_view::<u64>()?;
    let dict_bytes_view = dict_bytes_d.cuda_view::<u8>()?;
    let packed_view = packed_d.cuda_view::<u64>()?;
    let boundaries_view = boundaries_d.cuda_view::<u32>()?;
    let num_rows_u64 = num_rows as u64;

    // ── Pass 1: per-row decoded byte counts on GPU ──
    let row_lengths_dev = ctx.device_alloc::<u32>(num_rows)?;
    ctx.launch_kernel(&lengths_function, num_rows, |args| {
        args.arg(&dict_table_view)
            .arg(&packed_view)
            .arg(&boundaries_view)
            .arg(&row_lengths_dev)
            .arg(&num_rows_u64);
    })?;

    // D2H the row lengths, scan on host, H2D the per-row u64 output
    // offsets. The scan is microseconds at 1e7 rows.
    let row_lengths_host = CudaDeviceBuffer::new(row_lengths_dev)
        .copy_to_host(Alignment::new(4))?
        .await?;
    let row_lengths_bytes = row_lengths_host.as_slice();
    // SAFETY: the kernel writes u32-aligned u32s; we copied with align-4 above.
    let row_lengths: &[u32] = unsafe {
        std::slice::from_raw_parts(
            row_lengths_bytes.as_ptr().cast::<u32>(),
            row_lengths_bytes.len() / size_of::<u32>(),
        )
    };
    let row_lengths = &row_lengths[..num_rows];

    let (output_offsets_u64, host_offsets) = build_output_offsets(row_lengths)?;
    let total_size = usize::try_from(*output_offsets_u64.last().unwrap_or(&0))
        .map_err(|_| vortex_err!("OnPair: total decoded size overflows usize"))?;
    let output_offsets_d = ctx.copy_to_device(output_offsets_u64)?.await?;
    let output_offsets_view = output_offsets_d.cuda_view::<u64>()?;

    // ── Pass 2: decode every row into the contiguous output buffer ──
    let device_output = ctx.device_alloc::<u8>(total_size.max(1))?;
    ctx.launch_kernel(&decode_function, num_rows, |args| {
        args.arg(&dict_table_view)
            .arg(&dict_bytes_view)
            .arg(&packed_view)
            .arg(&boundaries_view)
            .arg(&output_offsets_view)
            .arg(&device_output)
            .arg(&num_rows_u64);
    })?;

    // ── D2H output bytes ──
    let host_bytes = CudaDeviceBuffer::new(device_output)
        .copy_to_host(Alignment::new(1))?
        .await?;
    let bytes = host_bytes.as_slice()[..total_size].to_vec();

    Ok(OnPairGpuDecoded {
        bytes,
        offsets: host_offsets,
    })
}
