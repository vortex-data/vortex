// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::{InstBuilder, MemFlags};
use vortex_error::{VortexResult, vortex_bail};

use crate::emit::{ArgKey, EmitCtx, Lanes, LaneSlice, SigBuilder};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// Leaf stage: unpack W-bit values from a packed buffer into
/// `Lane(I32, Linear)` SIMD chunks.
///
/// Layout — fastlanes-style interleaved across `simd_lanes` streams:
///   Logical input `input[r*S + s]` (`S = simd_lanes`, `s` is the stream id,
///   `r` is the row within the stream) is packed densely into stream `s`,
///   and the per-stream u32 words are interleaved at `S` granularity in the
///   packed buffer. Concretely, the i-th u32 word of stream `s` lives at
///   `packed[i * S + s]`.
///
/// Why interleaved: loading 4 consecutive u32s with `load.i32x4` gives us the
/// i-th word of all 4 streams *simultaneously*, so a single vector shift +
/// mask extracts row i from all 4 streams at once. No `insertlane`.
///
/// Constraint for clean per-block addressing:
///   `n_chunks_per_emit * W` must be a multiple of 32. This guarantees the
///   per-block byte offset is a compile-time constant and word reads stay
///   aligned across block boundaries.
#[derive(Debug, Clone, Copy)]
pub struct BitPackedLoad {
    pub ptype: PType,
    pub bit_width: u8,
}

impl JitStage for BitPackedLoad {
    fn tag(&self) -> &'static str {
        "BitPackedLoad"
    }

    fn fingerprint(&self) -> Vec<u8> {
        vec![self.ptype as u8, self.bit_width]
    }

    fn output(&self) -> Form {
        Form::Lane(self.ptype, Layout::Linear)
    }

    fn declare(&self, sig: &mut SigBuilder) {
        sig.request_arg(ArgKey::InPtr);
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        if !matches!(self.ptype, PType::I32) {
            vortex_bail!("BitPackedLoad v1 supports only I32");
        }
        let w = self.bit_width as usize;
        if w == 0 || w >= 32 {
            vortex_bail!("BitPackedLoad: bit_width must be 1..32, got {}", w);
        }

        let in_ptr = cx.runtime_arg(&ArgKey::InPtr)?;
        let block_idx = cx.block_idx();
        let n_lanes = cx.chunk_count();
        let simd_lanes = self.ptype.simd_lanes() as usize;
        let n_chunks = n_lanes / simd_lanes;

        let bits_per_block_per_stream = n_chunks * w;
        if !bits_per_block_per_stream.is_multiple_of(32) {
            vortex_bail!(
                "BitPackedLoad: n_chunks ({}) * bit_width ({}) must be a multiple of 32, got {}",
                n_chunks,
                w,
                bits_per_block_per_stream
            );
        }
        let words_per_stream_per_block = bits_per_block_per_stream / 32;
        // Each "word" in the interleaved layout is `simd_lanes` u32s wide.
        let bytes_per_block = words_per_stream_per_block * simd_lanes * 4;

        let bpb_v = cx.const_int(PType::I64, bytes_per_block as i64);
        let block_byte_offset = cx.fb().ins().imul(block_idx, bpb_v);
        let base = cx.fb().ins().iadd(in_ptr, block_byte_offset);

        let simd_t = self.ptype.simd_type();

        // Pre-build broadcast masks. `band_imm` works on scalars only; for
        // vector AND we need `band(vec, splat(imm))`. Splat the masks once
        // per emit() so Cranelift's LICM can hoist them out of the block loop.
        let mask_full_scalar = cx.fb().ins().iconst(
            PType::I32.cl_type(),
            ((1u64 << w) - 1) as i64,
        );
        let mask_full = cx.fb().ins().splat(simd_t, mask_full_scalar);

        let mut chunks = Vec::with_capacity(n_chunks);
        for row in 0..n_chunks {
            let bit_start = row * w;
            let word_idx = bit_start / 32;
            let bit_off = bit_start % 32;
            let bits_in_lo = (32 - bit_off).min(w);
            let bits_in_hi = w - bits_in_lo;

            let lo_off = i32::try_from(word_idx * simd_lanes * 4)
                .expect("packed offset fits in i32");
            let lo_vec = cx
                .fb()
                .ins()
                .load(simd_t, MemFlags::trusted(), base, lo_off);

            // `ushr_imm` for vectors: applies scalar shift amount to each
            // lane. This one works.
            let shifted_lo = if bit_off == 0 {
                lo_vec
            } else {
                cx.fb().ins().ushr_imm(lo_vec, bit_off as i64)
            };

            let value = if bits_in_hi == 0 {
                if w == 32 {
                    shifted_lo
                } else {
                    cx.fb().ins().band(shifted_lo, mask_full)
                }
            } else {
                // Per-row partial masks for the straddle case.
                let mask_lo_scalar = cx
                    .fb()
                    .ins()
                    .iconst(PType::I32.cl_type(), ((1u64 << bits_in_lo) - 1) as i64);
                let mask_lo = cx.fb().ins().splat(simd_t, mask_lo_scalar);
                let mask_hi_scalar = cx
                    .fb()
                    .ins()
                    .iconst(PType::I32.cl_type(), ((1u64 << bits_in_hi) - 1) as i64);
                let mask_hi = cx.fb().ins().splat(simd_t, mask_hi_scalar);

                let hi_off = i32::try_from((word_idx + 1) * simd_lanes * 4)
                    .expect("packed offset fits in i32");
                let hi_vec = cx
                    .fb()
                    .ins()
                    .load(simd_t, MemFlags::trusted(), base, hi_off);
                let masked_lo = cx.fb().ins().band(shifted_lo, mask_lo);
                let masked_hi = cx.fb().ins().band(hi_vec, mask_hi);
                let shifted_hi = cx.fb().ins().ishl_imm(masked_hi, bits_in_lo as i64);
                cx.fb().ins().bor(masked_lo, shifted_hi)
            };

            chunks.push(value);
        }

        cx.put_output(Lanes::Of(LaneSlice::new_simd(
            chunks,
            self.ptype,
            Layout::Linear,
        )));
        Ok(())
    }
}

/// Reference packer: dense bit-packing in the interleaved layout described
/// above. `input.len()` must be a multiple of `simd_lanes` (4 for I32).
///
/// Output u32s = `ceil(input.len() / simd_lanes * w / 32) * simd_lanes`.
pub fn pack_dense(input: &[i32], w: u8) -> Vec<u32> {
    pack_dense_with_lanes(input, w, PType::I32.simd_lanes() as usize)
}

fn pack_dense_with_lanes(input: &[i32], w: u8, simd_lanes: usize) -> Vec<u32> {
    assert!((1..32).contains(&w), "bit_width must be 1..32");
    assert!(
        input.len().is_multiple_of(simd_lanes),
        "input.len() must be multiple of simd_lanes"
    );
    let elems_per_stream = input.len() / simd_lanes;
    let bits_per_stream = elems_per_stream * w as usize;
    let words_per_stream = bits_per_stream.div_ceil(32);
    let mut out = vec![0u32; words_per_stream * simd_lanes];
    let mask = ((1u64 << w) - 1) as u32;
    for stream_idx in 0..simd_lanes {
        for row in 0..elems_per_stream {
            let v = (input[row * simd_lanes + stream_idx] as u32) & mask;
            let bit_start = row * w as usize;
            let word_idx = bit_start / 32;
            let bit_off = bit_start % 32;
            out[word_idx * simd_lanes + stream_idx] |= v.wrapping_shl(bit_off as u32);
            let bits_in_lo = (32 - bit_off).min(w as usize);
            let bits_in_hi = w as usize - bits_in_lo;
            if bits_in_hi > 0 {
                out[(word_idx + 1) * simd_lanes + stream_idx] |=
                    v.wrapping_shr(bits_in_lo as u32);
            }
        }
    }
    out
}

/// Scalar reference unpacker that matches the layout used by `pack_dense`.
/// Helpful for ground-truth comparison in bench code.
pub fn unpack_one(packed: &[u32], idx: usize, w: u8, simd_lanes: usize) -> i32 {
    let s = idx % simd_lanes;
    let row = idx / simd_lanes;
    let bit_start = row * w as usize;
    let word_idx = bit_start / 32;
    let bit_off = bit_start % 32;
    let lo = packed[word_idx * simd_lanes + s] >> bit_off;
    let val = if bit_off + w as usize <= 32 {
        lo & ((1u32 << w) - 1)
    } else {
        let bits_in_lo = 32 - bit_off;
        let bits_in_hi = w as usize - bits_in_lo;
        let hi = packed[(word_idx + 1) * simd_lanes + s] & ((1u32 << bits_in_hi) - 1);
        (lo | (hi << bits_in_lo)) & ((1u32 << w) - 1)
    };
    val as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trips_interleaved() {
        let input: Vec<i32> = (0..128).map(|i| i & 0x7FF).collect();
        let packed = pack_dense(&input, 11);
        for (i, &expected) in input.iter().enumerate() {
            let got = unpack_one(&packed, i, 11, 4);
            assert_eq!(got, expected, "mismatch at i={i}");
        }
    }
}
