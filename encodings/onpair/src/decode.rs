// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Pure-Rust decoder for an [`OnPair`][crate::OnPair] array.
//!
//! The decode loop is intentionally simple — three independent array
//! lookups and a `memcpy` — so the autovectoriser keeps the hot bytes-out
//! path SIMD-friendly. We materialise the children once into `Buffer<u16>`
//! / `Buffer<u32>` (always at native alignment) so the inner loop can index
//! straight into raw slices without branches.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;

/// Materialised, host-resident copies of every read path's input.
///
/// All four byte arrays come from the outer `OnPair` array as raw
/// `BufferHandle`s, which Vortex's flat-segment writer pads to the buffer's
/// own alignment on disk. To insulate the decoder from arbitrary host
/// alignment (e.g. a file segment that started mid-byte), we copy each
/// buffer into a `Buffer<uN>` at the right type. The decode hot loop then
/// indexes raw slices with no branches.
pub(crate) struct OwnedDecodeInputs {
    pub dict_bytes: ByteBuffer,
    pub dict_offsets: Buffer<u32>,
    pub codes: Buffer<u16>,
    pub codes_offsets: Buffer<u32>,
}

impl OwnedDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, _ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_offsets: bytes_to_buffer_u32(array.dict_offsets_bytes())?,
            codes: bytes_to_buffer_u16(array.codes_bytes_raw())?,
            codes_offsets: bytes_to_buffer_u32(array.codes_offsets_bytes())?,
        })
    }

    pub fn view(&self) -> DecodeView<'_> {
        DecodeView {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_offsets: self.dict_offsets.as_slice(),
            codes: self.codes.as_slice(),
            codes_offsets: self.codes_offsets.as_slice(),
        }
    }
}

/// Decode `bytes` (little-endian-packed u32s) into an aligned `Buffer<u32>`.
/// Goes through a typed `Vec<u32>` so the result is always 4-aligned.
/// LLVM autovectorises the inner `from_le_bytes` loop to a single load on
/// little-endian targets.
#[inline]
fn bytes_to_buffer_u32(bytes: &ByteBuffer) -> VortexResult<Buffer<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(vortex_err!(
            "OnPair: byte buffer of length {} is not a multiple of 4",
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    let mut out: Vec<u32> = Vec::with_capacity(n);
    let slice = bytes.as_slice();
    let mut i = 0;
    while i + 4 <= slice.len() {
        // SAFETY: bounds checked by the while condition.
        let arr: [u8; 4] = unsafe { slice.get_unchecked(i..i + 4).try_into().unwrap_unchecked() };
        out.push(u32::from_le_bytes(arr));
        i += 4;
    }
    Ok(Buffer::<u32>::copy_from(out))
}

/// Same as `bytes_to_buffer_u32` for u16.
#[inline]
fn bytes_to_buffer_u16(bytes: &ByteBuffer) -> VortexResult<Buffer<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(vortex_err!(
            "OnPair: byte buffer of length {} is not a multiple of 2",
            bytes.len()
        ));
    }
    let n = bytes.len() / 2;
    let mut out: Vec<u16> = Vec::with_capacity(n);
    let slice = bytes.as_slice();
    let mut i = 0;
    while i + 2 <= slice.len() {
        // SAFETY: bounds checked by the while condition.
        let arr: [u8; 2] = unsafe { slice.get_unchecked(i..i + 2).try_into().unwrap_unchecked() };
        out.push(u16::from_le_bytes(arr));
        i += 2;
    }
    Ok(Buffer::<u16>::copy_from(out))
}

/// Borrowed slices for the decode loop.
#[derive(Copy, Clone)]
pub(crate) struct DecodeView<'a> {
    pub dict_bytes: &'a [u8],
    pub dict_offsets: &'a [u32],
    pub codes: &'a [u16],
    pub codes_offsets: &'a [u32],
}

impl<'a> DecodeView<'a> {
    /// Decode row `row` into `out` (appended).
    ///
    /// Fast path matching OnPair's C++ decoder: a fixed [`MAX_TOKEN_SIZE`]
    /// memcpy per token, regardless of the token's true length. The output
    /// cursor advances by the *true* length, so the next memcpy overwrites
    /// the trailing slop from the previous one. Requires:
    ///
    /// * `dict_bytes` padded with `MAX_TOKEN_SIZE` trailing bytes (the
    ///   compress path enforces this).
    /// * `out` has at least `MAX_TOKEN_SIZE` bytes of headroom past the
    ///   decoded end. The function reserves this implicitly.
    ///
    /// On x86_64 / aarch64 LLVM lowers the fixed-size copy to a single
    /// 16-byte unaligned vector store, making each token an O(1) SIMD op.
    #[inline]
    pub fn decode_row_into(&self, row: usize, out: &mut Vec<u8>) {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        let row_codes = &self.codes[lo..hi];

        // Pre-compute the true decoded length so we can size `out` once and
        // use the unchecked-write fast loop below.
        let mut decoded_len = 0usize;
        for &c in row_codes {
            let dlo = self.dict_offsets[c as usize] as usize;
            let dhi = self.dict_offsets[c as usize + 1] as usize;
            decoded_len += dhi - dlo;
        }

        let written_start = out.len();
        out.reserve(decoded_len + crate::MAX_TOKEN_SIZE);
        // SAFETY: we just reserved at least `decoded_len + MAX_TOKEN_SIZE`
        // bytes past `written_start`. The over-copy writes
        // `MAX_TOKEN_SIZE` bytes per token, but we only advance the cursor
        // by the true token length, so the final `set_len` reflects the
        // true decoded length.
        unsafe {
            let dst_base = out.as_mut_ptr().add(written_start);
            let mut cursor = 0usize;
            for &c in row_codes {
                let dlo = *self.dict_offsets.get_unchecked(c as usize) as usize;
                let dhi = *self.dict_offsets.get_unchecked(c as usize + 1) as usize;
                let src = self.dict_bytes.as_ptr().add(dlo);
                let dst = dst_base.add(cursor);
                // Fixed 16-byte copy — LLVM lowers to a SIMD store.
                std::ptr::copy_nonoverlapping(src, dst, crate::MAX_TOKEN_SIZE);
                cursor += dhi - dlo;
            }
            out.set_len(written_start + decoded_len);
        }
    }

    /// Bulk decode rows `[start, start + count)` contiguously into `out`.
    /// Reuses the same over-copy strategy as [`Self::decode_row_into`] but
    /// computes lengths only once across the full window, which removes the
    /// per-row reserve / set_len overhead in the canonicalise hot path.
    pub fn decode_rows_into(&self, start: usize, count: usize, out: &mut Vec<u8>) {
        if count == 0 {
            return;
        }
        let lo = self.codes_offsets[start] as usize;
        let hi = self.codes_offsets[start + count] as usize;
        let codes = &self.codes[lo..hi];

        let mut decoded_len = 0usize;
        for &c in codes {
            let dlo = self.dict_offsets[c as usize] as usize;
            let dhi = self.dict_offsets[c as usize + 1] as usize;
            decoded_len += dhi - dlo;
        }

        let written_start = out.len();
        out.reserve(decoded_len + crate::MAX_TOKEN_SIZE);
        // SAFETY: same invariants as `decode_row_into` — pad written by
        // `MAX_TOKEN_SIZE`, advance cursor by true length, then truncate.
        unsafe {
            let dst_base = out.as_mut_ptr().add(written_start);
            let mut cursor = 0usize;
            for &c in codes {
                let dlo = *self.dict_offsets.get_unchecked(c as usize) as usize;
                let dhi = *self.dict_offsets.get_unchecked(c as usize + 1) as usize;
                let src = self.dict_bytes.as_ptr().add(dlo);
                let dst = dst_base.add(cursor);
                std::ptr::copy_nonoverlapping(src, dst, crate::MAX_TOKEN_SIZE);
                cursor += dhi - dlo;
            }
            out.set_len(written_start + decoded_len);
        }
    }

    /// Decoded byte length of row `row` without actually copying bytes.
    #[inline]
    pub fn decoded_len(&self, row: usize) -> usize {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        let row_codes = &self.codes[lo..hi];
        // Closed-form length sum — branch-free, autovectorises to gather + sub.
        row_codes
            .iter()
            .map(|&c| {
                let dlo = self.dict_offsets[c as usize] as usize;
                let dhi = self.dict_offsets[c as usize + 1] as usize;
                dhi - dlo
            })
            .sum()
    }

    /// Iterate the decoded bytes of `row` without materialising them, calling
    /// `f` on each contiguous dict slice. Returns early if `f` returns
    /// `false`. Useful for predicates that can short-circuit (e.g. `equals`,
    /// `starts_with`).
    #[inline]
    pub fn for_each_dict_slice<F: FnMut(&'a [u8]) -> bool>(&self, row: usize, mut f: F) -> bool {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        for &c in &self.codes[lo..hi] {
            let dlo = self.dict_offsets[c as usize] as usize;
            let dhi = self.dict_offsets[c as usize + 1] as usize;
            if !f(&self.dict_bytes[dlo..dhi]) {
                return false;
            }
        }
        true
    }
}
