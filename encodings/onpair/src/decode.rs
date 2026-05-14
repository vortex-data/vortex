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

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;

/// Materialised, host-resident copies of every read path's input.
///
/// Each integer child (`dict_offsets`, `codes`, `codes_offsets`) is a slot
/// on the outer `OnPair` array, possibly wrapped in a non-canonical encoding
/// the cascading compressor chose (e.g. FastLanes-bit-packed `codes`,
/// `narrow`-ed dict offsets) and `execute::<PrimitiveArray>` may hand us
/// back a narrower ptype than the decode loop wants (`u8`/`u16` instead of
/// `u32`). `collect` widens each child to the decoder's native width
/// (`u32` for both offset arrays, `u16` for codes) once so the inner loop
/// is branch-free pointer arithmetic.
pub struct OwnedDecodeInputs {
    pub dict_bytes: ByteBuffer,
    pub dict_offsets: Buffer<u32>,
    pub codes: Buffer<u16>,
    pub codes_offsets: Buffer<u32>,
}

impl OwnedDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_offsets: widen_to_u32(&to_primitive(array.dict_offsets(), ctx)?),
            codes: widen_to_u16(&to_primitive(array.codes(), ctx)?),
            codes_offsets: widen_to_u32(&to_primitive(array.codes_offsets(), ctx)?),
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

fn to_primitive(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    arr.clone().execute::<PrimitiveArray>(ctx)
}

/// Widen any integer-typed PrimitiveArray to `Buffer<u32>`. Used when the
/// cascading compressor narrowed an offset array (e.g. `u32` → `u16`) and
/// the decode loop wants the canonical wide type. The macro covers `i64` /
/// `u64` too; for OnPair-produced offsets those values always fit in u32
/// (we cap at `dict_offsets[last] = dict_bytes.len() ≤ u32::MAX`).
#[allow(clippy::cast_lossless, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::unnecessary_cast)]
fn widen_to_u32(arr: &PrimitiveArray) -> Buffer<u32> {
    match_each_integer_ptype!(arr.ptype(), |P| {
        Buffer::<u32>::copy_from(
            arr.as_slice::<P>()
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>(),
        )
    })
}

#[allow(clippy::cast_lossless, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::unnecessary_cast)]
fn widen_to_u16(arr: &PrimitiveArray) -> Buffer<u16> {
    match_each_integer_ptype!(arr.ptype(), |P| {
        Buffer::<u16>::copy_from(
            arr.as_slice::<P>()
                .iter()
                .map(|&v| v as u16)
                .collect::<Vec<_>>(),
        )
    })
}

/// Borrowed slices for the decode loop.
#[derive(Copy, Clone)]
pub struct DecodeView<'a> {
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
        self.decode_rows_into(row, 1, out);
    }

    /// Bulk decode rows `[start, start + count)` contiguously into `out`.
    /// Pre-computes the decoded length, reserves once, then delegates to
    /// the unrolled fast path. Callers that already know the size (e.g.
    /// canonicalize from `uncompressed_lengths`) should call
    /// [`Self::decode_rows_into_with_size`] to skip the size pre-pass.
    pub fn decode_rows_into(&self, start: usize, count: usize, out: &mut Vec<u8>) {
        if count == 0 {
            return;
        }
        // Closed-form sum over the token window — autovectorises.
        let decoded_len = {
            let lo = self.codes_offsets[start] as usize;
            let hi = self.codes_offsets[start + count] as usize;
            let mut total = 0usize;
            // SAFETY: bounds checked by indexing above.
            unsafe {
                for i in lo..hi {
                    let c = *self.codes.get_unchecked(i) as usize;
                    let dlo = *self.dict_offsets.get_unchecked(c) as usize;
                    let dhi = *self.dict_offsets.get_unchecked(c + 1) as usize;
                    total += dhi - dlo;
                }
            }
            total
        };

        let written_start = out.len();
        out.reserve(decoded_len + crate::MAX_TOKEN_SIZE);
        // SAFETY: capacity reserved above; `decode_rows_unchecked`'s
        // invariants are upheld by the [`OnPair::try_new`] validation.
        unsafe {
            let written =
                self.decode_rows_unchecked(start, count, out.as_mut_ptr().add(written_start));
            debug_assert_eq!(written, decoded_len);
            out.set_len(written_start + written);
        }
    }

    /// Single-pass over-copy decode of a token window into raw `dst`.
    ///
    /// Mirrors OnPair C++ `decode_all<Bits = 16>` (and `decompress`) exactly:
    /// each iteration loads one `u16` code, two adjacent `u32` dict
    /// offsets, issues a fixed [`MAX_TOKEN_SIZE`][crate::MAX_TOKEN_SIZE]
    /// `copy_nonoverlapping` (which LLVM lowers to a single unaligned
    /// 128-bit SIMD store on x86_64 / aarch64), and advances the cursor by
    /// the *true* token length. The body is hand-unrolled four times so
    /// the CPU can keep four independent stores in flight, matching the
    /// `ONPAIR_EMIT4` block of the upstream `decode_all.h`.
    ///
    /// Returns the number of *true* bytes written.
    ///
    /// # Safety
    /// * `dst` must point into a region with at least
    ///   `decoded_byte_length + MAX_TOKEN_SIZE` bytes of writable
    ///   uninitialised capacity.
    /// * `self.dict_bytes` must have at least `MAX_TOKEN_SIZE` trailing
    ///   pad bytes past the last real token byte (`compress.rs` enforces
    ///   this).
    /// * Every `code` in the window must be `< dict_offsets.len() - 1`.
    #[inline]
    pub unsafe fn decode_rows_unchecked(&self, start: usize, count: usize, dst: *mut u8) -> usize {
        if count == 0 {
            return 0;
        }
        // SAFETY: caller invariants.
        let lo = unsafe { *self.codes_offsets.get_unchecked(start) } as usize;
        let hi = unsafe { *self.codes_offsets.get_unchecked(start + count) } as usize;

        let codes_ptr = self.codes.as_ptr();
        let off_ptr = self.dict_offsets.as_ptr();
        let dict_ptr = self.dict_bytes.as_ptr();

        let mut cursor = dst;
        let unroll_end = lo + ((hi - lo) & !3);
        let mut i = lo;
        // SAFETY: indices derived from validated offsets; the 16-byte
        // over-copy reads stay within `dict_bytes`'s trailing pad; writes
        // stay within the caller-promised capacity.
        unsafe {
            while i < unroll_end {
                macro_rules! emit {
                    ($k:expr) => {{
                        let c = *codes_ptr.add(i + $k) as usize;
                        let off_lo = *off_ptr.add(c) as usize;
                        let off_hi = *off_ptr.add(c + 1) as usize;
                        std::ptr::copy_nonoverlapping(
                            dict_ptr.add(off_lo),
                            cursor,
                            crate::MAX_TOKEN_SIZE,
                        );
                        cursor = cursor.add(off_hi - off_lo);
                    }};
                }
                emit!(0);
                emit!(1);
                emit!(2);
                emit!(3);
                i += 4;
            }
            while i < hi {
                let c = *codes_ptr.add(i) as usize;
                let off_lo = *off_ptr.add(c) as usize;
                let off_hi = *off_ptr.add(c + 1) as usize;
                std::ptr::copy_nonoverlapping(dict_ptr.add(off_lo), cursor, crate::MAX_TOKEN_SIZE);
                cursor = cursor.add(off_hi - off_lo);
                i += 1;
            }
            cursor.offset_from(dst) as usize
        }
    }

    /// Single-pass decode when the caller already knows the total decoded
    /// byte length (e.g. from summing `uncompressed_lengths`). Skips the
    /// size-precomputation pass.
    ///
    /// # Safety
    /// `out.capacity() - out.len() >= total_size + MAX_TOKEN_SIZE` and
    /// `total_size` equals the true decoded length.
    #[inline]
    pub unsafe fn decode_rows_into_with_size(
        &self,
        start: usize,
        count: usize,
        total_size: usize,
        out: &mut Vec<u8>,
    ) {
        let written_start = out.len();
        debug_assert!(out.capacity() - written_start >= total_size + crate::MAX_TOKEN_SIZE);
        // SAFETY: caller's invariants.
        let written = unsafe {
            self.decode_rows_unchecked(start, count, out.as_mut_ptr().add(written_start))
        };
        debug_assert_eq!(written, total_size);
        // SAFETY: `written` ≤ reserved capacity (caller invariants).
        unsafe { out.set_len(written_start + written) };
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
