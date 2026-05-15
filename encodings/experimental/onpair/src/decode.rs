// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Pure-Rust decoder for an [`OnPair`][crate::OnPair] array.
//!
//! The decode loop is intentionally simple — one `u16` code load, one
//! `u64` table load, one fixed 16-byte over-copy `memcpy` — so the
//! autovectoriser keeps the hot path SIMD-friendly. We materialise the
//! children once into native-aligned `Buffer<u_N>`s (and pack the dict
//! offsets + lengths into a single `Buffer<u64>` lookup table) so the
//! inner loop indexes straight into raw slices with no branches.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;

/// Materialised, host-resident copies of every read path's input.
///
/// Each integer child (`dict_offsets`, `codes`, `codes_offsets`) is a slot
/// on the outer `OnPair` array, possibly wrapped in a non-canonical
/// encoding the cascading compressor chose (e.g. FastLanes-bit-packed
/// `codes`, `narrow`-ed dict offsets). `execute::<PrimitiveArray>` may
/// hand us back a narrower ptype than the decode loop wants. `collect`
/// widens each child to the decoder's native width (`u32` for both offset
/// arrays, `u16` for codes) once so the inner loop is branch-free pointer
/// arithmetic.
///
/// Construction also packs `dict_offsets` into the combined
/// `(offset << 16) | length` `dict_table` so the decode hot loop loads a
/// single `u64` per token instead of two adjacent `u32`s.
pub struct OwnedDecodeInputs {
    pub dict_bytes: ByteBuffer,
    /// `(dict_offset << 16) | dict_len` per token. `dict_len` ≤
    /// `MAX_TOKEN_SIZE = 16` so 16 bits suffice.
    pub dict_table: Buffer<u64>,
    pub codes: Buffer<u16>,
    pub codes_offsets: Buffer<u32>,
}

impl OwnedDecodeInputs {
    pub fn collect(array: ArrayView<'_, OnPair>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let dict_offsets_arr = to_primitive(array.dict_offsets(), ctx)?;
        let dict_table = build_dict_table(&dict_offsets_arr);
        Ok(Self {
            dict_bytes: array.dict_bytes().clone(),
            dict_table,
            codes: widen_to_u16(&to_primitive(array.codes(), ctx)?),
            codes_offsets: widen_to_u32(&to_primitive(array.codes_offsets(), ctx)?),
        })
    }

    pub fn view(&self) -> DecodeView<'_> {
        DecodeView {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_table: self.dict_table.as_slice(),
            codes: self.codes.as_slice(),
            codes_offsets: self.codes_offsets.as_slice(),
        }
    }
}

/// Pack `dict_offsets` directly into `(offset << 16) | length` per token.
/// Reads through the integer-ptype macro once so we don't have to widen
/// the offsets buffer first — saves one `Vec` allocation in the common
/// (non-narrowed) case.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn build_dict_table(arr: &PrimitiveArray) -> Buffer<u64> {
    match_each_integer_ptype!(arr.ptype(), |P| {
        let slice = arr.as_slice::<P>();
        if slice.is_empty() {
            return Buffer::<u64>::copy_from(Vec::<u64>::new());
        }
        let dict_size = slice.len() - 1;
        let mut table = BufferMut::<u64>::with_capacity(dict_size);
        for i in 0..dict_size {
            let off = slice[i] as u64;
            let len = (slice[i + 1] - slice[i]) as u64;
            // SAFETY: capacity reserved above; we push exactly dict_size times.
            unsafe { table.push_unchecked((off << 16) | len) };
        }
        table.freeze()
    })
}

fn to_primitive(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    arr.clone().execute::<PrimitiveArray>(ctx)
}

/// Widen any integer-typed `PrimitiveArray` to `Buffer<u32>`. When the
/// underlying ptype already matches we transmute the buffer instead of
/// allocating a new one. Used when the cascading compressor narrowed an
/// offset array (e.g. `u32` → `u16`).
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn widen_to_u32(arr: &PrimitiveArray) -> Buffer<u32> {
    if arr.ptype() == PType::U32 {
        // Cheap: PrimitiveArray's underlying buffer is Arc-shared, so
        // `into_buffer` on a clone is effectively a refcount bump.
        return arr.clone().into_buffer::<u32>();
    }
    match_each_integer_ptype!(arr.ptype(), |P| {
        let slice = arr.as_slice::<P>();
        let mut out = BufferMut::<u32>::with_capacity(slice.len());
        for &v in slice {
            // SAFETY: capacity reserved above.
            unsafe { out.push_unchecked(v as u32) };
        }
        out.freeze()
    })
}

/// As `widen_to_u32` but for `Buffer<u16>`.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn widen_to_u16(arr: &PrimitiveArray) -> Buffer<u16> {
    if arr.ptype() == PType::U16 {
        return arr.clone().into_buffer::<u16>();
    }
    match_each_integer_ptype!(arr.ptype(), |P| {
        let slice = arr.as_slice::<P>();
        let mut out = BufferMut::<u16>::with_capacity(slice.len());
        for &v in slice {
            // SAFETY: capacity reserved above.
            unsafe { out.push_unchecked(v as u16) };
        }
        out.freeze()
    })
}

/// Borrowed slices for the decode loop.
#[derive(Copy, Clone)]
pub struct DecodeView<'a> {
    pub dict_bytes: &'a [u8],
    pub dict_table: &'a [u64],
    pub codes: &'a [u16],
    pub codes_offsets: &'a [u32],
}

impl<'a> DecodeView<'a> {
    /// Decode row `row` into `out` (appended). Thin wrapper around
    /// [`Self::decode_rows_into`].
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
        let decoded_len = self.decoded_len_rows(start, count);
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
    /// Mirrors OnPair C++ `decode_all<Bits = 16>` (and `decompress`)
    /// exactly: each iteration loads one `u16` code, one `u64` dict-table
    /// entry, issues a fixed [`MAX_TOKEN_SIZE`][crate::MAX_TOKEN_SIZE]
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
    /// * Every `code` in the window must be `< self.dict_table.len()`.
    #[inline]
    pub unsafe fn decode_rows_unchecked(&self, start: usize, count: usize, dst: *mut u8) -> usize {
        if count == 0 {
            return 0;
        }
        // SAFETY: caller invariants.
        let lo = unsafe { *self.codes_offsets.get_unchecked(start) } as usize;
        let hi = unsafe { *self.codes_offsets.get_unchecked(start + count) } as usize;

        let codes_ptr = self.codes.as_ptr();
        let table_ptr = self.dict_table.as_ptr();
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
                        let entry = *table_ptr.add(c);
                        let off = (entry >> 16) as usize;
                        let len = (entry & 0xffff) as usize;
                        std::ptr::copy_nonoverlapping(
                            dict_ptr.add(off),
                            cursor,
                            crate::MAX_TOKEN_SIZE,
                        );
                        cursor = cursor.add(len);
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
                let entry = *table_ptr.add(c);
                let off = (entry >> 16) as usize;
                let len = (entry & 0xffff) as usize;
                std::ptr::copy_nonoverlapping(dict_ptr.add(off), cursor, crate::MAX_TOKEN_SIZE);
                cursor = cursor.add(len);
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

    /// Decoded byte length of row `row` without copying any bytes.
    #[inline]
    pub fn decoded_len(&self, row: usize) -> usize {
        self.decoded_len_rows(row, 1)
    }

    /// Decoded byte length of rows `[start, start + count)`. Uses the
    /// combined `dict_table` — one `u64` load per token.
    #[inline]
    pub fn decoded_len_rows(&self, start: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        let lo = self.codes_offsets[start] as usize;
        let hi = self.codes_offsets[start + count] as usize;
        let mut total = 0usize;
        // SAFETY: bounds checked by indexing above.
        unsafe {
            for i in lo..hi {
                let c = *self.codes.get_unchecked(i) as usize;
                total += (*self.dict_table.get_unchecked(c) & 0xffff) as usize;
            }
        }
        total
    }

    /// Iterate the decoded bytes of `row` without materialising the full
    /// row, calling `f` on each contiguous dict slice. Returns
    ///
    /// * `true` if every slice was visited (i.e. `f` always returned
    ///   `true`),
    /// * `false` if `f` short-circuited with `false`.
    ///
    /// Useful for predicates that can short-circuit, e.g. `equals` and
    /// `starts_with`.
    #[inline]
    pub fn for_each_dict_slice<F: FnMut(&'a [u8]) -> bool>(&self, row: usize, mut f: F) -> bool {
        let lo = self.codes_offsets[row] as usize;
        let hi = self.codes_offsets[row + 1] as usize;
        let codes = &self.codes[lo..hi];
        // SAFETY: codes were validated at construction time.
        unsafe {
            for &c in codes {
                let entry = *self.dict_table.get_unchecked(c as usize);
                let off = (entry >> 16) as usize;
                let len = (entry & 0xffff) as usize;
                let slice = self.dict_bytes.get_unchecked(off..off + len);
                if !f(slice) {
                    return false;
                }
            }
        }
        true
    }
}
