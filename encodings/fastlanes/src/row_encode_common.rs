// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for the FastLanes row-encode kernels (BitPacked, FoR, Delta).
//!
//! Each kernel walks the compressed storage in 1024-element chunks, unpacks each chunk into
//! a stack-local buffer, and writes the row-encoded bytes in one pass. This module defines
//! the per-row write primitive used after a chunk has been unpacked.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "row encoding indexes into u32-sized buffers"
)]

use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;

/// Trait implemented by primitive types that can be written into a row-encoded byte slot.
///
/// Mirrors `vortex_row::codec::RowEncode` for the integer types that show up as the output
/// of BitPacked/FoR/Delta.
pub trait PrimRowEncode: Copy {
    /// Encode this value into `out`, inverting the bytes for descending order.
    fn row_encode_to(self, out: &mut [u8], descending: bool);
}

macro_rules! impl_unsigned {
    ($t:ty) => {
        impl PrimRowEncode for $t {
            #[inline]
            fn row_encode_to(self, out: &mut [u8], descending: bool) {
                let bytes = self.to_be_bytes();
                if descending {
                    for (i, b) in bytes.iter().enumerate() {
                        out[i] = b ^ 0xFF;
                    }
                } else {
                    out.copy_from_slice(&bytes);
                }
            }
        }
    };
}

macro_rules! impl_signed {
    ($t:ty) => {
        impl PrimRowEncode for $t {
            #[inline]
            fn row_encode_to(self, out: &mut [u8], descending: bool) {
                let mut bytes = self.to_be_bytes();
                bytes[0] ^= 0x80;
                if descending {
                    for (i, b) in bytes.iter().enumerate() {
                        out[i] = b ^ 0xFF;
                    }
                } else {
                    out.copy_from_slice(&bytes);
                }
            }
        }
    };
}

impl_unsigned!(u8);
impl_unsigned!(u16);
impl_unsigned!(u32);
impl_unsigned!(u64);
impl_signed!(i8);
impl_signed!(i16);
impl_signed!(i32);
impl_signed!(i64);

/// Encoded row width (sentinel + value bytes) for the given primitive type.
#[inline]
pub fn encoded_size_for_ptype(ptype: PType) -> u32 {
    1 + (ptype.byte_width() as u32)
}

/// Write a contiguous slice of unpacked values (one chunk) into the row-encoded output buffer.
///
/// `chunk[j]` is the value for logical row `row_start + j`. The output position for row `i`
/// is `offsets[i] + cursors[i]`; the cursor is advanced by `stride` after each row write.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn encode_primitive_chunk<T: NativePType + PrimRowEncode>(
    chunk: &[T],
    row_start: usize,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    mask: Option<&vortex_mask::Mask>,
    non_null: u8,
    null: u8,
    descending: bool,
    value_bytes: usize,
    stride: u32,
) {
    match mask {
        None => {
            for (j, &v) in chunk.iter().enumerate() {
                let row = row_start + j;
                let pos = (offsets[row] + cursors[row]) as usize;
                out[pos] = non_null;
                v.row_encode_to(&mut out[pos + 1..pos + 1 + value_bytes], descending);
                cursors[row] += stride;
            }
        }
        Some(m) => {
            for (j, &v) in chunk.iter().enumerate() {
                let row = row_start + j;
                let pos = (offsets[row] + cursors[row]) as usize;
                if m.value(row) {
                    out[pos] = non_null;
                    v.row_encode_to(&mut out[pos + 1..pos + 1 + value_bytes], descending);
                } else {
                    out[pos] = null;
                    for b in &mut out[pos + 1..pos + 1 + value_bytes] {
                        *b = 0;
                    }
                }
                cursors[row] += stride;
            }
        }
    }
}
