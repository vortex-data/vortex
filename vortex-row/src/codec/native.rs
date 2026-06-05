// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `RowEncode` trait and its implementations for fixed-width native value types.

use super::*;

/// Internal trait for encoding a fixed-width native value into byte slots.
///
/// Implementations must produce a sequence of `size_of::<Self>()` bytes that is
/// lexicographically byte-comparable according to the natural ordering of the type.
pub(crate) trait RowEncode: Copy {
    /// Encode this value into `out`, inverting the bytes for descending order.
    fn encode_to(self, out: &mut [u8], descending: bool);
}

macro_rules! impl_row_encode_unsigned {
    ($t:ty) => {
        impl RowEncode for $t {
            #[inline]
            fn encode_to(self, out: &mut [u8], descending: bool) {
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

macro_rules! impl_row_encode_signed {
    ($t:ty) => {
        impl RowEncode for $t {
            #[inline]
            fn encode_to(self, out: &mut [u8], descending: bool) {
                let mut bytes = self.to_be_bytes();
                // Flip sign bit so negatives < non-negatives lexicographically.
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

impl_row_encode_unsigned!(u8);
impl_row_encode_unsigned!(u16);
impl_row_encode_unsigned!(u32);
impl_row_encode_unsigned!(u64);
impl_row_encode_signed!(i8);
impl_row_encode_signed!(i16);
impl_row_encode_signed!(i32);
impl_row_encode_signed!(i64);
impl_row_encode_signed!(i128);

impl RowEncode for f32 {
    fn encode_to(self, out: &mut [u8], descending: bool) {
        let bits = self.to_bits();
        let mask: u32 = if (bits >> 31) == 0 {
            0x8000_0000
        } else {
            0xFFFF_FFFF
        };
        let mut bytes = (bits ^ mask).to_be_bytes();
        if descending {
            for b in bytes.iter_mut() {
                *b ^= 0xFF;
            }
        }
        out.copy_from_slice(&bytes);
    }
}

impl RowEncode for f64 {
    fn encode_to(self, out: &mut [u8], descending: bool) {
        let bits = self.to_bits();
        let mask: u64 = if (bits >> 63) == 0 {
            0x8000_0000_0000_0000
        } else {
            0xFFFF_FFFF_FFFF_FFFF
        };
        let mut bytes = (bits ^ mask).to_be_bytes();
        if descending {
            for b in bytes.iter_mut() {
                *b ^= 0xFF;
            }
        }
        out.copy_from_slice(&bytes);
    }
}

impl RowEncode for f16 {
    fn encode_to(self, out: &mut [u8], descending: bool) {
        let bits = self.to_bits();
        let mask: u16 = if (bits >> 15) == 0 { 0x8000 } else { 0xFFFF };
        let mut bytes = (bits ^ mask).to_be_bytes();
        if descending {
            for b in bytes.iter_mut() {
                *b ^= 0xFF;
            }
        }
        out.copy_from_slice(&bytes);
    }
}
