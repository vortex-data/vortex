// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression measurement for the split vs interleaved layouts.
//!
//! Two metrics, both relevant to how Vortex actually stores integers:
//!   * `ffor_bits` - bits per value after frame-of-reference (subtract min,
//!     bit-pack the range). This is what FastLanes bit-packing achieves and is
//!     the cleanest signal that a limb stream is "free" (0 bits = constant).
//!   * `zstd` - bytes after a general-purpose compressor, a proxy for what a
//!     cascading compressor with entropy coding gets.

use crate::layout::SplitI128;
use crate::layout::SplitI256;

/// Bytes after zstd at the given level (falls back to raw length on error).
pub fn zstd_size(bytes: &[u8], level: i32) -> usize {
    zstd::encode_all(bytes, level).map_or(bytes.len(), |v| v.len())
}

/// Little-endian bytes of a u64 limb stream.
pub fn limb_bytes(stream: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(stream.len() * 8);
    for &w in stream {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Contiguous 16-byte little-endian bytes of an i128 slice (Arrow layout).
pub fn i128_aos_bytes(values: &[i128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 16);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Bits per value after frame-of-reference encoding of a limb stream, treating
/// limbs as unsigned. A constant stream (e.g. all-zero high limbs of small
/// decimals) reports 0.
pub fn ffor_bits(stream: &[u64]) -> u32 {
    let Some(&first) = stream.first() else {
        return 0;
    };
    let mut min = first;
    let mut max = first;
    for &w in stream {
        min = min.min(w);
        max = max.max(w);
    }
    let range = max - min;
    if range == 0 {
        0
    } else {
        64 - range.leading_zeros()
    }
}

/// Compression facts for one i128 column.
#[derive(Debug)]
pub struct I128Report {
    pub n: usize,
    pub aos_zstd: usize,
    pub lo_zstd: usize,
    pub hi_zstd: usize,
    pub lo_ffor_bits: u32,
    pub hi_ffor_bits: u32,
}

impl I128Report {
    pub fn split_zstd(&self) -> usize {
        self.lo_zstd + self.hi_zstd
    }

    pub fn zstd_ratio(&self) -> f64 {
        if self.split_zstd() == 0 {
            0.0
        } else {
            self.aos_zstd as f64 / self.split_zstd() as f64
        }
    }

    /// Raw interleaved size: 16 bytes per value.
    pub fn raw_bytes(&self) -> usize {
        self.n * 16
    }

    /// Size if each limb is FFoR bit-packed (what Vortex's FastLanes path does
    /// to ≤64-bit lanes). The interleaved i128 cannot be bit-packed this way -
    /// FastLanes has no 128-bit lane - so this compression is only reachable
    /// via the split.
    pub fn split_bitpacked_bytes(&self) -> usize {
        ((self.lo_ffor_bits + self.hi_ffor_bits) as usize * self.n).div_ceil(8)
    }

    /// Bit-packed split size vs raw interleaved.
    pub fn bitpack_ratio(&self) -> f64 {
        let packed = self.split_bitpacked_bytes();
        if packed == 0 {
            f64::INFINITY
        } else {
            self.raw_bytes() as f64 / packed as f64
        }
    }
}

pub fn analyze_i128(values: &[i128], level: i32) -> I128Report {
    let split = SplitI128::from_aos(values);
    I128Report {
        n: values.len(),
        aos_zstd: zstd_size(&i128_aos_bytes(values), level),
        lo_zstd: zstd_size(&limb_bytes(&split.lo), level),
        hi_zstd: zstd_size(&limb_bytes(&split.hi), level),
        lo_ffor_bits: ffor_bits(&split.lo),
        hi_ffor_bits: ffor_bits(&split.hi),
    }
}

/// Compression facts for one i256 column.
#[derive(Debug)]
pub struct I256Report {
    pub n: usize,
    pub aos_zstd: usize,
    pub limb_zstd: [usize; 4],
    pub limb_ffor_bits: [u32; 4],
}

impl I256Report {
    pub fn split_zstd(&self) -> usize {
        self.limb_zstd.iter().sum()
    }

    pub fn zstd_ratio(&self) -> f64 {
        if self.split_zstd() == 0 {
            0.0
        } else {
            self.aos_zstd as f64 / self.split_zstd() as f64
        }
    }

    pub fn raw_bytes(&self) -> usize {
        self.n * 32
    }

    pub fn split_bitpacked_bytes(&self) -> usize {
        let bits: u32 = self.limb_ffor_bits.iter().sum();
        (bits as usize * self.n).div_ceil(8)
    }

    pub fn bitpack_ratio(&self) -> f64 {
        let packed = self.split_bitpacked_bytes();
        if packed == 0 {
            f64::INFINITY
        } else {
            self.raw_bytes() as f64 / packed as f64
        }
    }
}

pub fn analyze_i256(values: &[arrow_buffer::i256], level: i32) -> I256Report {
    let split = SplitI256::from_aos(values);
    let mut aos = Vec::with_capacity(values.len() * 32);
    for v in values {
        aos.extend_from_slice(&v.to_le_bytes());
    }
    let mut limb_zstd = [0usize; 4];
    let mut limb_ffor_bits = [0u32; 4];
    for k in 0..4 {
        limb_zstd[k] = zstd_size(&limb_bytes(&split.limbs[k]), level);
        limb_ffor_bits[k] = ffor_bits(&split.limbs[k]);
    }
    I256Report {
        n: values.len(),
        aos_zstd: zstd_size(&aos, level),
        limb_zstd,
        limb_ffor_bits,
    }
}
