// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Struct-of-arrays ("split") layouts for 128- and 256-bit decimal storage.
//!
//! Arrow stores decimals array-of-structs: each value is a contiguous 16-byte
//! (i128) or 32-byte (i256) little-endian integer. That interleaving defeats
//! lane-parallel arithmetic because the carry between the low and high halves
//! of one value sits inside a single SIMD lane.
//!
//! The split layout instead stores one `Vec<u64>` per 64-bit limb. With 8
//! values' low limbs packed contiguously we can add them with a single
//! `vpaddq`, detect per-lane unsigned overflow with `vpcmpuq`, and add the
//! carry into the high limbs. It is also better for compression: for "small"
//! decimals the high limbs are all zero (or a single repeated sign word) and
//! collapse to nothing under bit-packing / RLE / general compressors.

use arrow_buffer::i256;

/// i128 values stored as two little-endian 64-bit limb streams.
#[derive(Clone, Debug, Default)]
pub struct SplitI128 {
    /// Low 64 bits of each value (unsigned limb).
    pub lo: Vec<u64>,
    /// High 64 bits of each value (top limb, two's-complement).
    pub hi: Vec<u64>,
}

impl SplitI128 {
    /// Split a contiguous (Arrow-style) `i128` slice into limb streams.
    pub fn from_aos(values: &[i128]) -> Self {
        let mut lo = Vec::with_capacity(values.len());
        let mut hi = Vec::with_capacity(values.len());
        for &v in values {
            let bits = v as u128;
            lo.push(bits as u64);
            hi.push((bits >> 64) as u64);
        }
        Self { lo, hi }
    }

    /// Reassemble the contiguous `i128` representation.
    pub fn to_aos(&self) -> Vec<i128> {
        self.lo
            .iter()
            .zip(&self.hi)
            .map(|(&lo, &hi)| (((hi as u128) << 64) | (lo as u128)) as i128)
            .collect()
    }

    /// Allocate an output buffer of the same length.
    pub fn zeroed_like(&self) -> Self {
        Self {
            lo: vec![0; self.lo.len()],
            hi: vec![0; self.hi.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.lo.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lo.is_empty()
    }
}

/// i256 values stored as four little-endian 64-bit limb streams.
///
/// `limbs[0]` is the least significant limb, `limbs[3]` the most significant
/// (two's-complement top limb).
#[derive(Clone, Debug, Default)]
pub struct SplitI256 {
    pub limbs: [Vec<u64>; 4],
}

impl SplitI256 {
    /// Split a contiguous (Arrow-style) `i256` slice into four limb streams.
    pub fn from_aos(values: &[i256]) -> Self {
        let n = values.len();
        let mut limbs: [Vec<u64>; 4] = [
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        ];
        for v in values {
            let bytes = v.to_le_bytes();
            for (k, limb) in limbs.iter_mut().enumerate() {
                let mut word = [0u8; 8];
                word.copy_from_slice(&bytes[k * 8..k * 8 + 8]);
                limb.push(u64::from_le_bytes(word));
            }
        }
        Self { limbs }
    }

    /// Reassemble the contiguous `i256` representation.
    pub fn to_aos(&self) -> Vec<i256> {
        (0..self.len())
            .map(|i| {
                let mut bytes = [0u8; 32];
                for (k, limb) in self.limbs.iter().enumerate() {
                    bytes[k * 8..k * 8 + 8].copy_from_slice(&limb[i].to_le_bytes());
                }
                i256::from_le_bytes(bytes)
            })
            .collect()
    }

    pub fn zeroed_like(&self) -> Self {
        let n = self.len();
        Self {
            limbs: [vec![0; n], vec![0; n], vec![0; n], vec![0; n]],
        }
    }

    pub fn len(&self) -> usize {
        self.limbs[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.limbs[0].is_empty()
    }
}
