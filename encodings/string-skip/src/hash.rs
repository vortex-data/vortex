// SPDX-License-Identifier: Apache-2.0
//! Hash helpers used by the bloom and bigram tables.

use serde::Deserialize;
use serde::Serialize;

/// 32-bit splitmix step. Tiny, branch-free, no SIMD needed.
#[inline]
pub fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

/// Hash a pair of code IDs into two 32-bit hashes for double-hashing.
#[inline]
pub fn pair_hash(a: u16, b: u16) -> (u32, u32) {
    let key = ((a as u32) << 16) | (b as u32);
    let h1 = splitmix32(key);
    let h2 = splitmix32(key ^ 0x27d4_eb2f);
    (h1, h2)
}

/// FNV-1a 32-bit hash over byte slices (used for byte n-grams).
#[inline]
pub fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Hash a byte slice into a pair of 32-bit hashes.
#[inline]
pub fn hash_pair(bytes: &[u8]) -> (u32, u32) {
    let h1 = fnv1a_32(bytes);
    let h2 = h1 ^ 0x27d4_eb2f;
    (h1, h2)
}

/// Packed bigram key for ubiquity and tier tables. Sorted as a u32 for
/// efficient binary search.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BigramKey(pub u32);

impl BigramKey {
    /// Pack two code IDs into a single sortable key.
    #[inline]
    pub fn new(a: u16, b: u16) -> Self {
        Self(((a as u32) << 16) | (b as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix32_distributes() {
        // Sanity: at least some bits differ for sequential inputs.
        let h0 = splitmix32(0);
        let h1 = splitmix32(1);
        assert_ne!(h0, h1);
        let diff = (h0 ^ h1).count_ones();
        assert!(
            diff > 8,
            "splitmix32 has poor avalanche: only {diff} bits diff"
        );
    }

    #[test]
    fn pair_hash_distinguishes_order() {
        // (a,b) and (b,a) should produce different hashes.
        let (h1a, h2a) = pair_hash(100, 200);
        let (h1b, h2b) = pair_hash(200, 100);
        assert_ne!((h1a, h2a), (h1b, h2b));
    }

    #[test]
    fn bigram_key_packs() {
        let k = BigramKey::new(0xabcd, 0x1234);
        assert_eq!(k.0, 0xabcd1234);
    }
}
