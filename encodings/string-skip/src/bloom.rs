// SPDX-License-Identifier: Apache-2.0
//! A simple bloom filter with **variable-k** support per insert/probe.
//!
//! The standard `fastbloom`/`bloomfilter` crates assume a fixed `k` at
//! construction. We need variable `k` for the BitFunnel-style
//! [`crate::tiers::BigramTiers`] where common bigrams get fewer bit
//! probes than rare ones. That single requirement makes existing
//! crates a non-fit, so we ship a small bloom of our own.
//!
//! The implementation uses double-hashing (`h1 + i*h2`), the same trick
//! that high-performance blooms use internally. For 2–16 KB blooms
//! that are L1-resident, the perf difference to SIMD blooms is below
//! noise.

use serde::{Deserialize, Serialize};

/// Bit-level approximate-membership filter.
///
/// The number of bits is rounded up to the next power of two for
/// fast modular indexing via mask.
#[derive(Clone, Serialize, Deserialize)]
pub struct Bloom {
    bits: Vec<u64>,
    mask: u64,
    /// Default `k` for the convenience methods that omit `k`.
    default_k: u32,
}

impl Bloom {
    /// Construct a bloom with `num_bits` (rounded up to next power of
    /// two, min 64) and the given default `k`.
    pub fn new(num_bits: usize, default_k: u32) -> Self {
        let num_bits = num_bits.next_power_of_two().max(64);
        let mask = (num_bits as u64) - 1;
        let bits = vec![0u64; num_bits / 64];
        Self { bits, mask, default_k }
    }

    /// Bytes occupied by the bit array (not counting `mask`/`k`).
    pub fn byte_size(&self) -> usize {
        self.bits.len() * 8
    }

    /// Number of bits in the underlying bit array.
    pub fn num_bits(&self) -> usize {
        self.bits.len() * 64
    }

    /// Insert with the default `k`.
    #[inline]
    pub fn insert(&mut self, h1: u32, h2: u32) {
        self.insert_k(h1, h2, self.default_k);
    }

    /// Probe with the default `k`.
    #[inline]
    pub fn contains(&self, h1: u32, h2: u32) -> bool {
        self.contains_k(h1, h2, self.default_k)
    }

    /// Insert with an explicit `k`. `k=0` is a no-op (BitFunnel "skip").
    #[inline]
    pub fn insert_k(&mut self, h1: u32, h2: u32, k: u32) {
        let mask = self.mask;
        for probe_idx in 0..k {
            let raw = (h1 as u64).wrapping_add((probe_idx as u64).wrapping_mul(h2 as u64));
            let pos = raw & mask;
            self.bits[(pos / 64) as usize] |= 1u64 << (pos % 64);
        }
    }

    /// Probe with an explicit `k`. `k=0` returns `true` (item is
    /// treated as "always present" — its bits weren't inserted).
    #[inline]
    pub fn contains_k(&self, h1: u32, h2: u32, k: u32) -> bool {
        let mask = self.mask;
        for probe_idx in 0..k {
            let raw = (h1 as u64).wrapping_add((probe_idx as u64).wrapping_mul(h2 as u64));
            let pos = raw & mask;
            if (self.bits[(pos / 64) as usize] >> (pos % 64)) & 1 == 0 {
                return false;
            }
        }
        true
    }

    /// Estimate the fill ratio (fraction of bits set). For diagnostics.
    pub fn fill_ratio(&self) -> f64 {
        let set: u32 = self.bits.iter().map(|w| w.count_ones()).sum();
        set as f64 / (self.bits.len() * 64) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_inserted_items() {
        let mut b = Bloom::new(4096, 3);
        for i in 0..100u32 {
            b.insert(i, i.wrapping_mul(0x27d4_eb2f));
        }
        for i in 0..100u32 {
            assert!(b.contains(i, i.wrapping_mul(0x27d4_eb2f)),
                "missed item {i}");
        }
    }

    #[test]
    fn no_false_negatives_variable_k() {
        let mut b = Bloom::new(8192, 3);
        for (i, k) in (0..200u32).zip([1u32, 2, 3].iter().cycle()) {
            b.insert_k(i, i.wrapping_mul(0x27d4_eb2f), *k);
        }
        for (i, k) in (0..200u32).zip([1u32, 2, 3].iter().cycle()) {
            assert!(b.contains_k(i, i.wrapping_mul(0x27d4_eb2f), *k),
                "missed item {i} k={k}");
        }
    }

    #[test]
    fn k_zero_always_returns_true() {
        let b = Bloom::new(64, 3);
        assert!(b.contains_k(123, 456, 0));
    }

    #[test]
    fn fpr_below_target() {
        // 10K items in 100K bits with k=7. In real use, the caller
        // passes already-mixed hashes (e.g. via `pair_hash`). We
        // splitmix the inputs here to simulate that.
        use crate::hash::splitmix32;
        let mut b = Bloom::new(100_000, 7);
        for i in 0..10_000u32 {
            let h1 = splitmix32(i);
            let h2 = splitmix32(i ^ 0x27d4_eb2f);
            b.insert(h1, h2);
        }
        let mut fp = 0u32;
        let tries = 10_000;
        for i in 100_000..100_000 + tries {
            let h1 = splitmix32(i);
            let h2 = splitmix32(i ^ 0x27d4_eb2f);
            if b.contains(h1, h2) {
                fp += 1;
            }
        }
        let rate = fp as f64 / tries as f64;
        // Theoretical FPR for 10K items, 131072 bits, k=7 is ~1.3%.
        // Be generous on the bound to avoid flakiness.
        assert!(rate < 0.03, "FPR too high: {rate}");
    }

    #[test]
    fn serializes_with_bincode() {
        let mut b = Bloom::new(256, 3);
        b.insert(42, 1337);
        let bytes = bincode::serialize(&b).unwrap();
        let b2: Bloom = bincode::deserialize(&bytes).unwrap();
        assert!(b2.contains(42, 1337));
    }
}
