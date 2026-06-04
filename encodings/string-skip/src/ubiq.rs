// SPDX-License-Identifier: Apache-2.0
//! [`UbiquitousBigrams`] — column-level table of code bigrams that
//! appear in too many chunks to be discriminative. Inspired by
//! BitFunnel's frequency-conscious row allocation (Bing, SIGIR 2017).
//!
//! Skipping these bigrams on both build and probe concentrates bloom
//! bits on the rare bigrams that actually carry pruning signal.

use hashbrown::HashMap;
use hashbrown::HashSet;
use serde::Deserialize;
use serde::Serialize;

use crate::hash::BigramKey;

/// Sorted set of code bigrams that exceed a global frequency threshold.
#[derive(Clone, Serialize, Deserialize)]
pub struct UbiquitousBigrams {
    /// Sorted packed bigram IDs for O(log n) lookup.
    sorted: Vec<u32>,
}

impl UbiquitousBigrams {
    /// Build from the column's full code stream and offsets.
    ///
    /// A bigram is "ubiquitous" iff it appears in `> threshold_pct`% of
    /// chunks. Set `threshold_pct = 0` to disable (returns empty set).
    ///
    /// `codes_offsets[i..i+1]` defines row `i`'s slice of `codes`.
    /// `chunk_size` rows go in each chunk.
    pub fn build(
        codes: &[u16],
        codes_offsets: &[u32],
        chunk_size: usize,
        threshold_pct: u8,
    ) -> Self {
        let n_rows = codes_offsets.len().saturating_sub(1);
        let n_chunks = n_rows / chunk_size;
        if n_chunks == 0 || threshold_pct == 0 {
            return Self { sorted: Vec::new() };
        }
        let threshold = (n_chunks * threshold_pct as usize) / 100;

        let mut counts: HashMap<u32, u32> = HashMap::new();
        let mut seen: HashSet<u32> = HashSet::new();

        for c in 0..n_chunks {
            seen.clear();
            let row_lo = c * chunk_size;
            let row_hi = (c + 1) * chunk_size;
            for r in row_lo..row_hi {
                let rlo = codes_offsets[r] as usize;
                let rhi = codes_offsets[r + 1] as usize;
                for w in codes[rlo..rhi].windows(2) {
                    seen.insert(BigramKey::new(w[0], w[1]).0);
                }
            }
            for &key in &seen {
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<u32> = counts
            .into_iter()
            .filter(|&(_, c)| c as usize > threshold)
            .map(|(k, _)| k)
            .collect();
        sorted.sort_unstable();
        Self { sorted }
    }

    /// Empty table (no filtering).
    pub fn empty() -> Self {
        Self { sorted: Vec::new() }
    }

    /// True iff bigram `(a, b)` is ubiquitous.
    #[inline]
    pub fn contains(&self, a: u16, b: u16) -> bool {
        let key = BigramKey::new(a, b).0;
        self.sorted.binary_search(&key).is_ok()
    }

    /// Number of bigrams in the table.
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Bytes of memory the table occupies.
    pub fn byte_size(&self) -> usize {
        self.sorted.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_when_threshold_zero() {
        let codes = vec![1u16, 2, 3, 4];
        let offsets = vec![0u32, 2, 4];
        let ubiq = UbiquitousBigrams::build(&codes, &offsets, 1, 0);
        assert!(ubiq.is_empty());
    }

    #[test]
    fn finds_ubiquitous_bigram() {
        // 4 chunks of 1 row each, every row is (1,2) + (3,4).
        let codes = vec![1u16, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let offsets = vec![0u32, 4, 8, 12, 16];
        // Bigrams per row: (1,2), (2,3), (3,4). Each in 100% of chunks.
        let ubiq = UbiquitousBigrams::build(&codes, &offsets, 1, 75);
        assert!(ubiq.contains(1, 2));
        assert!(ubiq.contains(2, 3));
        assert!(ubiq.contains(3, 4));
        assert!(!ubiq.contains(5, 6));
    }

    #[test]
    fn excludes_rare_bigrams() {
        // 5 chunks, but bigram (9,10) only in 1 chunk.
        let codes = vec![1u16, 2, 1, 2, 1, 2, 1, 2, 9, 10];
        let offsets = vec![0u32, 2, 4, 6, 8, 10];
        let ubiq = UbiquitousBigrams::build(&codes, &offsets, 1, 50);
        assert!(ubiq.contains(1, 2)); // in 4/5 = 80% chunks
        assert!(!ubiq.contains(9, 10)); // in 1/5 = 20% chunks
    }
}
