// SPDX-License-Identifier: Apache-2.0
//! [`BigramTiers`] — variable-k allocation per code bigram based on
//! frequency. Implements the full BitFunnel idea: common bigrams get
//! fewer hash bits, rare bigrams get more.

use crate::hash::BigramKey;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Per-bigram tier table.
///
/// Tiers (by % of chunks containing the bigram, configurable):
///   * `> top_pct%`       → k=0 (skip entirely)
///   * `common..top`      → k=1
///   * `medium..common`   → k=2
///   * `≤ medium_pct`     → default_k (= 3)
///
/// Bigrams in the default tier are **not stored** to save space — we
/// just look them up and fall back to `default_k`.
#[derive(Clone, Serialize, Deserialize)]
pub struct BigramTiers {
    /// Sorted (packed_bigram_id, k) pairs.
    entries: Vec<(u32, u8)>,
    /// Default k for bigrams not in the table.
    default_k: u8,
}

impl BigramTiers {
    /// Build tier table from the full column codes.
    ///
    /// `top_pct`, `common_pct`, `medium_pct` are increasing thresholds:
    /// e.g. `(50, 25, 10)` means
    ///   - >50% chunks: k=0
    ///   - 25-50%: k=1
    ///   - 10-25%: k=2
    ///   - ≤10%: k=3 (default, not stored)
    pub fn build(
        codes: &[u16],
        codes_offsets: &[u32],
        chunk_size: usize,
        top_pct: u8,
        common_pct: u8,
        medium_pct: u8,
    ) -> Self {
        let n_rows = codes_offsets.len().saturating_sub(1);
        let n_chunks = n_rows / chunk_size;
        if n_chunks == 0 {
            return Self::empty();
        }
        let t_top = (n_chunks * top_pct as usize) / 100;
        let t_common = (n_chunks * common_pct as usize) / 100;
        let t_medium = (n_chunks * medium_pct as usize) / 100;

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

        let mut entries: Vec<(u32, u8)> = counts
            .into_iter()
            .filter_map(|(k, c)| {
                let c = c as usize;
                let tier = if c > t_top {
                    0u8
                } else if c > t_common {
                    1
                } else if c > t_medium {
                    2
                } else {
                    return None;
                };
                Some((k, tier))
            })
            .collect();
        entries.sort_unstable_by_key(|&(k, _)| k);
        Self { entries, default_k: 3 }
    }

    /// Empty tier table — every bigram uses `default_k=3`.
    pub fn empty() -> Self {
        Self { entries: Vec::new(), default_k: 3 }
    }

    /// Get `k` for a given bigram. Returns `default_k` if not in table.
    #[inline]
    pub fn k_for(&self, a: u16, b: u16) -> u32 {
        let key = BigramKey::new(a, b).0;
        match self.entries.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(i) => self.entries[i].1 as u32,
            Err(_) => self.default_k as u32,
        }
    }

    /// Number of stored tier entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no entries (all bigrams use `default_k`).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Memory size in bytes.
    pub fn byte_size(&self) -> usize {
        self.entries.len() * 5 // u32 + u8 packed
    }

    /// Count of bigrams per tier (k=0,1,2). For diagnostics.
    pub fn tier_counts(&self) -> [usize; 4] {
        let mut c = [0usize; 4];
        for &(_, k) in &self.entries {
            c[k as usize] += 1;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_assignment() {
        // 10 chunks, bigram (1,2) in all 10 (100%) → k=0
        // bigram (3,4) in 5 (50%) → k=1 (between common=25 and top=50)
        // bigram (5,6) in 2 (20%) → k=2 (between medium=10 and common=25)
        // bigram (7,8) in 1 (10%) → default k=3 (not stored)
        // Note: 50% of 10 is 5, and `t_top = 5`, so 5 chunks is NOT > 5 → tier 1
        // bigram (3,4) in 6 chunks → t_top=5, t_common=2, t_medium=1: 6>5 → k=0
        // Let me use a setup where the comparisons are clear:
        let mut codes = Vec::new();
        let mut offsets = vec![0u32];
        // 10 chunks of 1 row each.
        for c in 0..10 {
            let mut row = vec![1u16, 2]; // (1,2) bigram in every chunk
            if c < 8 {
                row.extend_from_slice(&[3, 4]); // (2,3), (3,4) bigrams; (3,4) in 80% chunks
            }
            if c < 3 {
                row.extend_from_slice(&[5, 6]); // (5,6) bigram in 30%
            }
            codes.extend(row);
            offsets.push(codes.len() as u32);
        }
        // top=50, common=25, medium=10
        // thresholds: t_top=5, t_common=2, t_medium=1
        // (1,2): 10 → >5 → k=0
        // (3,4): 8 → >5 → k=0
        // (5,6): 3 → >2 → k=1
        let tiers = BigramTiers::build(&codes, &offsets, 1, 50, 25, 10);
        assert_eq!(tiers.k_for(1, 2), 0);
        assert_eq!(tiers.k_for(3, 4), 0);
        assert_eq!(tiers.k_for(5, 6), 1);
        // unknown bigram → default
        assert_eq!(tiers.k_for(99, 100), 3);
    }

    #[test]
    fn empty_table_uses_default() {
        let t = BigramTiers::empty();
        assert_eq!(t.k_for(42, 43), 3);
    }
}
