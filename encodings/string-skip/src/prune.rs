// SPDX-License-Identifier: Apache-2.0
//! Chunk-level predicate evaluation.
//!
//! This module wires the stats, presence, and bloom variants together
//! into a single `chunk_might_match` function — the entry point used
//! by query planners to decide whether to scan or skip a chunk.

use crate::bloom::Bloom;
use crate::chunk_stats::ChunkStats;
use crate::dict::{DictIndex, TokenDict, is_safe_position, tokenize_needle};
use crate::hash::pair_hash;
use crate::pred::Pred;
use crate::presence::DictPresence;
use crate::tiers::BigramTiers;
use crate::ubiq::UbiquitousBigrams;
use serde::{Deserialize, Serialize};

/// BitFunnel-style code-bigram bloom with ubiquitous-bigram skipping.
#[derive(Clone, Serialize, Deserialize)]
pub struct HybridBloom {
    bloom: Bloom,
}

impl HybridBloom {
    /// Build the bloom from `codes` (the row's flat token stream) and
    /// `codes_offsets` (one per row + sentinel). Bigrams in `ubiq` are
    /// skipped on insert (and at probe time).
    pub fn build(
        codes: &[u16],
        codes_offsets: &[u32],
        row_lo: usize,
        row_hi: usize,
        bits_per_row: usize,
        ubiq: &UbiquitousBigrams,
    ) -> Self {
        let n_rows = row_hi - row_lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        for r in row_lo..row_hi {
            let rlo = codes_offsets[r] as usize;
            let rhi = codes_offsets[r + 1] as usize;
            for w in codes[rlo..rhi].windows(2) {
                if ubiq.contains(w[0], w[1]) {
                    continue;
                }
                let (h1, h2) = pair_hash(w[0], w[1]);
                bloom.insert(h1, h2);
            }
        }
        Self { bloom }
    }

    /// Bytes occupied by the per-chunk bloom.
    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'`.
    pub fn might_contain<D: TokenDict>(
        &self,
        dict: &D,
        index: &DictIndex,
        presence: &DictPresence,
        ubiq: &UbiquitousBigrams,
        needle: &[u8],
    ) -> bool {
        contains_via_bloom(&self.bloom, dict, index, presence, ubiq, None, needle)
    }
}

/// Full BitFunnel-style tiered bloom — variable hash count per bigram.
#[derive(Clone, Serialize, Deserialize)]
pub struct TieredBloom {
    bloom: Bloom,
}

impl TieredBloom {
    /// Build using `tiers` for per-bigram k. Bigrams with `k=0` are
    /// skipped entirely.
    pub fn build(
        codes: &[u16],
        codes_offsets: &[u32],
        row_lo: usize,
        row_hi: usize,
        bits_per_row: usize,
        tiers: &BigramTiers,
    ) -> Self {
        let n_rows = row_hi - row_lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        for r in row_lo..row_hi {
            let rlo = codes_offsets[r] as usize;
            let rhi = codes_offsets[r + 1] as usize;
            for w in codes[rlo..rhi].windows(2) {
                let k = tiers.k_for(w[0], w[1]);
                if k == 0 {
                    continue;
                }
                let (h1, h2) = pair_hash(w[0], w[1]);
                bloom.insert_k(h1, h2, k);
            }
        }
        Self { bloom }
    }

    /// Bytes occupied by the per-chunk bloom.
    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'`, tier-aware.
    pub fn might_contain<D: TokenDict>(
        &self,
        dict: &D,
        index: &DictIndex,
        presence: &DictPresence,
        tiers: &BigramTiers,
        needle: &[u8],
    ) -> bool {
        contains_via_tiered(&self.bloom, dict, index, presence, tiers, needle)
    }
}

// ---------------------------------------------------------------------
//                  Substring matching (the novel part)
// ---------------------------------------------------------------------

/// Generic substring check for the HybridBloom path.
fn contains_via_bloom<D: TokenDict>(
    bloom: &Bloom,
    dict: &D,
    index: &DictIndex,
    presence: &DictPresence,
    ubiq: &UbiquitousBigrams,
    _entry: Option<u16>,
    needle: &[u8],
) -> bool {
    if needle.is_empty() {
        return true;
    }

    // Case: needle fits inside one dict token.
    for id in 0..dict.len() {
        let bytes = dict.token_bytes(id as u16);
        if bytes.len() < needle.len() || !presence.is_set(id) {
            continue;
        }
        if memchr::memmem::find(bytes, needle).is_some() {
            return true;
        }
    }

    // cover = 0: needle starts at token boundary.
    if check_aligned_ubiq(bloom, dict, index, needle, None, ubiq) {
        return true;
    }

    // cover = 1..MAX_TOKEN_SIZE: needle starts inside a token.
    let max_cover = 16.min(needle.len());
    for cover in 1..=max_cover {
        let suffix = &needle[..cover];
        let remainder = &needle[cover..];
        if remainder.is_empty() {
            continue;
        }
        for id in 0..dict.len() {
            let bytes = dict.token_bytes(id as u16);
            if bytes.len() < cover || !presence.is_set(id) {
                continue;
            }
            if &bytes[bytes.len() - cover..] != suffix {
                continue;
            }
            if check_aligned_ubiq(bloom, dict, index, remainder, Some(id as u16), ubiq) {
                return true;
            }
        }
    }

    false
}

/// Same for tiered bloom — uses `tiers.k_for(...)` per probe.
fn contains_via_tiered<D: TokenDict>(
    bloom: &Bloom,
    dict: &D,
    index: &DictIndex,
    presence: &DictPresence,
    tiers: &BigramTiers,
    needle: &[u8],
) -> bool {
    if needle.is_empty() {
        return true;
    }
    for id in 0..dict.len() {
        let bytes = dict.token_bytes(id as u16);
        if bytes.len() < needle.len() || !presence.is_set(id) {
            continue;
        }
        if memchr::memmem::find(bytes, needle).is_some() {
            return true;
        }
    }
    if check_aligned_tiered(bloom, dict, index, needle, None, tiers) {
        return true;
    }
    let max_cover = 16.min(needle.len());
    for cover in 1..=max_cover {
        let suffix = &needle[..cover];
        let remainder = &needle[cover..];
        if remainder.is_empty() {
            continue;
        }
        for id in 0..dict.len() {
            let bytes = dict.token_bytes(id as u16);
            if bytes.len() < cover || !presence.is_set(id) {
                continue;
            }
            if &bytes[bytes.len() - cover..] != suffix {
                continue;
            }
            if check_aligned_tiered(bloom, dict, index, remainder, Some(id as u16), tiers) {
                return true;
            }
        }
    }
    false
}

/// One-alignment check for HybridBloom: bigrams checked unless ubiq.
fn check_aligned_ubiq<D: TokenDict>(
    bloom: &Bloom,
    dict: &D,
    index: &DictIndex,
    remainder: &[u8],
    entry: Option<u16>,
    ubiq: &UbiquitousBigrams,
) -> bool {
    let Some(toks) = tokenize_needle(dict, index, remainder) else {
        return false;
    };
    if toks.is_empty() {
        return entry.is_none();
    }
    let mut starts = Vec::with_capacity(toks.len());
    let mut pos = 0usize;
    for &t in &toks {
        starts.push(pos);
        pos += dict.token_bytes(t).len();
    }
    let safe: Vec<bool> = starts.iter()
        .map(|&p| is_safe_position(dict, index, remainder, p))
        .collect();

    // Entry bigram
    if let Some(e) = entry {
        if safe[0] && !ubiq.contains(e, toks[0]) {
            let (h1, h2) = pair_hash(e, toks[0]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }
    // Interior bigrams
    for (i, w) in toks.windows(2).enumerate() {
        if safe[i] && safe[i + 1] && !ubiq.contains(w[0], w[1]) {
            let (h1, h2) = pair_hash(w[0], w[1]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }
    true
}

/// One-alignment check for TieredBloom: variable k per bigram.
fn check_aligned_tiered<D: TokenDict>(
    bloom: &Bloom,
    dict: &D,
    index: &DictIndex,
    remainder: &[u8],
    entry: Option<u16>,
    tiers: &BigramTiers,
) -> bool {
    let Some(toks) = tokenize_needle(dict, index, remainder) else {
        return false;
    };
    if toks.is_empty() {
        return entry.is_none();
    }
    let mut starts = Vec::with_capacity(toks.len());
    let mut pos = 0usize;
    for &t in &toks {
        starts.push(pos);
        pos += dict.token_bytes(t).len();
    }
    let safe: Vec<bool> = starts.iter()
        .map(|&p| is_safe_position(dict, index, remainder, p))
        .collect();

    if let Some(e) = entry {
        if safe[0] {
            let k = tiers.k_for(e, toks[0]);
            if k > 0 {
                let (h1, h2) = pair_hash(e, toks[0]);
                if !bloom.contains_k(h1, h2, k) {
                    return false;
                }
            }
        }
    }
    for (i, w) in toks.windows(2).enumerate() {
        if safe[i] && safe[i + 1] {
            let k = tiers.k_for(w[0], w[1]);
            if k > 0 {
                let (h1, h2) = pair_hash(w[0], w[1]);
                if !bloom.contains_k(h1, h2, k) {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------
//                     Top-level chunk-level dispatch
// ---------------------------------------------------------------------

/// Per-chunk skip-index state. Borrow-only; lives at column read time.
pub struct ChunkSkipState<'a, D: TokenDict> {
    /// Per-chunk byte/length/null statistics.
    pub stats: &'a ChunkStats,
    /// Per-chunk dict presence bitmap.
    pub presence: &'a DictPresence,
    /// Per-chunk bloom (Hybrid variant).
    pub bloom: Option<&'a HybridBloom>,
    /// Per-chunk tiered bloom (alternative to bloom).
    pub tiered: Option<&'a TieredBloom>,
    /// Column-shared ubiquity table.
    pub ubiq: &'a UbiquitousBigrams,
    /// Column-shared tier table (only used if `tiered` is Some).
    pub tiers: &'a BigramTiers,
    /// Column-shared dict.
    pub dict: &'a D,
    /// Column-shared first-byte index.
    pub index: &'a DictIndex,
}

/// Result of running a predicate against a chunk.
#[derive(Clone, Copy, Debug, Default)]
pub struct PruneResult {
    /// True iff the chunk *might* contain a matching row.
    pub might_match: bool,
}

/// **The main entry point.** Returns `true` if this chunk might
/// contain a row matching `pred`, `false` if it provably can't.
///
/// Sound: never returns `false` when the chunk truly contains a match.
pub fn chunk_might_match<D: TokenDict>(
    pred: &Pred,
    state: &ChunkSkipState<'_, D>,
) -> bool {
    let s = state.stats;
    match pred {
        Pred::Eq(x) => {
            let xb = x.as_slice();
            s.min.as_slice() <= xb && xb <= s.max.as_slice()
                && state.presence.might_eq(state.dict, state.index, xb)
        }
        Pred::Lt(x) => s.min.as_slice() < x.as_slice(),
        Pred::Gt(x) => s.max.as_slice() > x.as_slice(),
        Pred::Le(x) => s.min.as_slice() <= x.as_slice(),
        Pred::Ge(x) => s.max.as_slice() >= x.as_slice(),
        Pred::Between(a, b) => {
            !(s.max.as_slice() < a.as_slice() || s.min.as_slice() > b.as_slice())
        }
        Pred::Prefix(p) => {
            let pb = p.as_slice();
            if s.max.as_slice() < pb {
                return false;
            }
            let mut upper = pb.to_vec();
            upper.extend(std::iter::repeat(0xffu8).take(s.max_len));
            if s.min.as_slice() > upper.as_slice() {
                return false;
            }
            state.presence.might_starts_with(state.dict, state.index, pb)
        }
        Pred::Suffix(suf) => {
            // No reliable min/max pruning for suffixes — bloom only.
            match (state.bloom, state.tiered) {
                (Some(b), _) => b.might_contain(
                    state.dict, state.index, state.presence, state.ubiq, suf),
                (_, Some(t)) => t.might_contain(
                    state.dict, state.index, state.presence, state.tiers, suf),
                _ => true, // no bloom → can't prune
            }
        }
        Pred::Contains(s_bytes) => {
            match (state.bloom, state.tiered) {
                (Some(b), _) => b.might_contain(
                    state.dict, state.index, state.presence, state.ubiq, s_bytes),
                (_, Some(t)) => t.might_contain(
                    state.dict, state.index, state.presence, state.tiers, s_bytes),
                _ => true,
            }
        }
        Pred::PrefixSuffix(p, suf) => {
            // Prefix range pruning + bloom for suffix
            let pb = p.as_slice();
            if s.max.as_slice() < pb {
                return false;
            }
            let mut upper = pb.to_vec();
            upper.extend(std::iter::repeat(0xffu8).take(s.max_len));
            if s.min.as_slice() > upper.as_slice() {
                return false;
            }
            match (state.bloom, state.tiered) {
                (Some(b), _) => b.might_contain(
                    state.dict, state.index, state.presence, state.ubiq, suf),
                (_, Some(t)) => t.might_contain(
                    state.dict, state.index, state.presence, state.tiers, suf),
                _ => true,
            }
        }
        Pred::SingleWildcard(p, suf) => {
            // Both anchored parts must be in the chunk (as substrings)
            let check = |needle: &[u8]| match (state.bloom, state.tiered) {
                (Some(b), _) => b.might_contain(
                    state.dict, state.index, state.presence, state.ubiq, needle),
                (_, Some(t)) => t.might_contain(
                    state.dict, state.index, state.presence, state.tiers, needle),
                _ => true,
            };
            if !p.is_empty() && !check(p) {
                return false;
            }
            if !suf.is_empty() && !check(suf) {
                return false;
            }
            true
        }
        Pred::MultiFragment(frags) => {
            let check = |needle: &[u8]| match (state.bloom, state.tiered) {
                (Some(b), _) => b.might_contain(
                    state.dict, state.index, state.presence, state.ubiq, needle),
                (_, Some(t)) => t.might_contain(
                    state.dict, state.index, state.presence, state.tiers, needle),
                _ => true,
            };
            for f in frags {
                if !check(f) {
                    return false;
                }
            }
            true
        }
        Pred::LengthGt(k) => s.max_len > *k,
        Pred::LengthBetween(lo, hi) => !(s.max_len < *lo || s.min_len > *hi),
        Pred::IsNull => s.null_count > 0,
        Pred::IsNotNull => s.n_rows > 0,
        Pred::InSet(xs) => xs.iter().any(|x| {
            let xb = x.as_slice();
            s.min.as_slice() <= xb && xb <= s.max.as_slice()
                && state.presence.might_eq(state.dict, state.index, xb)
        }),
    }
}
