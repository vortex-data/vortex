// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Greedy tokenisation over a forward token trie with hashed transitions.
//!
//! The reversed-DFA parsers compute the longest match at *every* byte, which
//! forces a completed transition table whose size scales with states × 256.
//! Greedy parsing only needs the longest match at token starts, so a forward
//! walk down a trie of the dictionary answers each start directly. Because
//! each trie node has exactly one incoming edge, the whole transition relation
//! fits an open-addressing table of one `u64` per edge — a few megabytes even
//! for 65,536-token dictionaries, small enough to stay cache-resident where
//! the completed reversed DFA spills to DRAM.

use hashbrown::HashMap;

use crate::bits::BitWriter;
use crate::dict::Dictionary;
use crate::store::Store;
use crate::types::BitWidth;
use crate::types::MAX_TOKEN_SIZE;
use crate::types::Token;

/// Marker for trie nodes that do not end a dictionary token.
const NO_TOKEN: u32 = u32::MAX;

/// Empty slot marker in the open-addressing transition tag array.
const EMPTY: u32 = u32::MAX;

/// 32-bit Fibonacci multiplier for transition-key hashing.
const HASH_MULTIPLIER: u32 = 0x9E37_79B9;

/// Forward token trie with transitions in a flat open-addressing hash table.
///
/// Dictionaries produced by OnPair contain all 256 one-byte tokens, so the
/// greedy walk always finds a match at every start position.
pub struct TrieLpm {
    /// Edge key `(state << 8) | byte` per occupied slot, [`EMPTY`] otherwise;
    /// linear probing, capacity is a power of two at ≤ 50% load.
    tags: Vec<u32>,
    /// Child node id for the same slot as `tags`.
    children: Vec<u32>,
    /// `capacity - 1` for masking hashed probe positions.
    mask: usize,
    /// `32 - log2(capacity)`: right shift applied to the hashed key.
    hash_shift: u32,
    /// Token ending exactly at each node, or [`NO_TOKEN`].
    token_at: Vec<u32>,
    num_nodes: usize,
    num_tokens: usize,
}

#[inline]
fn probe_start(key: u32, hash_shift: u32) -> usize {
    (key.wrapping_mul(HASH_MULTIPLIER) >> hash_shift) as usize
}

impl TrieLpm {
    /// Build the forward trie and hashed transition table for `dict`.
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        assert!(
            dict.num_tokens() <= usize::from(Token::MAX) + 1,
            "OnPair dictionary has more than 65,536 tokens"
        );

        let mut children: Vec<HashMap<u8, u32>> = vec![HashMap::new()];
        let mut token_at = vec![NO_TOKEN];
        for id in 0..dict.num_tokens() {
            let token = dict.data(id as Token);
            assert!(
                !token.is_empty() && token.len() <= MAX_TOKEN_SIZE,
                "OnPair token length must be in 1..={MAX_TOKEN_SIZE}"
            );
            let mut state = 0u32;
            for &byte in token {
                let next = children[state as usize].get(&byte).copied();
                state = match next {
                    Some(next) => next,
                    None => {
                        let next = children.len() as u32;
                        children.push(HashMap::new());
                        token_at.push(NO_TOKEN);
                        children[state as usize].insert(byte, next);
                        next
                    }
                };
            }
            // Duplicate token bytes keep the largest id, matching the greedy
            // hash matcher's tie-break.
            token_at[state as usize] = id as u32;
        }

        let num_nodes = children.len();
        assert!(num_nodes <= 1 << 20, "token trie exceeds 2^20 nodes");
        let edges = num_nodes - 1;
        let capacity = (edges * 2).next_power_of_two().max(1024);
        let mask = capacity - 1;
        let hash_shift = 32 - capacity.trailing_zeros();
        let mut tags = vec![EMPTY; capacity];
        let mut child_slots = vec![0u32; capacity];
        for (state, node_children) in children.iter().enumerate() {
            for (&byte, &child) in node_children {
                let key = ((state as u32) << 8) | u32::from(byte);
                let mut slot = probe_start(key, hash_shift);
                while tags[slot] != EMPTY {
                    slot = (slot + 1) & mask;
                }
                tags[slot] = key;
                child_slots[slot] = child;
            }
        }

        Self {
            tags,
            children: child_slots,
            mask,
            hash_shift,
            token_at,
            num_nodes,
            num_tokens: dict.num_tokens(),
        }
    }

    /// Number of dictionary patterns in the trie.
    pub fn size(&self) -> usize {
        self.num_tokens
    }

    /// Number of trie nodes, including the root.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Heap bytes used by the transition tables and per-node token ids.
    pub fn automaton_bytes(&self) -> usize {
        (self.tags.len() + self.children.len() + self.token_at.len()) * size_of::<u32>()
    }

    /// Child of `state` on `byte`, if the trie has that edge.
    #[inline]
    fn child(&self, state: u32, byte: u8) -> Option<u32> {
        let key = (state << 8) | u32::from(byte);
        let mut slot = probe_start(key, self.hash_shift);
        loop {
            // SAFETY: `slot` is masked into the table's power-of-two range.
            let tag = unsafe { *self.tags.get_unchecked(slot) };
            if tag == key {
                // SAFETY: `children` has the same capacity as `tags`.
                return Some(unsafe { *self.children.get_unchecked(slot) });
            }
            if tag == EMPTY {
                return None;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Longest dictionary token that is a prefix of `data[pos..end]`, packed
    /// as `(len << 16) | token`.
    #[inline]
    fn longest_match(&self, data: &[u8], pos: usize, end: usize) -> u32 {
        let limit = (end - pos).min(MAX_TOKEN_SIZE);
        let mut state = 0u32;
        let mut best = 0u32;
        for depth in 0..limit {
            // SAFETY: `pos + depth < end <= data.len()`.
            let byte = unsafe { *data.get_unchecked(pos + depth) };
            match self.child(state, byte) {
                Some(child) => {
                    state = child;
                    // SAFETY: `child` indexes the node arrays by construction.
                    let token = unsafe { *self.token_at.get_unchecked(child as usize) };
                    if token != NO_TOKEN {
                        best = ((depth as u32 + 1) << 16) | token;
                    }
                }
                None => break,
            }
        }
        best
    }
}

/// Encode all rows by walking the forward trie greedily, one row at a time.
pub fn parse_trie(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &TrieLpm,
    bits: BitWidth,
    store: &mut Store,
) {
    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();
    let mut writer = BitWriter::new(store);
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0);
    for row in 0..n {
        let mut pos = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        while pos < end {
            let best = matcher.longest_match(data, pos, end);
            let len = (best >> 16) as usize;
            debug_assert!(len != 0, "OnPair dictionary is missing a single-byte token");
            writer.write(best as Token);
            pos += len;
        }
        boundaries.push(writer.tokens_written() as u32);
    }
    drop(writer);
    store.boundaries = boundaries;
}

/// Rows interleaved by the trie parser to keep several independent probe
/// chains in flight.
const TRIE_LANES: usize = 12;

/// Per-lane cursor for the interleaved trie walk.
#[derive(Clone, Copy, Default)]
struct TrieLane {
    /// Start of the match currently being extended.
    pos: usize,
    end: usize,
    state: u32,
    depth: u32,
    limit: u32,
    /// Best match so far for this start: `(len << 16) | token`.
    best: u32,
}

/// Encode all rows with `TRIE_LANES` interleaved greedy walks. Matches are
/// first recorded per start position, then packed in row order.
pub fn parse_trie_interleaved(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &TrieLpm,
    bits: BitWidth,
    store: &mut Store,
) {
    let mut best = vec![0u32; offsets[n] as usize];
    let rows = crate::reversed_lpm::rows_sorted_by_length(offsets, n);
    for chunk in rows.chunks(TRIE_LANES) {
        let mut lanes = [TrieLane::default(); TRIE_LANES];
        let mut active = 0usize;
        for (lane, &row) in lanes.iter_mut().zip(chunk) {
            lane.pos = offsets[row] as usize;
            lane.end = offsets[row + 1] as usize;
            lane.limit = (lane.end - lane.pos).min(MAX_TOKEN_SIZE) as u32;
            if lane.pos < lane.end {
                active += 1;
            }
        }
        while active > 0 {
            for lane in lanes.iter_mut() {
                if lane.pos >= lane.end {
                    continue;
                }
                let advanced = if lane.depth < lane.limit {
                    // SAFETY: `pos + depth < end <= data.len()`.
                    let byte = unsafe { *data.get_unchecked(lane.pos + lane.depth as usize) };
                    match matcher.child(lane.state, byte) {
                        Some(child) => {
                            lane.state = child;
                            lane.depth += 1;
                            // SAFETY: `child` indexes the node arrays.
                            let token = unsafe { *matcher.token_at.get_unchecked(child as usize) };
                            if token != NO_TOKEN {
                                lane.best = (lane.depth << 16) | token;
                            }
                            true
                        }
                        None => false,
                    }
                } else {
                    false
                };
                if !advanced {
                    let len = (lane.best >> 16) as usize;
                    debug_assert!(len != 0, "OnPair dictionary is missing a single-byte token");
                    // SAFETY: `pos` is a valid in-row position.
                    unsafe { *best.get_unchecked_mut(lane.pos) = lane.best };
                    lane.pos += len;
                    lane.state = 0;
                    lane.depth = 0;
                    lane.best = 0;
                    if lane.pos >= lane.end {
                        active -= 1;
                    } else {
                        lane.limit = (lane.end - lane.pos).min(MAX_TOKEN_SIZE) as u32;
                    }
                }
            }
        }
    }

    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();
    let mut writer = BitWriter::new(store);
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0);
    for row in 0..n {
        let mut pos = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        while pos < end {
            let packed = best[pos];
            let len = (packed >> 16) as usize;
            debug_assert!(len != 0, "OnPair dictionary is missing a single-byte token");
            writer.write(packed as Token);
            pos += len;
        }
        boundaries.push(writer.tokens_written() as u32);
    }
    drop(writer);
    store.boundaries = boundaries;
}

/// Record the greedy match for every token-start position of the given rows,
/// walking all sixteen lanes branchlessly with masked AVX-512 gathers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn walk_trie_avx512_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    matcher: &TrieLpm,
    best_out: &mut [u32],
) {
    use std::arch::x86_64::*;

    let mut pos_array = [0i32; 16];
    let mut end_array = [0i32; 16];
    let mut lane_mask = 0u16;
    for (lane, &row) in rows.iter().enumerate() {
        pos_array[lane] = offsets[row] as i32;
        end_array[lane] = offsets[row + 1] as i32;
        lane_mask |= 1 << lane;
    }

    // SAFETY: cursor loads read complete local arrays; every gather/scatter is
    // masked to lanes whose indices the caller has bounds-checked.
    unsafe {
        let mut pos = _mm512_loadu_si512(pos_array.as_ptr().cast());
        let end = _mm512_loadu_si512(end_array.as_ptr().cast());
        let mut state = _mm512_setzero_si512();
        let mut depth = _mm512_setzero_si512();
        let mut best = _mm512_setzero_si512();
        let mut limit = _mm512_min_epi32(
            _mm512_sub_epi32(end, pos),
            _mm512_set1_epi32(MAX_TOKEN_SIZE as i32),
        );

        let byte_mask = _mm512_set1_epi32(0xff);
        let one = _mm512_set1_epi32(1);
        let empty = _mm512_set1_epi32(EMPTY as i32);
        let multiplier = _mm512_set1_epi32(HASH_MULTIPLIER as i32);
        let slot_mask = _mm512_set1_epi32(matcher.mask as i32);
        let hash_shift = _mm_cvtsi32_si128(matcher.hash_shift as i32);

        loop {
            let active = _mm512_mask_cmplt_epi32_mask(lane_mask, pos, end);
            if active == 0 {
                break;
            }
            let walking = _mm512_mask_cmplt_epi32_mask(active, depth, limit);

            // One probe step for every walking lane.
            let byte_index = _mm512_add_epi32(pos, depth);
            let words = _mm512_mask_i32gather_epi32::<1>(
                _mm512_setzero_si512(),
                walking,
                byte_index,
                data.as_ptr().cast(),
            );
            let key = _mm512_or_si512(
                _mm512_slli_epi32::<8>(state),
                _mm512_and_si512(words, byte_mask),
            );
            let mut slot = _mm512_srl_epi32(_mm512_mullo_epi32(key, multiplier), hash_shift);
            let mut probing = walking;
            let mut hit = 0u16;
            loop {
                let tags = _mm512_mask_i32gather_epi32::<4>(
                    empty,
                    probing,
                    slot,
                    matcher.tags.as_ptr().cast(),
                );
                hit |= _mm512_mask_cmpeq_epi32_mask(probing, tags, key);
                let missing = _mm512_mask_cmpeq_epi32_mask(probing, tags, empty);
                probing &= !(hit | missing);
                if probing == 0 {
                    break;
                }
                slot = _mm512_mask_and_epi32(slot, probing, _mm512_add_epi32(slot, one), slot_mask);
            }

            let child = _mm512_mask_i32gather_epi32::<4>(
                _mm512_setzero_si512(),
                hit,
                slot,
                matcher.children.as_ptr().cast(),
            );
            let token = _mm512_mask_i32gather_epi32::<4>(
                empty,
                hit,
                child,
                matcher.token_at.as_ptr().cast(),
            );
            state = _mm512_mask_mov_epi32(state, hit, child);
            depth = _mm512_mask_add_epi32(depth, hit, depth, one);
            let accept = _mm512_mask_cmpneq_epi32_mask(hit, token, empty);
            best = _mm512_mask_mov_epi32(
                best,
                accept,
                _mm512_or_si512(_mm512_slli_epi32::<16>(depth), token),
            );

            // Lanes that missed or exhausted their window emit and restart.
            let fail = active & !hit;
            _mm512_mask_i32scatter_epi32::<4>(best_out.as_mut_ptr().cast(), fail, pos, best);
            pos = _mm512_mask_add_epi32(pos, fail, pos, _mm512_srli_epi32::<16>(best));
            state = _mm512_mask_mov_epi32(state, fail, _mm512_setzero_si512());
            depth = _mm512_mask_mov_epi32(depth, fail, _mm512_setzero_si512());
            best = _mm512_mask_mov_epi32(best, fail, _mm512_setzero_si512());
            limit = _mm512_mask_min_epi32(
                limit,
                fail,
                _mm512_sub_epi32(end, pos),
                _mm512_set1_epi32(MAX_TOKEN_SIZE as i32),
            );
        }
    }
}

/// Encode all rows with sixteen branchless AVX-512 trie-walk lanes. Falls back
/// to [`parse_trie`] when AVX-512 is unavailable.
pub fn parse_trie_avx512(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &TrieLpm,
    bits: BitWidth,
    store: &mut Store,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && data.len() <= i32::MAX as usize {
        use crate::reversed_lpm::gather_reads_past_data;
        use crate::reversed_lpm::rows_sorted_by_length;

        let rows = rows_sorted_by_length(offsets, n);
        let mut best = vec![0u32; offsets[n] as usize];
        for chunk in rows.chunks(16) {
            if gather_reads_past_data(chunk, offsets, data.len()) {
                for &row in chunk {
                    let mut pos = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    while pos < end {
                        let packed = matcher.longest_match(data, pos, end);
                        best[pos] = packed;
                        pos += (packed >> 16) as usize;
                    }
                }
            } else {
                // SAFETY: runtime feature detection and gather bounds are
                // checked above.
                unsafe { walk_trie_avx512_group(data, offsets, chunk, matcher, &mut best) };
            }
        }
        pack_trie_matches(offsets, n, &best, bits, store);
        return;
    }

    parse_trie(data, offsets, n, matcher, bits, store);
}

/// Pack per-start `(len << 16) | token` matches into the store in row order.
fn pack_trie_matches(offsets: &[u32], n: usize, best: &[u32], bits: BitWidth, store: &mut Store) {
    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();
    let mut writer = BitWriter::new(store);
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0);
    for row in 0..n {
        let mut pos = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        while pos < end {
            let packed = best[pos];
            let len = (packed >> 16) as usize;
            debug_assert!(len != 0, "OnPair dictionary is missing a single-byte token");
            writer.write(packed as Token);
            pos += len;
        }
        boundaries.push(writer.tokens_written() as u32);
    }
    drop(writer);
    store.boundaries = boundaries;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FixedThreshold;
    use crate::config::ThresholdSpec;
    use crate::config::TrainingConfig;
    use crate::parser::parse;
    use crate::test_corpus::binary_strings;
    use crate::test_corpus::make_raw;
    use crate::test_corpus::random_ascii_strings;
    use crate::test_corpus::user_strings;
    use crate::trainer::train;

    fn assert_matches_greedy(rows: &[Vec<u8>], bits: BitWidth) {
        let raw = make_raw(rows);
        let config = TrainingConfig {
            bits,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(7),
        };
        let trained = train(&raw.data, &raw.offsets, raw.n, &config);
        let mut greedy = Store::default();
        parse(
            &raw.data,
            &raw.offsets,
            raw.n,
            &trained.lpm,
            bits,
            &mut greedy,
        );

        let matcher = TrieLpm::from_dictionary(&trained.dict);
        let mut scalar = Store::default();
        parse_trie(&raw.data, &raw.offsets, raw.n, &matcher, bits, &mut scalar);
        assert_eq!(scalar.packed, greedy.packed);
        assert_eq!(scalar.boundaries, greedy.boundaries);

        let mut interleaved = Store::default();
        parse_trie_interleaved(
            &raw.data,
            &raw.offsets,
            raw.n,
            &matcher,
            bits,
            &mut interleaved,
        );
        assert_eq!(interleaved.packed, greedy.packed);
        assert_eq!(interleaved.boundaries, greedy.boundaries);

        let mut simd = Store::default();
        parse_trie_avx512(&raw.data, &raw.offsets, raw.n, &matcher, bits, &mut simd);
        assert_eq!(simd.packed, greedy.packed);
        assert_eq!(simd.boundaries, greedy.boundaries);
    }

    #[test]
    fn trie_variants_match_greedy_tokens() {
        for bits in [12u8, 16u8] {
            let user: Vec<Vec<u8>> = user_strings(200)
                .into_iter()
                .map(String::into_bytes)
                .collect();
            assert_matches_greedy(&user, bits);
            assert_matches_greedy(&random_ascii_strings(512, 80, 97), bits);
            assert_matches_greedy(&binary_strings(256, 80, 31), bits);
        }
    }

    #[test]
    fn empty_rows_and_boundaries_are_preserved() {
        let rows = vec![
            b"".to_vec(),
            b"abcabcabc".to_vec(),
            b"".to_vec(),
            b"abc".to_vec(),
        ];
        assert_matches_greedy(&rows, 12);
    }
}
