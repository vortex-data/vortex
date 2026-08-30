// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Length-only reversed-automaton tokenisation.
//!
//! The reversed Aho--Corasick automaton in [`crate::reversed_lpm`] reports
//! `(token, length)` at every byte, which prevents state merging for large
//! dictionaries: almost every state's output names a distinct token id. This
//! variant drops token identity from the automaton entirely — the backward
//! pass computes only the *length* of the longest dictionary token starting at
//! each byte. With outputs drawn from `1..=16` instead of a 64Ki token space,
//! Mealy minimization collapses states that differ only in which token they
//! would report, shrinking the transition table dramatically. The forward pass
//! then resolves each emitted token with one exact-length hash probe over the
//! matched bytes.

use std::collections::VecDeque;

use hashbrown::HashMap;

use crate::bits::BitWriter;
use crate::dict::Dictionary;
use crate::reversed_lpm::AVX512_GROUPS;
use crate::reversed_lpm::advise_huge_pages;
use crate::reversed_lpm::gather_reads_past_data;
use crate::reversed_lpm::minimize_dense;
use crate::reversed_lpm::pack_dense16;
use crate::reversed_lpm::read_packed_transition;
use crate::reversed_lpm::rows_sorted_by_length;
use crate::store::Store;
use crate::types::BitWidth;
use crate::types::MAX_TOKEN_SIZE;
use crate::types::Token;

/// Reversed longest-match automaton that reports match lengths only.
///
/// Token ids are recovered afterwards by hashing the matched bytes, so the
/// automaton state space depends only on the token byte set's structure, not
/// on token identity. Dictionaries produced by OnPair contain all 256
/// single-byte tokens, making matching total at every byte position.
pub struct ReversedLengthMatcher {
    transitions: LengthTransitions,
    num_tokens: usize,
    num_states: usize,
    trie_states: usize,
    /// Tokens of length `1..=8`, keyed by their bytes packed little-endian.
    short: [HashMap<u64, Token>; 8],
    /// Tokens of length `9..=16`, keyed by both packed halves.
    long: [HashMap<(u64, u64), Token>; 8],
}

enum LengthTransitions {
    /// One `u32` per `(state, byte)`: `((len - 1) << 16) | next_state`.
    Fused(Vec<u32>),
    /// 20-bit packed next states plus one match length per state.
    Packed { table: Vec<u8>, lens: Vec<u8> },
}

#[derive(Default)]
struct LengthNode {
    children: HashMap<u8, u32>,
    fail: u32,
    /// Length of the longest token ending at this trie state (0 for none);
    /// failure-link outputs are propagated during the BFS.
    best_len: u32,
}

/// Pack `len` bytes starting at `pos` into `(lo, hi)` little-endian halves,
/// zero-padded past `len`.
#[inline]
fn load_key(data: &[u8], pos: usize, len: usize) -> (u64, u64) {
    debug_assert!((1..=MAX_TOKEN_SIZE).contains(&len));
    debug_assert!(pos + len <= data.len());
    let mut buf = [0u8; 16];
    if pos + 16 <= data.len() {
        buf.copy_from_slice(&data[pos..pos + 16]);
    } else {
        buf[..len].copy_from_slice(&data[pos..pos + len]);
    }
    let lo = u64::from_le_bytes(buf[..8].try_into().expect("eight bytes"));
    let hi = u64::from_le_bytes(buf[8..].try_into().expect("eight bytes"));
    if len >= 16 {
        (lo, hi)
    } else if len >= 8 {
        (lo, hi & ((1u64 << ((len - 8) * 8)) - 1))
    } else {
        (lo & ((1u64 << (len * 8)) - 1), 0)
    }
}

/// Mealy-minimize a completed length DFA: states merge whenever every byte
/// leads to targets with equal match lengths and equal classes. Returns the
/// class of each state; the root keeps class 0.
fn minimize_mealy_lengths(table: &[u32], lens: &[u32]) -> (Vec<u32>, usize) {
    let states = table.len() / 256;
    let mut classes = vec![0u32; states];

    loop {
        let mut partitions: HashMap<u64, Vec<usize>> = HashMap::with_capacity(states);
        let mut next_classes = vec![0u32; states];
        let mut num_classes = 0u32;

        for state in 0..states {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for byte in 0..256 {
                let target = table[state * 256 + byte] as usize;
                let value = (u64::from(lens[target]) << 32) | u64::from(classes[target]);
                hash ^= value;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
            let equivalent = partitions.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&candidate| {
                    (0..256).all(|byte| {
                        let left = table[state * 256 + byte] as usize;
                        let right = table[candidate * 256 + byte] as usize;
                        lens[left] == lens[right] && classes[left] == classes[right]
                    })
                })
            });
            let next_class = match equivalent {
                Some(candidate) => next_classes[candidate],
                None => {
                    let class = num_classes;
                    num_classes += 1;
                    partitions.entry(hash).or_default().push(state);
                    class
                }
            };
            next_classes[state] = next_class;
        }

        if next_classes == classes {
            break;
        }
        classes = next_classes;
    }

    let num_classes = classes.iter().copied().max().unwrap_or(0) as usize + 1;
    (classes, num_classes)
}

impl ReversedLengthMatcher {
    /// Build the length-only reversed automaton and token-resolution maps for
    /// `dict`.
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        assert!(
            dict.num_tokens() <= usize::from(Token::MAX) + 1,
            "OnPair dictionary has more than 65,536 tokens"
        );

        let mut short: [HashMap<u64, Token>; 8] = Default::default();
        let mut long: [HashMap<(u64, u64), Token>; 8] = Default::default();
        let mut nodes = vec![LengthNode::default()];
        for id in 0..dict.num_tokens() {
            let token = dict.data(id as Token);
            assert!(
                !token.is_empty() && token.len() <= MAX_TOKEN_SIZE,
                "OnPair token length must be in 1..={MAX_TOKEN_SIZE}"
            );
            let (lo, hi) = load_key(token, 0, token.len());
            if token.len() <= 8 {
                short[token.len() - 1].insert(lo, id as Token);
            } else {
                long[token.len() - 9].insert((lo, hi), id as Token);
            }

            let mut state = 0u32;
            for &byte in token.iter().rev() {
                let next = nodes[state as usize].children.get(&byte).copied();
                state = match next {
                    Some(next) => next,
                    None => {
                        let next = nodes.len() as u32;
                        nodes.push(LengthNode::default());
                        nodes[state as usize].children.insert(byte, next);
                        next
                    }
                };
            }
            nodes[state as usize].best_len = nodes[state as usize].best_len.max(token.len() as u32);
        }

        // Breadth-first failure links with match lengths propagated so each
        // state directly stores the longest accepting suffix length.
        let mut queue = VecDeque::new();
        let mut bfs_order = Vec::with_capacity(nodes.len());
        bfs_order.push(0u32);
        let root_children: Vec<u32> = nodes[0].children.values().copied().collect();
        for child in root_children {
            nodes[child as usize].fail = 0;
            queue.push_back(child);
        }
        while let Some(state) = queue.pop_front() {
            bfs_order.push(state);
            let inherited = nodes[nodes[state as usize].fail as usize].best_len;
            nodes[state as usize].best_len = nodes[state as usize].best_len.max(inherited);

            let children: Vec<(u8, u32)> = nodes[state as usize]
                .children
                .iter()
                .map(|(&label, &target)| (label, target))
                .collect();
            for (label, child) in children {
                let mut fallback = nodes[state as usize].fail;
                let fail = loop {
                    if let Some(&target) = nodes[fallback as usize].children.get(&label) {
                        break target;
                    }
                    if fallback == 0 {
                        break 0;
                    }
                    fallback = nodes[fallback as usize].fail;
                };
                nodes[child as usize].fail = fail;
                queue.push_back(child);
            }
        }

        let lens: Vec<u32> = nodes.iter().map(|node| node.best_len).collect();
        let mut table = vec![0u32; nodes.len() * 256];
        for &state in &bfs_order {
            for byte in 0u16..=255 {
                let target = nodes[state as usize]
                    .children
                    .get(&(byte as u8))
                    .copied()
                    .unwrap_or_else(|| {
                        if state == 0 {
                            0
                        } else {
                            table[nodes[state as usize].fail as usize * 256 + byte as usize]
                        }
                    });
                table[state as usize * 256 + byte as usize] = target;
            }
        }

        let (classes, num_classes) = minimize_mealy_lengths(&table, &lens);
        let transitions = if num_classes <= usize::from(u16::MAX) + 1 {
            let mut representatives = vec![usize::MAX; num_classes];
            for (state, &class) in classes.iter().enumerate() {
                if representatives[class as usize] == usize::MAX {
                    representatives[class as usize] = state;
                }
            }
            let mut fused = vec![0u32; num_classes * 256];
            for (class, &state) in representatives.iter().enumerate() {
                for byte in 0..256 {
                    let target = table[state * 256 + byte] as usize;
                    let len = lens[target];
                    debug_assert!((1..=MAX_TOKEN_SIZE as u32).contains(&len));
                    fused[class * 256 + byte] = classes[target] | ((len - 1) << 16);
                }
            }
            // SAFETY: a `u32` slice reinterprets as bytes for page advice only.
            unsafe {
                advise_huge_pages(std::slice::from_raw_parts_mut(
                    fused.as_mut_ptr().cast::<u8>(),
                    fused.len() * size_of::<u32>(),
                ));
            }
            LengthTransitions::Fused(fused)
        } else {
            // Too many classes for 16-bit fused states: fall back to the
            // Moore-minimized packed form with per-state lengths.
            let (minimized, outputs) = minimize_dense(&table, &lens);
            LengthTransitions::Packed {
                table: pack_dense16(&minimized),
                lens: outputs.iter().map(|&len| len as u8).collect(),
            }
        };
        let num_states = match &transitions {
            LengthTransitions::Fused(fused) => fused.len() / 256,
            LengthTransitions::Packed { lens, .. } => lens.len(),
        };

        Self {
            transitions,
            num_tokens: dict.num_tokens(),
            num_states,
            trie_states: nodes.len(),
            short,
            long,
        }
    }

    /// Number of dictionary patterns in the automaton.
    pub fn size(&self) -> usize {
        self.num_tokens
    }

    /// Number of runtime automaton states after minimization.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Number of states before equivalent continuations were merged.
    pub fn trie_states(&self) -> usize {
        self.trie_states
    }

    /// Whether transitions use the one-lookup fused `u32` representation.
    pub fn uses_fused_transitions(&self) -> bool {
        matches!(self.transitions, LengthTransitions::Fused(_))
    }

    /// Heap bytes used by the transition table and per-state lengths.
    pub fn automaton_bytes(&self) -> usize {
        match &self.transitions {
            LengthTransitions::Fused(fused) => fused.len() * size_of::<u32>(),
            LengthTransitions::Packed { table, lens } => table.len() + lens.len(),
        }
    }

    /// Resolve the dictionary token for the `len` bytes at `data[pos..]`.
    #[inline]
    fn resolve(&self, data: &[u8], pos: usize, len: usize) -> Token {
        let (lo, hi) = load_key(data, pos, len);
        let token = if len <= 8 {
            self.short[len - 1].get(&lo)
        } else {
            self.long[len - 9].get(&(lo, hi))
        };
        *token.expect("length automaton reported a match absent from the dictionary")
    }

    /// Write the match length for every byte of `data[start..end]` into
    /// `best[start..end]` (fused form: `len - 1`; packed form: the state,
    /// whose length lives in `lens`).
    fn fill_scalar(&self, data: &[u8], start: usize, end: usize, best: &mut [u32]) {
        match &self.transitions {
            LengthTransitions::Fused(table) => {
                let mut state = 0usize;
                for pos in (start..end).rev() {
                    // SAFETY: `pos` is in bounds by construction; every fused
                    // entry's low 16 bits name a class in the same table.
                    unsafe {
                        let byte = *data.get_unchecked(pos) as usize;
                        let entry = *table.get_unchecked((state << 8) | byte);
                        state = (entry & u32::from(u16::MAX)) as usize;
                        *best.get_unchecked_mut(pos) = entry >> 16;
                    }
                }
            }
            LengthTransitions::Packed { table, .. } => {
                let mut state = 0usize;
                for pos in (start..end).rev() {
                    let byte = data[pos] as usize;
                    state = read_packed_transition(table, (state << 8) | byte) as usize;
                    best[pos] = state as u32;
                }
            }
        }
    }

    #[inline]
    fn match_len(&self, best: u32) -> usize {
        match &self.transitions {
            LengthTransitions::Fused(_) => (best & 0xf) as usize + 1,
            LengthTransitions::Packed { lens, .. } => lens[best as usize] as usize,
        }
    }
}

/// Encode all rows with the length-only reversed automaton: a backward pass
/// records the longest match length per byte, then the forward greedy pass
/// resolves each emitted token with one exact-length hash probe.
pub fn parse_reversed_lengths(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &ReversedLengthMatcher,
    bits: BitWidth,
    store: &mut Store,
) {
    let mut best = vec![0u32; offsets[n] as usize];
    for row in 0..n {
        matcher.fill_scalar(data, offsets[row] as usize, offsets[row + 1] as usize, &mut best);
    }
    pack_length_matches(data, offsets, n, &best, matcher, bits, store);
}

/// Encode rows using length-grouped AVX-512 DFA lanes over the length-only
/// automaton. Falls back to [`parse_reversed_lengths`] when AVX-512 is
/// unavailable.
pub fn parse_reversed_lengths_avx512(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &ReversedLengthMatcher,
    bits: BitWidth,
    store: &mut Store,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && data.len() <= i32::MAX as usize {
        use crate::reversed_lpm::fill_avx512_group;
        use crate::reversed_lpm::fill_avx512_packed_group;

        let rows = rows_sorted_by_length(offsets, n);
        let mut best = vec![0u32; offsets[n] as usize];
        match &matcher.transitions {
            LengthTransitions::Fused(table) if table.len() <= i32::MAX as usize => {
                for chunk in rows.chunks(AVX512_GROUPS * 16) {
                    if gather_reads_past_data(chunk, offsets, data.len()) {
                        for &row in chunk {
                            matcher.fill_scalar(
                                data,
                                offsets[row] as usize,
                                offsets[row + 1] as usize,
                                &mut best,
                            );
                        }
                    } else {
                        // SAFETY: runtime feature detection and gather bounds
                        // are checked above.
                        unsafe { fill_avx512_group(data, offsets, chunk, table, &mut best) };
                    }
                }
            }
            LengthTransitions::Packed { table, .. } if table.len() <= i32::MAX as usize => {
                for chunk in rows.chunks(AVX512_GROUPS * 16) {
                    if gather_reads_past_data(chunk, offsets, data.len()) {
                        for &row in chunk {
                            matcher.fill_scalar(
                                data,
                                offsets[row] as usize,
                                offsets[row + 1] as usize,
                                &mut best,
                            );
                        }
                    } else {
                        // SAFETY: runtime feature detection and gather bounds
                        // are checked above.
                        unsafe { fill_avx512_packed_group(data, offsets, chunk, table, &mut best) };
                    }
                }
            }
            _ => {
                parse_reversed_lengths(data, offsets, n, matcher, bits, store);
                return;
            }
        }
        pack_length_matches(data, offsets, n, &best, matcher, bits, store);
        return;
    }

    parse_reversed_lengths(data, offsets, n, matcher, bits, store);
}

fn pack_length_matches(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    best: &[u32],
    matcher: &ReversedLengthMatcher,
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
            let len = matcher.match_len(best[pos]);
            debug_assert!(len != 0, "OnPair dictionary is missing a single-byte token");
            writer.write(matcher.resolve(data, pos, len));
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
        parse(&raw.data, &raw.offsets, raw.n, &trained.lpm, bits, &mut greedy);

        let matcher = ReversedLengthMatcher::from_dictionary(&trained.dict);
        let mut scalar = Store::default();
        parse_reversed_lengths(&raw.data, &raw.offsets, raw.n, &matcher, bits, &mut scalar);
        assert_eq!(scalar.packed, greedy.packed);
        assert_eq!(scalar.boundaries, greedy.boundaries);

        let mut simd = Store::default();
        parse_reversed_lengths_avx512(&raw.data, &raw.offsets, raw.n, &matcher, bits, &mut simd);
        assert_eq!(simd.packed, greedy.packed);
        assert_eq!(simd.boundaries, greedy.boundaries);
    }

    #[test]
    fn length_variants_match_greedy_tokens() {
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
