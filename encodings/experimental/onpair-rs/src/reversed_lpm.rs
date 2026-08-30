// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Two-pass greedy tokenisation using a reversed Aho--Corasick automaton.
//!
//! The first pass scans one row backwards. At each byte, the automaton state's
//! precomputed output is exactly the longest dictionary token starting at that
//! byte in the original row. The second, forward pass follows those matches
//! with the same greedy rule as [`crate::LongestPrefixMatcher`].

use std::collections::VecDeque;

use hashbrown::HashMap;

use crate::bits::BitWriter;
use crate::dict::Dictionary;
use crate::store::Store;
use crate::types::BitWidth;
use crate::types::MAX_TOKEN_SIZE;
use crate::types::Token;

/// A dictionary matcher built from every token with its bytes reversed.
///
/// Token IDs are dictionary indices. Dictionaries produced by OnPair contain
/// all 256 one-byte tokens, making matching total at every byte position.
pub struct ReversedAhoCorasickMatcher {
    transitions: Transitions,
    /// `(token << 8) | length`; failure-link outputs are already propagated.
    best_output: Vec<u32>,
    num_tokens: usize,
    num_states: usize,
    trie_states: usize,
}

enum Transitions {
    /// Dict-12 DFA entry: `(compact_output << 16) | next_state`.
    Dense12(Vec<u32>),
    /// Larger-dictionary next states packed as little-endian 20-bit integers.
    Dense16(Vec<u8>),
    /// Completed DFA for tries or dictionaries that do not fit the fused form.
    Dense(Vec<u32>),
    /// Memory-bounded fallback for unusually large (notably dict-16) tries.
    Sparse {
        offsets: Vec<u32>,
        labels: Vec<u8>,
        targets: Vec<u32>,
        fail: Vec<u32>,
    },
}

#[derive(Default)]
struct BuildNode {
    children: HashMap<u8, u32>,
    fail: u32,
    direct_output: u32,
    best_output: u32,
}

/// A dict-12 trie has at most 61,697 states, so its worst-case full byte DFA
/// is about 60.3 MiB. Dict-16 can exceed 1 GiB and uses the sparse fallback.
const MAX_DENSE_TRANSITION_BYTES: usize = 512 * 1024 * 1024;

#[inline]
fn better_output(left: u32, right: u32) -> u32 {
    let left_len = left & 0xff;
    let right_len = right & 0xff;
    if right_len > left_len || (right_len == left_len && right > left) {
        right
    } else {
        left
    }
}

#[inline]
fn read_packed_transition(table: &[u8], index: usize) -> u32 {
    let byte_offset = index * 2 + index / 2;
    let shift = (index & 1) * 4;
    // SAFETY: Dense16 appends four padding bytes, so the unaligned four-byte
    // read beginning at any 20-bit entry remains inside the allocation.
    unsafe {
        table
            .as_ptr()
            .add(byte_offset)
            .cast::<u32>()
            .read_unaligned()
            >> shift
            & 0x000f_ffff
    }
}

#[cfg(target_os = "linux")]
fn advise_huge_pages(bytes: &mut [u8]) {
    use std::ffi::c_void;

    if std::env::var_os("ONPAIR_DISABLE_HUGEPAGES").is_some() {
        return;
    }

    const HUGE_PAGE: usize = 2 * 1024 * 1024;
    const MADV_HUGEPAGE: i32 = 14;
    const MADV_COLLAPSE: i32 = 25;

    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }

    let start = bytes.as_mut_ptr() as usize;
    let aligned_start = start.div_ceil(HUGE_PAGE) * HUGE_PAGE;
    let aligned_end = (start + bytes.len()) / HUGE_PAGE * HUGE_PAGE;
    if aligned_start >= aligned_end {
        return;
    }
    // SAFETY: the advised range is page-aligned and lies wholly inside the
    // allocation. Both advice values affect backing/page promotion only.
    unsafe {
        let pointer = aligned_start as *mut c_void;
        let length = aligned_end - aligned_start;
        let _ = madvise(pointer, length, MADV_HUGEPAGE);
        let _ = madvise(pointer, length, MADV_COLLAPSE);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge_pages(_bytes: &mut [u8]) {}

#[inline]
fn find_label(labels: &[u8], byte: u8) -> Option<usize> {
    match labels {
        [] => None,
        [label] => (*label == byte).then_some(0),
        [first, second] => {
            if *first == byte {
                Some(0)
            } else {
                (*second == byte).then_some(1)
            }
        }
        labels => labels.binary_search(&byte).ok(),
    }
}

/// Minimize the completed matcher as a Mealy machine. The observable output
/// belongs to each transition, so states with different trie paths can share a
/// row whenever all future byte streams produce identical matches.
fn minimize_and_fuse_dense12(table: Vec<u32>, best_output: &[u32]) -> Vec<u32> {
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
                let value = (u64::from(best_output[target]) << 32) | u64::from(classes[target]);
                hash ^= value;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
            let equivalent = partitions.get(&hash).and_then(|candidates| {
                candidates.iter().copied().find(|&candidate| {
                    equivalent_signatures(state, candidate, &table, best_output, &classes)
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

    let minimized_states = classes.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut representatives = vec![usize::MAX; minimized_states];
    for (state, &class) in classes.iter().enumerate() {
        representatives[class as usize] = representatives[class as usize].min(state);
    }

    let mut minimized = vec![0u32; minimized_states * 256];
    for (class, &state) in representatives.iter().enumerate() {
        for byte in 0..256 {
            let target = table[state * 256 + byte] as usize;
            let output = best_output[target];
            let token = output >> 8;
            let len = output & 0xff;
            debug_assert!(token < 1 << 12);
            debug_assert!((1..=MAX_TOKEN_SIZE as u32).contains(&len));
            let compact_output = (token << 4) | (len - 1);
            minimized[class * 256 + byte] = classes[target] | (compact_output << 16);
        }
    }
    minimized
}

fn equivalent_signatures(
    left: usize,
    right: usize,
    table: &[u32],
    best_output: &[u32],
    classes: &[u32],
) -> bool {
    (0..256).all(|byte| {
        let left_target = table[left * 256 + byte] as usize;
        let right_target = table[right * 256 + byte] as usize;
        best_output[left_target] == best_output[right_target]
            && classes[left_target] == classes[right_target]
    })
}

impl ReversedAhoCorasickMatcher {
    /// Build the reversed automaton for `dict`.
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        assert!(
            dict.num_tokens() <= usize::from(Token::MAX) + 1,
            "OnPair dictionary has more than 65,536 tokens"
        );

        let mut nodes = vec![BuildNode::default()];
        for id in 0..dict.num_tokens() {
            let token = dict.data(id as Token);
            assert!(
                !token.is_empty() && token.len() <= MAX_TOKEN_SIZE,
                "OnPair token length must be in 1..={MAX_TOKEN_SIZE}"
            );
            let mut state = 0u32;
            for &byte in token.iter().rev() {
                let next = nodes[state as usize].children.get(&byte).copied();
                state = match next {
                    Some(next) => next,
                    None => {
                        let next = nodes.len() as u32;
                        nodes.push(BuildNode::default());
                        nodes[state as usize].children.insert(byte, next);
                        next
                    }
                };
            }
            let output = ((id as u32) << 8) | token.len() as u32;
            nodes[state as usize].direct_output =
                better_output(nodes[state as usize].direct_output, output);
        }

        // Build failure links breadth-first. Propagating the best output now
        // avoids enumerating every accepting suffix during parsing.
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
            let inherited = nodes[nodes[state as usize].fail as usize].best_output;
            nodes[state as usize].best_output =
                better_output(nodes[state as usize].direct_output, inherited);

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

        let best_output: Vec<u32> = nodes.iter().map(|node| node.best_output).collect();
        let dense_bytes = nodes.len().saturating_mul(256 * size_of::<u32>());
        let transitions = if dense_bytes <= MAX_DENSE_TRANSITION_BYTES {
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
            let num_states;
            if dict.num_tokens() <= 1 << 12 && nodes.len() <= usize::from(u16::MAX) + 1 {
                table = minimize_and_fuse_dense12(table, &best_output);
                num_states = table.len() / 256;
                (Transitions::Dense12(table), num_states)
            } else if nodes.len() < 1 << 20 {
                num_states = nodes.len();
                debug_assert_eq!(table.len() & 1, 0);
                let mut packed = vec![0u8; table.len() * 5 / 2 + 8];
                for (pair, targets) in table.chunks_exact(2).enumerate() {
                    let word = u64::from(targets[0]) | (u64::from(targets[1]) << 20);
                    // SAFETY: eight bytes fit at every five-byte pair offset
                    // due to the allocation padding. The following store
                    // overwrites only the previous store's zero upper bytes.
                    unsafe {
                        let pointer = packed.as_mut_ptr().add(pair * 5).cast::<u64>();
                        pointer.write_unaligned(word);
                    }
                }
                advise_huge_pages(&mut packed);
                (Transitions::Dense16(packed), num_states)
            } else {
                num_states = nodes.len();
                (Transitions::Dense(table), num_states)
            }
        } else {
            let mut offsets = Vec::with_capacity(nodes.len() + 1);
            let mut labels = Vec::with_capacity(nodes.len().saturating_sub(1));
            let mut targets = Vec::with_capacity(nodes.len().saturating_sub(1));
            let mut fail = Vec::with_capacity(nodes.len());
            for node in &nodes {
                offsets.push(labels.len() as u32);
                let mut edges: Vec<_> = node.children.iter().collect();
                edges.sort_unstable_by_key(|(label, _)| **label);
                for (&label, &target) in edges {
                    labels.push(label);
                    targets.push(target);
                }
                fail.push(node.fail);
            }
            offsets.push(labels.len() as u32);
            (
                Transitions::Sparse {
                    offsets,
                    labels,
                    targets,
                    fail,
                },
                nodes.len(),
            )
        };

        let (transitions, num_states) = transitions;

        Self {
            transitions,
            best_output,
            num_tokens: dict.num_tokens(),
            num_states,
            trie_states: nodes.len(),
        }
    }

    /// Tokenise one row using the reversed first pass and greedy forward pass.
    pub fn tokenize(&self, row: &[u8]) -> Vec<Token> {
        let mut best = Vec::with_capacity(row.len());
        let mut tokens = Vec::with_capacity(row.len());
        self.tokenize_into(row, &mut best, &mut tokens);
        tokens
    }

    /// Number of dictionary patterns in the automaton.
    pub fn size(&self) -> usize {
        self.num_tokens
    }

    /// Number of runtime automaton states, including the root.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Number of states before equivalent DFA continuations are merged.
    pub fn trie_states(&self) -> usize {
        self.trie_states
    }

    /// Whether parsing uses the one-lookup completed DFA representation.
    pub fn uses_dense_transitions(&self) -> bool {
        matches!(
            self.transitions,
            Transitions::Dense12(_) | Transitions::Dense16(_) | Transitions::Dense(_)
        )
    }

    /// Heap bytes used by transitions, failure links, and state outputs.
    pub fn automaton_bytes(&self) -> usize {
        let transition_bytes = match &self.transitions {
            Transitions::Dense12(table) | Transitions::Dense(table) => {
                table.len() * size_of::<u32>()
            }
            Transitions::Dense16(table) => table.len(),
            Transitions::Sparse {
                offsets,
                labels,
                targets,
                fail,
            } => {
                offsets.len() * size_of::<u32>()
                    + labels.len()
                    + targets.len() * size_of::<u32>()
                    + fail.len() * size_of::<u32>()
            }
        };
        transition_bytes + self.best_output.len() * size_of::<u32>()
    }

    #[inline]
    fn advance(&self, mut state: u32, byte: u8) -> u32 {
        match &self.transitions {
            Transitions::Dense12(table) => {
                table[state as usize * 256 + byte as usize] & u32::from(u16::MAX)
            }
            Transitions::Dense16(table) => {
                read_packed_transition(table, state as usize * 256 + byte as usize)
            }
            Transitions::Dense(table) => table[state as usize * 256 + byte as usize],
            Transitions::Sparse {
                offsets,
                labels,
                targets,
                fail,
            } => loop {
                let start = offsets[state as usize] as usize;
                let end = offsets[state as usize + 1] as usize;
                if let Some(index) = find_label(&labels[start..end], byte) {
                    return targets[start + index];
                }
                if state == 0 {
                    return 0;
                }
                state = fail[state as usize];
            },
        }
    }

    /// Store the best token and length for every row byte.
    fn find_best_matches(&self, row: &[u8], best: &mut Vec<u32>) {
        best.clear();
        best.reserve(row.len());
        let output = best.spare_capacity_mut();
        match &self.transitions {
            Transitions::Dense12(table) => {
                let mut state = 0usize;
                for pos in (0..row.len()).rev() {
                    // SAFETY: `pos` comes from the row range. Every constructed
                    // DFA has 256 entries per state, and every transition's low
                    // 16 bits name a state in that same table.
                    unsafe {
                        let byte = *row.get_unchecked(pos) as usize;
                        let entry = *table.get_unchecked((state << 8) | byte);
                        state = (entry & u32::from(u16::MAX)) as usize;
                        output.get_unchecked_mut(pos).write(entry >> 16);
                    }
                }
            }
            Transitions::Dense16(table) => {
                let mut state = 0usize;
                for pos in (0..row.len()).rev() {
                    let byte = row[pos] as usize;
                    state = read_packed_transition(table, (state << 8) | byte) as usize;
                    output[pos].write(self.best_output[state]);
                }
            }
            Transitions::Dense(table) => {
                let mut state = 0usize;
                for pos in (0..row.len()).rev() {
                    // SAFETY: the same construction invariants as Dense12
                    // apply; best_output has exactly one entry per DFA state.
                    unsafe {
                        let byte = *row.get_unchecked(pos) as usize;
                        state = *table.get_unchecked((state << 8) | byte) as usize;
                        output
                            .get_unchecked_mut(pos)
                            .write(*self.best_output.get_unchecked(state));
                    }
                }
            }
            Transitions::Sparse { .. } => {
                let mut state = 0;
                for pos in (0..row.len()).rev() {
                    state = self.advance(state, row[pos]);
                    output[pos].write(self.best_output[state as usize]);
                }
            }
        }
        // SAFETY: every branch above writes each position in `0..row.len()`
        // exactly once before exposing the initialized elements.
        unsafe { best.set_len(row.len()) };
    }

    #[inline]
    fn decode_match(&self, packed: u32) -> (Token, usize) {
        if matches!(self.transitions, Transitions::Dense12(_)) {
            ((packed >> 4) as Token, ((packed & 0xf) + 1) as usize)
        } else {
            ((packed >> 8) as Token, (packed & 0xff) as usize)
        }
    }

    fn tokenize_into(&self, row: &[u8], best: &mut Vec<u32>, tokens: &mut Vec<Token>) -> usize {
        self.find_best_matches(row, best);
        let tokens_before = tokens.len();
        let mut pos = 0;
        while pos < row.len() {
            let (token, len) = self.decode_match(best[pos]);
            assert!(len != 0, "OnPair dictionary is missing a single-byte token");
            tokens.push(token);
            pos += len;
        }
        tokens.len() - tokens_before
    }
}

/// Encode all rows with a reversed-Aho--Corasick first pass followed by the
/// ordinary forward greedy walk. The automaton is reset implicitly because
/// each row is searched independently, so tokens can never cross boundaries.
pub fn parse_reversed(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &ReversedAhoCorasickMatcher,
    bits: BitWidth,
    store: &mut Store,
) {
    store.bit_width = bits;
    store.packed.clear();
    store.boundaries.clear();

    let mut writer = BitWriter::new(store);
    let mut boundaries = Vec::with_capacity(n + 1);
    boundaries.push(0);
    let mut best = Vec::new();

    for row in 0..n {
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        let bytes = &data[start..end];
        matcher.find_best_matches(bytes, &mut best);

        let mut pos = 0;
        while pos < bytes.len() {
            let (token, len) = matcher.decode_match(best[pos]);
            assert!(len != 0, "OnPair dictionary is missing a single-byte token");
            writer.write(token);
            pos += len;
        }
        boundaries.push(writer.tokens_written() as u32);
    }

    drop(writer);
    store.boundaries = boundaries;
}

/// Encode rows on one thread while interleaving eight similarly-sized DFA
/// streams to expose independent transition loads to the CPU.
pub fn parse_reversed_interleaved(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &ReversedAhoCorasickMatcher,
    bits: BitWidth,
    store: &mut Store,
) {
    let rows = rows_sorted_by_length(offsets, n);
    let mut best = vec![0u32; offsets[n] as usize];
    match &matcher.transitions {
        Transitions::Dense12(table) => {
            for group in rows.chunks(8) {
                fill_interleaved_group(data, offsets, group, table, &mut best);
            }
        }
        Transitions::Dense16(table) => {
            for group in rows.chunks(16) {
                fill_interleaved_packed_group(data, offsets, group, table, &mut best);
            }
        }
        Transitions::Dense(table) => {
            for group in rows.chunks(16) {
                fill_interleaved_dense_group(
                    data,
                    offsets,
                    group,
                    table,
                    &matcher.best_output,
                    &mut best,
                );
            }
        }
        Transitions::Sparse { .. } => {
            parse_reversed(data, offsets, n, matcher, bits, store);
            return;
        }
    }
    pack_matches(offsets, n, &best, matcher, bits, store);
}

/// Encode rows on one thread using sixteen length-grouped AVX-512 DFA lanes.
/// Falls back to [`parse_reversed_interleaved`] when AVX-512 is unavailable.
pub fn parse_reversed_avx512(
    data: &[u8],
    offsets: &[u32],
    n: usize,
    matcher: &ReversedAhoCorasickMatcher,
    bits: BitWidth,
    store: &mut Store,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") && data.len() <= i32::MAX as usize {
        let rows = rows_sorted_by_length(offsets, n);
        let mut best = vec![0u32; offsets[n] as usize];
        match &matcher.transitions {
            Transitions::Dense12(table) if table.len() <= i32::MAX as usize => {
                for group in rows.chunks(16) {
                    if gather_reads_past_data(group, offsets, data.len()) {
                        fill_interleaved_group(data, offsets, group, table, &mut best);
                    } else {
                        // SAFETY: runtime feature detection and gather bounds
                        // are checked above.
                        unsafe { fill_avx512_group(data, offsets, group, table, &mut best) };
                    }
                }
            }
            Transitions::Dense16(table) if table.len() <= i32::MAX as usize => {
                for group in rows.chunks(16) {
                    if gather_reads_past_data(group, offsets, data.len()) {
                        fill_interleaved_packed_group(data, offsets, group, table, &mut best);
                    } else {
                        // SAFETY: runtime feature detection and gather bounds
                        // are checked above.
                        unsafe { fill_avx512_packed_group(data, offsets, group, table, &mut best) };
                    }
                }
            }
            Transitions::Dense(table) if table.len() <= i32::MAX as usize => {
                for group in rows.chunks(16) {
                    if gather_reads_past_data(group, offsets, data.len()) {
                        fill_interleaved_dense_group(
                            data,
                            offsets,
                            group,
                            table,
                            &matcher.best_output,
                            &mut best,
                        );
                    } else {
                        // SAFETY: runtime feature detection and gather bounds
                        // are checked above.
                        unsafe {
                            fill_avx512_dense_group(
                                data,
                                offsets,
                                group,
                                table,
                                &matcher.best_output,
                                &mut best,
                            )
                        };
                    }
                }
            }
            _ => {
                parse_reversed(data, offsets, n, matcher, bits, store);
                return;
            }
        }
        pack_matches(offsets, n, &best, matcher, bits, store);
        return;
    }

    parse_reversed_interleaved(data, offsets, n, matcher, bits, store);
}

fn gather_reads_past_data(rows: &[usize], offsets: &[u32], data_len: usize) -> bool {
    rows.iter()
        .any(|&row| offsets[row + 1] as usize > data_len.saturating_sub(3))
}

fn rows_sorted_by_length(offsets: &[u32], n: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..n).collect();
    rows.sort_unstable_by_key(|&row| offsets[row + 1] - offsets[row]);
    rows
}

fn fill_interleaved_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    table: &[u32],
    best: &mut [u32],
) {
    let mut starts = [0usize; 16];
    let mut positions = [0usize; 16];
    let mut states = [0usize; 16];
    for (lane, &row) in rows.iter().enumerate() {
        starts[lane] = offsets[row] as usize;
        positions[lane] = offsets[row + 1] as usize;
    }

    loop {
        let mut active = false;
        for lane in 0..rows.len() {
            if positions[lane] == starts[lane] {
                continue;
            }
            active = true;
            positions[lane] -= 1;
            let pos = positions[lane];
            let entry = table[(states[lane] << 8) | data[pos] as usize];
            states[lane] = (entry & u32::from(u16::MAX)) as usize;
            best[pos] = entry >> 16;
        }
        if !active {
            break;
        }
    }
}

fn fill_interleaved_packed_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    table: &[u8],
    best: &mut [u32],
) {
    let mut starts = [0usize; 16];
    let mut positions = [0usize; 16];
    let mut states = [0usize; 16];
    for (lane, &row) in rows.iter().enumerate() {
        starts[lane] = offsets[row] as usize;
        positions[lane] = offsets[row + 1] as usize;
    }

    loop {
        let mut active = false;
        for lane in 0..rows.len() {
            if positions[lane] == starts[lane] {
                continue;
            }
            active = true;
            positions[lane] -= 1;
            let pos = positions[lane];
            let state = read_packed_transition(table, (states[lane] << 8) | data[pos] as usize);
            states[lane] = state as usize;
            best[pos] = state;
        }
        if !active {
            break;
        }
    }
}

fn fill_interleaved_dense_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    table: &[u32],
    _state_output: &[u32],
    best: &mut [u32],
) {
    let mut starts = [0usize; 16];
    let mut positions = [0usize; 16];
    let mut states = [0usize; 16];
    for (lane, &row) in rows.iter().enumerate() {
        starts[lane] = offsets[row] as usize;
        positions[lane] = offsets[row + 1] as usize;
    }

    loop {
        let mut active = false;
        for lane in 0..rows.len() {
            if positions[lane] == starts[lane] {
                continue;
            }
            active = true;
            positions[lane] -= 1;
            let pos = positions[lane];
            let state = table[(states[lane] << 8) | data[pos] as usize] as usize;
            states[lane] = state;
            best[pos] = state as u32;
        }
        if !active {
            break;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fill_avx512_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    table: &[u32],
    best: &mut [u32],
) {
    use std::arch::x86_64::*;

    let mut starts_array = [0i32; 16];
    let mut positions_array = [0i32; 16];
    let mut lane_mask = 0u16;
    for (lane, &row) in rows.iter().enumerate() {
        starts_array[lane] = offsets[row] as i32;
        positions_array[lane] = offsets[row + 1] as i32;
        lane_mask |= 1 << lane;
    }

    // SAFETY: all loads are from complete local arrays. The caller establishes
    // valid gather/scatter bounds and AVX-512 support.
    unsafe {
        let starts = _mm512_loadu_si512(starts_array.as_ptr().cast());
        let mut positions = _mm512_loadu_si512(positions_array.as_ptr().cast());
        let mut states = _mm512_setzero_si512();
        let one = _mm512_set1_epi32(1);
        let byte_mask = _mm512_set1_epi32(0xff);
        let state_mask = _mm512_set1_epi32(i32::from(u16::MAX));

        loop {
            let active = _mm512_mask_cmpgt_epi32_mask(lane_mask, positions, starts);
            if active == 0 {
                break;
            }
            positions = _mm512_mask_sub_epi32(positions, active, positions, one);
            let words = _mm512_mask_i32gather_epi32::<1>(
                _mm512_setzero_si512(),
                active,
                positions,
                data.as_ptr().cast(),
            );
            let bytes = _mm512_and_si512(words, byte_mask);
            let indices = _mm512_or_si512(_mm512_slli_epi32::<8>(states), bytes);
            let entries = _mm512_mask_i32gather_epi32::<4>(
                _mm512_setzero_si512(),
                active,
                indices,
                table.as_ptr().cast(),
            );
            states = _mm512_and_si512(entries, state_mask);
            let outputs = _mm512_srli_epi32::<16>(entries);
            _mm512_mask_i32scatter_epi32::<4>(best.as_mut_ptr().cast(), active, positions, outputs);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fill_avx512_packed_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    table: &[u8],
    best: &mut [u32],
) {
    use std::arch::x86_64::*;

    let mut starts_array = [0i32; 16];
    let mut positions_array = [0i32; 16];
    let mut lane_mask = 0u16;
    for (lane, &row) in rows.iter().enumerate() {
        starts_array[lane] = offsets[row] as i32;
        positions_array[lane] = offsets[row + 1] as i32;
        lane_mask |= 1 << lane;
    }

    // SAFETY: all loads are from complete local arrays. The caller establishes
    // valid gather/scatter bounds and AVX-512 support.
    unsafe {
        let starts = _mm512_loadu_si512(starts_array.as_ptr().cast());
        let mut positions = _mm512_loadu_si512(positions_array.as_ptr().cast());
        let mut states = _mm512_setzero_si512();
        let one = _mm512_set1_epi32(1);
        let byte_mask = _mm512_set1_epi32(0xff);
        let state_mask = _mm512_set1_epi32(0x000f_ffff);

        loop {
            let active = _mm512_mask_cmpgt_epi32_mask(lane_mask, positions, starts);
            if active == 0 {
                break;
            }
            positions = _mm512_mask_sub_epi32(positions, active, positions, one);
            let words = _mm512_mask_i32gather_epi32::<1>(
                _mm512_setzero_si512(),
                active,
                positions,
                data.as_ptr().cast(),
            );
            let bytes = _mm512_and_si512(words, byte_mask);
            let indices = _mm512_or_si512(_mm512_slli_epi32::<8>(states), bytes);
            let byte_offsets = _mm512_add_epi32(
                _mm512_slli_epi32::<1>(indices),
                _mm512_srli_epi32::<1>(indices),
            );
            let entries = _mm512_mask_i32gather_epi32::<1>(
                _mm512_setzero_si512(),
                active,
                byte_offsets,
                table.as_ptr().cast(),
            );
            let shifts = _mm512_slli_epi32::<2>(_mm512_and_si512(indices, one));
            states = _mm512_and_si512(_mm512_srlv_epi32(entries, shifts), state_mask);
            _mm512_mask_i32scatter_epi32::<4>(best.as_mut_ptr().cast(), active, positions, states);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fill_avx512_dense_group(
    data: &[u8],
    offsets: &[u32],
    rows: &[usize],
    table: &[u32],
    _state_output: &[u32],
    best: &mut [u32],
) {
    use std::arch::x86_64::*;

    let mut starts_array = [0i32; 16];
    let mut positions_array = [0i32; 16];
    let mut lane_mask = 0u16;
    for (lane, &row) in rows.iter().enumerate() {
        starts_array[lane] = offsets[row] as i32;
        positions_array[lane] = offsets[row + 1] as i32;
        lane_mask |= 1 << lane;
    }

    // SAFETY: all loads are from complete local arrays. The caller establishes
    // valid gather/scatter bounds and AVX-512 support.
    unsafe {
        let starts = _mm512_loadu_si512(starts_array.as_ptr().cast());
        let mut positions = _mm512_loadu_si512(positions_array.as_ptr().cast());
        let mut states = _mm512_setzero_si512();
        let one = _mm512_set1_epi32(1);
        let byte_mask = _mm512_set1_epi32(0xff);

        loop {
            let active = _mm512_mask_cmpgt_epi32_mask(lane_mask, positions, starts);
            if active == 0 {
                break;
            }
            positions = _mm512_mask_sub_epi32(positions, active, positions, one);
            let words = _mm512_mask_i32gather_epi32::<1>(
                _mm512_setzero_si512(),
                active,
                positions,
                data.as_ptr().cast(),
            );
            let bytes = _mm512_and_si512(words, byte_mask);
            let indices = _mm512_or_si512(_mm512_slli_epi32::<8>(states), bytes);
            states = _mm512_mask_i32gather_epi32::<4>(
                _mm512_setzero_si512(),
                active,
                indices,
                table.as_ptr().cast(),
            );
            _mm512_mask_i32scatter_epi32::<4>(best.as_mut_ptr().cast(), active, positions, states);
        }
    }
}

fn pack_matches(
    offsets: &[u32],
    n: usize,
    best: &[u32],
    matcher: &ReversedAhoCorasickMatcher,
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
            let packed = if matches!(
                matcher.transitions,
                Transitions::Dense16(_) | Transitions::Dense(_)
            ) {
                matcher.best_output[best[pos] as usize]
            } else {
                best[pos]
            };
            let (token, len) = matcher.decode_match(packed);
            writer.write(token);
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
    use crate::lpm::LongestPrefixMatcher;
    use crate::parser::parse;
    use crate::test_corpus::binary_strings;
    use crate::test_corpus::make_raw;
    use crate::test_corpus::random_ascii_strings;
    use crate::test_corpus::user_strings;
    use crate::tokenize::tokenize_with;
    use crate::trainer::train;

    fn dictionary(tokens: &[&[u8]]) -> Dictionary {
        let mut dict = Dictionary::default();
        dict.offsets.push(0);
        for byte in 0u16..=255 {
            dict.bytes.push(byte as u8);
            dict.offsets.push(dict.bytes.len() as u32);
        }
        for token in tokens {
            dict.bytes.extend_from_slice(token);
            dict.offsets.push(dict.bytes.len() as u32);
        }
        dict
    }

    #[test]
    fn overlapping_prefixes_choose_the_longest_at_each_position() {
        let dict = dictionary(&[b"ab", b"abc", b"bc", b"abcd"]);
        let reference = LongestPrefixMatcher::from_dictionary(&dict);
        let reversed = ReversedAhoCorasickMatcher::from_dictionary(&dict);
        assert_eq!(
            reversed.tokenize(b"zabcdabc"),
            tokenize_with(b"zabcdabc", &reference)
        );
    }

    #[test]
    fn duplicate_tokens_use_the_last_dictionary_id() {
        let dict = dictionary(&[b"abc", b"abc"]);
        let reference = LongestPrefixMatcher::from_dictionary(&dict);
        let reversed = ReversedAhoCorasickMatcher::from_dictionary(&dict);
        assert_eq!(reversed.tokenize(b"abc"), tokenize_with(b"abc", &reference));
        assert_eq!(reversed.tokenize(b"abc"), vec![257]);
    }

    #[test]
    fn rows_are_independent_and_empty_rows_are_preserved() {
        let dict = dictionary(&[b"abcd", b"ab", b"cd"]);
        let raw = make_raw(&[b"ab".as_slice(), b"".as_slice(), b"cd".as_slice()]);
        let lpm = LongestPrefixMatcher::from_dictionary(&dict);
        let reversed = ReversedAhoCorasickMatcher::from_dictionary(&dict);
        let mut expected = Store::default();
        let mut actual = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &lpm, 12, &mut expected);
        parse_reversed(&raw.data, &raw.offsets, raw.n, &reversed, 12, &mut actual);
        assert_eq!(actual.boundaries, vec![0, 1, 1, 2]);
        assert_eq!(actual.boundaries, expected.boundaries);
        assert_eq!(actual.packed, expected.packed);
    }

    #[test]
    fn dense16_variants_match_greedy_tokens() {
        let mut dict = dictionary(&[]);
        for first in 0u8..20 {
            for second in 0u16..=255 {
                dict.bytes.extend_from_slice(&[first, second as u8]);
                dict.offsets.push(dict.bytes.len() as u32);
            }
        }
        let raw = make_raw(&[
            b"\x03\xfe\x12\x00\x04\x80".as_slice(),
            b"\x00\x00\x13\xff".as_slice(),
        ]);
        let greedy = LongestPrefixMatcher::from_dictionary(&dict);
        let reversed = ReversedAhoCorasickMatcher::from_dictionary(&dict);
        assert!(matches!(reversed.transitions, Transitions::Dense16(_)));

        let mut expected = Store::default();
        let mut interleaved = Store::default();
        let mut avx512 = Store::default();
        parse(&raw.data, &raw.offsets, raw.n, &greedy, 16, &mut expected);
        parse_reversed_interleaved(
            &raw.data,
            &raw.offsets,
            raw.n,
            &reversed,
            16,
            &mut interleaved,
        );
        parse_reversed_avx512(&raw.data, &raw.offsets, raw.n, &reversed, 16, &mut avx512);
        assert_eq!(interleaved.boundaries, expected.boundaries);
        assert_eq!(interleaved.packed, expected.packed);
        assert_eq!(avx512.boundaries, expected.boundaries);
        assert_eq!(avx512.packed, expected.packed);
    }

    #[test]
    fn trained_dictionaries_are_bit_exact_for_varied_corpora() {
        let corpora = [
            user_strings(200)
                .into_iter()
                .map(String::into_bytes)
                .collect(),
            random_ascii_strings(200, 80, 17),
            binary_strings(200, 80, 91),
        ];
        for (corpus_index, corpus) in corpora.iter().enumerate() {
            let raw = make_raw(corpus);
            for bits in [9, 10, 12] {
                let cfg = TrainingConfig {
                    bits,
                    threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
                    seed: Some(42),
                };
                let trained = train(&raw.data, &raw.offsets, raw.n, &cfg);
                let reversed = ReversedAhoCorasickMatcher::from_dictionary(&trained.dict);
                let mut expected = Store::default();
                let mut actual = Store::default();
                let mut interleaved = Store::default();
                let mut avx512 = Store::default();
                parse(
                    &raw.data,
                    &raw.offsets,
                    raw.n,
                    &trained.lpm,
                    bits,
                    &mut expected,
                );
                parse_reversed(&raw.data, &raw.offsets, raw.n, &reversed, bits, &mut actual);
                parse_reversed_interleaved(
                    &raw.data,
                    &raw.offsets,
                    raw.n,
                    &reversed,
                    bits,
                    &mut interleaved,
                );
                parse_reversed_avx512(&raw.data, &raw.offsets, raw.n, &reversed, bits, &mut avx512);
                assert_eq!(
                    actual.boundaries, expected.boundaries,
                    "boundaries differ for corpus {corpus_index}, bits {bits}"
                );
                assert_eq!(
                    actual.packed, expected.packed,
                    "codes differ for corpus {corpus_index}, bits {bits}"
                );
                assert_eq!(interleaved.boundaries, expected.boundaries);
                assert_eq!(interleaved.packed, expected.packed);
                assert_eq!(avx512.boundaries, expected.boundaries);
                assert_eq!(avx512.packed, expected.packed);
            }
        }
    }
}
