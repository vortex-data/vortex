//! Picky-OnPair: OnPair training with Picky-BPE-style intermediate-token pruning.
//!
//! During training we maintain unigram counts for every token. When a new
//! merge `(a, b) -> c` is created, we compute the Intersection-over-Self
//! metric `IoS(a) = pair_count(a,b) / unigram(a)` and `IoS(b)` analogously.
//! If either side is dominated by occurrences inside this merge (IoS >= tau),
//! the side is evicted: removed from the longest-prefix matcher and its
//! 16-bit slot returned to a free list. Future merges reuse those slots,
//! letting the 16-bit dictionary admit more high-utility merges than the
//! vanilla algorithm.
//!
//! After training, we run the standard parse pass against the (now slightly
//! denser) dictionary.

use crate::lpm::LongestPrefixMatcher;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustc_hash::FxHashMap;

const FAST_COPY_SIZE: usize = 16;

pub struct OnPairPicky {
    threshold: u16,
    /// IoS threshold for eviction (e.g. 0.85).
    tau_num: u32,
    tau_den: u32,
    /// Minimum unigram count before a token is even considered for eviction.
    /// Guards against evicting tokens with thin statistics.
    min_unigram: u32,
    /// Number of training passes over the data.
    passes: u32,

    compressed_data: Vec<u16>,
    string_boundaries: Vec<usize>,

    dictionary_tokens: Vec<Vec<u8>>, // token_id -> raw bytes (Vec so we can rewrite slots)
    n_used_slots: usize,
}

impl OnPairPicky {
    pub fn new(threshold: u16, tau_num: u32, tau_den: u32) -> Self {
        Self::with_params(threshold, tau_num, tau_den, 4, 1)
    }

    pub fn with_params(threshold: u16, tau_num: u32, tau_den: u32, min_unigram: u32, passes: u32) -> Self {
        assert!(threshold > 1);
        Self {
            threshold,
            tau_num,
            tau_den,
            min_unigram,
            passes: passes.max(1),
            compressed_data: Vec::new(),
            string_boundaries: Vec::new(),
            dictionary_tokens: Vec::new(),
            n_used_slots: 0,
        }
    }

    pub fn with_capacity(threshold: u16, tau_num: u32, tau_den: u32, n_strings: usize, n_bytes: usize) -> Self {
        Self {
            threshold,
            tau_num,
            tau_den,
            min_unigram: 4,
            passes: 1,
            compressed_data: Vec::with_capacity(n_bytes),
            string_boundaries: Vec::with_capacity(n_strings),
            dictionary_tokens: Vec::with_capacity(1 << 16),
            n_used_slots: 0,
        }
    }

    pub fn with_capacity_params(
        threshold: u16,
        tau_num: u32,
        tau_den: u32,
        min_unigram: u32,
        passes: u32,
        n_strings: usize,
        n_bytes: usize,
    ) -> Self {
        assert!(threshold > 1);
        Self {
            threshold,
            tau_num,
            tau_den,
            min_unigram,
            passes: passes.max(1),
            compressed_data: Vec::with_capacity(n_bytes),
            string_boundaries: Vec::with_capacity(n_strings),
            dictionary_tokens: Vec::with_capacity(1 << 16),
            n_used_slots: 0,
        }
    }

    pub fn compress_strings<S: AsRef<str>>(&mut self, strings: &[S]) {
        let total_len: usize = strings.iter().map(|s| s.as_ref().len()).sum();
        let mut data = Vec::with_capacity(total_len);
        let mut end_positions = Vec::with_capacity(strings.len() + 1);
        end_positions.push(0);
        for s in strings {
            data.extend_from_slice(s.as_ref().as_bytes());
            end_positions.push(data.len());
        }
        self.compress_bytes(&data, &end_positions);
    }

    pub fn compress_bytes(&mut self, data: &[u8], end_positions: &[usize]) {
        let lpm = self.train_dictionary(data, end_positions);
        self.parse_data(data, end_positions, &lpm);
    }

    fn train_dictionary(&mut self, data: &[u8], end_positions: &[usize]) -> LongestPrefixMatcher<u16> {
        // Initialize 256 byte tokens. We never evict byte tokens.
        self.dictionary_tokens.clear();
        self.dictionary_tokens.resize(65536, Vec::new());
        let mut lpm = LongestPrefixMatcher::new();
        for i in 0..256u16 {
            self.dictionary_tokens[i as usize] = vec![i as u8];
            lpm.insert(&[i as u8], i);
        }
        self.n_used_slots = 256;

        let mut next_token_id: u32 = 256;
        let mut free_slots: Vec<u16> = Vec::new();

        let mut pair_freq: FxHashMap<(u16, u16), u32> = FxHashMap::default();
        let mut unigram: Vec<u32> = vec![0u32; 65536];
        let mut alive: Vec<bool> = vec![false; 65536];
        for i in 0..256 {
            alive[i] = true;
        }

        let tau_num = self.tau_num as u64;
        let tau_den = self.tau_den as u64;
        let min_unigram = self.min_unigram;

        let mut shuffled: Vec<usize> = (0..end_positions.len() - 1).collect();

        for _pass in 0..self.passes {
            shuffled.shuffle(&mut thread_rng());
            // Reset pair frequencies between passes so each pass adds merges that
            // emerge under the *current* tokenization, not stale pairs.
            pair_freq.clear();

            for &idx in shuffled.iter() {
                let start = end_positions[idx];
                let end = end_positions[idx + 1];
                if start == end {
                    continue;
                }

                let (mut prev_id, mut prev_len) = lpm.find_longest_match(&data[start..end]).unwrap();
                unigram[prev_id as usize] = unigram[prev_id as usize].saturating_add(1);
                let mut pos = start + prev_len;

                while pos < end {
                    let (cur_id, cur_len) = lpm.find_longest_match(&data[pos..end]).unwrap();
                    unigram[cur_id as usize] = unigram[cur_id as usize].saturating_add(1);

                    let key = (prev_id, cur_id);
                    let entry = pair_freq.entry(key).or_insert(0);
                    *entry += 1;

                    if *entry >= self.threshold as u32 {
                        let pair_count = *entry as u64;
                        pair_freq.remove(&key);

                        // Allocate slot
                        let new_id: u16 = if let Some(reused) = free_slots.pop() {
                            reused
                        } else if next_token_id < 65536 {
                            let id = next_token_id as u16;
                            next_token_id += 1;
                            id
                        } else {
                            // Dictionary full and nothing to reuse; advance.
                            prev_id = cur_id;
                            prev_len = cur_len;
                            pos += cur_len;
                            continue;
                        };

                        let merged: Vec<u8> = data[pos - prev_len..pos + cur_len].to_vec();
                        if !self.dictionary_tokens[new_id as usize].is_empty() {
                            let old = std::mem::take(&mut self.dictionary_tokens[new_id as usize]);
                            lpm.remove(&old);
                        }
                        lpm.insert(&merged, new_id);
                        self.dictionary_tokens[new_id as usize] = merged;
                        alive[new_id as usize] = true;
                        unigram[new_id as usize] = 0;
                        self.n_used_slots = self.n_used_slots.max(new_id as usize + 1);

                        // Picky eviction: drop prev/cur if IoS w.r.t. this merge is high
                        // AND we have enough unigram evidence.
                        for &cand in &[prev_id, cur_id] {
                            if (cand as usize) < 256 || !alive[cand as usize] {
                                continue;
                            }
                            let u = unigram[cand as usize] as u64;
                            if u < min_unigram as u64 {
                                continue;
                            }
                            if pair_count * tau_den >= u * tau_num {
                                let bytes = std::mem::take(&mut self.dictionary_tokens[cand as usize]);
                                lpm.remove(&bytes);
                                alive[cand as usize] = false;
                                free_slots.push(cand);
                            }
                        }

                        prev_id = new_id;
                        prev_len += cur_len;
                    } else {
                        prev_id = cur_id;
                        prev_len = cur_len;
                    }
                    pos += cur_len;
                }
            }
        }
        let _ = (tau_num, tau_den);
        lpm
    }

    fn parse_data(&mut self, data: &[u8], end_positions: &[usize], lpm: &LongestPrefixMatcher<u16>) {
        self.string_boundaries.push(0);
        for window in end_positions.windows(2) {
            let (start, end) = (window[0], window[1]);
            if start == end {
                self.string_boundaries.push(self.compressed_data.len());
                continue;
            }
            let mut pos = start;
            while pos < end {
                let (id, len) = lpm.find_longest_match(&data[pos..end]).unwrap();
                self.compressed_data.push(id);
                pos += len;
            }
            self.string_boundaries.push(self.compressed_data.len());
        }
    }

    pub fn space_used(&self) -> usize {
        // Same accounting as OnPair: stream as u16, dict bytes, boundaries as u32 per dict entry.
        let dict_bytes: usize = self.dictionary_tokens.iter().map(|t| t.len()).sum();
        let boundary_bytes = (self.n_used_slots + 1) * 4;
        self.compressed_data.len() * 2 + dict_bytes + boundary_bytes
    }

    pub fn compressed_data(&self) -> &[u16] {
        &self.compressed_data
    }
    pub fn token_len(&self, id: u16) -> usize {
        self.dictionary_tokens[id as usize].len()
    }
    pub fn dictionary_token_count(&self) -> usize {
        // count of live (non-empty) slots
        self.dictionary_tokens.iter().filter(|t| !t.is_empty()).count()
    }
    pub fn compressed_token_count(&self) -> usize {
        self.compressed_data.len()
    }
    pub fn dictionary_bytes(&self) -> usize {
        self.dictionary_tokens.iter().map(|t| t.len()).sum()
    }

    pub fn decompress_string(&self, index: usize, buffer: &mut [u8]) -> usize {
        let s = self.string_boundaries[index];
        let e = self.string_boundaries[index + 1];
        let mut size = 0;
        for &id in &self.compressed_data[s..e] {
            let tok = &self.dictionary_tokens[id as usize];
            buffer[size..size + tok.len()].copy_from_slice(tok);
            size += tok.len();
        }
        let _ = FAST_COPY_SIZE;
        size
    }
}
