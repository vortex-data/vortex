//! OnPair4Tier: OnPair training (Picky-BPE style) plus a four-tier bit-packed
//! token-stream code.
//!
//! Construction trains the dictionary identically to `OnPairOpt` (Picky-BPE
//! pass with `tau_num / tau_den`, `min_unigram`, multipass). After parsing,
//! tokens are ranked by stream frequency and the rank space is partitioned
//! into four contiguous tiers with sizes that are powers of two:
//!
//!     k0 = 2^a0   (head)
//!     k1 = 2^a1   (warm)
//!     k2 = 2^a2   (cool)
//!     k3 = 2^a3   (tail)
//!
//! Each token is encoded as a 2-bit prefix selecting the tier (00,01,10,11)
//! followed by a payload of a_i bits identifying the token within that tier.
//!
//! The partition (a0,a1,a2,a3) is chosen by brute-force sweep over all
//! combinations where each a_i ∈ {2,..,15} and k0+k1+k2+k3 ≥ N_used, picking
//! the one that minimises total stream bits.

use crate::lpm::LongestPrefixMatcher;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustc_hash::FxHashMap;

use super::onpair_opt::OnPairOptParams;

/// Four-tier OnPair encoder with brute-force tier-size optimisation.
pub struct OnPair4Tier {
    params: OnPairOptParams,
    /// Raw token bytes packed contiguously.
    dictionary: Vec<u8>,
    /// Boundary indices into `dictionary`, length = n_tokens + 1.
    token_boundaries: Vec<u32>,
    /// Map from frequency rank (0 = most frequent) to dictionary id.
    rank_to_id: Vec<u16>,
    /// Inverse mapping for encoding.
    id_to_rank: Vec<u16>,
    /// Per-tier widths in bits (a_i = log2(k_i)).
    widths: [u32; 4],
    /// Cumulative tier sizes: cum[0] = 0, cum[i+1] = cum[i] + k_i.
    cum: [u32; 5],
    /// Total tokens in the final dictionary.
    n_tokens: usize,
    /// Packed stream bits.
    stream_bits: Vec<u64>,
    stream_bit_len: u64,
    /// Per-string bit offsets into stream_bits; length = n_strings + 1.
    string_bit_offsets: Vec<u64>,
}

impl OnPair4Tier {
    pub fn new(params: OnPairOptParams) -> Self {
        Self {
            params,
            dictionary: Vec::new(),
            token_boundaries: Vec::new(),
            rank_to_id: Vec::new(),
            id_to_rank: Vec::new(),
            widths: [0; 4],
            cum: [0; 5],
            n_tokens: 0,
            stream_bits: Vec::new(),
            stream_bit_len: 0,
            string_bit_offsets: Vec::new(),
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
        // Phase A: train + parse (identical to OnPairOpt).
        let (lpm, dict_tokens) = self.train(data, end_positions);
        let (raw_stream, raw_string_boundaries) = self.parse(data, end_positions, &lpm);

        // Phase B: rank tokens by stream frequency, build compact dictionary.
        let stream_n = raw_stream.len();
        let mut counts = vec![0u32; dict_tokens.len()];
        for &id in &raw_stream {
            counts[id as usize] += 1;
        }
        let mut entries: Vec<(u32, u16)> = counts.iter().enumerate()
            .filter_map(|(i, &c)| if c > 0 { Some((c, i as u16)) } else { None })
            .collect();
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let n_used = entries.len();
        self.n_tokens = n_used;
        self.rank_to_id = entries.iter().map(|&(_, id)| id).collect();

        self.id_to_rank = vec![u16::MAX; dict_tokens.len()];
        for (rank, &id) in self.rank_to_id.iter().enumerate() {
            self.id_to_rank[id as usize] = rank as u16;
        }

        self.dictionary.clear();
        self.token_boundaries.clear();
        self.token_boundaries.push(0);
        for &id in &self.rank_to_id {
            let tok = &dict_tokens[id as usize];
            self.dictionary.extend_from_slice(tok);
            self.token_boundaries.push(self.dictionary.len() as u32);
        }

        // Phase C: choose the best 4-tier partition by brute force.
        let sorted_counts: Vec<u32> = entries.iter().map(|&(c, _)| c).collect();
        // Cumulative-count prefix sum for O(1) coverage queries.
        let mut prefix = vec![0u64; n_used + 1];
        for i in 0..n_used {
            prefix[i + 1] = prefix[i] + sorted_counts[i] as u64;
        }

        let stream_n_u = stream_n as u64;
        let mut best_widths = [0u32; 4];
        let mut best_cum = [0u32; 5];
        let mut best_bits = u64::MAX;

        // Sweep all (a0,a1,a2,a3) where each a_i ∈ {1,..,15} and the
        // cumulative size first reaches n_used at tier 3.
        // Allow a_i down to 1 for flexibility; bounds keep the sweep tiny.
        const A_MIN: u32 = 1;
        const A_MAX: u32 = 15;
        for a0 in A_MIN..=A_MAX {
            let k0 = 1u64 << a0;
            if k0 >= n_used as u64 {
                // Single tier would suffice; still allow the rest to be tiny.
                // We still need to fill four tiers, so just try minimal extras.
            }
            // cov0 covers ranks [0, min(k0, n_used))
            let end0 = (k0 as usize).min(n_used);
            let cov0 = prefix[end0];
            let rem0 = stream_n_u - cov0;
            // Early prune: if even with best-case rest we can't beat current best, skip.
            let head_bits = 2 * stream_n_u + cov0 * a0 as u64;
            if head_bits + rem0 >= best_bits { continue; }
            if end0 == n_used {
                // All tokens in tier 0; still need three more (unused) tiers.
                // Just pick minimal widths for the others.
                let total = head_bits;
                if total < best_bits {
                    best_bits = total;
                    best_widths = [a0, A_MIN, A_MIN, A_MIN];
                    let k1 = 1u32 << A_MIN;
                    best_cum = [0, k0 as u32, k0 as u32 + k1,
                                k0 as u32 + 2 * k1, k0 as u32 + 3 * k1];
                }
                continue;
            }
            for a1 in A_MIN..=A_MAX {
                let k1 = 1u64 << a1;
                let end1 = ((k0 + k1) as usize).min(n_used);
                let cov1 = prefix[end1] - cov0;
                let rem1 = stream_n_u - cov0 - cov1;
                let head_bits1 = head_bits + cov1 * a1 as u64;
                if head_bits1 + rem1 >= best_bits { continue; }
                if end1 == n_used {
                    let total = head_bits1;
                    if total < best_bits {
                        best_bits = total;
                        best_widths = [a0, a1, A_MIN, A_MIN];
                        let k2 = 1u32 << A_MIN;
                        best_cum = [0, k0 as u32, (k0 + k1) as u32,
                                    (k0 + k1) as u32 + k2,
                                    (k0 + k1) as u32 + 2 * k2];
                    }
                    continue;
                }
                for a2 in A_MIN..=A_MAX {
                    let k2 = 1u64 << a2;
                    let end2 = ((k0 + k1 + k2) as usize).min(n_used);
                    let cov2 = prefix[end2] - cov0 - cov1;
                    let rem2 = stream_n_u - cov0 - cov1 - cov2;
                    let head_bits2 = head_bits1 + cov2 * a2 as u64;
                    if head_bits2 + rem2 >= best_bits { continue; }
                    if end2 == n_used {
                        let total = head_bits2;
                        if total < best_bits {
                            best_bits = total;
                            best_widths = [a0, a1, a2, A_MIN];
                            let k3 = 1u32 << A_MIN;
                            best_cum = [0, k0 as u32, (k0 + k1) as u32,
                                        (k0 + k1 + k2) as u32,
                                        (k0 + k1 + k2) as u32 + k3];
                        }
                        continue;
                    }
                    // Need tier 3 to cover the rest.
                    let remaining = n_used as u64 - (k0 + k1 + k2);
                    if remaining == 0 { continue; }
                    // Smallest a3 with 2^a3 >= remaining.
                    let mut a3 = A_MIN;
                    while (1u64 << a3) < remaining && a3 < A_MAX { a3 += 1; }
                    if (1u64 << a3) < remaining { continue; } // can't cover
                    let cov3 = rem2; // all leftover tokens fall in tier 3
                    let total = head_bits2 + cov3 * a3 as u64;
                    if total < best_bits {
                        best_bits = total;
                        best_widths = [a0, a1, a2, a3];
                        let k3 = 1u64 << a3;
                        best_cum = [0, k0 as u32, (k0 + k1) as u32,
                                    (k0 + k1 + k2) as u32,
                                    (k0 + k1 + k2 + k3) as u32];
                    }
                }
            }
        }

        if best_bits == u64::MAX {
            // Degenerate fallback (e.g. n_used == 0).
            best_widths = [1, 1, 1, 1];
            best_cum = [0, 2, 4, 6, 8];
            best_bits = 0;
        }
        self.widths = best_widths;
        self.cum = best_cum;

        // Phase D: encode stream.
        let n_words = ((best_bits + 63) / 64) as usize + 2;
        self.stream_bits = vec![0u64; n_words];
        self.string_bit_offsets = vec![0u64; raw_string_boundaries.len()];
        let mut bit_pos: u64 = 0;
        let mut sidx: usize = 0;
        for (i, &id) in raw_stream.iter().enumerate() {
            while sidx + 1 < raw_string_boundaries.len()
                && raw_string_boundaries[sidx + 1] == i
            {
                sidx += 1;
                self.string_bit_offsets[sidx] = bit_pos;
            }
            let rank = self.id_to_rank[id as usize] as u32;
            // Locate tier.
            let tier = if rank < self.cum[1] { 0 }
                else if rank < self.cum[2] { 1 }
                else if rank < self.cum[3] { 2 }
                else { 3 };
            let payload = (rank - self.cum[tier]) as u64;
            Self::push_bits(&mut self.stream_bits, &mut bit_pos, tier as u64, 2);
            Self::push_bits(&mut self.stream_bits, &mut bit_pos, payload, self.widths[tier]);
        }
        while sidx + 1 < raw_string_boundaries.len() {
            sidx += 1;
            self.string_bit_offsets[sidx] = bit_pos;
        }
        self.stream_bit_len = bit_pos;
    }

    fn train(&self, data: &[u8], end_positions: &[usize]) -> (LongestPrefixMatcher<u16>, Vec<Vec<u8>>) {
        let mut dict_tokens: Vec<Vec<u8>> = vec![Vec::new(); 65536];
        let mut lpm = LongestPrefixMatcher::new();
        for i in 0..256u16 {
            dict_tokens[i as usize] = vec![i as u8];
            lpm.insert(&[i as u8], i);
        }
        let mut next_id: u32 = 256;
        let mut free_slots: Vec<u16> = Vec::new();
        let mut pair_freq: FxHashMap<(u16, u16), u32> = FxHashMap::default();
        let mut unigram: Vec<u32> = vec![0; 65536];
        let mut alive: Vec<bool> = vec![false; 65536];
        for i in 0..256 { alive[i] = true; }

        let tau_num = self.params.tau_num as u64;
        let tau_den = self.params.tau_den as u64;
        let min_unigram = self.params.min_unigram as u64;
        let threshold = self.params.threshold as u32;

        let mut shuffled: Vec<usize> = (0..end_positions.len() - 1).collect();
        for _pass in 0..self.params.passes.max(1) {
            shuffled.shuffle(&mut thread_rng());
            pair_freq.clear();

            for &idx in shuffled.iter() {
                let start = end_positions[idx];
                let end = end_positions[idx + 1];
                if start == end { continue; }

                let (mut prev_id, mut prev_len) = lpm.find_longest_match(&data[start..end]).unwrap();
                unigram[prev_id as usize] = unigram[prev_id as usize].saturating_add(1);
                let mut pos = start + prev_len;
                while pos < end {
                    let (cur_id, cur_len) = lpm.find_longest_match(&data[pos..end]).unwrap();
                    unigram[cur_id as usize] = unigram[cur_id as usize].saturating_add(1);

                    let key = (prev_id, cur_id);
                    let entry = pair_freq.entry(key).or_insert(0);
                    *entry += 1;

                    if *entry >= threshold {
                        let pair_count = *entry as u64;
                        pair_freq.remove(&key);

                        let new_id: u16 = if let Some(reused) = free_slots.pop() {
                            reused
                        } else if next_id < 65536 {
                            let id = next_id as u16;
                            next_id += 1;
                            id
                        } else {
                            prev_id = cur_id;
                            prev_len = cur_len;
                            pos += cur_len;
                            continue;
                        };

                        let merged: Vec<u8> = data[pos - prev_len..pos + cur_len].to_vec();
                        if !dict_tokens[new_id as usize].is_empty() {
                            let old = std::mem::take(&mut dict_tokens[new_id as usize]);
                            lpm.remove(&old);
                        }
                        lpm.insert(&merged, new_id);
                        dict_tokens[new_id as usize] = merged;
                        alive[new_id as usize] = true;
                        unigram[new_id as usize] = 0;

                        if tau_num > 0 {
                            for &cand in &[prev_id, cur_id] {
                                if (cand as usize) < 256 || !alive[cand as usize] { continue; }
                                let u = unigram[cand as usize] as u64;
                                if u < min_unigram { continue; }
                                if pair_count * tau_den >= u * tau_num {
                                    let bytes = std::mem::take(&mut dict_tokens[cand as usize]);
                                    lpm.remove(&bytes);
                                    alive[cand as usize] = false;
                                    free_slots.push(cand);
                                }
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
        (lpm, dict_tokens)
    }

    fn parse(&self, data: &[u8], end_positions: &[usize], lpm: &LongestPrefixMatcher<u16>) -> (Vec<u16>, Vec<usize>) {
        let mut stream: Vec<u16> = Vec::new();
        let mut boundaries: Vec<usize> = Vec::with_capacity(end_positions.len());
        boundaries.push(0);
        for w in end_positions.windows(2) {
            let (s, e) = (w[0], w[1]);
            let mut pos = s;
            while pos < e {
                let (id, len) = lpm.find_longest_match(&data[pos..e]).unwrap();
                stream.push(id);
                pos += len;
            }
            boundaries.push(stream.len());
        }
        (stream, boundaries)
    }

    #[inline]
    fn push_bits(buf: &mut [u64], bit_pos: &mut u64, value: u64, width: u32) {
        if width == 0 { return; }
        let bp = *bit_pos;
        let word_idx = (bp / 64) as usize;
        let bit_in_word = (bp % 64) as u32;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let masked = value & mask;
        buf[word_idx] |= masked << bit_in_word;
        let bits_into_next = (bit_in_word as i64 + width as i64) - 64;
        if bits_into_next > 0 {
            buf[word_idx + 1] |= masked >> (64 - bit_in_word);
        }
        *bit_pos = bp + width as u64;
    }

    #[inline]
    fn read_bits(buf: &[u64], bit_pos: u64, width: u32) -> u64 {
        if width == 0 { return 0; }
        let word_idx = (bit_pos / 64) as usize;
        let bit_in_word = (bit_pos % 64) as u32;
        let lo = buf[word_idx] >> bit_in_word;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        if bit_in_word + width <= 64 {
            lo & mask
        } else {
            let hi = buf[word_idx + 1] << (64 - bit_in_word);
            (lo | hi) & mask
        }
    }

    pub fn decompress_string(&self, index: usize, buffer: &mut [u8]) -> usize {
        let mut bit_pos = self.string_bit_offsets[index];
        let end_bit_pos = self.string_bit_offsets[index + 1];
        let mut size = 0;
        while bit_pos < end_bit_pos {
            let tier = Self::read_bits(&self.stream_bits, bit_pos, 2) as usize;
            bit_pos += 2;
            let w = self.widths[tier];
            let payload = Self::read_bits(&self.stream_bits, bit_pos, w) as u32;
            bit_pos += w as u64;
            let rank = (self.cum[tier] + payload) as usize;
            let start = self.token_boundaries[rank] as usize;
            let end = self.token_boundaries[rank + 1] as usize;
            let n = end - start;
            buffer[size..size + n].copy_from_slice(&self.dictionary[start..end]);
            size += n;
        }
        size
    }

    pub fn space_used(&self) -> usize {
        let stream_bytes = ((self.stream_bit_len + 7) / 8) as usize;
        let dict_bytes = self.dictionary.len();
        let bound_bytes = self.token_boundaries.len() * 4;
        stream_bytes + dict_bytes + bound_bytes
    }

    pub fn n_tokens(&self) -> usize { self.n_tokens }
    pub fn widths(&self) -> [u32; 4] { self.widths }
    pub fn tier_sizes(&self) -> [u32; 4] {
        [
            self.cum[1] - self.cum[0],
            self.cum[2] - self.cum[1],
            self.cum[3] - self.cum[2],
            self.cum[4] - self.cum[3],
        ]
    }
    pub fn stream_bit_len(&self) -> u64 { self.stream_bit_len }
    pub fn dictionary_bytes(&self) -> usize { self.dictionary.len() }
}
