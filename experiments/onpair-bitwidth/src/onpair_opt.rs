//! OnPairOpt: OnPair training plus a two-tier bit-packed token-stream code.
//!
//! Construction (training) optionally uses Picky-BPE-style intermediate-token
//! pruning (`tau_num / tau_den`, `min_unigram`). After the standard parse pass,
//! we rank tokens by stream frequency, choose a short-tier size `k` (power of
//! two), and re-encode the stream as:
//!
//!     for each token:
//!         1 flag bit: 0 = short tier, 1 = long tier
//!         then b1 = log2(k)        bits if short
//!              b2 = log2(N - k)    bits if long
//!
//! `k` is picked to minimise total packed bytes by scanning powers of two.
//!
//! Ratio accounting matches OnPair: stream bits packed to bytes, plus raw
//! dictionary bytes, plus 4 bytes per dictionary token for boundaries.
//! String-boundary bit-offsets are tracked but excluded from `space_used`,
//! same as the reference impl.

use crate::lpm::LongestPrefixMatcher;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy)]
pub struct OnPairOptParams {
    pub threshold: u16,
    pub tau_num: u32,
    pub tau_den: u32,
    pub min_unigram: u32,
    pub passes: u32,
    /// Force a specific log2(k); if None, pick the best automatically.
    pub force_log2_k: Option<u32>,
}

impl Default for OnPairOptParams {
    fn default() -> Self {
        Self {
            threshold: 2,
            tau_num: 80,
            tau_den: 100,
            min_unigram: 4,
            passes: 1,
            force_log2_k: None,
        }
    }
}

pub struct OnPairOpt {
    params: OnPairOptParams,
    /// Raw token bytes packed contiguously.
    dictionary: Vec<u8>,
    /// Boundary indices into `dictionary`, length = n_tokens + 1.
    token_boundaries: Vec<u32>,
    /// Map from frequency rank (0 = most frequent) to dictionary id.
    rank_to_id: Vec<u16>,
    /// Inverse mapping for encoding.
    id_to_rank: Vec<u16>,
    /// Selected log2(k) for the short tier.
    log2_k: u32,
    /// Bits per short / long code (b1 = log2_k, b2 = ceil(log2(n - 2^log2_k))).
    b1: u32,
    b2: u32,
    /// Total tokens in the dictionary (n_tokens).
    n_tokens: usize,
    /// Packed stream bits.
    stream_bits: Vec<u64>, // 64-bit words
    stream_bit_len: u64,
    /// Per-string bit offsets into stream_bits; length = n_strings + 1.
    string_bit_offsets: Vec<u64>,
}

impl OnPairOpt {
    pub fn new(params: OnPairOptParams) -> Self {
        Self {
            params,
            dictionary: Vec::new(),
            token_boundaries: Vec::new(),
            rank_to_id: Vec::new(),
            id_to_rank: Vec::new(),
            log2_k: 0,
            b1: 0,
            b2: 0,
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
        // Phase A: train (Picky-ish) + parse → token stream + dictionary.
        let (lpm, dict_tokens) = self.train(data, end_positions);
        let (raw_stream, raw_string_boundaries) = self.parse(data, end_positions, &lpm);

        // Phase B: build the final dictionary by compacting live tokens.
        // We renumber dict ids to 0..n_used contiguous range, ordered by stream frequency.
        let stream_n = raw_stream.len();
        let mut counts = vec![0u32; dict_tokens.len()];
        for &id in &raw_stream {
            counts[id as usize] += 1;
        }

        // Collect (count, id) for tokens that are either live (have bytes) AND used (>0 stream count),
        // OR are byte tokens (we must keep all 256 for parsing fallback even if some have count=0;
        // but byte tokens with count=0 contribute nothing — they're already in the dict though).
        // To match OnPair's accounting we keep only used tokens.
        let mut entries: Vec<(u32, u16)> = counts.iter().enumerate()
            .filter_map(|(i, &c)| if c > 0 { Some((c, i as u16)) } else { None })
            .collect();
        // Sort descending by count.
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let n_used = entries.len();
        self.n_tokens = n_used;
        self.rank_to_id = entries.iter().map(|&(_, id)| id).collect();

        // Build id_to_rank lookup (size = dict_tokens.len()).
        self.id_to_rank = vec![u16::MAX; dict_tokens.len()];
        for (rank, &id) in self.rank_to_id.iter().enumerate() {
            self.id_to_rank[id as usize] = rank as u16;
        }

        // Materialise the final dictionary bytes + boundaries in rank order.
        self.dictionary.clear();
        self.token_boundaries.clear();
        self.token_boundaries.push(0);
        for &id in &self.rank_to_id {
            let tok = &dict_tokens[id as usize];
            self.dictionary.extend_from_slice(tok);
            self.token_boundaries.push(self.dictionary.len() as u32);
        }

        // Phase C: pick best log2_k by exact byte counting.
        let mut sorted_counts: Vec<u32> = entries.iter().map(|&(c, _)| c).collect();
        // sorted_counts already in descending order.
        let _ = &mut sorted_counts;

        let stream_n_u = stream_n as u64;
        let mut best_log2_k = 0u32;
        let mut best_bits = u64::MAX;
        let try_log2_k = |log2_k: u32| -> u64 {
            let k = 1usize << log2_k;
            if k >= n_used { return u64::MAX; }
            let cov: u64 = sorted_counts[..k].iter().map(|&c| c as u64).sum();
            let rare = stream_n_u - cov;
            let b1 = log2_k as u64;
            let b2 = ((n_used - k) as f64).log2().ceil() as u64;
            let b2 = b2.max(1);
            // 1 flag bit per token + b1/b2 payload
            stream_n_u + cov * b1 + rare * b2
        };
        if let Some(fk) = self.params.force_log2_k {
            best_log2_k = fk;
            best_bits = try_log2_k(fk);
        } else {
            for lk in 1..16 {
                let bits = try_log2_k(lk);
                if bits < best_bits {
                    best_bits = bits;
                    best_log2_k = lk;
                }
            }
        }
        self.log2_k = best_log2_k;
        let k = 1usize << self.log2_k;
        self.b1 = self.log2_k;
        self.b2 = ((n_used - k) as f64).log2().ceil() as u32;
        if self.b2 == 0 { self.b2 = 1; }

        // Phase D: encode stream into bit buffer.
        let n_words = ((best_bits + 63) / 64) as usize + 2;
        self.stream_bits = vec![0u64; n_words];
        self.string_bit_offsets = vec![0u64; raw_string_boundaries.len()];
        let mut bit_pos: u64 = 0;
        let mut sidx: usize = 0; // index of the next boundary to fill (0 already = 0)
        // string i's encoded range is [string_bit_offsets[i], string_bit_offsets[i+1]).
        // Boundaries at the same stream index (i.e. empty strings) all share the same bit_pos.
        for (i, &id) in raw_stream.iter().enumerate() {
            while sidx + 1 < raw_string_boundaries.len()
                && raw_string_boundaries[sidx + 1] == i
            {
                sidx += 1;
                self.string_bit_offsets[sidx] = bit_pos;
            }
            let rank = self.id_to_rank[id as usize] as usize;
            if rank < k {
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, 0, 1);
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, rank as u64, self.b1);
            } else {
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, 1, 1);
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, (rank - k) as u64, self.b2);
            }
        }
        // Flush trailing boundaries (last string end + any trailing empties).
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

                        // Picky: try to evict prev / cur if their IoS is high.
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
        let bp = *bit_pos;
        let word_idx = (bp / 64) as usize;
        let bit_in_word = (bp % 64) as u32;
        let masked = value & ((1u64 << width) - 1);
        buf[word_idx] |= masked << bit_in_word;
        let bits_into_next = (bit_in_word as i64 + width as i64) - 64;
        if bits_into_next > 0 {
            buf[word_idx + 1] |= masked >> (64 - bit_in_word);
        }
        *bit_pos = bp + width as u64;
    }

    #[inline]
    fn read_bits(buf: &[u64], bit_pos: u64, width: u32) -> u64 {
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
        let k = 1usize << self.log2_k;
        let mut size = 0;
        while bit_pos < end_bit_pos {
            let flag = Self::read_bits(&self.stream_bits, bit_pos, 1);
            bit_pos += 1;
            let rank = if flag == 0 {
                Self::read_bits(&self.stream_bits, bit_pos, self.b1) as usize
            } else {
                k + Self::read_bits(&self.stream_bits, bit_pos, self.b2) as usize
            };
            bit_pos += if flag == 0 { self.b1 as u64 } else { self.b2 as u64 };
            let start = self.token_boundaries[rank] as usize;
            let end = self.token_boundaries[rank + 1] as usize;
            let n = end - start;
            buffer[size..size + n].copy_from_slice(&self.dictionary[start..end]);
            size += n;
        }
        size
    }

    pub fn space_used(&self) -> usize {
        // OnPair-style accounting: stream bytes + dict bytes + 4-byte boundaries.
        // Note: we EXCLUDE string_bit_offsets (matches OnPair's space_used).
        let stream_bytes = ((self.stream_bit_len + 7) / 8) as usize;
        let dict_bytes = self.dictionary.len();
        let bound_bytes = self.token_boundaries.len() * 4;
        stream_bytes + dict_bytes + bound_bytes
    }

    pub fn n_tokens(&self) -> usize { self.n_tokens }
    pub fn log2_k(&self) -> u32 { self.log2_k }
    pub fn b1(&self) -> u32 { self.b1 }
    pub fn b2(&self) -> u32 { self.b2 }
    pub fn stream_bit_len(&self) -> u64 { self.stream_bit_len }
    pub fn dictionary_bytes(&self) -> usize { self.dictionary.len() }
}
