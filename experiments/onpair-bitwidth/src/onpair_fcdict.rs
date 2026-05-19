//! OnPairFcDict: same training/parsing/stream layout as OnPairOpt, but the
//! dictionary bytes are stored using *plain front coding (PFC)* over a
//! lex-sorted bucketed dictionary.
//!
//! Front-coded dictionary layout:
//!   * Tokens are first sorted lexicographically.
//!   * They are grouped into buckets of B (bucket size). Within a bucket the
//!     first ("header") token is stored verbatim with a 1-byte length prefix,
//!     and the remaining B-1 tokens are stored as (LCP-length byte, suffix-length
//!     byte, suffix bytes) relative to the previous token in the bucket.
//!   * A 4-byte offset per bucket gives O(1) random access to the bucket head.
//!   * A `freq_rank -> lex_rank` permutation (2 bytes per token) is stored so
//!     the stream (which encodes frequency ranks) can still look up token
//!     bytes.
//!
//! The decoder, when asked for `freq_rank`, looks up the lex rank, computes
//! `bucket = lex_rank / B` and `offset = lex_rank % B`, seeks to the bucket
//! head and decodes forward `offset` steps to materialise the requested token.
//!
//! Space accounting:
//!   stream_bytes
//!   + front_coded_bytes
//!   + 4 * n_buckets        (bucket offsets)
//!   + 2 * n_used           (permutation)
//!
//! Stream layout (b1/b2/log2_k, flag bit, etc.) is identical to OnPairOpt so
//! we only change the dictionary side of accounting.

use crate::lpm::LongestPrefixMatcher;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy)]
pub struct OnPairFcDictParams {
    pub threshold: u16,
    pub tau_num: u32,
    pub tau_den: u32,
    pub min_unigram: u32,
    pub passes: u32,
    pub force_log2_k: Option<u32>,
    /// Force a specific bucket size; if None, sweep {8,16,32,64} and pick best.
    pub force_bucket: Option<u32>,
}

impl Default for OnPairFcDictParams {
    fn default() -> Self {
        Self {
            threshold: 2,
            tau_num: 80,
            tau_den: 100,
            min_unigram: 4,
            passes: 1,
            force_log2_k: None,
            force_bucket: None,
        }
    }
}

pub struct OnPairFcDict {
    params: OnPairFcDictParams,
    // ---- Front-coded dictionary ----
    /// Encoded byte buffer: per bucket head we store (1-byte length, bytes);
    /// per non-head we store (1-byte LCP length, 1-byte suffix length, suffix bytes).
    fc_bytes: Vec<u8>,
    /// Byte offset of the start of each bucket within `fc_bytes`. Length = n_buckets + 1.
    bucket_offsets: Vec<u32>,
    /// Permutation: freq_rank -> lex_rank.
    freq_to_lex: Vec<u16>,
    bucket_size: u32,
    /// Maximum token length (for verifying lengths fit in a byte).
    max_token_len: usize,

    // ---- Stream side (identical to OnPairOpt) ----
    log2_k: u32,
    b1: u32,
    b2: u32,
    n_tokens: usize,
    stream_bits: Vec<u64>,
    stream_bit_len: u64,
    string_bit_offsets: Vec<u64>,
}

impl OnPairFcDict {
    pub fn new(params: OnPairFcDictParams) -> Self {
        Self {
            params,
            fc_bytes: Vec::new(),
            bucket_offsets: Vec::new(),
            freq_to_lex: Vec::new(),
            bucket_size: 0,
            max_token_len: 0,
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
        let (lpm, dict_tokens) = self.train(data, end_positions);
        let (raw_stream, raw_string_boundaries) = self.parse(data, end_positions, &lpm);

        let stream_n = raw_stream.len();
        let mut counts = vec![0u32; dict_tokens.len()];
        for &id in &raw_stream {
            counts[id as usize] += 1;
        }

        // Frequency-sorted live tokens.
        let mut entries: Vec<(u32, u16)> = counts.iter().enumerate()
            .filter_map(|(i, &c)| if c > 0 { Some((c, i as u16)) } else { None })
            .collect();
        entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let n_used = entries.len();
        self.n_tokens = n_used;
        // freq_rank -> dict_id
        let freq_to_id: Vec<u16> = entries.iter().map(|&(_, id)| id).collect();

        // id_to_freq_rank for stream encoding.
        let mut id_to_freq_rank = vec![u16::MAX; dict_tokens.len()];
        for (rank, &id) in freq_to_id.iter().enumerate() {
            id_to_freq_rank[id as usize] = rank as u16;
        }

        // Compute lex-sorted order over freq_ranks (not raw ids).
        // sorted_freq_ranks[i] gives the freq_rank whose token comes i-th in lex order.
        let mut sorted_freq_ranks: Vec<u32> = (0..n_used as u32).collect();
        sorted_freq_ranks.sort_unstable_by(|&a, &b| {
            let ta = &dict_tokens[freq_to_id[a as usize] as usize];
            let tb = &dict_tokens[freq_to_id[b as usize] as usize];
            ta.cmp(tb)
        });

        // freq_to_lex: index by freq_rank, output is lex_rank.
        self.freq_to_lex = vec![0u16; n_used];
        for (lex_rank, &freq_rank) in sorted_freq_ranks.iter().enumerate() {
            self.freq_to_lex[freq_rank as usize] = lex_rank as u16;
        }

        // Materialise lex-sorted token bytes for FC encoding.
        let lex_sorted_tokens: Vec<&[u8]> = sorted_freq_ranks.iter()
            .map(|&fr| dict_tokens[freq_to_id[fr as usize] as usize].as_slice())
            .collect();

        self.max_token_len = lex_sorted_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        assert!(self.max_token_len <= 255,
            "max_token_len {} exceeds 255 -- 1-byte length prefix insufficient",
            self.max_token_len);

        // Pick best bucket size.
        let bucket_candidates: Vec<u32> = if let Some(b) = self.params.force_bucket {
            vec![b]
        } else {
            vec![4, 8, 16, 32, 64, 128]
        };

        let mut best_b = bucket_candidates[0];
        let mut best_total = usize::MAX;
        let mut best_fc: Vec<u8> = Vec::new();
        let mut best_offsets: Vec<u32> = Vec::new();
        for &b in &bucket_candidates {
            let (fc, offs) = encode_front_coded(&lex_sorted_tokens, b as usize);
            // total dict-side bytes: fc + offsets (4 bytes each) + permutation (2 bytes each)
            let dict_bytes = fc.len() + offs.len() * 4 + n_used * 2;
            if dict_bytes < best_total {
                best_total = dict_bytes;
                best_b = b;
                best_fc = fc;
                best_offsets = offs;
            }
        }
        self.bucket_size = best_b;
        self.fc_bytes = best_fc;
        self.bucket_offsets = best_offsets;

        // ---- Stream encode: identical logic to OnPairOpt ----
        let mut sorted_counts: Vec<u32> = entries.iter().map(|&(c, _)| c).collect();
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
            let rank = id_to_freq_rank[id as usize] as usize;
            if rank < k {
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, 0, 1);
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, rank as u64, self.b1);
            } else {
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, 1, 1);
                Self::push_bits(&mut self.stream_bits, &mut bit_pos, (rank - k) as u64, self.b2);
            }
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

    /// Decode a token by its lex rank, writing its bytes into `out` and
    /// returning the number of bytes written. `scratch` is used to hold the
    /// previous-in-bucket token while walking forward.
    fn decode_lex_token(&self, lex_rank: usize, scratch: &mut Vec<u8>) -> usize {
        let b = self.bucket_size as usize;
        let bucket = lex_rank / b;
        let off = lex_rank % b;
        let mut p = self.bucket_offsets[bucket] as usize;
        // Read header token: 1-byte length + bytes.
        let head_len = self.fc_bytes[p] as usize;
        p += 1;
        scratch.clear();
        scratch.extend_from_slice(&self.fc_bytes[p..p + head_len]);
        p += head_len;
        for _ in 0..off {
            // (lcp, suf_len, suf_bytes)
            let lcp = self.fc_bytes[p] as usize;
            let suf_len = self.fc_bytes[p + 1] as usize;
            p += 2;
            scratch.truncate(lcp);
            scratch.extend_from_slice(&self.fc_bytes[p..p + suf_len]);
            p += suf_len;
        }
        scratch.len()
    }

    pub fn decompress_string(&self, index: usize, buffer: &mut [u8]) -> usize {
        let mut bit_pos = self.string_bit_offsets[index];
        let end_bit_pos = self.string_bit_offsets[index + 1];
        let k = 1usize << self.log2_k;
        let mut size = 0;
        let mut scratch: Vec<u8> = Vec::with_capacity(self.max_token_len);
        while bit_pos < end_bit_pos {
            let flag = Self::read_bits(&self.stream_bits, bit_pos, 1);
            bit_pos += 1;
            let rank = if flag == 0 {
                Self::read_bits(&self.stream_bits, bit_pos, self.b1) as usize
            } else {
                k + Self::read_bits(&self.stream_bits, bit_pos, self.b2) as usize
            };
            bit_pos += if flag == 0 { self.b1 as u64 } else { self.b2 as u64 };
            let lex = self.freq_to_lex[rank] as usize;
            let n = self.decode_lex_token(lex, &mut scratch);
            buffer[size..size + n].copy_from_slice(&scratch);
            size += n;
        }
        size
    }

    pub fn space_used(&self) -> usize {
        let stream_bytes = ((self.stream_bit_len + 7) / 8) as usize;
        let fc = self.fc_bytes.len();
        let bucket_off_bytes = self.bucket_offsets.len() * 4;
        let perm_bytes = self.freq_to_lex.len() * 2;
        stream_bytes + fc + bucket_off_bytes + perm_bytes
    }

    // --- Inspection accessors ---
    pub fn n_tokens(&self) -> usize { self.n_tokens }
    pub fn log2_k(&self) -> u32 { self.log2_k }
    pub fn b1(&self) -> u32 { self.b1 }
    pub fn b2(&self) -> u32 { self.b2 }
    pub fn stream_bit_len(&self) -> u64 { self.stream_bit_len }
    pub fn bucket_size(&self) -> u32 { self.bucket_size }
    pub fn fc_bytes_len(&self) -> usize { self.fc_bytes.len() }
    pub fn bucket_offsets_len(&self) -> usize { self.bucket_offsets.len() }
    pub fn perm_bytes(&self) -> usize { self.freq_to_lex.len() * 2 }
    pub fn max_token_len(&self) -> usize { self.max_token_len }
}

/// Encode a lex-sorted slice of tokens with plain front coding using bucket size `b`.
/// Returns `(fc_bytes, bucket_offsets)`. `bucket_offsets.len() == n_buckets + 1`.
fn encode_front_coded(tokens: &[&[u8]], b: usize) -> (Vec<u8>, Vec<u32>) {
    let n = tokens.len();
    let n_buckets = (n + b - 1) / b;
    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(n_buckets + 1);
    for bi in 0..n_buckets {
        offsets.push(out.len() as u32);
        let start = bi * b;
        let end = (start + b).min(n);
        // Header token: 1-byte length + bytes.
        let head = tokens[start];
        debug_assert!(head.len() <= 255);
        out.push(head.len() as u8);
        out.extend_from_slice(head);
        let mut prev = head;
        for i in start + 1..end {
            let cur = tokens[i];
            let max_lcp = prev.len().min(cur.len()).min(255);
            let mut lcp = 0;
            while lcp < max_lcp && prev[lcp] == cur[lcp] {
                lcp += 1;
            }
            let suf_len = cur.len() - lcp;
            debug_assert!(suf_len <= 255);
            out.push(lcp as u8);
            out.push(suf_len as u8);
            out.extend_from_slice(&cur[lcp..]);
            prev = cur;
        }
    }
    offsets.push(out.len() as u32);
    (out, offsets)
}
