//! OnPairCombined: Picky-BPE training + 4-tier bit-packed stream code +
//! front-coded sorted dictionary.
//!
//! Fuses the two winners from the parallel experiment sweep:
//!   * `onpair_4tier`: 2-bit prefix + four power-of-two width tiers,
//!     partition chosen by brute-force sweep to minimise total stream bits.
//!   * `onpair_fcdict`: lex-sorted bucketed plain front-coded dictionary
//!     with a `freq_rank -> lex_rank` permutation.
//!
//! Decompression: read 2-bit tier + payload from stream → frequency rank →
//! permutation → lex rank → bucket lookup + sequential FC walk.

use crate::lpm::LongestPrefixMatcher;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustc_hash::FxHashMap;

use super::onpair_opt::OnPairOptParams;

pub struct OnPairCombined {
    params: OnPairOptParams,
    // ---- Front-coded dictionary ----
    fc_bytes: Vec<u8>,
    bucket_offsets: Vec<u32>,
    freq_to_lex: Vec<u16>,
    bucket_size: u32,
    max_token_len: usize,
    // ---- 4-tier stream ----
    widths: [u32; 4],
    cum: [u32; 5],
    n_tokens: usize,
    stream_bits: Vec<u64>,
    stream_bit_len: u64,
    string_bit_offsets: Vec<u64>,
}

impl OnPairCombined {
    pub fn new(params: OnPairOptParams) -> Self {
        Self {
            params,
            fc_bytes: Vec::new(),
            bucket_offsets: Vec::new(),
            freq_to_lex: Vec::new(),
            bucket_size: 0,
            max_token_len: 0,
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
        let (lpm, dict_tokens) = self.train(data, end_positions);
        let (raw_stream, raw_string_boundaries) = self.parse(data, end_positions, &lpm);

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
        let freq_to_id: Vec<u16> = entries.iter().map(|&(_, id)| id).collect();

        // id -> freq rank
        let mut id_to_freq_rank = vec![u16::MAX; dict_tokens.len()];
        for (rank, &id) in freq_to_id.iter().enumerate() {
            id_to_freq_rank[id as usize] = rank as u16;
        }

        // ---- Build lex-sorted permutation ----
        let mut sorted_freq_ranks: Vec<u32> = (0..n_used as u32).collect();
        sorted_freq_ranks.sort_unstable_by(|&a, &b| {
            let ta = &dict_tokens[freq_to_id[a as usize] as usize];
            let tb = &dict_tokens[freq_to_id[b as usize] as usize];
            ta.cmp(tb)
        });
        self.freq_to_lex = vec![0u16; n_used];
        for (lex_rank, &freq_rank) in sorted_freq_ranks.iter().enumerate() {
            self.freq_to_lex[freq_rank as usize] = lex_rank as u16;
        }

        let lex_sorted_tokens: Vec<&[u8]> = sorted_freq_ranks.iter()
            .map(|&fr| dict_tokens[freq_to_id[fr as usize] as usize].as_slice())
            .collect();
        self.max_token_len = lex_sorted_tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        assert!(self.max_token_len <= 255, "max_token_len {} > 255", self.max_token_len);

        // Pick best bucket size (sweep).
        let bucket_candidates: [u32; 6] = [4, 8, 16, 32, 64, 128];
        let mut best_b: u32 = bucket_candidates[0];
        let mut best_total = usize::MAX;
        let mut best_fc: Vec<u8> = Vec::new();
        let mut best_offsets: Vec<u32> = Vec::new();
        for &b in &bucket_candidates {
            let (fc, offs) = encode_front_coded(&lex_sorted_tokens, b as usize);
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

        // ---- 4-tier partition sweep ----
        let sorted_counts: Vec<u32> = entries.iter().map(|&(c, _)| c).collect();
        let mut prefix = vec![0u64; n_used + 1];
        for i in 0..n_used {
            prefix[i + 1] = prefix[i] + sorted_counts[i] as u64;
        }
        let stream_n_u = stream_n as u64;
        let mut best_widths = [0u32; 4];
        let mut best_cum = [0u32; 5];
        let mut best_bits = u64::MAX;
        const A_MIN: u32 = 1;
        const A_MAX: u32 = 15;
        for a0 in A_MIN..=A_MAX {
            let k0 = 1u64 << a0;
            let end0 = (k0 as usize).min(n_used);
            let cov0 = prefix[end0];
            let rem0 = stream_n_u - cov0;
            let head_bits = 2 * stream_n_u + cov0 * a0 as u64;
            if head_bits + rem0 >= best_bits { continue; }
            if end0 == n_used {
                if head_bits < best_bits {
                    best_bits = head_bits;
                    best_widths = [a0, A_MIN, A_MIN, A_MIN];
                    let k1 = 1u32 << A_MIN;
                    best_cum = [0, k0 as u32, k0 as u32 + k1, k0 as u32 + 2 * k1, k0 as u32 + 3 * k1];
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
                    if head_bits1 < best_bits {
                        best_bits = head_bits1;
                        best_widths = [a0, a1, A_MIN, A_MIN];
                        let k2 = 1u32 << A_MIN;
                        best_cum = [0, k0 as u32, (k0 + k1) as u32, (k0 + k1) as u32 + k2, (k0 + k1) as u32 + 2 * k2];
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
                        if head_bits2 < best_bits {
                            best_bits = head_bits2;
                            best_widths = [a0, a1, a2, A_MIN];
                            let k3 = 1u32 << A_MIN;
                            best_cum = [0, k0 as u32, (k0 + k1) as u32, (k0 + k1 + k2) as u32, (k0 + k1 + k2) as u32 + k3];
                        }
                        continue;
                    }
                    let remaining = n_used as u64 - (k0 + k1 + k2);
                    if remaining == 0 { continue; }
                    let mut a3 = A_MIN;
                    while (1u64 << a3) < remaining && a3 < A_MAX { a3 += 1; }
                    if (1u64 << a3) < remaining { continue; }
                    let cov3 = rem2;
                    let total = head_bits2 + cov3 * a3 as u64;
                    if total < best_bits {
                        best_bits = total;
                        best_widths = [a0, a1, a2, a3];
                        let k3 = 1u64 << a3;
                        best_cum = [0, k0 as u32, (k0 + k1) as u32, (k0 + k1 + k2) as u32, (k0 + k1 + k2 + k3) as u32];
                    }
                }
            }
        }
        if best_bits == u64::MAX {
            best_widths = [1, 1, 1, 1];
            best_cum = [0, 2, 4, 6, 8];
            best_bits = 0;
        }
        self.widths = best_widths;
        self.cum = best_cum;

        // ---- Encode stream (4-tier) ----
        let n_words = ((best_bits + 63) / 64) as usize + 2;
        self.stream_bits = vec![0u64; n_words];
        self.string_bit_offsets = vec![0u64; raw_string_boundaries.len()];
        let mut bit_pos: u64 = 0;
        let mut sidx: usize = 0;
        for (i, &id) in raw_stream.iter().enumerate() {
            while sidx + 1 < raw_string_boundaries.len() && raw_string_boundaries[sidx + 1] == i {
                sidx += 1;
                self.string_bit_offsets[sidx] = bit_pos;
            }
            let rank = id_to_freq_rank[id as usize] as u32;
            let tier = if rank < self.cum[1] { 0 }
                else if rank < self.cum[2] { 1 }
                else if rank < self.cum[3] { 2 }
                else { 3 };
            let payload = (rank - self.cum[tier]) as u64;
            push_bits(&mut self.stream_bits, &mut bit_pos, tier as u64, 2);
            push_bits(&mut self.stream_bits, &mut bit_pos, payload, self.widths[tier]);
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
                        if merged.len() > 255 {
                            // Front-coding uses 1-byte length prefixes; skip merges that don't fit.
                            // Restore the free slot for later use.
                            if next_id - 1 == new_id as u32 {
                                next_id -= 1;
                            } else {
                                free_slots.push(new_id);
                            }
                            prev_id = cur_id;
                            prev_len = cur_len;
                            pos += cur_len;
                            continue;
                        }
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

    fn decode_lex_token(&self, lex_rank: usize, scratch: &mut Vec<u8>) -> usize {
        let b = self.bucket_size as usize;
        let bucket = lex_rank / b;
        let off = lex_rank % b;
        let mut p = self.bucket_offsets[bucket] as usize;
        let head_len = self.fc_bytes[p] as usize;
        p += 1;
        scratch.clear();
        scratch.extend_from_slice(&self.fc_bytes[p..p + head_len]);
        p += head_len;
        for _ in 0..off {
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
        let mut size = 0;
        let mut scratch: Vec<u8> = Vec::with_capacity(self.max_token_len);
        while bit_pos < end_bit_pos {
            let tier = read_bits(&self.stream_bits, bit_pos, 2) as usize;
            bit_pos += 2;
            let w = self.widths[tier];
            let payload = read_bits(&self.stream_bits, bit_pos, w) as u32;
            bit_pos += w as u64;
            let rank = (self.cum[tier] + payload) as usize;
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
    pub fn bucket_size(&self) -> u32 { self.bucket_size }
    pub fn fc_bytes_len(&self) -> usize { self.fc_bytes.len() }
    pub fn bucket_offsets_len(&self) -> usize { self.bucket_offsets.len() }
    pub fn perm_bytes(&self) -> usize { self.freq_to_lex.len() * 2 }
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

fn encode_front_coded(tokens: &[&[u8]], b: usize) -> (Vec<u8>, Vec<u32>) {
    let n = tokens.len();
    let n_buckets = (n + b - 1) / b;
    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::with_capacity(n_buckets + 1);
    for bi in 0..n_buckets {
        offsets.push(out.len() as u32);
        let start = bi * b;
        let end = (start + b).min(n);
        let head = tokens[start];
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
            out.push(lcp as u8);
            out.push(suf_len as u8);
            out.extend_from_slice(&cur[lcp..]);
            prev = cur;
        }
    }
    offsets.push(out.len() as u32);
    (out, offsets)
}
