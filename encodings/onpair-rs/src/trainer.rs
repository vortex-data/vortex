// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `src/onpair/encoding/training/trainer.cpp` plus the
// `DynamicThresholdController` from
// `include/onpair/encoding/training/dynamic_threshold.h`.

use hashbrown::HashMap;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::config::{ThresholdSpec, TrainingConfig};
use crate::dict::Dictionary;
use crate::lpm::LongestPrefixMatcher;
use crate::types::{MAX_TOKEN_SIZE, Token, max_dict_size};

/// Result of [`train`]: a sorted dictionary and a matching LPM whose token
/// IDs correspond to the dictionary's sorted order.
#[derive(Debug, Clone)]
pub struct TrainResult {
    pub dict: Dictionary,
    pub lpm: LongestPrefixMatcher,
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicThresholdController — adaptive merge threshold.
// ─────────────────────────────────────────────────────────────────────────────

struct DynamicThresholdController {
    capacity: usize,
    scan_budget: usize,
    check_interval: usize,
    threshold: u8,
    entries_created: usize,
    bytes_scanned: usize,
    entries_at_check: usize,
    bytes_at_check: usize,
    next_checkpoint: usize,
}

impl DynamicThresholdController {
    fn new(capacity: usize, total_bytes: usize, scan_fraction: f64) -> Self {
        let scan_budget = (total_bytes as f64 * scan_fraction) as usize;
        let check_interval = (capacity / 128).max(64);
        Self {
            capacity,
            scan_budget,
            check_interval,
            threshold: 2,
            entries_created: 0,
            bytes_scanned: 0,
            entries_at_check: 0,
            bytes_at_check: 0,
            next_checkpoint: check_interval,
        }
    }

    #[inline]
    fn get(&self) -> u8 {
        self.threshold
    }

    #[inline]
    fn budget_exhausted(&self) -> bool {
        self.bytes_scanned > self.scan_budget
    }

    #[inline]
    fn on_bytes_scanned(&mut self, n: usize) {
        self.bytes_scanned += n;
    }

    fn on_entry_created(&mut self) {
        self.entries_created += 1;
        if self.entries_created >= self.next_checkpoint {
            self.rebalance();
        }
    }

    fn rebalance(&mut self) {
        let delta_e = self.entries_created - self.entries_at_check;
        let delta_b = self.bytes_scanned - self.bytes_at_check;

        let recent_rate = if delta_b > 0 {
            delta_e as f64 / delta_b as f64
        } else {
            1e9
        };

        let e_rem = if self.capacity > self.entries_created {
            self.capacity - self.entries_created
        } else {
            1
        };
        let b_rem = if self.scan_budget > self.bytes_scanned {
            self.scan_budget - self.bytes_scanned
        } else {
            1
        };

        let target_rate = e_rem as f64 / b_rem as f64;
        let ratio = if target_rate > 0.0 {
            recent_rate / target_rate
        } else {
            1e9
        };

        if ratio > 2.0 && self.threshold < 255 {
            self.threshold += 1;
        } else if ratio < 0.5 {
            self.threshold = if self.threshold > 2 { self.threshold - 1 } else { 2 };
        }

        self.entries_at_check = self.entries_created;
        self.bytes_at_check = self.bytes_scanned;
        self.next_checkpoint = self.entries_created + self.check_interval;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// train()
// ─────────────────────────────────────────────────────────────────────────────

/// Discover merge tokens via frequency-threshold scanning, then sort the
/// dictionary lexicographically and pad it for decoder over-copy.
///
/// `offsets` has length `n + 1`; string `i` occupies
/// `data[offsets[i]..offsets[i + 1]]`.
pub fn train(data: &[u8], offsets: &[u32], n: usize, cfg: &TrainingConfig) -> TrainResult {
    let dict_capacity = max_dict_size(cfg.bits);

    // ── Initialise with the 256 single-byte base tokens ────────────────────
    let mut dict = Dictionary::default();
    dict.offsets.reserve(dict_capacity + 1);
    dict.bytes.reserve(dict_capacity * MAX_TOKEN_SIZE);
    dict.offsets.push(0);
    for i in 0u16..=255 {
        dict.bytes.push(i as u8);
        dict.offsets.push(dict.bytes.len() as u32);
    }
    let mut lpm = LongestPrefixMatcher::new();

    // ── Threshold setup ────────────────────────────────────────────────────
    let mut threshold: u8;
    let mut dyn_ctrl: Option<DynamicThresholdController> = None;
    match cfg.threshold {
        ThresholdSpec::Fixed(ft) => {
            threshold = ft.value;
        }
        ThresholdSpec::Dynamic(dt) => {
            let total_bytes = if n == 0 { 0 } else { offsets[n] as usize };
            let capacity = dict_capacity - 256;
            let ctrl = DynamicThresholdController::new(capacity, total_bytes, dt.sample_fraction);
            threshold = ctrl.get();
            dyn_ctrl = Some(ctrl);
        }
    }

    // ── Shuffle training order ─────────────────────────────────────────────
    // The C++ trainer uses `std::mt19937_64` with `std::shuffle`. Pure-Rust
    // bit-exact compatibility would require reimplementing both. We use the
    // default Rng (with deterministic seed) and document this as a known
    // divergence — cross-impl comparison tests assert structural equality
    // (decompression equivalence, predicate equivalence), not bit-exact
    // dictionary equality.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let seed = cfg.seed.unwrap_or_else(|| {
        use rand::RngExt;
        rand::rng().random()
    });
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    order.shuffle(&mut rng);

    // ── Pair frequency map. Key packs two Token values into a u32. ─────────
    let mut freq: HashMap<u32, u8> = HashMap::new();

    let mut full_dictionary = false;
    let mut budget_exhausted = false;

    for idx in order {
        if full_dictionary || budget_exhausted {
            break;
        }

        let s_start = offsets[idx as usize] as usize;
        let s_end = offsets[idx as usize + 1] as usize;
        if s_end == s_start {
            continue;
        }
        let str_bytes = &data[s_start..s_end];
        let len = str_bytes.len();

        // First match.
        let (mut prev_id, mut prev_len) = lpm.find_longest_match(str_bytes);
        let mut pos = prev_len;

        if let Some(ref mut dyn_) = dyn_ctrl {
            dyn_.on_bytes_scanned(prev_len);
            budget_exhausted = dyn_.budget_exhausted();
            if budget_exhausted {
                break;
            }
        }

        while pos < len {
            let (curr_id, curr_len) = lpm.find_longest_match(&str_bytes[pos..]);

            if let Some(ref mut dyn_) = dyn_ctrl {
                dyn_.on_bytes_scanned(curr_len);
                budget_exhausted = dyn_.budget_exhausted();
                if budget_exhausted {
                    break;
                }
            }

            let pair_len = prev_len + curr_len;

            if pair_len <= MAX_TOKEN_SIZE {
                let key = ((prev_id as u32) << 16) | (curr_id as u32);
                // Saturating increment branchless of C++: f += (f < 255).
                let f_slot = freq.entry(key).or_insert(0);
                *f_slot = f_slot.saturating_add(1);
                if *f_slot >= threshold {
                    // Merge: create new token for this pair.
                    let pair_start = pos - prev_len;
                    let pair_end = pos + curr_len;
                    let new_id = lpm.insert(&str_bytes[pair_start..pair_end]);
                    dict.bytes.extend_from_slice(&str_bytes[pair_start..pair_end]);
                    dict.offsets.push(dict.bytes.len() as u32);

                    if lpm.size() == dict_capacity {
                        full_dictionary = true;
                        break;
                    }

                    if let Some(ref mut dyn_) = dyn_ctrl {
                        dyn_.on_entry_created();
                        threshold = dyn_.get();
                    }

                    freq.remove(&key);
                    prev_id = new_id;
                    prev_len = pair_len;
                    pos += curr_len;
                    continue;
                }
            }

            prev_id = curr_id;
            prev_len = curr_len;
            pos += curr_len;
        }
    }

    let mut result = TrainResult { dict, lpm };
    sort_dictionary(&mut result);
    result.dict.pad_for_decoder();
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// sort_dictionary — internal helper.
//
// Sorts the dictionary lexicographically and rebuilds the LPM so token IDs
// match the new positions. Mirrors the same-named function in trainer.cpp.
// ─────────────────────────────────────────────────────────────────────────────

fn sort_dictionary(result: &mut TrainResult) {
    let n = result.dict.num_tokens();

    let mut perm: Vec<Token> = (0..n as Token).collect();
    perm.sort_by(|&a, &b| {
        let pa = result.dict.data(a);
        let pb = result.dict.data(b);
        pa.cmp(pb)
    });

    let mut sorted = Dictionary::default();
    sorted.bytes.reserve(result.dict.bytes_used());
    sorted.offsets.reserve(n + 1);
    sorted.offsets.push(0);

    for &old_id in &perm {
        let s = result.dict.span(old_id);
        sorted
            .bytes
            .extend_from_slice(&result.dict.bytes[s.begin as usize..s.end as usize]);
        sorted.offsets.push(sorted.bytes.len() as u32);
    }

    result.dict = sorted;
    result.lpm = LongestPrefixMatcher::from_dictionary(&result.dict);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — ported from `tests/encoding/test_trainer.cpp`.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::{DynamicThreshold, FixedThreshold};
    use crate::test_corpus::{
        alternating_strings as make_alternating_strings, binary_strings as make_binary_strings,
        fixed_length_strings as make_fixed_length_strings, homogeneous_strings as make_homogeneous_strings,
        make_raw, mixed_length_strings as make_mixed_length_strings,
        random_ascii_strings as make_random_strings, user_strings as make_user_strings,
    };

    fn train_strings<S: AsRef<[u8]>>(strings: &[S], cfg: &TrainingConfig) -> TrainResult {
        let raw = make_raw(strings);
        train(&raw.data, &raw.offsets, raw.n, cfg)
    }

    fn check_base_tokens(d: &Dictionary) {
        assert!(d.num_tokens() >= 256);
        let mut found = [false; 256];
        for i in 0..d.num_tokens() {
            let s = d.span(i as Token);
            if s.size() == 1 {
                found[d.bytes[s.begin as usize] as usize] = true;
            }
        }
        for (i, &f) in found.iter().enumerate() {
            assert!(f, "base token for byte {i} not found in dictionary");
        }
    }

    fn is_lex_sorted(d: &Dictionary) -> bool {
        let n = d.num_tokens();
        for i in 1..n {
            let a = d.data((i - 1) as Token);
            let b = d.data(i as Token);
            if a > b {
                return false;
            }
        }
        true
    }

    // ── Baseline invariant ─────────────────────────────────────────────────

    #[test]
    fn base_tokens_always_single_bytes() {
        let result = train_strings(&make_user_strings(50), &TrainingConfig::default());
        check_base_tokens(&result.dict);
    }

    #[test]
    fn base_tokens_on_empty_input() {
        let data: Vec<u8> = vec![];
        let offsets = vec![0u32];
        let result = train(&data, &offsets, 0, &TrainingConfig::default());
        check_base_tokens(&result.dict);
        assert_eq!(result.dict.num_tokens(), 256);
    }

    #[test]
    fn base_tokens_on_single_empty_string() {
        let data: Vec<u8> = vec![];
        let offsets = vec![0u32, 0];
        let result = train(&data, &offsets, 1, &TrainingConfig::default());
        check_base_tokens(&result.dict);
        assert_eq!(result.dict.num_tokens(), 256);
    }

    // ── Dictionary size bounds ─────────────────────────────────────────────

    #[test]
    fn dictionary_size_does_not_exceed_capacity() {
        let cfg = TrainingConfig {
            bits: 12,
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
        };
        let result = train_strings(&make_user_strings(500), &cfg);
        assert!(result.dict.num_tokens() <= max_dict_size(cfg.bits));
    }

    // ── FixedThreshold ─────────────────────────────────────────────────────

    #[test]
    fn threshold_gates_merges() {
        // 100 copies of "ab": pair (a,b) appears exactly 100 times.
        let corpus: Vec<&str> = (0..100).map(|_| "ab").collect();

        let cfg_low = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        assert!(train_strings(&corpus, &cfg_low).dict.num_tokens() > 256);

        let cfg_high = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 101 }),
            seed: Some(42),
            ..Default::default()
        };
        assert_eq!(train_strings(&corpus, &cfg_high).dict.num_tokens(), 256);
    }

    #[test]
    fn fixed_threshold_2_merges_frequent_pairs() {
        let corpus: Vec<&str> = (0..50).map(|_| "aabb").collect();
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        assert!(train_strings(&corpus, &cfg).dict.num_tokens() > 256);
    }

    #[test]
    fn merged_token_content_is_correct() {
        let corpus: Vec<&str> = (0..50).map(|_| "ab").collect();
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        let result = train_strings(&corpus, &cfg);
        let found = (0..result.dict.num_tokens()).any(|i| {
            let s = result.dict.data(i as Token);
            s == b"ab"
        });
        assert!(found, "merged token \"ab\" not found in dictionary");
    }

    // ── Seed reproducibility ───────────────────────────────────────────────

    #[test]
    fn same_seed_produces_identical_dictionaries() {
        let corpus = make_random_strings(100, 40, 12345);
        let cfg = TrainingConfig { seed: Some(42), ..Default::default() };
        let r1 = train_strings(&corpus, &cfg);
        let r2 = train_strings(&corpus, &cfg);
        assert_eq!(r1.dict.num_tokens(), r2.dict.num_tokens());
        assert_eq!(r1.dict.bytes, r2.dict.bytes);
        assert_eq!(r1.dict.offsets, r2.dict.offsets);
    }

    // ── sort_dictionary / LPM remap ────────────────────────────────────────

    #[test]
    fn dictionary_is_always_sorted() {
        let result = train_strings(&make_user_strings(100), &TrainingConfig::default());
        assert!(is_lex_sorted(&result.dict));
    }

    #[test]
    fn lpm_remaps_correctly() {
        let strings = make_user_strings(30);
        let result = train_strings(&strings, &TrainingConfig::default());
        let n = result.dict.num_tokens();
        for id in 0..n {
            let bytes = result.dict.data(id as Token);
            let (tok, len) = result.lpm.find_longest_match(bytes);
            assert_eq!(tok, id as Token, "ID mismatch for token {id}");
            assert_eq!(len, bytes.len(), "length mismatch for token {id}");
        }
    }

    // ── Token byte length ──────────────────────────────────────────────────

    #[test]
    fn no_token_exceeds_max_token_size() {
        let strings = make_random_strings(100, 50, 99);
        let result = train_strings(&strings, &TrainingConfig::default());
        for i in 0..result.dict.num_tokens() {
            let len = result.dict.token_size(i as Token);
            assert!(len <= MAX_TOKEN_SIZE, "token {i} exceeds MAX_TOKEN_SIZE");
        }
    }

    #[test]
    fn no_token_has_zero_length() {
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        let corpora: Vec<(&str, Vec<Vec<u8>>)> = vec![
            ("random", make_random_strings(100, 50, 77)),
            (
                "user",
                make_user_strings(50).into_iter().map(|s| s.into_bytes()).collect(),
            ),
            ("binary", make_binary_strings(50, 30, 13)),
            ("fixed_len", make_fixed_length_strings(20, MAX_TOKEN_SIZE)),
        ];
        for (name, c) in &corpora {
            let result = train_strings(c, &cfg);
            for i in 0..result.dict.num_tokens() {
                let len = result.dict.token_size(i as Token);
                assert!(len > 0, "corpus={name} token {i} has zero length");
            }
        }
    }

    // ── DynamicThreshold ───────────────────────────────────────────────────

    #[test]
    fn dynamic_threshold_produces_merged_tokens() {
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Dynamic(DynamicThreshold { sample_fraction: 0.5 }),
            seed: Some(42),
            ..Default::default()
        };
        let result = train_strings(&make_user_strings(200), &cfg);
        assert!(result.dict.num_tokens() > 256);
    }

    #[test]
    fn dynamic_threshold_does_not_exceed_capacity() {
        let cfg = TrainingConfig {
            bits: 12,
            threshold: ThresholdSpec::Dynamic(DynamicThreshold { sample_fraction: 1.0 }),
            seed: Some(42),
        };
        let result = train_strings(&make_user_strings(500), &cfg);
        assert!(result.dict.num_tokens() <= max_dict_size(cfg.bits));
    }

    #[test]
    fn dynamic_threshold_smaller_fraction_produces_fewer_tokens() {
        let corpus = make_user_strings(500);

        let cfg_small = TrainingConfig {
            bits: 14,
            threshold: ThresholdSpec::Dynamic(DynamicThreshold { sample_fraction: 0.05 }),
            seed: Some(42),
        };
        let cfg_large = TrainingConfig {
            bits: 14,
            threshold: ThresholdSpec::Dynamic(DynamicThreshold { sample_fraction: 1.0 }),
            seed: Some(42),
        };

        let r_small = train_strings(&corpus, &cfg_small);
        let r_large = train_strings(&corpus, &cfg_large);
        assert!(r_small.dict.num_tokens() <= r_large.dict.num_tokens());
    }

    #[test]
    fn dynamic_threshold_dictionary_is_sorted() {
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Dynamic(DynamicThreshold { sample_fraction: 0.3 }),
            seed: Some(42),
            ..Default::default()
        };
        let result = train_strings(&make_user_strings(100), &cfg);
        assert!(is_lex_sorted(&result.dict));
    }

    // ── Dictionary is padded for decoder ───────────────────────────────────

    #[test]
    fn dictionary_is_padded_for_decoder() {
        let result = train_strings(&make_user_strings(50), &TrainingConfig::default());
        let last_start = result.dict.offsets[result.dict.offsets.len() - 2] as usize;
        assert!(result.dict.bytes.len() >= last_start + MAX_TOKEN_SIZE);
    }

    // ── No duplicate tokens ────────────────────────────────────────────────

    #[test]
    fn no_duplicate_tokens_in_dictionary() {
        let result = train_strings(&make_user_strings(100), &TrainingConfig::default());
        let n = result.dict.num_tokens();
        for i in 1..n {
            let a = result.dict.data((i - 1) as Token);
            let b = result.dict.data(i as Token);
            assert!(a != b, "duplicate token at positions {} and {}", i - 1, i);
        }
    }

    // ── Corpus type coverage ───────────────────────────────────────────────

    #[test]
    fn homogeneous_corpus_produces_merges() {
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        let result = train_strings(&make_homogeneous_strings(50, 16, b'a'), &cfg);
        assert!(result.dict.num_tokens() > 256);
        check_base_tokens(&result.dict);
    }

    #[test]
    fn alternating_corpus_produces_merges() {
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        let result = train_strings(&make_alternating_strings(50, 16), &cfg);
        assert!(result.dict.num_tokens() > 256);
        check_base_tokens(&result.dict);
    }

    #[test]
    fn mixed_length_corpus_produces_valid_dictionary() {
        let cfg = TrainingConfig {
            threshold: ThresholdSpec::Fixed(FixedThreshold { value: 2 }),
            seed: Some(42),
            ..Default::default()
        };
        let result = train_strings(&make_mixed_length_strings(200, 64, 7), &cfg);
        check_base_tokens(&result.dict);
        assert!(is_lex_sorted(&result.dict));
        assert!(result.dict.num_tokens() <= max_dict_size(cfg.bits));
    }

    // ── All bit widths produce valid dictionary ────────────────────────────

    #[test]
    fn all_bit_widths_produce_valid_dictionary() {
        let corpus = make_user_strings(50);
        for b in 9u8..=16 {
            let cfg = TrainingConfig { bits: b, seed: Some(42), ..Default::default() };
            let result = train_strings(&corpus, &cfg);
            check_base_tokens(&result.dict);
            assert!(is_lex_sorted(&result.dict), "not sorted for bits={b}");
            assert!(result.dict.num_tokens() <= max_dict_size(b), "overflow for bits={b}");
        }
    }
}
