// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/encoding/lpm.h`.
//
// Two-tier storage mirroring the C++ design:
//   * **short map** — tokens of length 1..=8 keyed by their bytes packed
//     into a `u64` plus the length. Almost all dictionary entries land here
//     (256 single-byte base tokens + the BPE merges, which tend to stay
//     short on real data).
//   * **long map** — tokens of length 9..=16 keyed by `(u128, u8)`.
//
// `find_longest_match` reads up to 16 bytes from the input as a `u128`, then
// probes the long map for lengths `min(max_len, 16) .. 9` (only if at least
// 9 bytes are available) and falls through to the short map for lengths
// `min(max_len, 8) .. 1`. Most calls return after one or two short-map
// hits; the long map is only consulted for inputs with a multi-byte token
// match.

use hashbrown::HashMap;

use crate::dict::Dictionary;
use crate::types::MAX_TOKEN_SIZE;
use crate::types::Token;

const SHORT_LEN: usize = 8;

/// Load up to 16 bytes from `data` as a little-endian `u128`. Bytes beyond
/// `data.len()` are read as zero.
#[inline]
fn load_le_u128(data: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    let n = data.len().min(16);
    buf[..n].copy_from_slice(&data[..n]);
    u128::from_le_bytes(buf)
}

/// Mask of the low `len * 8` bits in a `u128`.
#[inline]
fn mask_u128(len: usize) -> u128 {
    if len >= 16 {
        u128::MAX
    } else {
        (1u128 << (len * 8)) - 1
    }
}

#[inline]
fn mask_u64(len: usize) -> u64 {
    if len >= 8 {
        u64::MAX
    } else {
        (1u64 << (len * 8)) - 1
    }
}

/// Maps byte sequences (1..=`MAX_TOKEN_SIZE` bytes) to `Token` IDs. Always
/// holds the 256 single-byte tokens after construction so
/// `find_longest_match` is total.
#[derive(Default, Debug, Clone)]
pub struct LongestPrefixMatcher {
    /// Length 1..=8 tokens keyed by (low-8-byte u64, length).
    short_map: HashMap<(u64, u8), Token>,
    /// Length 9..=16 tokens keyed by (full u128, length).
    long_map: HashMap<(u128, u8), Token>,
    /// Next ID to assign. Stored as u32 so we can represent the full 16-bit
    /// token space (65 536 entries) without overflow.
    next_id: u32,
}

impl LongestPrefixMatcher {
    /// Pre-inserts the 256 single-byte tokens with IDs 0..=255.
    pub fn new() -> Self {
        let mut short_map = HashMap::with_capacity(256);
        for i in 0u16..=255 {
            short_map.insert((i as u64, 1u8), i);
        }
        Self {
            short_map,
            long_map: HashMap::new(),
            next_id: 256,
        }
    }

    /// Build a matcher from a complete dictionary: token at index `i`
    /// receives ID `i`. Caller guarantees the dictionary contains every
    /// single-byte token so `find_longest_match` remains total.
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        let n = dict.num_tokens();
        let mut me = Self {
            short_map: HashMap::with_capacity(n.min(SHORT_LEN * 256)),
            long_map: HashMap::new(),
            next_id: n as u32,
        };
        for i in 0..n {
            let id = i as Token;
            me.insert_internal(dict.data(id), id);
        }
        me
    }

    /// Insert `data` and assign it the next available token id.
    ///
    /// Precondition: `1 <= data.len() <= MAX_TOKEN_SIZE` and
    /// `size() < 65_536`.
    pub fn insert(&mut self, data: &[u8]) -> Token {
        let id = self.next_id as Token;
        self.next_id += 1;
        self.insert_internal(data, id);
        id
    }

    #[inline]
    fn insert_internal(&mut self, data: &[u8], id: Token) {
        debug_assert!(!data.is_empty() && data.len() <= MAX_TOKEN_SIZE);
        let len = data.len();
        if len <= SHORT_LEN {
            let key = (load_le_u128(data) as u64) & mask_u64(len);
            self.short_map.insert((key, len as u8), id);
        } else {
            let key = load_le_u128(data) & mask_u128(len);
            self.long_map.insert((key, len as u8), id);
        }
    }

    /// Longest token whose bytes are a prefix of `data`, together with that
    /// prefix's length.
    ///
    /// Precondition: `!data.is_empty()` and the matcher contains every
    /// single-byte token (always true after [`new`] or [`from_dictionary`]
    /// with a complete dictionary).
    #[inline]
    pub fn find_longest_match(&self, data: &[u8]) -> (Token, usize) {
        let max_len = data.len().min(MAX_TOKEN_SIZE);
        let packed = load_le_u128(data);
        // Long map: only relevant when at least 9 bytes of input are available.
        if max_len > SHORT_LEN && !self.long_map.is_empty() {
            for len in (SHORT_LEN + 1..=max_len).rev() {
                let key = packed & mask_u128(len);
                if let Some(&t) = self.long_map.get(&(key, len as u8)) {
                    return (t, len);
                }
            }
        }
        // Short map: lengths min(max_len, 8) down to 1.
        let short_max = max_len.min(SHORT_LEN);
        let low64 = packed as u64;
        for len in (1..=short_max).rev() {
            let key = low64 & mask_u64(len);
            if let Some(&t) = self.short_map.get(&(key, len as u8)) {
                return (t, len);
            }
        }
        unreachable!("LPM precondition: every single-byte token must be present")
    }

    /// Number of tokens currently in the matcher.
    #[inline]
    pub fn size(&self) -> usize {
        self.next_id as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_str(lpm: &mut LongestPrefixMatcher, s: &str) -> Token {
        lpm.insert(s.as_bytes())
    }

    fn find_str(lpm: &LongestPrefixMatcher, s: &str) -> (Token, usize) {
        lpm.find_longest_match(s.as_bytes())
    }

    fn make_test_dictionary(extra: &[&str]) -> Dictionary {
        let mut d = Dictionary::default();
        d.offsets.push(0);
        for i in 0u16..=255 {
            d.bytes.push(i as u8);
            d.offsets.push(d.bytes.len() as u32);
        }
        for &s in extra {
            d.bytes.extend_from_slice(s.as_bytes());
            d.offsets.push(d.bytes.len() as u32);
        }
        d
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn default_constructor_size_is_256() {
        let lpm = LongestPrefixMatcher::new();
        assert_eq!(lpm.size(), 256);
    }

    #[test]
    fn all_single_bytes_found_after_construction() {
        let lpm = LongestPrefixMatcher::new();
        for i in 0u16..=255 {
            let b = [i as u8];
            let (tok, len) = lpm.find_longest_match(&b);
            assert_eq!(tok, i, "wrong token for byte {i}");
            assert_eq!(len, 1, "wrong length for byte {i}");
        }
    }

    #[test]
    fn zero_byte_is_token_0() {
        let lpm = LongestPrefixMatcher::new();
        let (tok, len) = lpm.find_longest_match(&[0x00]);
        assert_eq!(tok, 0);
        assert_eq!(len, 1);
    }

    #[test]
    fn max_byte_is_token_255() {
        let lpm = LongestPrefixMatcher::new();
        let (tok, len) = lpm.find_longest_match(&[0xFF]);
        assert_eq!(tok, 255);
        assert_eq!(len, 1);
    }

    // ── Insert ───────────────────────────────────────────────────────────────

    #[test]
    fn first_insert_returns_id_256() {
        let mut lpm = LongestPrefixMatcher::new();
        assert_eq!(insert_str(&mut lpm, "ab"), 256);
    }

    #[test]
    fn subsequent_inserts_increment_id() {
        let mut lpm = LongestPrefixMatcher::new();
        assert_eq!(insert_str(&mut lpm, "ab"), 256);
        assert_eq!(insert_str(&mut lpm, "cd"), 257);
        assert_eq!(insert_str(&mut lpm, "ef"), 258);
    }

    #[test]
    fn size_grows_with_each_insert() {
        let mut lpm = LongestPrefixMatcher::new();
        assert_eq!(lpm.size(), 256);
        insert_str(&mut lpm, "ab");
        assert_eq!(lpm.size(), 257);
        insert_str(&mut lpm, "cd");
        assert_eq!(lpm.size(), 258);
    }

    #[test]
    fn exactly_eight_bytes_short_store() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "12345678");
        let (tok, len) = find_str(&lpm, "12345678");
        assert_eq!(tok, id);
        assert_eq!(len, 8);
    }

    #[test]
    fn exactly_nine_bytes_long_store() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "123456789");
        let (tok, len) = find_str(&lpm, "123456789X");
        assert_eq!(tok, id);
        assert_eq!(len, 9);
    }

    #[test]
    fn max_token_size_insert_and_find() {
        let mut lpm = LongestPrefixMatcher::new();
        let pat = "0123456789abcdef";
        assert_eq!(pat.len(), MAX_TOKEN_SIZE);
        let id = lpm.insert(pat.as_bytes());
        let (tok, len) = lpm.find_longest_match(pat.as_bytes());
        assert_eq!(tok, id);
        assert_eq!(len, MAX_TOKEN_SIZE);
    }

    #[test]
    fn sequence_with_embedded_zero_bytes() {
        let mut lpm = LongestPrefixMatcher::new();
        let data = [0x00u8, 0x01, 0x02];
        let id = lpm.insert(&data);
        let (tok, len) = lpm.find_longest_match(&data);
        assert_eq!(tok, id);
        assert_eq!(len, 3);
    }

    // ── find_longest_match ───────────────────────────────────────────────────

    #[test]
    fn single_byte_found_with_correct_id() {
        let lpm = LongestPrefixMatcher::new();
        let (tok, len) = lpm.find_longest_match(&[0x42]);
        assert_eq!(tok, 0x42);
        assert_eq!(len, 1);
    }

    #[test]
    fn longest_match_wins_over_shorter() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "abc");
        let long_id = insert_str(&mut lpm, "abcdefghi");
        let (tok, len) = find_str(&lpm, "abcdefghi");
        assert_eq!(tok, long_id);
        assert_eq!(len, 9);
    }

    #[test]
    fn falls_back_to_shorter_if_long_not_present() {
        let mut lpm = LongestPrefixMatcher::new();
        let short_id = insert_str(&mut lpm, "abc");
        let (tok, len) = find_str(&lpm, "abcdef");
        assert_eq!(tok, short_id);
        assert_eq!(len, 3);
    }

    #[test]
    fn falls_back_to_single_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "XY");
        let (tok, len) = find_str(&lpm, "XZ");
        assert_eq!(tok, b'X' as Token);
        assert_eq!(len, 1);
    }

    #[test]
    fn exact_match_no_trailing_bytes() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "hello");
        let (tok, len) = find_str(&lpm, "hello");
        assert_eq!(tok, id);
        assert_eq!(len, 5);
    }

    #[test]
    fn input_shorter_than_stored_pattern_falls_to_single_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "abcde");
        let (tok, len) = find_str(&lpm, "ab");
        assert_eq!(tok, b'a' as Token);
        assert_eq!(len, 1);
    }

    #[test]
    fn input_shorter_than_longest_matches_shorter_token() {
        let mut lpm = LongestPrefixMatcher::new();
        let id2 = insert_str(&mut lpm, "ab");
        insert_str(&mut lpm, "abcde");
        let (tok, len) = find_str(&lpm, "ab");
        assert_eq!(tok, id2);
        assert_eq!(len, 2);
    }

    #[test]
    fn eight_byte_token_with_longer_input() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "ABCDEFGH");
        let (tok, len) = find_str(&lpm, "ABCDEFGHIJ");
        assert_eq!(tok, id);
        assert_eq!(len, 8);
    }

    #[test]
    fn nine_byte_beats_eight_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "ABCDEFGH");
        let id9 = insert_str(&mut lpm, "ABCDEFGHI");
        let (tok, len) = find_str(&lpm, "ABCDEFGHIJ");
        assert_eq!(tok, id9);
        assert_eq!(len, 9);
    }

    #[test]
    fn short_long_token_is_prefix_of_longer_long_token() {
        let mut lpm = LongestPrefixMatcher::new();
        let id_short = insert_str(&mut lpm, "ABCDEFGHI"); // 9 bytes
        let id_long = insert_str(&mut lpm, "ABCDEFGHIJK"); // 11 bytes

        let (t_s, l_s) = find_str(&lpm, "ABCDEFGHIJx");
        assert_eq!(t_s, id_short);
        assert_eq!(l_s, 9);

        let (t_l, l_l) = find_str(&lpm, "ABCDEFGHIJKx");
        assert_eq!(t_l, id_long);
        assert_eq!(l_l, 11);
    }

    #[test]
    fn multiple_tokens_same_long_prefix() {
        let mut lpm = LongestPrefixMatcher::new();
        let id1 = insert_str(&mut lpm, "ABCDEFGHX");
        let id2 = insert_str(&mut lpm, "ABCDEFGHYZ");

        let (t1, l1) = find_str(&lpm, "ABCDEFGHX__");
        assert_eq!(t1, id1);
        assert_eq!(l1, 9);

        let (t2, l2) = find_str(&lpm, "ABCDEFGHYZ_");
        assert_eq!(t2, id2);
        assert_eq!(l2, 10);
    }

    #[test]
    fn max_token_size_pattern_found() {
        let mut lpm = LongestPrefixMatcher::new();
        let pat = "0123456789abcdef";
        let id = insert_str(&mut lpm, pat);
        let (tok, len) = find_str(&lpm, pat);
        assert_eq!(tok, id);
        assert_eq!(len, MAX_TOKEN_SIZE);
    }

    #[test]
    fn binary_all_zeros_long_sequence() {
        let mut lpm = LongestPrefixMatcher::new();
        let data = [0u8; 10];
        let id = lpm.insert(&data);
        let (tok, len) = lpm.find_longest_match(&data);
        assert_eq!(tok, id);
        assert_eq!(len, 10);
    }

    #[test]
    fn binary_all_ff_long_sequence() {
        let mut lpm = LongestPrefixMatcher::new();
        let data = [0xFFu8; 10];
        let id = lpm.insert(&data);
        let (tok, len) = lpm.find_longest_match(&data);
        assert_eq!(tok, id);
        assert_eq!(len, 10);
    }

    // ── Behavioural equivalent of LinearBucket→TrieBucket promotion tests ──

    #[test]
    fn all_tokens_findable_with_shared_long_prefix() {
        let mut lpm = LongestPrefixMatcher::new();
        let prefix = vec![b'X'; 8];
        let mut inserted = Vec::with_capacity(130);
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(i as u8);
            inserted.push(lpm.insert(&buf));
        }
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(i as u8);
            buf.push(0xFF);
            let (tok, len) = lpm.find_longest_match(&buf);
            assert_eq!(tok, inserted[i as usize], "token index {i}");
            assert_eq!(len, 9, "token index {i}");
        }
    }

    #[test]
    fn size_correct_after_many_long_inserts() {
        let mut lpm = LongestPrefixMatcher::new();
        let prefix = vec![b'Y'; 8];
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(i as u8);
            lpm.insert(&buf);
        }
        assert_eq!(lpm.size(), 256 + 130);
    }

    #[test]
    fn deep_trie_multi_level_suffix() {
        let mut lpm = LongestPrefixMatcher::new();
        let prefix = vec![b'Z'; 8];
        let mut inserted = Vec::with_capacity(130);
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(0x00);
            buf.push(i as u8);
            inserted.push(lpm.insert(&buf));
        }
        for i in 0..130u32 {
            let mut buf = prefix.clone();
            buf.push(0x00);
            buf.push(i as u8);
            buf.push(0xFF);
            let (tok, len) = lpm.find_longest_match(&buf);
            assert_eq!(tok, inserted[i as usize], "token index {i}");
            assert_eq!(len, 10, "token index {i}");
        }
    }

    // ── from_dictionary ──────────────────────────────────────────────────────

    #[test]
    fn from_dict_size_matches_base_only() {
        let d = make_test_dictionary(&[]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        assert_eq!(lpm.size(), 256);
    }

    #[test]
    fn from_dict_size_matches_extra_tokens() {
        let d = make_test_dictionary(&["ab", "abcde"]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        assert_eq!(lpm.size(), 258);
    }

    #[test]
    fn from_dict_all_single_bytes_found() {
        let d = make_test_dictionary(&[]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        for i in 0u16..=255 {
            let (tok, len) = lpm.find_longest_match(&[i as u8]);
            assert_eq!(tok, i, "byte {i}");
            assert_eq!(len, 1, "byte {i}");
        }
    }

    #[test]
    fn from_dict_single_byte_uses_positional_id() {
        let d = make_test_dictionary(&[]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        let (tok, len) = lpm.find_longest_match(&[0x41]);
        assert_eq!(tok, 0x41);
        assert_eq!(len, 1);
    }

    #[test]
    fn from_dict_multi_byte_token_found_with_correct_id() {
        let d = make_test_dictionary(&["ab", "abcde"]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        let (tok, len) = find_str(&lpm, "abcde");
        assert_eq!(tok, 257);
        assert_eq!(len, 5);
    }

    #[test]
    fn from_dict_shorter_multi_byte_token_fallback() {
        let d = make_test_dictionary(&["ab", "abcde"]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        let (tok, len) = find_str(&lpm, "abc");
        assert_eq!(tok, 256);
        assert_eq!(len, 2);
    }

    #[test]
    fn from_dict_long_token_from_dictionary() {
        let d = make_test_dictionary(&["ABCDEFGHI"]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        let (tok, len) = find_str(&lpm, "ABCDEFGHIX");
        assert_eq!(tok, 256);
        assert_eq!(len, 9);
    }

    #[test]
    fn from_dict_max_size_token_from_dictionary() {
        let pat = "0123456789abcdef";
        let d = make_test_dictionary(&[pat]);
        let lpm = LongestPrefixMatcher::from_dictionary(&d);
        let (tok, len) = find_str(&lpm, pat);
        assert_eq!(tok, 256);
        assert_eq!(len, MAX_TOKEN_SIZE);
    }

    #[test]
    fn from_dict_insert_continues_id() {
        let d = make_test_dictionary(&["ab", "cd"]);
        let mut lpm = LongestPrefixMatcher::from_dictionary(&d);
        let new_id = insert_str(&mut lpm, "ef");
        assert_eq!(new_id, 258);
        assert_eq!(lpm.size(), 259);
    }

    #[test]
    fn from_dict_inserted_token_is_searchable() {
        let d = make_test_dictionary(&["ab"]);
        let mut lpm = LongestPrefixMatcher::from_dictionary(&d);
        let id = insert_str(&mut lpm, "xyz");
        let (tok, len) = find_str(&lpm, "xyzW");
        assert_eq!(tok, id);
        assert_eq!(len, 3);
    }
}
