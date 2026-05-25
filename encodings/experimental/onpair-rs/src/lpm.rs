// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/encoding/lpm.h`.
//
// Two-tier storage mirroring the C++ design:
//   * **short map** — tokens of length 1..=8 keyed by their bytes packed
//     into a `u64` plus the length.
//   * **long map** — tokens of length 9..=16 bucketed by their 8-byte
//     prefix. Each bucket holds the (suffix, length, token) triples that
//     share that prefix and is searched for the longest matching suffix. A
//     bucket starts as a sorted vector (`LinearBucket`) and is promoted to a
//     byte-trie (`TrieBucket`) once it grows past `PROMOTE_THRESHOLD`.
//
// `find_longest_match` issues a *single* hash probe on the 8-byte prefix to
// reach the long bucket (vs. the previous design's per-length probe loop),
// which matters because on real corpora a large fraction of dictionary
// tokens exceed 8 bytes. It then falls through to the short map, probing
// lengths `min(max_len, 8) .. 1`.

use crate::dict::Dictionary;
use crate::hash::Map;
use crate::hash::map;
use crate::hash::map_with_capacity;
use crate::types::MAX_TOKEN_SIZE;
use crate::types::Token;

/// Tokens of this length or shorter live in the short map; longer tokens are
/// bucketed by their first `BUCKET_PREFIX_LEN` bytes.
const BUCKET_PREFIX_LEN: usize = 8;

/// A long bucket is promoted from a linear vector to a trie once it holds
/// more than this many entries, bounding worst-case suffix search.
const PROMOTE_THRESHOLD: usize = 128;

/// Pack the low `min(len, data.len(), 8)` bytes of `data` into a
/// little-endian `u64`; higher bytes read as zero. The full-8-byte case (the
/// hot path for any input with a multi-byte token ahead) is a single load.
#[inline]
fn load_le_u64(data: &[u8], len: usize) -> u64 {
    if len >= BUCKET_PREFIX_LEN && data.len() >= BUCKET_PREFIX_LEN {
        return u64::from_le_bytes(data[..BUCKET_PREFIX_LEN].try_into().unwrap());
    }
    let mut buf = [0u8; 8];
    let n = len.min(data.len());
    buf[..n].copy_from_slice(&data[..n]);
    u64::from_le_bytes(buf)
}

/// Mask of the low `len * 8` bits in a `u64`.
#[inline]
fn mask_u64(len: usize) -> u64 {
    if len >= 8 {
        u64::MAX
    } else {
        (1u64 << (len * 8)) - 1
    }
}

/// One long-token entry within a bucket: the suffix bytes after the shared
/// 8-byte prefix (`slen` of them, packed little-endian) and the token id.
#[derive(Copy, Clone, Debug)]
struct LongEntry {
    suffix: u64,
    slen: u8,
    token: Token,
}

/// A node in the shared trie pool. `children` is a small linear-scanned
/// association list of `(byte, node_index)`.
#[derive(Default, Debug, Clone)]
struct TrieNode {
    token: Option<Token>,
    children: Vec<(u8, u32)>,
}

/// A long bucket: entries sharing an 8-byte prefix. Starts linear (sorted by
/// descending suffix length so the first match is the longest) and is
/// promoted to a trie rooted at a pool index once it grows large.
#[derive(Debug, Clone)]
enum Bucket {
    Linear(Vec<LongEntry>),
    Trie(u32),
}

/// Search a sorted-descending `LinearBucket` for the longest suffix that
/// matches the low bytes of `val` (the input suffix, ≤ `max_slen` bytes).
#[inline]
fn search_linear(entries: &[LongEntry], val: u64, max_slen: usize) -> Option<(Token, usize)> {
    for e in entries {
        let elen = e.slen as usize;
        // Matching low bytes = trailing-zero bytes of the XOR.
        if elen <= max_slen && ((val ^ e.suffix).trailing_zeros() >> 3) as usize >= elen {
            return Some((e.token, elen));
        }
    }
    None
}

/// Walk the trie at `root` against `suf`, returning the deepest node that
/// carries a token id together with the matched suffix length.
#[inline]
fn search_trie(pool: &[TrieNode], root: u32, suf: &[u8]) -> Option<(Token, usize)> {
    let mut best = None;
    let mut cur = root;
    for (pos, &b) in suf.iter().enumerate() {
        match trie_find_child(pool, cur, b) {
            Some(child) => {
                cur = child;
                if let Some(t) = pool[cur as usize].token {
                    best = Some((t, pos + 1));
                }
            }
            None => break,
        }
    }
    best
}

#[inline]
fn trie_find_child(pool: &[TrieNode], node: u32, byte: u8) -> Option<u32> {
    pool[node as usize]
        .children
        .iter()
        .find_map(|&(b, idx)| (b == byte).then_some(idx))
}

fn trie_alloc(pool: &mut Vec<TrieNode>) -> u32 {
    let idx = pool.len() as u32;
    pool.push(TrieNode::default());
    idx
}

fn trie_insert(pool: &mut Vec<TrieNode>, root: u32, suf: &[u8], token: Token) {
    let mut cur = root;
    for &b in suf {
        match trie_find_child(pool, cur, b) {
            Some(child) => cur = child,
            None => {
                let new_idx = trie_alloc(pool);
                pool[cur as usize].children.push((b, new_idx));
                cur = new_idx;
            }
        }
    }
    pool[cur as usize].token = Some(token);
}

/// Build a `TrieBucket` from the entries of a `LinearBucket`.
fn build_trie(pool: &mut Vec<TrieNode>, entries: &[LongEntry]) -> Bucket {
    let root = trie_alloc(pool);
    for e in entries {
        let buf = e.suffix.to_le_bytes();
        trie_insert(pool, root, &buf[..e.slen as usize], e.token);
    }
    Bucket::Trie(root)
}

/// Maps byte sequences (1..=`MAX_TOKEN_SIZE` bytes) to `Token` IDs. Always
/// holds the 256 single-byte tokens after construction so
/// `find_longest_match` is total.
#[derive(Default, Debug, Clone)]
pub struct LongestPrefixMatcher {
    /// Length 1..=8 tokens keyed by (low-`len`-byte u64, length).
    short_map: Map<(u64, u8), Token>,
    /// Length 9..=16 tokens bucketed by their 8-byte prefix.
    long_map: Map<u64, Bucket>,
    /// Trie node arena shared by every promoted long bucket.
    pool: Vec<TrieNode>,
    /// Longest short-map token length present (1..=8). The short-map probe
    /// loop starts here instead of at 8, which avoids wasted probes while the
    /// dictionary is still small during training.
    max_short_len: u8,
    /// Next ID to assign. Stored as u32 so we can represent the full 16-bit
    /// token space (65 536 entries) without overflow.
    next_id: u32,
}

impl LongestPrefixMatcher {
    /// Pre-inserts the 256 single-byte tokens with IDs 0..=255.
    pub fn new() -> Self {
        let mut short_map = map_with_capacity(256);
        for i in 0u16..=255 {
            short_map.insert((i as u64, 1u8), i);
        }
        Self {
            short_map,
            long_map: map(),
            pool: Vec::new(),
            max_short_len: 1,
            next_id: 256,
        }
    }

    /// Build a matcher from a complete dictionary: token at index `i`
    /// receives ID `i`. Caller guarantees the dictionary contains every
    /// single-byte token so `find_longest_match` remains total.
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        let n = dict.num_tokens();
        let mut me = Self {
            short_map: map_with_capacity(n.min(BUCKET_PREFIX_LEN * 256)),
            long_map: map(),
            pool: Vec::new(),
            max_short_len: 1,
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
        if len <= BUCKET_PREFIX_LEN {
            let key = load_le_u64(data, len);
            self.short_map.insert((key, len as u8), id);
            self.max_short_len = self.max_short_len.max(len as u8);
            return;
        }

        let prefix = load_le_u64(data, BUCKET_PREFIX_LEN);
        let slen = len - BUCKET_PREFIX_LEN;
        let suffix = load_le_u64(&data[BUCKET_PREFIX_LEN..], slen);
        // Split borrows: `pool` and `long_map` are disjoint fields.
        let pool = &mut self.pool;
        let bucket = self
            .long_map
            .entry(prefix)
            .or_insert_with(|| Bucket::Linear(Vec::new()));
        match bucket {
            Bucket::Linear(entries) => {
                entries.push(LongEntry {
                    suffix,
                    slen: slen as u8,
                    token: id,
                });
                // Keep descending-by-length order so the first match wins.
                entries.sort_by(|a, b| b.slen.cmp(&a.slen));
                if entries.len() > PROMOTE_THRESHOLD {
                    *bucket = build_trie(pool, entries);
                }
            }
            Bucket::Trie(root) => {
                let buf = suffix.to_le_bytes();
                trie_insert(pool, *root, &buf[..slen], id);
            }
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
        // The first up-to-8 bytes serve as both the long-bucket prefix key and
        // the short-map probe window, so load them once.
        let low64 = load_le_u64(data, max_len.min(BUCKET_PREFIX_LEN));
        // Long bucket: a single prefix probe, only when ≥ 9 input bytes exist.
        if max_len > BUCKET_PREFIX_LEN
            && !self.long_map.is_empty()
            && let Some(bucket) = self.long_map.get(&low64)
        {
            let suf = &data[BUCKET_PREFIX_LEN..max_len];
            let hit = match bucket {
                Bucket::Linear(entries) => {
                    search_linear(entries, load_le_u64(suf, suf.len()), suf.len())
                }
                Bucket::Trie(root) => search_trie(&self.pool, *root, suf),
            };
            if let Some((t, slen)) = hit {
                return (t, BUCKET_PREFIX_LEN + slen);
            }
        }
        // Short map: probe from the longest short token that actually exists
        // (≤ the input window) down to length 1.
        let short_max = max_len.min(self.max_short_len as usize);
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
