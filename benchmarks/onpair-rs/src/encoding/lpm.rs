// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Longest-prefix matcher: maps byte sequences (`1..=MAX_TOKEN_SIZE` bytes) to
//! token ids and answers "what is the longest dictionary token that is a prefix
//! of this input?".
//!
//! Two-tier storage:
//!   * **short map** — tokens of length `1..=8` keyed by their bytes packed into
//!     a `u64` plus the length.
//!   * **long map** — tokens of length `9..=16` bucketed by their 8-byte prefix.
//!     Each bucket holds the `(suffix, length, token)` triples sharing that
//!     prefix and is searched for the longest matching suffix. A bucket starts
//!     as a sorted vector and is promoted to a byte-trie once it grows past
//!     `PROMOTE_THRESHOLD`.
//!
//! Longest-match lookup issues a single probe on the 8-byte prefix to reach the
//! long bucket, then falls through to short-token matching.

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use hashbrown::HashTable;

use crate::encoding::hash::{Map, map, map_with_capacity};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{
    _mm256_and_si256, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256,
    _mm256_movemask_pd, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_xor_si256,
};

/// Tokens of this length or shorter live in the short map; longer tokens are
/// bucketed by their first `BUCKET_PREFIX_LEN` bytes.
const BUCKET_PREFIX_LEN: usize = 8;
const SHORT_PREFIX_COUNT: usize = 1 << 16;

/// A long bucket is promoted from a linear vector to a trie once it holds more
/// than this many entries, bounding worst-case suffix search.
const PROMOTE_THRESHOLD: usize = 128;

#[inline(always)]
fn hash_long_prefix(key: u64) -> u64 {
    let hash = (key ^ key.rotate_left(32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^ (hash >> 32)
}

/// Pack the low `min(len, data.len(), 8)` bytes of `data` into a little-endian
/// `u64`; higher bytes read as zero. The full-8-byte case is a single load.
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

/// Static short-token index used after dictionary training. Tokens sharing
/// their first two bytes are contiguous and sorted longest-first. This turns
/// several unrelated hash probes into sequential packed-prefix comparisons.
#[derive(Debug, Clone)]
struct ShortBuckets {
    offsets: Box<[u32]>,
    length_index: Option<ShortLengthIndexes>,
    keys: Box<[u64]>,
    masks: Box<[u64]>,
    tokens: Box<[Token]>,
    lengths: Box<[u8]>,
    single_tokens: Box<[Token; 256]>,
}

#[derive(Debug, Clone)]
struct ShortLengthIndex {
    length_bitmap: u8,
    length_boundaries: [u16; BUCKET_PREFIX_LEN],
}

#[derive(Debug, Clone)]
struct ShortLengthIndexes {
    ids: Box<[u16]>,
    indexes: Box<[ShortLengthIndex]>,
}

impl ShortBuckets {
    fn from_dictionary(dict: CompactDictionaryView<'_>, build_length_index: bool) -> Self {
        let mut single_tokens = Box::new([0; 256]);
        let mut entries = Vec::with_capacity(dict.num_tokens());
        for index in 0..dict.num_tokens() {
            let token = index as Token;
            let bytes = dict.token(token);
            if bytes.len() == 1 {
                single_tokens[bytes[0] as usize] = token;
            } else if bytes.len() <= BUCKET_PREFIX_LEN {
                let key = load_le_u64(bytes, bytes.len());
                entries.push(((key & 0xffff) as u16, key, bytes.len() as u8, token));
            }
        }
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.2.cmp(&a.2)));

        let mut offsets = vec![0u32; SHORT_PREFIX_COUNT + 1];
        for &(prefix, _, _, _) in &entries {
            offsets[prefix as usize + 1] += 1;
        }
        for prefix in 0..SHORT_PREFIX_COUNT {
            offsets[prefix + 1] += offsets[prefix];
        }

        let length_index = build_length_index.then(|| {
            let mut ids = vec![0u16; SHORT_PREFIX_COUNT];
            let mut indexes = Vec::new();
            for prefix in 0..SHORT_PREFIX_COUNT {
                let start = offsets[prefix] as usize;
                let end = offsets[prefix + 1] as usize;
                if start == end {
                    continue;
                }
                ids[prefix] = indexes.len() as u16 + 1;
                let mut length_bitmap = 0u8;
                let mut length_boundaries = [0u16; BUCKET_PREFIX_LEN];
                let mut cursor = start;
                for length in (2..=BUCKET_PREFIX_LEN).rev() {
                    let boundary = BUCKET_PREFIX_LEN - length;
                    length_boundaries[boundary] = (cursor - start) as u16;
                    if cursor < end && entries[cursor].2 as usize == length {
                        length_bitmap |= 1 << (length - 2);
                        while cursor < end && entries[cursor].2 as usize == length {
                            cursor += 1;
                        }
                    }
                }
                length_boundaries[BUCKET_PREFIX_LEN - 1] = (end - start) as u16;
                indexes.push(ShortLengthIndex {
                    length_bitmap,
                    length_boundaries,
                });
            }
            ShortLengthIndexes {
                ids: ids.into_boxed_slice(),
                indexes: indexes.into_boxed_slice(),
            }
        });

        let mut keys = Vec::with_capacity(entries.len());
        let mut masks = Vec::with_capacity(entries.len());
        let mut tokens = Vec::with_capacity(entries.len());
        let mut lengths = Vec::with_capacity(entries.len());
        for (_, key, length, token) in entries {
            keys.push(key);
            masks.push(mask_u64(length as usize));
            tokens.push(token);
            lengths.push(length);
        }
        Self {
            offsets: offsets.into_boxed_slice(),
            length_index,
            keys: keys.into_boxed_slice(),
            masks: masks.into_boxed_slice(),
            tokens: tokens.into_boxed_slice(),
            lengths: lengths.into_boxed_slice(),
            single_tokens,
        }
    }

    #[inline]
    fn find(&self, value: u64, max_len: usize, first_byte: u8) -> Option<(Token, usize)> {
        if max_len >= 2 {
            let prefix = (value & 0xffff) as usize;
            let mut start = self.offsets[prefix] as usize;
            let end = self.offsets[prefix + 1] as usize;
            if start < end && max_len < BUCKET_PREFIX_LEN {
                if let Some(length_indexes) = &self.length_index {
                    let length_index =
                        &length_indexes.indexes[length_indexes.ids[prefix] as usize - 1];
                    let eligible = length_index.length_bitmap & ((1u16 << (max_len - 1)) - 1) as u8;
                    if eligible == 0 {
                        return Some((self.single_tokens[first_byte as usize], 1));
                    }
                    let length = u8::BITS as usize - eligible.leading_zeros() as usize + 1;
                    start += length_index.length_boundaries[BUCKET_PREFIX_LEN - length] as usize;
                } else {
                    while start < end && self.lengths[start] as usize > max_len {
                        start += 1;
                    }
                }
            }
            if end - start >= 4 {
                if let Some(hit) = self.find_simd(value, start, end) {
                    return Some(hit);
                }
            } else {
                for index in start..end {
                    if (value ^ self.keys[index]) & self.masks[index] == 0 {
                        return Some((self.tokens[index], self.lengths[index] as usize));
                    }
                }
            }
        }
        Some((self.single_tokens[first_byte as usize], 1))
    }

    #[inline]
    fn find_simd(&self, value: u64, start: usize, end: usize) -> Option<(Token, usize)> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let mut index = start;
            let input = unsafe { _mm256_set1_epi64x(value as i64) };
            let zero = unsafe { _mm256_setzero_si256() };
            while index + 4 <= end {
                let keys = unsafe { _mm256_loadu_si256(self.keys.as_ptr().add(index).cast()) };
                let masks = unsafe { _mm256_loadu_si256(self.masks.as_ptr().add(index).cast()) };
                let difference = unsafe { _mm256_and_si256(_mm256_xor_si256(input, keys), masks) };
                let equal = unsafe { _mm256_cmpeq_epi64(difference, zero) };
                let matches = unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(equal)) } as u32;
                if matches != 0 {
                    let hit = index + matches.trailing_zeros() as usize;
                    return Some((self.tokens[hit], self.lengths[hit] as usize));
                }
                index += 4;
            }
            for index in index..end {
                if (value ^ self.keys[index]) & self.masks[index] == 0 {
                    return Some((self.tokens[index], self.lengths[index] as usize));
                }
            }
            None
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for index in start..end {
                if (value ^ self.keys[index]) & self.masks[index] == 0 {
                    return Some((self.tokens[index], self.lengths[index] as usize));
                }
            }
            None
        }
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
/// descending suffix length so the first match is the longest) and is promoted
/// to a trie rooted at a pool index once it grows large.
#[derive(Debug, Clone)]
enum Bucket {
    Linear(Vec<LongEntry>),
    Trie(u32),
}

const LONG_FILTER_BITS: usize = 1 << 16;
const MAX_FILTERED_LONG_PREFIXES: usize = 512;

/// Read-only long-prefix index used when its compact membership filter can
/// reject the dominant miss path before touching the hash table.
#[derive(Debug, Clone)]
struct FrozenLongMap {
    table: HashTable<(u64, Bucket)>,
    filter: Box<[u64]>,
}

impl FrozenLongMap {
    fn from_map(map: Map<u64, Bucket>) -> Result<Self, Map<u64, Bucket>> {
        if map.len() > MAX_FILTERED_LONG_PREFIXES {
            return Err(map);
        }

        let mut table = HashTable::with_capacity(map.len());
        let mut filter = vec![0u64; LONG_FILTER_BITS / 64].into_boxed_slice();
        for entry in map {
            let hash = hash_long_prefix(entry.0);
            let bit = hash as usize & (LONG_FILTER_BITS - 1);
            filter[bit >> 6] |= 1 << (bit & 63);
            table.insert_unique(hash, entry, |(prefix, _)| hash_long_prefix(*prefix));
        }
        Ok(Self { table, filter })
    }

    #[inline]
    fn get(&self, key: u64) -> Option<&Bucket> {
        let hash = hash_long_prefix(key);
        let bit = hash as usize & (LONG_FILTER_BITS - 1);
        if self.filter[bit >> 6] & (1 << (bit & 63)) == 0 {
            return None;
        }
        self.table
            .find(hash, |(candidate, _)| *candidate == key)
            .map(|(_, bucket)| bucket)
    }

    fn into_map(self) -> Map<u64, Bucket> {
        let mut map = map_with_capacity(self.table.len());
        for (key, bucket) in self.table {
            map.insert(key, bucket);
        }
        map
    }
}

/// Search a sorted-descending linear bucket for the longest suffix that matches
/// the low bytes of `val` (the input suffix, `<= max_slen` bytes).
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

/// Build a trie bucket from the entries of a linear bucket.
fn build_trie(pool: &mut Vec<TrieNode>, entries: &[LongEntry]) -> Bucket {
    let root = trie_alloc(pool);
    for e in entries {
        let buf = e.suffix.to_le_bytes();
        trie_insert(pool, root, &buf[..e.slen as usize], e.token);
    }
    Bucket::Trie(root)
}

/// Maps byte sequences (`1..=MAX_TOKEN_SIZE` bytes) to [`Token`] ids. Always
/// holds the 256 single-byte tokens after construction, so
/// Longest-match lookup is total.
#[derive(Default, Debug, Clone)]
pub(crate) struct LongestPrefixMatcher {
    /// Length `1..=8` tokens keyed by (low-`len`-byte u64, length).
    short_map: Map<(u64, u8), Token>,
    /// Read-only prefix buckets built for the final parsing dictionary.
    short_buckets: Option<ShortBuckets>,
    /// Length `9..=16` tokens bucketed by their 8-byte prefix.
    long_map: Map<u64, Bucket>,
    /// Filtered, prehashed replacement for small completed long-prefix maps.
    frozen_long_map: Option<FrozenLongMap>,
    /// Trie node arena shared by every promoted long bucket.
    pool: Vec<TrieNode>,
    /// Longest short-map token length present (`1..=8`).
    max_short_len: u8,
    /// Next id to assign. `u32` so the full 16-bit token space (65 536 entries)
    /// is representable without overflow.
    next_id: u32,
}

impl LongestPrefixMatcher {
    /// Pre-inserts the 256 single-byte tokens with ids `0..=255`.
    pub(crate) fn new() -> Self {
        let mut short_map = map_with_capacity(256);
        for i in 0u16..=255 {
            short_map.insert((i as u64, 1u8), i);
        }
        Self {
            short_map,
            short_buckets: None,
            long_map: map(),
            frozen_long_map: None,
            pool: Vec::new(),
            max_short_len: 1,
            next_id: 256,
        }
    }

    /// Reserve training-time maps for the configured dictionary budget.
    pub(crate) fn reserve(&mut self, token_capacity: usize) {
        self.short_map
            .reserve(token_capacity.saturating_sub(self.short_map.len()));
        let long_capacity = (token_capacity / 4).max(16);
        self.long_map
            .reserve(long_capacity.saturating_sub(self.long_map.len()));
    }

    /// Build a matcher from a complete dictionary: token at index `i` receives
    /// id `i`. The caller guarantees the dictionary contains every single-byte
    /// token so longest-match lookup stays total.
    pub(crate) fn from_dictionary(
        dict: CompactDictionaryView<'_>,
        build_short_buckets: bool,
        build_length_index: bool,
    ) -> Self {
        let n = dict.num_tokens();
        let mut me = Self {
            short_map: map_with_capacity(n.min(BUCKET_PREFIX_LEN * 256)),
            short_buckets: None,
            long_map: map(),
            frozen_long_map: None,
            pool: Vec::new(),
            max_short_len: 1,
            next_id: n as u32,
        };
        for i in 0..n {
            let id = i as Token;
            me.insert_internal(dict.token(id), id);
        }
        match FrozenLongMap::from_map(std::mem::take(&mut me.long_map)) {
            Ok(frozen) => me.frozen_long_map = Some(frozen),
            Err(map) => me.long_map = map,
        }
        me.short_buckets =
            build_short_buckets.then(|| ShortBuckets::from_dictionary(dict, build_length_index));
        me
    }

    /// Insert `data` and assign it the next available token id.
    ///
    /// Precondition: `1 <= data.len() <= MAX_TOKEN_SIZE` and `size() < 65_536`.
    pub(crate) fn insert(&mut self, data: &[u8]) -> Token {
        self.short_buckets = None;
        if let Some(frozen) = self.frozen_long_map.take() {
            self.long_map = frozen.into_map();
        }
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
                entries.sort_by_key(|entry| std::cmp::Reverse(entry.slen));
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

    /// Longest token whose bytes are a prefix of `data`, with that prefix's
    /// length.
    ///
    /// Precondition: `!data.is_empty()` and the matcher contains every
    /// single-byte token (always true after [`new`](Self::new) or
    /// [`from_dictionary`](Self::from_dictionary) with a complete dictionary).
    #[cfg(test)]
    #[inline]
    pub(crate) fn find_longest_match(&self, data: &[u8]) -> (Token, usize) {
        if self.frozen_long_map.is_some() {
            self.find_longest_match_frozen(data)
        } else {
            self.find_longest_match_unfrozen(data)
        }
    }

    #[inline]
    pub(crate) fn has_frozen_long_map(&self) -> bool {
        self.frozen_long_map.is_some()
    }

    /// Parsing lookup for a completed small long-prefix map. The caller hoists
    /// selection of this path outside the per-token loop.
    #[inline]
    pub(crate) fn find_longest_match_frozen(&self, data: &[u8]) -> (Token, usize) {
        let max_len = data.len().min(MAX_TOKEN_SIZE);
        let low64 = load_le_u64(data, max_len.min(BUCKET_PREFIX_LEN));
        if max_len > BUCKET_PREFIX_LEN
            && let Some(bucket) = self.frozen_long_map.as_ref().and_then(|map| map.get(low64))
        {
            let suf = &data[BUCKET_PREFIX_LEN..max_len];
            let hit = match bucket {
                Bucket::Linear(entries) => {
                    search_linear(entries, load_le_u64(suf, suf.len()), suf.len())
                }
                Bucket::Trie(root) => search_trie(&self.pool, *root, suf),
            };
            if let Some((token, suffix_len)) = hit {
                return (token, BUCKET_PREFIX_LEN + suffix_len);
            }
        }
        if let Some(short_buckets) = &self.short_buckets
            && let Some(hit) = short_buckets.find(low64, max_len.min(BUCKET_PREFIX_LEN), data[0])
        {
            return hit;
        }
        // Short map: probe from the longest short token that exists (<= the
        // input window) down to length 1.
        let short_max = max_len.min(self.max_short_len as usize);
        for len in (1..=short_max).rev() {
            let key = low64 & mask_u64(len);
            if let Some(&token) = self.short_map.get(&(key, len as u8)) {
                return (token, len);
            }
        }
        unreachable!("LPM precondition: every single-byte token must be present")
    }

    /// Parsing lookup retaining the mutable long map for larger dictionaries.
    #[inline]
    pub(crate) fn find_longest_match_unfrozen(&self, data: &[u8]) -> (Token, usize) {
        let max_len = data.len().min(MAX_TOKEN_SIZE);
        let low64 = load_le_u64(data, max_len.min(BUCKET_PREFIX_LEN));
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
            if let Some((token, suffix_len)) = hit {
                return (token, BUCKET_PREFIX_LEN + suffix_len);
            }
        }
        if let Some(short_buckets) = &self.short_buckets
            && let Some(hit) = short_buckets.find(low64, max_len.min(BUCKET_PREFIX_LEN), data[0])
        {
            return hit;
        }
        let short_max = max_len.min(self.max_short_len as usize);
        for len in (1..=short_max).rev() {
            let key = low64 & mask_u64(len);
            if let Some(&token) = self.short_map.get(&(key, len as u8)) {
                return (token, len);
            }
        }
        unreachable!("LPM precondition: every single-byte token must be present")
    }

    /// Training-only lookup over the mutable maps. Keeping this path separate
    /// avoids a frozen-index mode branch in the merge loop.
    #[inline]
    pub(crate) fn find_longest_match_training(&self, data: &[u8]) -> (Token, usize) {
        let max_len = data.len().min(MAX_TOKEN_SIZE);
        let low64 = load_le_u64(data, max_len.min(BUCKET_PREFIX_LEN));
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
            if let Some((token, suffix_len)) = hit {
                return (token, BUCKET_PREFIX_LEN + suffix_len);
            }
        }
        let short_max = max_len.min(self.max_short_len as usize);
        for len in (1..=short_max).rev() {
            let key = low64 & mask_u64(len);
            if let Some(&token) = self.short_map.get(&(key, len as u8)) {
                return (token, len);
            }
        }
        unreachable!("LPM precondition: every single-byte token must be present")
    }

    /// Number of tokens currently in the matcher.
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.next_id as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionary, Dictionary};

    fn insert_str(lpm: &mut LongestPrefixMatcher, s: &str) -> Token {
        lpm.insert(s.as_bytes())
    }

    fn find_str(lpm: &LongestPrefixMatcher, s: &str) -> (Token, usize) {
        lpm.find_longest_match(s.as_bytes())
    }

    fn make_test_dictionary(extra: &[&str]) -> CompactDictionary {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for i in 0u16..=255 {
            bytes.push(i as u8);
            offsets.push(bytes.len() as u32);
        }
        for &s in extra {
            bytes.extend_from_slice(s.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        CompactDictionary::from_raw(bytes, offsets)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn default_constructor_size_is_256() {
        assert_eq!(LongestPrefixMatcher::new().size(), 256);
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
    fn exactly_eight_bytes_short_store() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "12345678");
        let (tok, len) = find_str(&lpm, "12345678");
        assert_eq!((tok, len), (id, 8));
    }

    #[test]
    fn exactly_nine_bytes_long_store() {
        let mut lpm = LongestPrefixMatcher::new();
        let id = insert_str(&mut lpm, "123456789");
        let (tok, len) = find_str(&lpm, "123456789X");
        assert_eq!((tok, len), (id, 9));
    }

    #[test]
    fn max_token_size_insert_and_find() {
        let mut lpm = LongestPrefixMatcher::new();
        let pat = "0123456789abcdef";
        assert_eq!(pat.len(), MAX_TOKEN_SIZE);
        let id = lpm.insert(pat.as_bytes());
        let (tok, len) = lpm.find_longest_match(pat.as_bytes());
        assert_eq!((tok, len), (id, MAX_TOKEN_SIZE));
    }

    // ── find_longest_match ───────────────────────────────────────────────────

    #[test]
    fn longest_match_wins_over_shorter() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "abc");
        let long_id = insert_str(&mut lpm, "abcdefghi");
        let (tok, len) = find_str(&lpm, "abcdefghi");
        assert_eq!((tok, len), (long_id, 9));
    }

    #[test]
    fn falls_back_to_shorter_if_long_not_present() {
        let mut lpm = LongestPrefixMatcher::new();
        let short_id = insert_str(&mut lpm, "abc");
        let (tok, len) = find_str(&lpm, "abcdef");
        assert_eq!((tok, len), (short_id, 3));
    }

    #[test]
    fn falls_back_to_single_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "XY");
        let (tok, len) = find_str(&lpm, "XZ");
        assert_eq!((tok, len), (b'X' as Token, 1));
    }

    #[test]
    fn nine_byte_beats_eight_byte() {
        let mut lpm = LongestPrefixMatcher::new();
        insert_str(&mut lpm, "ABCDEFGH");
        let id9 = insert_str(&mut lpm, "ABCDEFGHI");
        let (tok, len) = find_str(&lpm, "ABCDEFGHIJ");
        assert_eq!((tok, len), (id9, 9));
    }

    #[test]
    fn multiple_tokens_same_long_prefix() {
        let mut lpm = LongestPrefixMatcher::new();
        let id1 = insert_str(&mut lpm, "ABCDEFGHX");
        let id2 = insert_str(&mut lpm, "ABCDEFGHYZ");
        assert_eq!(find_str(&lpm, "ABCDEFGHX__"), (id1, 9));
        assert_eq!(find_str(&lpm, "ABCDEFGHYZ_"), (id2, 10));
    }

    #[test]
    fn binary_all_zeros_long_sequence() {
        let mut lpm = LongestPrefixMatcher::new();
        let data = [0u8; 10];
        let id = lpm.insert(&data);
        assert_eq!(lpm.find_longest_match(&data), (id, 10));
    }

    // ── trie promotion (>128 entries in one bucket) ───────────────────────────

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
            assert_eq!((tok, len), (inserted[i as usize], 9), "token index {i}");
        }
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
            assert_eq!((tok, len), (inserted[i as usize], 10), "token index {i}");
        }
    }

    // ── from_dictionary ──────────────────────────────────────────────────────

    #[test]
    fn from_dict_size_matches_extra_tokens() {
        let d = make_test_dictionary(&["ab", "abcde"]);
        assert_eq!(
            LongestPrefixMatcher::from_dictionary(d.as_view(), true, true).size(),
            258
        );
    }

    #[test]
    fn from_dict_multi_byte_token_found_with_correct_id() {
        let d = make_test_dictionary(&["ab", "abcde"]);
        let lpm = LongestPrefixMatcher::from_dictionary(d.as_view(), true, true);
        assert_eq!(find_str(&lpm, "abcde"), (257, 5));
        assert_eq!(find_str(&lpm, "abc"), (256, 2));
    }

    #[test]
    fn from_dict_long_token_from_dictionary() {
        let d = make_test_dictionary(&["ABCDEFGHI"]);
        let lpm = LongestPrefixMatcher::from_dictionary(d.as_view(), true, true);
        assert_eq!(find_str(&lpm, "ABCDEFGHIX"), (256, 9));
    }

    #[test]
    fn from_dict_insert_continues_id() {
        let d = make_test_dictionary(&["ab", "cd"]);
        let mut lpm = LongestPrefixMatcher::from_dictionary(d.as_view(), true, true);
        assert_eq!(insert_str(&mut lpm, "ef"), 258);
        assert_eq!(lpm.size(), 259);
    }

    #[test]
    fn length_index_matches_linear_skip_for_short_windows() {
        let d = make_test_dictionary(&["ab", "abc", "abcde", "abcdefgh"]);
        let indexed = ShortBuckets::from_dictionary(d.as_view(), true);
        let linear = ShortBuckets::from_dictionary(d.as_view(), false);

        for input in ["a", "ab", "abc", "abcd", "abcde", "abcdefg"] {
            let bytes = input.as_bytes();
            let value = load_le_u64(bytes, bytes.len());
            assert_eq!(
                indexed.find(value, bytes.len(), bytes[0]),
                linear.find(value, bytes.len(), bytes[0]),
                "input {input}"
            );
        }
    }

    #[test]
    fn frozen_long_map_finds_members_and_rejects_misses() {
        let key = u64::from_le_bytes(*b"ABCDEFGH");
        let mut map = map();
        map.insert(
            key,
            Bucket::Linear(vec![LongEntry {
                suffix: b'I' as u64,
                slen: 1,
                token: 256,
            }]),
        );

        let frozen = FrozenLongMap::from_map(map).unwrap();
        assert!(frozen.get(key).is_some());
        assert!(frozen.get(u64::from_le_bytes(*b"abcdefgh")).is_none());
    }

    #[test]
    fn large_long_map_keeps_the_mutable_hash_map() {
        let mut map = map_with_capacity(MAX_FILTERED_LONG_PREFIXES + 1);
        for key in 0..=MAX_FILTERED_LONG_PREFIXES as u64 {
            map.insert(key, Bucket::Linear(Vec::new()));
        }

        assert!(FrozenLongMap::from_map(map).is_err());
    }
}
