// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Per-chunk skip indexes for OnPair-encoded columns.
//!
//! Each structure is a small per-chunk fingerprint that answers
//! "could this chunk contain *any* row satisfying predicate P?" without
//! decoding the chunk. All structures are **sound**: a `false` answer
//! guarantees there is no matching row in the chunk; a `true` answer
//! means "maybe" and full evaluation is required.
//!
//! Three independent structures are provided, each tuned for a
//! different predicate family. They are designed to be cheap enough to
//! keep all three resident next to a chunk:
//!
//! * [`DictPresence`] — one bit per dict id; tracks which dictionary
//!   tokens appear in the chunk. Exact, sound necessary condition for
//!   `eq` and `LIKE 'p%'`. Tiny (`dict_size / 8` bytes).
//! * [`TrigramBloom`] — Bloom filter over every overlapping 3-byte
//!   window of the chunk's raw bytes. Classic `LIKE '%substring%'`
//!   skip index (pg_trgm / ClickHouse `ngrambf_v1` style).
//! * [`SeamBloom`] — Bloom filter over only the 3-grams that *cross*
//!   a dict-token boundary. Combined with [`DictPresence`] and the
//!   column's shared dictionary, this reconstructs the chunk's full
//!   trigram set in much less space, exploiting OnPair-specific
//!   structure.
//!
//! All three are independent and may be combined: a chunk is kept iff
//! every enabled tier says "maybe". Each tier's answer is independent
//! so the conjunction is still sound.

use crate::decode::DecodeView;
use crate::lpm::DictIndex;
use crate::lpm::tokenize_needle;

// ---------------------------------------------------------------------------
//                              DictPresence
// ---------------------------------------------------------------------------

/// Bitmap over the shared OnPair dictionary: bit `i` is set iff dict id
/// `i` appears in *some* row of this chunk.
///
/// `bitmap.len() = ceil(dict_size / 64)` `u64` words. For the default
/// `dict-12` preset (4 096 entries) the bitmap is 512 bytes per chunk.
///
/// Predicate logic:
///
/// * `might_eq(needle)`: tokenise `needle` via greedy LPM against the
///   dictionary and require *every* token id of `tokenize(needle)` to
///   be set. Because OnPair compresses every row deterministically by
///   LPM against the same dict, two byte strings are equal iff their
///   token sequences are equal — so "all tokens of the needle present"
///   is a sound necessary condition. (Not sufficient: presence does
///   not imply adjacency; tier 2 covers that.)
/// * `might_starts_with(p)`: greedy-tokenise `p`. Every complete token
///   must be present. After the last complete token, the residual
///   suffix `r` of `p` (if any) must be a prefix of *some* dict entry
///   that is set in this chunk; because the dictionary is stored in
///   lexicographic order, those entries form a contiguous range that
///   we scan with the same first-byte index used by the eq kernel.
#[derive(Clone)]
pub struct DictPresence {
    bitmap: Vec<u64>,
    dict_size: usize,
}

impl DictPresence {
    /// Build the presence bitmap for rows `row_lo..row_hi`.
    pub fn build(dv: &DecodeView<'_>, row_lo: usize, row_hi: usize) -> Self {
        let dict_size = dv.dict_table.len();
        let mut bitmap = vec![0u64; dict_size.div_ceil(64).max(1)];
        let tok_lo = dv.codes_offsets[row_lo] as usize;
        let tok_hi = dv.codes_offsets[row_hi] as usize;
        for &tok in &dv.codes[tok_lo..tok_hi] {
            let i = tok as usize;
            bitmap[i / 64] |= 1u64 << (i % 64);
        }
        Self { bitmap, dict_size }
    }

    /// Bytes occupied by the bitmap.
    pub fn byte_size(&self) -> usize {
        self.bitmap.len() * size_of::<u64>()
    }

    #[inline]
    fn is_set(&self, dict_id: usize) -> bool {
        debug_assert!(dict_id < self.dict_size);
        (self.bitmap[dict_id / 64] >> (dict_id % 64)) & 1 != 0
    }

    /// Sound necessary condition for `col = needle`. Returns `false` only
    /// when no row in this chunk can possibly equal `needle`.
    pub fn might_eq(&self, dv: &DecodeView<'_>, index: &DictIndex, needle: &[u8]) -> bool {
        let Some(toks) = tokenize_needle(dv, index, needle) else {
            return false;
        };
        toks.iter().all(|&t| self.is_set(t as usize))
    }

    /// Sound necessary condition for `col LIKE 'prefix%'`.
    ///
    /// Greedy LPM is *not* prefix-consistent: the token decomposition
    /// the encoder picks for a row `r` may differ from the decomposition
    /// greedy-LPM picks for `prefix` (the boundary token in `r` can
    /// extend past byte `|prefix|`). So we cannot just tokenise `prefix`
    /// and check that every token is present. Instead we run a tiny NFA
    /// over byte positions in `prefix`: `reached[p] = true` iff there
    /// exists *some* sequence of dict tokens, all present in the chunk,
    /// whose concatenation has `prefix[..p]` as a prefix. The chunk
    /// might match if either (a) `reached[n]` for `n = prefix.len()`,
    /// or (b) at some reached position `p` there is a present dict
    /// token whose bytes start with `prefix[p..]` (the token consumes
    /// the remaining prefix and may extend past it).
    pub fn might_starts_with(&self, dv: &DecodeView<'_>, index: &DictIndex, prefix: &[u8]) -> bool {
        let n = prefix.len();
        if n == 0 {
            return true;
        }
        let mut reached = vec![false; n + 1];
        reached[0] = true;
        for p in 0..n {
            if !reached[p] {
                continue;
            }
            let remaining = &prefix[p..];
            let candidates = index.range_for(prefix[p]);
            for id in candidates {
                if !self.is_set(id) {
                    continue;
                }
                let entry = dv.dict_table[id];
                let off = (entry >> 16) as usize;
                let len = (entry & 0xffff) as usize;
                let bytes = &dv.dict_bytes[off..off + len];
                if bytes.len() >= remaining.len() {
                    if bytes.starts_with(remaining) {
                        return true; // consumes the whole prefix
                    }
                } else if remaining.starts_with(bytes) {
                    reached[p + len] = true;
                }
            }
        }
        reached[n]
    }

    /// Necessary condition for `LIKE '%needle%'`, using only the
    /// dictionary bitmap. Sound but very weak: it can only prove
    /// "this chunk contains the needle" via case 1 (some present
    /// dict token has the needle as a substring) and **cannot prove
    /// the negation** without information about token adjacency.
    /// When the needle would straddle two or more adjacent tokens
    /// (case 2/3), the bitmap alone has no signal — we conservatively
    /// return `true` and rely on the caller's `TrigramBloom` /
    /// `SeamBloom` for tight substring pruning.
    pub fn might_contain(&self, dv: &DecodeView<'_>, needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        // Case 1 is a fast confirmation, not a prune; if it fires we
        // can return `true` immediately. Without case 1 we still
        // return `true` because case 2/3 is unprovable from
        // presence-only data.
        for (id, &entry) in dv.dict_table.iter().enumerate() {
            let len = (entry & 0xffff) as usize;
            if len < needle.len() || !self.is_set(id) {
                continue;
            }
            let off = (entry >> 16) as usize;
            if memchr::memmem::find(&dv.dict_bytes[off..off + len], needle).is_some() {
                return true;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
//                                Bloom core
// ---------------------------------------------------------------------------

/// Minimal Bloom filter sized to a power-of-two number of bits (so
/// modulo is a mask). Two independent 32-bit hashes via FNV-1a + XOR
/// salt are mixed into `k` probe positions.
#[derive(Clone)]
struct Bloom {
    bits: Vec<u64>,
    mask: u64,
    k: u32,
}

impl Bloom {
    fn new(num_bits: usize, k: u32) -> Self {
        let num_bits = num_bits.next_power_of_two().max(64);
        let mask = (num_bits as u64) - 1;
        let bits = vec![0u64; num_bits / 64];
        Self { bits, mask, k }
    }

    #[inline]
    fn probes(&self, h1: u32, h2: u32) -> impl Iterator<Item = u64> + '_ {
        let mask = self.mask;
        (0..self.k).map(move |i| {
            // Double hashing: h1 + i * h2  (mod num_bits)
            let v = (h1 as u64).wrapping_add((i as u64).wrapping_mul(h2 as u64));
            v & mask
        })
    }

    #[inline]
    fn insert(&mut self, h1: u32, h2: u32) {
        let probes = self.k;
        let mask = self.mask;
        for probe_idx in 0..probes {
            let raw = (h1 as u64).wrapping_add((probe_idx as u64).wrapping_mul(h2 as u64));
            let pos = raw & mask;
            self.bits[(pos / 64) as usize] |= 1u64 << (pos % 64);
        }
    }

    #[inline]
    fn contains(&self, h1: u32, h2: u32) -> bool {
        for p in self.probes(h1, h2) {
            if (self.bits[(p / 64) as usize] >> (p % 64)) & 1 == 0 {
                return false;
            }
        }
        true
    }

    /// Insert with an explicit hash count (BitFunnel-style tiers).
    /// `k=0` is a no-op.
    #[inline]
    fn insert_k(&mut self, h1: u32, h2: u32, k: u32) {
        let mask = self.mask;
        for probe_idx in 0..k {
            let raw = (h1 as u64).wrapping_add((probe_idx as u64).wrapping_mul(h2 as u64));
            let pos = raw & mask;
            self.bits[(pos / 64) as usize] |= 1u64 << (pos % 64);
        }
    }

    /// Probe with an explicit hash count. `k=0` returns `true`
    /// (item is "always present" — its bits weren't actually inserted).
    #[inline]
    fn contains_k(&self, h1: u32, h2: u32, k: u32) -> bool {
        let mask = self.mask;
        for probe_idx in 0..k {
            let raw = (h1 as u64).wrapping_add((probe_idx as u64).wrapping_mul(h2 as u64));
            let pos = raw & mask;
            if (self.bits[(pos / 64) as usize] >> (pos % 64)) & 1 == 0 {
                return false;
            }
        }
        true
    }

    fn byte_size(&self) -> usize {
        self.bits.len() * size_of::<u64>()
    }
}

/// FNV-1a 32-bit hash. Used as `h1`; `h2 = h1 ^ 0x27d4eb2f`.
#[inline]
fn fnv1a_32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

#[inline]
fn hash_pair(bytes: &[u8]) -> (u32, u32) {
    let h1 = fnv1a_32(bytes);
    // Cheap, mostly-independent second hash. Murmur-style finaliser of h1.
    let mut h2 = h1 ^ 0x27d4_eb2f;
    h2 = h2.wrapping_mul(0x85eb_ca6b);
    h2 ^= h2 >> 13;
    h2 = h2.wrapping_mul(0xc2b2_ae35);
    h2 ^= h2 >> 16;
    (h1, h2)
}

// ---------------------------------------------------------------------------
//                              TrigramBloom
// ---------------------------------------------------------------------------

/// Bloom filter over every overlapping 3-byte window of the rows in a
/// chunk. The classic substring skip index (pg_trgm, ClickHouse
/// `ngrambf_v1`, Snowflake search optimization service).
///
/// Build cost: linear in total decoded bytes (one trigram hash per
/// byte). The Bloom is sized in bits; `bits_per_distinct_trigram = 10`
/// gives a ~1% false-positive rate per probe.
pub struct TrigramBloom {
    bloom: Bloom,
}

impl TrigramBloom {
    /// Build a trigram Bloom over rows `lo..hi` by streaming the
    /// decoded bytes via the OnPair `dict + codes` view. Never
    /// materialises the full decompressed buffer.
    ///
    /// `bits_per_row` controls the Bloom size in bits per row of the
    /// chunk; with rows of ~100 bytes and ~100 distinct trigrams per
    /// row, 64 bits/row sizes the Bloom for ~1% false-positive rate.
    pub fn build(dv: &DecodeView<'_>, lo: usize, hi: usize, bits_per_row: usize) -> Self {
        let n_rows = hi - lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        let mut row_buf: Vec<u8> = Vec::with_capacity(256);
        for r in lo..hi {
            row_buf.clear();
            dv.decode_row_into(r, &mut row_buf);
            for win in row_buf.windows(3) {
                let (h1, h2) = hash_pair(win);
                bloom.insert(h1, h2);
            }
        }
        Self { bloom }
    }

    /// Build a trigram Bloom from raw row bytes. Compression-agnostic:
    /// the caller supplies an iterator over each row's decoded bytes
    /// (which can come from any encoding, or from a raw Utf8 / Binary
    /// column with no compression at all).
    pub fn build_from_strings<I, S>(rows: I, n_rows: usize, bits_per_row: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        for row in rows {
            for win in row.as_ref().windows(3) {
                let (h1, h2) = hash_pair(win);
                bloom.insert(h1, h2);
            }
        }
        Self { bloom }
    }

    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'`. Returns
    /// `false` only when at least one 3-gram of `needle` is missing
    /// from the chunk. Needles shorter than 3 bytes are not skippable
    /// by this index (returns `true` unconditionally).
    pub fn might_contain(&self, needle: &[u8]) -> bool {
        if needle.len() < 3 {
            return true;
        }
        for win in needle.windows(3) {
            let (h1, h2) = hash_pair(win);
            if !self.bloom.contains(h1, h2) {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
//                              SeamBloom
// ---------------------------------------------------------------------------

/// Bloom filter over only the 3-grams that span a dict-token seam in
/// this chunk. Combined with [`DictPresence`] and the column's shared
/// dictionary, this reproduces the chunk's full trigram set while
/// storing only the *boundary* trigrams.
///
/// The trigrams of a chunk's decoded text decompose into:
/// * Interior trigrams that lie wholly inside a single dict token.
///   Recoverable from the dictionary (constant data shared across
///   chunks) + the chunk's `DictPresence` bitmap.
/// * Seam trigrams that straddle two adjacent tokens. These are
///   chunk-local and have to be stored explicitly.
///
/// For tokens of length ≥ 2, exactly two seam trigrams sit at each
/// boundary between consecutive tokens `t_i, t_{i+1}`:
/// `last2(t_i) || first1(t_{i+1})` and `last1(t_i) || first2(t_{i+1})`.
/// Tokens shorter than 2 bytes degenerate; we handle them by walking
/// the last 2 bytes of the running boundary buffer instead.
///
/// Per-chunk insert cost is `O(num_tokens)` and the structure is much
/// smaller than [`TrigramBloom`] for chunks where most trigrams are
/// interior (which is the OnPair common case).
pub struct SeamBloom {
    bloom: Bloom,
}

impl SeamBloom {
    /// Build a seam-trigram Bloom over rows `lo..hi`.
    pub fn build(dv: &DecodeView<'_>, lo: usize, hi: usize, bits_per_row: usize) -> Self {
        let n_rows = hi - lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        // We slide a 2-byte tail window across the row. At each token
        // boundary we emit up to 2 seam trigrams: `tail[0..2] ||
        // next_first_byte` and `tail[1] || next_first_two`.
        let row_lo = dv.codes_offsets[lo] as usize;
        // Walk row by row to reset the tail (no seams across rows).
        let mut rstart = row_lo;
        for r in lo..hi {
            let rend = dv.codes_offsets[r + 1] as usize;
            let tokens = &dv.codes[rstart..rend];
            // `tail` holds up to the last 2 bytes seen so far in this row.
            let mut tail: [u8; 2] = [0; 2];
            let mut tail_len: usize = 0;
            for (i, &tok) in tokens.iter().enumerate() {
                let entry = dv.dict_table[tok as usize];
                let off = (entry >> 16) as usize;
                let len = (entry & 0xffff) as usize;
                let bytes = &dv.dict_bytes[off..off + len];
                if i > 0 && tail_len > 0 && !bytes.is_empty() {
                    // Seam trigram(s).
                    if tail_len >= 2 && len >= 1 {
                        let t: [u8; 3] = [tail[0], tail[1], bytes[0]];
                        let (h1, h2) = hash_pair(&t);
                        bloom.insert(h1, h2);
                    }
                    if tail_len >= 1 && len >= 2 {
                        let t: [u8; 3] = [tail[tail_len - 1], bytes[0], bytes[1]];
                        let (h1, h2) = hash_pair(&t);
                        bloom.insert(h1, h2);
                    }
                }
                // Update the tail with the last 2 bytes of `bytes`.
                if len >= 2 {
                    tail[0] = bytes[len - 2];
                    tail[1] = bytes[len - 1];
                    tail_len = 2;
                } else if len == 1 {
                    if tail_len == 0 {
                        tail[0] = bytes[0];
                        tail_len = 1;
                    } else {
                        tail[0] = tail[tail_len - 1];
                        tail[1] = bytes[0];
                        tail_len = 2;
                    }
                }
            }
            rstart = rend;
        }
        Self { bloom }
    }

    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'` when combined
    /// with a [`DictPresence`] bitmap.
    ///
    /// For each 3-gram `g` of `needle`, `g` must either appear
    /// interior to some present dict token (checked against
    /// `DictPresence` + the shared dict) **or** in the seam Bloom.
    /// If any 3-gram is missing from both, the chunk cannot contain
    /// the needle.
    pub fn might_contain(
        &self,
        dv: &DecodeView<'_>,
        presence: &DictPresence,
        needle: &[u8],
    ) -> bool {
        if needle.len() < 3 {
            return true;
        }
        for win in needle.windows(3) {
            let (h1, h2) = hash_pair(win);
            if self.bloom.contains(h1, h2) {
                continue;
            }
            // 3-gram not in seam Bloom; must appear interior to some
            // present dict token.
            let mut found = false;
            for (id, &entry) in dv.dict_table.iter().enumerate() {
                let len = (entry & 0xffff) as usize;
                if len < 3 || !presence.is_set(id) {
                    continue;
                }
                let off = (entry >> 16) as usize;
                if memchr::memmem::find(&dv.dict_bytes[off..off + len], win).is_some() {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
//                             TokenPairBloom
// ---------------------------------------------------------------------------

/// Bloom filter over **consecutive code pairs** `(c_i, c_{i+1})` in the
/// chunk's OnPair code stream. The OnPair-structural alternative to a
/// byte-level [`TrigramBloom`].
///
/// What it catches that [`DictPresence`] does not: **adjacency**. Two
/// dict ids can both appear in a chunk without ever occurring next to
/// each other; the pair Bloom remembers the actual pairs that did.
///
/// Per-row build cost is O(tokens_in_row). Insertion count is roughly
/// `n_tokens - n_rows` per chunk, which for OnPair-encoded URLs is a
/// few thousand — typically smaller than the chunk's distinct trigram
/// set, giving slightly better FPR at the same byte budget.
///
/// Per-query cost is `O(|needle| × candidate_pairs)` because the
/// contains check enumerates dict-pair candidates per split position
/// of the needle. For URL-shaped data and short needles (≤ 16 bytes)
/// this is a few hundred dict-pair probes per chunk, which amortises
/// well across all chunks for a single query.
pub struct TokenPairBloom {
    bloom: Bloom,
}

impl TokenPairBloom {
    /// Build the pair Bloom for rows `lo..hi`.
    pub fn build(dv: &DecodeView<'_>, lo: usize, hi: usize, bits_per_row: usize) -> Self {
        let n_rows = hi - lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        let mut rstart = dv.codes_offsets[lo] as usize;
        for r in lo..hi {
            let rend = dv.codes_offsets[r + 1] as usize;
            let toks = &dv.codes[rstart..rend];
            for w in toks.windows(2) {
                let (h1, h2) = pair_hash(w[0], w[1]);
                bloom.insert(h1, h2);
            }
            rstart = rend;
        }
        Self { bloom }
    }

    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `col = needle` when paired with a
    /// [`DictPresence`]. Greedy-LPM tokenises the needle and requires
    /// every consecutive `(tᵢ, tᵢ₊₁)` to be present in the pair Bloom
    /// (and every `tᵢ` to be present in `presence`).
    pub fn might_eq(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        presence: &DictPresence,
        needle: &[u8],
    ) -> bool {
        let Some(toks) = tokenize_needle(dv, index, needle) else {
            return false;
        };
        if !toks.iter().all(|&t| presence.is_set(t as usize)) {
            return false;
        }
        for w in toks.windows(2) {
            let (h1, h2) = pair_hash(w[0], w[1]);
            if !self.bloom.contains(h1, h2) {
                return false;
            }
        }
        true
    }

    /// Sound necessary condition for `LIKE '%needle%'` when paired
    /// with a [`DictPresence`]. Covers:
    ///
    /// * Case 1 — the needle lies wholly inside a single present dict
    ///   token. Verified via the presence bitmap and the dictionary.
    /// * Case 2 — the needle straddles exactly two tokens `(a, b)`
    ///   such that `decode(a) · decode(b)` contains the needle. We
    ///   enumerate all `(s1, s2)` splits of the needle (`|s1|, |s2|
    ///   ≥ 1`); for each split we look up dict tokens whose bytes end
    ///   with `s1` and whose bytes start with `s2` (using the sorted
    ///   first-byte index), then probe the pair Bloom for each
    ///   `(a, b)` candidate.
    /// * Case 3+ — the needle spans three or more tokens. We do not
    ///   currently rule this out; the function returns `true`
    ///   conservatively after exhausting cases 1 and 2. For URL-
    ///   shaped data this case is rare because dict tokens are
    ///   typically longer than 2-3 bytes.
    pub fn might_contain(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        presence: &DictPresence,
        needle: &[u8],
    ) -> bool {
        if needle.is_empty() {
            return true;
        }
        // Case 1.
        for (id, &entry) in dv.dict_table.iter().enumerate() {
            let len = (entry & 0xffff) as usize;
            if len < needle.len() || !presence.is_set(id) {
                continue;
            }
            let off = (entry >> 16) as usize;
            if memchr::memmem::find(&dv.dict_bytes[off..off + len], needle).is_some() {
                return true;
            }
        }
        // Case 2: enumerate all (s1, s2) splits.
        for split in 1..needle.len() {
            let s1 = &needle[..split];
            let s2 = &needle[split..];
            // Tokens b candidates whose first byte is s2[0].
            let b_range = index.range_for(s2[0]);
            if b_range.is_empty() {
                continue;
            }
            // Linear scan dict for tokens ending with s1.
            for (a, &a_entry) in dv.dict_table.iter().enumerate() {
                let a_len = (a_entry & 0xffff) as usize;
                if a_len < s1.len() {
                    continue;
                }
                let a_off = (a_entry >> 16) as usize;
                let a_bytes = &dv.dict_bytes[a_off..a_off + a_len];
                if &a_bytes[a_len - s1.len()..] != s1 {
                    continue;
                }
                for b in b_range.clone() {
                    let b_entry = dv.dict_table[b];
                    let b_len = (b_entry & 0xffff) as usize;
                    if b_len < s2.len() {
                        continue;
                    }
                    let b_off = (b_entry >> 16) as usize;
                    let b_bytes = &dv.dict_bytes[b_off..b_off + b_len];
                    if !b_bytes.starts_with(s2) {
                        continue;
                    }
                    let (h1, h2) = pair_hash(a as u16, b as u16);
                    if self.bloom.contains(h1, h2) {
                        return true;
                    }
                }
            }
        }
        // Case 3+: needle spans ≥ 3 tokens. Reduce to a trigram check
        // that uses the pair Bloom as a *seam-trigram oracle*: every
        // 3-byte window of `needle` must be reachable in the chunk
        // either as an interior trigram of a present dict token or as
        // a seam trigram of a present token pair. If any trigram is
        // unaccounted for, the chunk cannot contain the needle.
        if needle.len() < 3 {
            return true;
        }
        for win in needle.windows(3) {
            if self.trigram_reachable(dv, index, presence, win) {
                continue;
            }
            return false;
        }
        true
    }

    /// True iff trigram `tri` (3 bytes) can appear in this chunk via
    /// either the interior of a present dict token or the seam of a
    /// present token pair.
    fn trigram_reachable(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        presence: &DictPresence,
        tri: &[u8],
    ) -> bool {
        debug_assert_eq!(tri.len(), 3);
        // (a) interior: any present dict token contains `tri` as a
        // substring.
        for (id, &entry) in dv.dict_table.iter().enumerate() {
            let len = (entry & 0xffff) as usize;
            if len < 3 || !presence.is_set(id) {
                continue;
            }
            let off = (entry >> 16) as usize;
            if memchr::memmem::find(&dv.dict_bytes[off..off + len], tri).is_some() {
                return true;
            }
        }
        // (b) seam, 2 tokens: try splits (1+2) and (2+1).
        for split in 1..3 {
            let s1 = &tri[..split];
            let s2 = &tri[split..];
            if self.has_pair_with(dv, index, s1, s2) {
                return true;
            }
        }
        // (c) seam, 3 tokens: the only fit for a 3-byte trigram is
        // 1+1+1 (a one-byte middle token). OnPair training guarantees
        // every single-byte token exists in the dictionary, so we can
        // look up `m = single-byte-token(tri[1])` directly.
        if let Some(m) = single_byte_token(dv, index, tri[1]) {
            // a ends with tri[0]; pair (a, m) in Bloom.
            // b starts with tri[2]; pair (m, b) in Bloom.
            let mut left_ok = false;
            for (a, &a_entry) in dv.dict_table.iter().enumerate() {
                let a_len = (a_entry & 0xffff) as usize;
                if a_len == 0 {
                    continue;
                }
                let a_off = (a_entry >> 16) as usize;
                if dv.dict_bytes[a_off + a_len - 1] != tri[0] {
                    continue;
                }
                let (h1, h2) = pair_hash(a as u16, m);
                if self.bloom.contains(h1, h2) {
                    left_ok = true;
                    break;
                }
            }
            if left_ok {
                let b_range = index.range_for(tri[2]);
                for b in b_range {
                    let (h1, h2) = pair_hash(m, b as u16);
                    if self.bloom.contains(h1, h2) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// True iff some dict pair `(a, b)` is in the Bloom with `a`'s
    /// bytes ending in `s1` and `b`'s bytes starting with `s2`.
    fn has_pair_with(&self, dv: &DecodeView<'_>, index: &DictIndex, s1: &[u8], s2: &[u8]) -> bool {
        if s2.is_empty() {
            return false;
        }
        let b_range = index.range_for(s2[0]);
        if b_range.is_empty() {
            return false;
        }
        for (a, &a_entry) in dv.dict_table.iter().enumerate() {
            let a_len = (a_entry & 0xffff) as usize;
            if a_len < s1.len() {
                continue;
            }
            let a_off = (a_entry >> 16) as usize;
            let a_bytes = &dv.dict_bytes[a_off..a_off + a_len];
            if &a_bytes[a_len - s1.len()..] != s1 {
                continue;
            }
            for b in b_range.clone() {
                let b_entry = dv.dict_table[b];
                let b_len = (b_entry & 0xffff) as usize;
                if b_len < s2.len() {
                    continue;
                }
                let b_off = (b_entry >> 16) as usize;
                let b_bytes = &dv.dict_bytes[b_off..b_off + b_len];
                if !b_bytes.starts_with(s2) {
                    continue;
                }
                let (h1, h2) = pair_hash(a as u16, b as u16);
                if self.bloom.contains(h1, h2) {
                    return true;
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
//                            CodeBigramBloom
// ---------------------------------------------------------------------------

/// Bloom filter over consecutive code pairs like [`TokenPairBloom`], but
/// with a sound, precise `might_contain` for `LIKE '%needle%'` that
/// enumerates **all valid tokenisations** of the needle instead of
/// falling back to trigram checks for needles spanning 3+ tokens.
///
/// The build phase is identical to `TokenPairBloom`. The query phase
/// runs a small DFA / DP over the needle bytes, trying every dict token
/// that matches at each position. Because OnPair tokens are 1–16 bytes
/// and needles are typically ≤ 30 bytes, the state space is tiny.
///
/// Sound: if a row truly contains the needle as a substring, then the
/// row's code stream includes the needle tokenised at some alignment.
/// The DP explores all such alignments and checks each code bigram
/// against the bloom. At least one alignment must pass.
pub struct CodeBigramBloom {
    bloom: Bloom,
}

impl CodeBigramBloom {
    /// Build the code-bigram Bloom for rows `lo..hi`.
    pub fn build(dv: &DecodeView<'_>, lo: usize, hi: usize, bits_per_row: usize) -> Self {
        let n_rows = hi - lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        let mut rstart = dv.codes_offsets[lo] as usize;
        for r in lo..hi {
            let rend = dv.codes_offsets[r + 1] as usize;
            let toks = &dv.codes[rstart..rend];
            for w in toks.windows(2) {
                let (h1, h2) = pair_hash(w[0], w[1]);
                bloom.insert(h1, h2);
            }
            rstart = rend;
        }
        Self { bloom }
    }

    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'`.
    ///
    /// Exploits the fact that greedy LPM is **deterministic**: once we
    /// know the first token boundary falls at byte `cover` in the
    /// needle, `tokenize_needle(needle[cover..])` gives the unique
    /// interior token sequence.  All interior code bigrams except the
    /// last must be in the bloom. The last token may differ from the
    /// actual row because the row has bytes beyond the needle that the
    /// greedy tokeniser can see.
    ///
    /// We enumerate `MAX_TOKEN_SIZE + 1` cover values (0 = needle
    /// starts at a token boundary, 1..=16 = needle starts `cover`
    /// bytes before the end of some entry token).
    pub fn might_contain(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        presence: &DictPresence,
        needle: &[u8],
    ) -> bool {
        if needle.is_empty() {
            return true;
        }

        // Case: needle fits entirely inside a single dict token.
        for (id, &entry) in dv.dict_table.iter().enumerate() {
            let tlen = (entry & 0xffff) as usize;
            if tlen < needle.len() || !presence.is_set(id) {
                continue;
            }
            let off = (entry >> 16) as usize;
            if memchr::memmem::find(&dv.dict_bytes[off..off + tlen], needle).is_some() {
                return true;
            }
        }

        // cover = 0: needle starts at a token boundary.
        if self.check_aligned(dv, index, needle, None) {
            return true;
        }

        // cover = 1..max_token_len: needle starts inside a token.
        let max_cover = 16.min(needle.len());
        for cover in 1..=max_cover {
            let suffix = &needle[..cover];
            let remainder = &needle[cover..];
            if remainder.is_empty() {
                continue; // handled by single-token case above
            }

            // Find all present entry tokens ending with `suffix`,
            // and check if any of them leads to a valid path.
            for (id, &entry) in dv.dict_table.iter().enumerate() {
                let tlen = (entry & 0xffff) as usize;
                if tlen < cover || !presence.is_set(id) {
                    continue;
                }
                let off = (entry >> 16) as usize;
                if &dv.dict_bytes[off + tlen - cover..off + tlen] != suffix {
                    continue;
                }
                if self.check_aligned(dv, index, remainder, Some(id as u16)) {
                    return true;
                }
            }
        }

        false
    }

    /// Debug: find a row containing needle and print its actual
    /// tokenization to understand why might_contain fails.
    pub fn debug_fn(
        &self,
        dv: &DecodeView<'_>,
        _index: &DictIndex,
        _presence: &DictPresence,
        needle: &[u8],
        lo: usize,
        hi: usize,
    ) {
        // Find a row containing the needle
        for r in lo..hi {
            let rlo = dv.codes_offsets[r] as usize;
            let rhi = dv.codes_offsets[r + 1] as usize;
            let toks = &dv.codes[rlo..rhi];
            // Decode the row
            let mut decoded = Vec::new();
            let mut tok_boundaries = vec![0usize]; // byte offset of each token start
            for &c in toks {
                let entry = dv.dict_table[c as usize];
                let off = (entry >> 16) as usize;
                let len = (entry & 0xffff) as usize;
                decoded.extend_from_slice(&dv.dict_bytes[off..off + len]);
                tok_boundaries.push(decoded.len());
            }
            // Check if needle is in this row
            if let Some(needle_pos) = memchr::memmem::find(&decoded, needle) {
                eprintln!("  Row {r}: needle at byte {needle_pos}");
                // Find which token(s) the needle spans
                for (i, &c) in toks.iter().enumerate() {
                    let tok_start = tok_boundaries[i];
                    let tok_end = tok_boundaries[i + 1];
                    if tok_end > needle_pos && tok_start < needle_pos + needle.len() {
                        let entry = dv.dict_table[c as usize];
                        let off = (entry >> 16) as usize;
                        let len = (entry & 0xffff) as usize;
                        let tb = &dv.dict_bytes[off..off + len];
                        let cover_bytes = if tok_start <= needle_pos {
                            tok_end - needle_pos
                        } else {
                            0
                        };
                        eprintln!(
                            "    tok[{i}] id={c} {:?} bytes={tok_start}..{tok_end} (cover from needle start: {cover_bytes})",
                            std::str::from_utf8(tb).unwrap_or("?")
                        );
                        if i > 0 {
                            let prev = toks[i - 1];
                            let (h1, h2) = pair_hash(prev, c);
                            eprintln!(
                                "      bigram({prev},{c}) in bloom: {}",
                                self.bloom.contains(h1, h2)
                            );
                        }
                    }
                }
                // Show what cover value this corresponds to
                let first_tok_idx =
                    tok_boundaries.iter().position(|&b| b > needle_pos).unwrap() - 1;
                let cover = tok_boundaries[first_tok_idx + 1] - needle_pos;
                eprintln!(
                    "    → cover = {cover} (needle starts {0} bytes from end of token {first_tok_idx})",
                    needle_pos - tok_boundaries[first_tok_idx]
                );
                return;
            }
        }
        eprintln!(
            "  No row found containing needle {:?} in range {lo}..{hi} ({} rows)",
            std::str::from_utf8(needle).unwrap_or("?"),
            hi - lo
        );
    }

    /// Check one alignment: delegates to the shared free function.
    fn check_aligned(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        remainder: &[u8],
        entry: Option<u16>,
    ) -> bool {
        check_aligned_on_bloom(&self.bloom, dv, index, remainder, entry)
    }
}

// ---------------------------------------------------------------------------
//                             HybridBloom
// ---------------------------------------------------------------------------

/// Code 2+3-gram Bloom filter.  Stores both consecutive **pairs**
/// and consecutive **triples** of OnPair codes in a single bloom.
/// Bigrams give signal on short needles (3+ tokens), trigrams add
/// extra specificity on longer needles (4+ tokens).  Total
/// insertions ~2N-3 per row — still ~15× sparser than byte trigrams.
///
/// Query: encode the needle via the dict, then check both code
/// bigrams and code trigrams.  If either finds a missing entry →
/// skip.
/// BitFunnel-style **frequency-conscious** code-bigram skip filter.
///
/// Builds a per-chunk bloom over OnPair code bigrams, but skips
/// inserting (and probing) bigrams that appear in too many chunks
/// to be discriminative.  The skipped "ubiquitous" set is column-
/// level metadata shared across all chunks.
///
/// Inspired by BitFunnel (Bing, SIGIR 2017): common terms saturate
/// every signature equally, so omitting them concentrates bits on
/// the rare bigrams that actually carry pruning signal.
///
/// Sound: skipping a ubiquitous bigram during query is equivalent
/// to assuming the chunk contains it (which it almost certainly
/// does — that's why it's ubiquitous).  Skipping during build saves
/// space without changing soundness.
pub struct HybridBloom {
    bloom: Bloom,
}

/// Column-level table of code bigrams that appear in too many
/// chunks to be discriminative. Shared across all per-chunk
/// [`HybridBloom`] instances.
///
/// Stored as a sorted `Vec<u32>` of packed bigram IDs
/// `(a << 16) | b` for O(log n) lookup. ~4 KB for 1000 entries.
pub struct UbiquitousBigrams {
    sorted: Vec<u32>,
}

impl UbiquitousBigrams {
    /// Build the ubiquitous-bigram set from the column's full code
    /// stream.  A bigram is considered ubiquitous iff it appears in
    /// more than `threshold_pct`% of chunks of size `chunk_size`.
    pub fn build(
        codes: &[u16],
        codes_offsets: &[u32],
        chunk_size: usize,
        threshold_pct: u8,
    ) -> Self {
        let n_rows = codes_offsets.len().saturating_sub(1);
        let n_chunks = n_rows / chunk_size;
        if n_chunks == 0 || threshold_pct == 0 {
            return Self { sorted: Vec::new() };
        }
        let threshold = (n_chunks * threshold_pct as usize) / 100;

        // Count chunks each bigram appears in. Use a HashMap keyed
        // on packed u32 bigram IDs.
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for c in 0..n_chunks {
            seen.clear();
            let row_lo = c * chunk_size;
            let row_hi = (c + 1) * chunk_size;
            for r in row_lo..row_hi {
                let rlo = codes_offsets[r] as usize;
                let rhi = codes_offsets[r + 1] as usize;
                for w in codes[rlo..rhi].windows(2) {
                    let key = ((w[0] as u32) << 16) | (w[1] as u32);
                    seen.insert(key);
                }
            }
            for &key in &seen {
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<u32> = counts
            .into_iter()
            .filter(|&(_, c)| c as usize > threshold)
            .map(|(k, _)| k)
            .collect();
        sorted.sort_unstable();
        Self { sorted }
    }

    /// Empty ubiquitous set (no filtering).
    pub fn empty() -> Self {
        Self { sorted: Vec::new() }
    }

    #[inline]
    pub fn contains(&self, a: u16, b: u16) -> bool {
        let key = ((a as u32) << 16) | (b as u32);
        self.sorted.binary_search(&key).is_ok()
    }

    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    pub fn byte_size(&self) -> usize {
        self.sorted.len() * size_of::<u32>()
    }
}

/// Per-bigram tier table — assigns each bigram a probe count `k`
/// based on its global frequency, in the spirit of BitFunnel's
/// frequency-conscious row allocation.
///
/// Common bigrams get `k=0` (skipped entirely), rare bigrams get
/// `k=3+` (high precision). Stored as sorted `Vec<(packed_id, k)>`
/// pairs for binary-search lookup. Default tier (for bigrams not in
/// the table) is `k=3`.
pub struct BigramTiers {
    /// Sorted (packed_bigram_id, k) pairs.
    entries: Vec<(u32, u8)>,
    /// Default k for bigrams not in the table.
    default_k: u8,
}

impl BigramTiers {
    /// Build tier table from full column codes.
    ///
    /// Tiers (by % of chunks containing the bigram):
    ///   > top_pct%        → k=0 (skip)
    ///   common_pct..top   → k=1
    ///   medium_pct..common→ k=2
    ///   ≤ medium_pct      → k=3 (default; not stored)
    pub fn build(
        codes: &[u16],
        codes_offsets: &[u32],
        chunk_size: usize,
        top_pct: u8,
        common_pct: u8,
        medium_pct: u8,
    ) -> Self {
        let n_rows = codes_offsets.len().saturating_sub(1);
        let n_chunks = n_rows / chunk_size;
        if n_chunks == 0 {
            return Self {
                entries: Vec::new(),
                default_k: 3,
            };
        }
        let t_top = (n_chunks * top_pct as usize) / 100;
        let t_common = (n_chunks * common_pct as usize) / 100;
        let t_medium = (n_chunks * medium_pct as usize) / 100;

        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for c in 0..n_chunks {
            seen.clear();
            let row_lo = c * chunk_size;
            let row_hi = (c + 1) * chunk_size;
            for r in row_lo..row_hi {
                let rlo = codes_offsets[r] as usize;
                let rhi = codes_offsets[r + 1] as usize;
                for w in codes[rlo..rhi].windows(2) {
                    let key = ((w[0] as u32) << 16) | (w[1] as u32);
                    seen.insert(key);
                }
            }
            for &key in &seen {
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        // Assign tier per bigram. Only store entries with k < default_k=3.
        let mut entries: Vec<(u32, u8)> = counts
            .into_iter()
            .filter_map(|(k, c)| {
                let c = c as usize;
                let tier = if c > t_top {
                    0u8
                } else if c > t_common {
                    1
                } else if c > t_medium {
                    2
                } else {
                    return None; // default tier, no entry needed
                };
                Some((k, tier))
            })
            .collect();
        entries.sort_unstable_by_key(|&(k, _)| k);
        Self {
            entries,
            default_k: 3,
        }
    }

    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            default_k: 3,
        }
    }

    /// Get the probe count `k` for a given bigram.
    #[inline]
    pub fn k_for(&self, a: u16, b: u16) -> u32 {
        let key = ((a as u32) << 16) | (b as u32);
        match self.entries.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(i) => self.entries[i].1 as u32,
            Err(_) => self.default_k as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn byte_size(&self) -> usize {
        self.entries.len() * (size_of::<u32>() + size_of::<u8>())
    }

    /// Diagnostic: count entries per tier.
    pub fn tier_counts(&self) -> [usize; 4] {
        let mut c = [0usize; 4];
        for &(_, k) in &self.entries {
            c[k as usize] += 1;
        }
        c
    }
}

impl HybridBloom {
    /// Build the code-bigram Bloom for rows `lo..hi`, skipping any
    /// bigrams that are in the column-level `ubiq` table.
    pub fn build(
        dv: &DecodeView<'_>,
        lo: usize,
        hi: usize,
        bits_per_row: usize,
        ubiq: &UbiquitousBigrams,
    ) -> Self {
        let n_rows = hi - lo;
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        let mut rstart = dv.codes_offsets[lo] as usize;
        for r in lo..hi {
            let rend = dv.codes_offsets[r + 1] as usize;
            let toks = &dv.codes[rstart..rend];
            for w in toks.windows(2) {
                if ubiq.contains(w[0], w[1]) {
                    continue;
                }
                let (h1, h2) = pair_hash(w[0], w[1]);
                bloom.insert(h1, h2);
            }
            rstart = rend;
        }
        Self { bloom }
    }

    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'`.
    ///
    /// Encode the needle via the OnPair dict, then check code bigrams
    /// against the bloom — skipping any ubiquitous bigrams (which
    /// would always probe "true" and contribute no signal).
    pub fn might_contain(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        presence: &DictPresence,
        ubiq: &UbiquitousBigrams,
        needle: &[u8],
    ) -> bool {
        if needle.is_empty() {
            return true;
        }

        // Single-token case: needle fits inside one dict token.
        for (id, &entry) in dv.dict_table.iter().enumerate() {
            let tlen = (entry & 0xffff) as usize;
            if tlen < needle.len() || !presence.is_set(id) {
                continue;
            }
            let off = (entry >> 16) as usize;
            if memchr::memmem::find(&dv.dict_bytes[off..off + tlen], needle).is_some() {
                return true;
            }
        }

        // cover = 0: needle starts at a token boundary.
        if check_aligned_with_ubiq(&self.bloom, dv, index, needle, None, ubiq) {
            return true;
        }

        // cover = 1..max_token_len: needle starts inside a token.
        let max_cover = 16.min(needle.len());
        for cover in 1..=max_cover {
            let suffix = &needle[..cover];
            let remainder = &needle[cover..];
            if remainder.is_empty() {
                continue;
            }
            for (id, &entry) in dv.dict_table.iter().enumerate() {
                let tlen = (entry & 0xffff) as usize;
                if tlen < cover || !presence.is_set(id) {
                    continue;
                }
                let off = (entry >> 16) as usize;
                if &dv.dict_bytes[off + tlen - cover..off + tlen] != suffix {
                    continue;
                }
                if check_aligned_with_ubiq(&self.bloom, dv, index, remainder, Some(id as u16), ubiq)
                {
                    return true;
                }
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
//                              TieredBloom
// ---------------------------------------------------------------------------

/// Full BitFunnel-style code-bigram filter with **variable hash count
/// per bigram** based on frequency tier. Rare bigrams get k=3+ hash
/// functions (high precision), common bigrams get k=1 (single bit),
/// ubiquitous bigrams get k=0 (skipped entirely).
///
/// The tier table is column-level metadata (shared across chunks).
pub struct TieredBloom {
    bloom: Bloom,
}

impl TieredBloom {
    /// Build the tiered bloom for rows `lo..hi` using `tiers` for
    /// the per-bigram k.
    pub fn build(
        dv: &DecodeView<'_>,
        lo: usize,
        hi: usize,
        bits_per_row: usize,
        tiers: &BigramTiers,
    ) -> Self {
        let n_rows = hi - lo;
        // Use k=3 as the bloom's default; per-insert k overrides it.
        let mut bloom = Bloom::new(bits_per_row * n_rows.max(1), 3);
        let mut rstart = dv.codes_offsets[lo] as usize;
        for r in lo..hi {
            let rend = dv.codes_offsets[r + 1] as usize;
            let toks = &dv.codes[rstart..rend];
            for w in toks.windows(2) {
                let k = tiers.k_for(w[0], w[1]);
                if k == 0 {
                    continue;
                }
                let (h1, h2) = pair_hash(w[0], w[1]);
                bloom.insert_k(h1, h2, k);
            }
            rstart = rend;
        }
        Self { bloom }
    }

    pub fn byte_size(&self) -> usize {
        self.bloom.byte_size()
    }

    /// Sound necessary condition for `LIKE '%needle%'` using tiered
    /// bigram probes. Bigrams with `k=0` are skipped (treated as
    /// always present), saving probes and avoiding zero-signal checks.
    pub fn might_contain(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        presence: &DictPresence,
        tiers: &BigramTiers,
        needle: &[u8],
    ) -> bool {
        if needle.is_empty() {
            return true;
        }

        for (id, &entry) in dv.dict_table.iter().enumerate() {
            let tlen = (entry & 0xffff) as usize;
            if tlen < needle.len() || !presence.is_set(id) {
                continue;
            }
            let off = (entry >> 16) as usize;
            if memchr::memmem::find(&dv.dict_bytes[off..off + tlen], needle).is_some() {
                return true;
            }
        }

        if check_aligned_tiered(&self.bloom, dv, index, needle, None, tiers) {
            return true;
        }

        let max_cover = 16.min(needle.len());
        for cover in 1..=max_cover {
            let suffix = &needle[..cover];
            let remainder = &needle[cover..];
            if remainder.is_empty() {
                continue;
            }
            for (id, &entry) in dv.dict_table.iter().enumerate() {
                let tlen = (entry & 0xffff) as usize;
                if tlen < cover || !presence.is_set(id) {
                    continue;
                }
                let off = (entry >> 16) as usize;
                if &dv.dict_bytes[off + tlen - cover..off + tlen] != suffix {
                    continue;
                }
                if check_aligned_tiered(&self.bloom, dv, index, remainder, Some(id as u16), tiers) {
                    return true;
                }
            }
        }

        false
    }
}

/// Tiered version of `check_aligned`: uses per-bigram k for probes.
/// Bigrams with k=0 are skipped (treated as always present).
fn check_aligned_tiered(
    bloom: &Bloom,
    dv: &DecodeView<'_>,
    index: &DictIndex,
    remainder: &[u8],
    entry: Option<u16>,
    tiers: &BigramTiers,
) -> bool {
    let Some(toks) = tokenize_needle(dv, index, remainder) else {
        return false;
    };
    if toks.is_empty() {
        return entry.is_none();
    }

    let mut starts = Vec::with_capacity(toks.len());
    let mut pos = 0usize;
    for &t in &toks {
        starts.push(pos);
        let e = dv.dict_table[t as usize];
        pos += (e & 0xffff) as usize;
    }

    let safe: Vec<bool> = starts
        .iter()
        .map(|&p| is_safe_position(dv, index, remainder, p))
        .collect();

    if let Some(e) = entry {
        if safe[0] {
            let k = tiers.k_for(e, toks[0]);
            if k > 0 {
                let (h1, h2) = pair_hash(e, toks[0]);
                if !bloom.contains_k(h1, h2, k) {
                    return false;
                }
            }
        }
    }

    for (i, w) in toks.windows(2).enumerate() {
        if safe[i] && safe[i + 1] {
            let k = tiers.k_for(w[0], w[1]);
            if k > 0 {
                let (h1, h2) = pair_hash(w[0], w[1]);
                if !bloom.contains_k(h1, h2, k) {
                    return false;
                }
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------
//                     Free functions for shared logic
// ---------------------------------------------------------------------------

/// Shared `check_aligned` logic that operates on an arbitrary `&Bloom`.
/// Used by both `CodeBigramBloom` and `HybridBloom`.
fn check_aligned_on_bloom(
    bloom: &Bloom,
    dv: &DecodeView<'_>,
    index: &DictIndex,
    remainder: &[u8],
    entry: Option<u16>,
) -> bool {
    let Some(toks) = tokenize_needle(dv, index, remainder) else {
        return false;
    };
    if toks.is_empty() {
        return entry.is_none();
    }

    let mut starts = Vec::with_capacity(toks.len());
    let mut pos = 0usize;
    for &t in &toks {
        starts.push(pos);
        let e = dv.dict_table[t as usize];
        pos += (e & 0xffff) as usize;
    }

    let safe: Vec<bool> = starts
        .iter()
        .map(|&p| is_safe_position(dv, index, remainder, p))
        .collect();

    if let Some(e) = entry {
        if safe[0] {
            let (h1, h2) = pair_hash(e, toks[0]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }

    for (i, w) in toks.windows(2).enumerate() {
        if safe[i] && safe[i + 1] {
            let (h1, h2) = pair_hash(w[0], w[1]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }

    true
}

/// Like `check_aligned_on_bloom` but also **skips probing ubiquitous
/// bigrams** (BitFunnel-style). Ubiquitous bigrams appear in nearly
/// every chunk's bloom and contribute zero pruning signal, so we
/// skip them on both insert and probe to free bloom space for the
/// rare bigrams that actually carry information.
fn check_aligned_with_ubiq(
    bloom: &Bloom,
    dv: &DecodeView<'_>,
    index: &DictIndex,
    remainder: &[u8],
    entry: Option<u16>,
    ubiq: &UbiquitousBigrams,
) -> bool {
    let Some(toks) = tokenize_needle(dv, index, remainder) else {
        return false;
    };
    if toks.is_empty() {
        return entry.is_none();
    }

    let mut starts = Vec::with_capacity(toks.len());
    let mut pos = 0usize;
    for &t in &toks {
        starts.push(pos);
        let e = dv.dict_table[t as usize];
        pos += (e & 0xffff) as usize;
    }

    let safe: Vec<bool> = starts
        .iter()
        .map(|&p| is_safe_position(dv, index, remainder, p))
        .collect();

    if let Some(e) = entry {
        if safe[0] && !ubiq.contains(e, toks[0]) {
            let (h1, h2) = pair_hash(e, toks[0]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }

    for (i, w) in toks.windows(2).enumerate() {
        if safe[i] && safe[i + 1] && !ubiq.contains(w[0], w[1]) {
            let (h1, h2) = pair_hash(w[0], w[1]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }

    true
}

/// Like `check_aligned_on_bloom` but checks code **trigrams** (windows
/// of 3 codes). Each probe captures ~3–9 bytes of content. Needs 4+
/// tokens to have any checkable trigram (skip the last window since
/// the exit token may differ).
fn check_aligned_trigram(
    bloom: &Bloom,
    dv: &DecodeView<'_>,
    index: &DictIndex,
    remainder: &[u8],
    entry: Option<u16>,
) -> bool {
    let Some(toks) = tokenize_needle(dv, index, remainder) else {
        return false;
    };
    if toks.is_empty() {
        return entry.is_none();
    }

    let mut starts = Vec::with_capacity(toks.len());
    let mut pos = 0usize;
    for &t in &toks {
        starts.push(pos);
        let e = dv.dict_table[t as usize];
        pos += (e & 0xffff) as usize;
    }

    let safe: Vec<bool> = starts
        .iter()
        .map(|&p| is_safe_position(dv, index, remainder, p))
        .collect();

    // Entry trigram: (entry, toks[0], toks[1]) — check if all safe.
    if let Some(e) = entry {
        if toks.len() >= 2 && safe[0] && safe[1] {
            let (h1, h2) = triple_hash(e, toks[0], toks[1]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }

    // Interior trigrams: (toks[i], toks[i+1], toks[i+2]).
    // Skip the last window (involves the exit token which may differ).
    for (i, w) in toks.windows(3).enumerate() {
        if safe[i] && safe[i + 1] && safe[i + 2] {
            let (h1, h2) = triple_hash(w[0], w[1], w[2]);
            if !bloom.contains(h1, h2) {
                return false;
            }
        }
    }

    true
}

/// True iff the greedy-LPM token at position `pos` within `remainder`
/// is guaranteed the same in the actual row. Shared by
/// `CodeBigramBloom` and `HybridBloom`.
#[inline]
fn is_safe_position(dv: &DecodeView<'_>, index: &DictIndex, remainder: &[u8], pos: usize) -> bool {
    let remaining = &remainder[pos..];
    if remaining.is_empty() {
        return false;
    }
    let candidates = index.range_for(remaining[0]);
    for id in candidates {
        let entry = dv.dict_table[id];
        let tlen = (entry & 0xffff) as usize;
        if tlen <= remaining.len() {
            continue;
        }
        let off = (entry >> 16) as usize;
        let tbytes = &dv.dict_bytes[off..off + tlen];
        if tbytes.starts_with(remaining) {
            return false;
        }
    }
    true
}

/// Find the single-byte dictionary token for byte `b`, if any. OnPair
/// training always includes the 256 single-byte tokens, so this is
/// `Some(_)` for any non-degenerate dictionary.
fn single_byte_token(dv: &DecodeView<'_>, index: &DictIndex, b: u8) -> Option<u16> {
    for id in index.range_for(b) {
        let entry = dv.dict_table[id];
        let len = (entry & 0xffff) as usize;
        if len == 1 {
            return Some(id as u16);
        }
    }
    None
}

#[inline]
fn pair_hash(a: u16, b: u16) -> (u32, u32) {
    let key = ((a as u32) << 16) | (b as u32);
    let h1 = splitmix32(key);
    let h2 = splitmix32(key ^ 0x27d4_eb2f);
    (h1, h2)
}

#[inline]
fn triple_hash(a: u16, b: u16, c: u16) -> (u32, u32) {
    // Mix three code IDs into two 32-bit hashes for the bloom.
    // Combine into a 48-bit key, then split-mix.
    let lo = ((a as u32) << 16) | (b as u32);
    let hi = c as u32;
    let h1 = splitmix32(lo ^ hi.wrapping_mul(0x9e37_79b9));
    let h2 = splitmix32(lo.wrapping_mul(0x85eb_ca6b) ^ hi);
    (h1, h2)
}

#[inline]
fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    x ^ (x >> 16)
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;
    use crate::DEFAULT_DICT12_CONFIG;
    use crate::decode::OwnedDecodeInputs;
    use crate::onpair_compress;

    fn build_inputs(strings: &[&str]) -> OwnedDecodeInputs {
        let varbin = VarBinArray::from_iter(
            strings.iter().map(|s| Some(s.as_bytes())),
            DType::Utf8(Nullability::NonNullable),
        );
        let arr =
            onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        OwnedDecodeInputs::collect(arr.as_view(), &mut ctx).unwrap()
    }

    #[test]
    fn presence_eq_no_false_negatives() {
        let strings: Vec<String> = (0..200).map(|i| format!("row-{i:04}-tail")).collect();
        let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let inputs = build_inputs(&str_refs);
        let dv = inputs.view();
        let index = DictIndex::build(&dv);

        let n_rows = dv.codes_offsets.len() - 1;
        let presence = DictPresence::build(&dv, 0, n_rows);

        for s in &strings {
            assert!(
                presence.might_eq(&dv, &index, s.as_bytes()),
                "false negative for present row {s:?}"
            );
        }
    }

    #[test]
    fn presence_starts_with_no_false_negatives() {
        let strings: &[&str] = &[
            "https://example.com/items/0001",
            "https://example.com/items/0002",
            "https://example.com/users/abc",
            "ftp://other.example.com/x",
        ];
        let inputs = build_inputs(strings);
        let dv = inputs.view();
        let index = DictIndex::build(&dv);
        let n_rows = dv.codes_offsets.len() - 1;
        let presence = DictPresence::build(&dv, 0, n_rows);

        assert!(presence.might_starts_with(&dv, &index, b"https://"));
        assert!(presence.might_starts_with(&dv, &index, b"https://example.com/items/"));
        assert!(presence.might_starts_with(&dv, &index, b"ftp://"));
        // Definitely-absent prefix uses bytes not all present together.
        assert!(!presence.might_starts_with(&dv, &index, b"zzzzz://"));
    }

    #[test]
    fn trigram_bloom_no_false_negatives() {
        let strings: Vec<String> = (0..200)
            .map(|i| format!("https://example.com/items/{i:04}"))
            .collect();
        let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let inputs = build_inputs(&str_refs);
        let dv = inputs.view();
        let n_rows = dv.codes_offsets.len() - 1;
        let bloom = TrigramBloom::build(&dv, 0, n_rows, 64);
        assert!(bloom.might_contain(b"example"));
        assert!(bloom.might_contain(b"/items/"));
        assert!(bloom.might_contain(b"0042"));
    }

    #[test]
    fn seam_bloom_no_false_negatives() {
        let strings: Vec<String> = (0..200)
            .map(|i| format!("https://example.com/items/{i:04}"))
            .collect();
        let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let inputs = build_inputs(&str_refs);
        let dv = inputs.view();
        let n_rows = dv.codes_offsets.len() - 1;
        let presence = DictPresence::build(&dv, 0, n_rows);
        let seam = SeamBloom::build(&dv, 0, n_rows, 64);

        // Substrings of actual rows must report `might_contain`.
        for s in [b"https".as_slice(), b"example", b"items", b"0042"] {
            assert!(
                seam.might_contain(&dv, &presence, s),
                "seam falsely rejected {:?}",
                std::str::from_utf8(s).unwrap()
            );
        }
    }
}
