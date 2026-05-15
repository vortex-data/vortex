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
    pub fn might_starts_with(
        &self,
        dv: &DecodeView<'_>,
        index: &DictIndex,
        prefix: &[u8],
    ) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_DICT12_CONFIG;
    use crate::decode::OwnedDecodeInputs;
    use crate::onpair_compress;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

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
