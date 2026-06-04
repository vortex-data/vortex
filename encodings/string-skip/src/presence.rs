// SPDX-License-Identifier: Apache-2.0
//! [`DictPresence`] — bitmap over dict ids that appear in a chunk.
//!
//! Tiny (`dict_size / 8` bytes per chunk; 512 B for the default 4096-
//! token dict). Exact necessary condition for `=` and `LIKE 'p%'`.

use serde::Deserialize;
use serde::Serialize;

use crate::dict::DictIndex;
use crate::dict::TokenDict;
use crate::dict::tokenize_needle;

/// Bitmap over dict ids: bit `i` is set iff token id `i` appears in
/// some row of this chunk.
#[derive(Clone, Serialize, Deserialize)]
pub struct DictPresence {
    bitmap: Vec<u64>,
    dict_size: usize,
}

impl DictPresence {
    /// Build the presence bitmap from a slice of code IDs.
    ///
    /// `dict_size` is the total number of dict entries (the bitmap is
    /// sized to fit this many bits).
    pub fn build(codes: &[u16], dict_size: usize) -> Self {
        let mut bitmap = vec![0u64; dict_size.div_ceil(64).max(1)];
        for &tok in codes {
            let i = tok as usize;
            bitmap[i / 64] |= 1u64 << (i % 64);
        }
        Self { bitmap, dict_size }
    }

    /// Bytes of metadata.
    pub fn byte_size(&self) -> usize {
        self.bitmap.len() * 8
    }

    /// Number of dict ids tracked.
    pub fn dict_size(&self) -> usize {
        self.dict_size
    }

    /// True iff dict id is present in this chunk.
    #[inline]
    pub fn is_set(&self, dict_id: usize) -> bool {
        debug_assert!(dict_id < self.dict_size);
        (self.bitmap[dict_id / 64] >> (dict_id % 64)) & 1 != 0
    }

    /// Sound necessary condition for `col = needle`. Returns `false`
    /// only when the chunk provably can't contain a row equal to
    /// `needle`.
    pub fn might_eq<D: TokenDict>(&self, dict: &D, index: &DictIndex, needle: &[u8]) -> bool {
        let Some(toks) = tokenize_needle(dict, index, needle) else {
            return false;
        };
        toks.iter().all(|&t| self.is_set(t as usize))
    }

    /// Sound necessary condition for `col LIKE 'prefix%'`.
    ///
    /// Greedy LPM isn't prefix-consistent — the encoder's token at the
    /// prefix boundary in a row may extend past `|prefix|`. We run an
    /// NFA over byte positions: `reached[p]` = true iff some chain of
    /// present tokens covers `prefix[..p]`. The chunk matches if
    /// either `reached[n]`, or at some reached `p` there's a present
    /// token whose bytes start with `prefix[p..]`.
    pub fn might_starts_with<D: TokenDict>(
        &self,
        dict: &D,
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
            for id in index.range_for(prefix[p]) {
                if !self.is_set(id) {
                    continue;
                }
                let bytes = dict.token_bytes(id as u16);
                if bytes.len() >= remaining.len() {
                    if bytes.starts_with(remaining) {
                        return true; // consumes the whole prefix
                    }
                } else if remaining.starts_with(bytes) {
                    reached[p + bytes.len()] = true;
                }
            }
        }
        reached[n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::TokenDict;

    struct TestDict {
        toks: Vec<Vec<u8>>,
    }
    impl TestDict {
        fn new(extras: Vec<&str>) -> Self {
            let mut toks: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
            for e in extras {
                toks.push(e.as_bytes().to_vec());
            }
            toks.sort();
            Self { toks }
        }
        fn id_of(&self, bytes: &[u8]) -> u16 {
            self.toks.iter().position(|t| t == bytes).unwrap() as u16
        }
    }
    impl TokenDict for TestDict {
        fn len(&self) -> usize {
            self.toks.len()
        }
        fn token_bytes(&self, id: u16) -> &[u8] {
            &self.toks[id as usize]
        }
    }

    #[test]
    fn might_eq_rejects_when_missing_token() {
        let d = TestDict::new(vec!["hello", "world"]);
        let idx = DictIndex::build(&d);
        let codes = vec![d.id_of(b"hello"), d.id_of(b"world")];
        let p = DictPresence::build(&codes, d.len());
        // "hello" tokenizes to one id (must be present)
        assert!(p.might_eq(&d, &idx, b"hello"));
        // "xyz" tokenizes to three single-byte ids: x, y, z
        // The codes only include 'hello' and 'world' tokens (not the
        // bytes inside them). The single-byte tokens for x, y, z are
        // NOT in the chunk's codes.
        assert!(!p.might_eq(&d, &idx, b"xyz"));
    }

    #[test]
    fn might_starts_with_via_single_token() {
        let d = TestDict::new(vec!["http://"]);
        let idx = DictIndex::build(&d);
        let codes = vec![d.id_of(b"http://")];
        let p = DictPresence::build(&codes, d.len());
        assert!(p.might_starts_with(&d, &idx, b"http"));
        assert!(p.might_starts_with(&d, &idx, b"http://"));
        assert!(!p.might_starts_with(&d, &idx, b"https"));
    }
}
