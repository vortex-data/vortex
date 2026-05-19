// SPDX-License-Identifier: Apache-2.0
//! Dictionary abstraction.
//!
//! Skip indexes need three things from any dictionary-coded string column:
//! 1. **Bytes of each token** (for tokenization and substring checks)
//! 2. **Lex-sorted dict** (for the first-byte range index)
//! 3. **Per-row code stream** (for building blooms over consecutive pairs)
//!
//! This module defines minimal traits so the algorithms work over OnPair,
//! FSST, BPE, or any custom encoding that exposes a sorted dictionary.

/// View of a dictionary-coded string column. Implementations: OnPair,
/// FSST, BPE, etc.
///
/// **Invariant**: tokens in `iter_tokens()` must be in **lex-ascending**
/// order. The skip-index algorithms depend on this for the first-byte
/// range index.
pub trait TokenDict {
    /// Number of tokens in the dictionary.
    fn len(&self) -> usize;
    /// True if the dictionary is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get the bytes of a single token by id.
    fn token_bytes(&self, id: u16) -> &[u8];
}

/// Per-byte first-byte index built from a [`TokenDict`].
///
/// `range_for(b) = lo..hi` is the range of dict ids whose first byte
/// equals `b`. Built in O(dict_size) once per column.
///
/// Storage: 257 `u32`s = 1 KB.
pub struct DictIndex {
    by_first_byte: [u32; 257],
}

impl DictIndex {
    /// Build the first-byte index from a dictionary.
    pub fn build<D: TokenDict>(dict: &D) -> Self {
        let mut by_first_byte = [0u32; 257];
        let dict_size = u32::try_from(dict.len()).expect("dict_size > u32::MAX");

        let mut last_first: usize = 0;
        for i in 0..dict.len() {
            let bytes = dict.token_bytes(i as u16);
            if bytes.is_empty() {
                continue;
            }
            let first = bytes[0] as usize;
            let i_u32 = i as u32;
            while last_first <= first {
                by_first_byte[last_first] = i_u32;
                last_first += 1;
            }
        }
        while last_first <= 256 {
            by_first_byte[last_first] = dict_size;
            last_first += 1;
        }
        Self { by_first_byte }
    }

    /// Range of dict ids whose first byte equals `b`. Empty if none.
    #[inline]
    pub fn range_for(&self, b: u8) -> core::ops::Range<usize> {
        let lo = self.by_first_byte[b as usize] as usize;
        let hi = self.by_first_byte[b as usize + 1] as usize;
        lo..hi
    }
}

/// Greedy longest-prefix-match tokenization of an arbitrary byte string
/// against the dict. Returns `None` if any byte has no matching dict
/// entry (normally impossible — dicts include 256 single-byte tokens).
///
/// **Determinism**: greedy LPM is deterministic. Two byte-equal strings
/// always tokenize to the same token sequence. This is the basis for
/// the soundness of code-bigram skip indexes.
pub fn tokenize_needle<D: TokenDict>(
    dict: &D,
    index: &DictIndex,
    needle: &[u8],
) -> Option<Vec<u16>> {
    let mut tokens = Vec::with_capacity(needle.len());
    let mut pos = 0usize;
    while pos < needle.len() {
        let candidates = index.range_for(needle[pos]);
        if candidates.is_empty() {
            return None;
        }
        let remaining = &needle[pos..];
        let mut best_len: usize = 0;
        let mut best_id: u16 = 0;
        for id in candidates {
            let entry_bytes = dict.token_bytes(id as u16);
            let len = entry_bytes.len();
            if len <= best_len || len > remaining.len() {
                continue;
            }
            if remaining.starts_with(entry_bytes) {
                best_len = len;
                best_id = id as u16;
            }
        }
        if best_len == 0 {
            return None;
        }
        tokens.push(best_id);
        pos += best_len;
    }
    Some(tokens)
}

/// True iff the greedy-LPM token at byte position `pos` in `remainder`
/// is guaranteed identical to the token the actual row's tokenizer
/// picks. False when some dict entry could "see past" the remainder
/// boundary and be picked in context but not by `tokenize_needle`.
///
/// This is the key soundness check for substring matching: only
/// bigrams between two safe positions can be probed against the bloom.
#[inline]
pub fn is_safe_position<D: TokenDict>(
    dict: &D,
    index: &DictIndex,
    remainder: &[u8],
    pos: usize,
) -> bool {
    let remaining = &remainder[pos..];
    if remaining.is_empty() {
        return false;
    }
    for id in index.range_for(remaining[0]) {
        let entry_bytes = dict.token_bytes(id as u16);
        let tlen = entry_bytes.len();
        if tlen <= remaining.len() {
            continue;
        }
        if entry_bytes.starts_with(remaining) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory test dict that holds bytes per id, lex-sorted.
    pub struct TestDict {
        tokens: Vec<Vec<u8>>,
    }

    impl TestDict {
        pub fn new(mut tokens: Vec<Vec<u8>>) -> Self {
            tokens.sort();
            Self { tokens }
        }
        /// Always include all 256 single-byte tokens.
        pub fn with_singletons(extras: Vec<&str>) -> Self {
            let mut toks: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
            for e in extras {
                toks.push(e.as_bytes().to_vec());
            }
            Self::new(toks)
        }
    }

    impl TokenDict for TestDict {
        fn len(&self) -> usize {
            self.tokens.len()
        }
        fn token_bytes(&self, id: u16) -> &[u8] {
            &self.tokens[id as usize]
        }
    }

    #[test]
    fn first_byte_index_covers_all_bytes() {
        let d = TestDict::with_singletons(vec!["abc", "xyz"]);
        let idx = DictIndex::build(&d);
        // Each byte b has at least the single-byte token for b in its range.
        for b in 0u8..=255 {
            let r = idx.range_for(b);
            assert!(!r.is_empty(), "byte {b} has empty range");
        }
    }

    #[test]
    fn tokenize_picks_longest() {
        let d = TestDict::with_singletons(vec!["ab", "abc", "abcd"]);
        let idx = DictIndex::build(&d);
        let toks = tokenize_needle(&d, &idx, b"abcd").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(d.token_bytes(toks[0]), b"abcd");
    }

    #[test]
    fn tokenize_falls_through_to_smaller() {
        let d = TestDict::with_singletons(vec!["ab", "abc"]);
        let idx = DictIndex::build(&d);
        let toks = tokenize_needle(&d, &idx, b"abx").unwrap();
        // Greedy picks "ab" then single "x"
        assert_eq!(toks.len(), 2);
        assert_eq!(d.token_bytes(toks[0]), b"ab");
        assert_eq!(d.token_bytes(toks[1]), b"x");
    }

    #[test]
    fn safe_position_when_no_extension() {
        let d = TestDict::with_singletons(vec!["abc"]);
        let idx = DictIndex::build(&d);
        // remainder = "abc", pos = 0
        // The longest token starting with 'a' is "abc" (3 bytes), which
        // fits exactly within remainder → safe.
        assert!(is_safe_position(&d, &idx, b"abc", 0));
    }

    #[test]
    fn unsafe_position_when_extension_exists() {
        let d = TestDict::with_singletons(vec!["abc", "abcdef"]);
        let idx = DictIndex::build(&d);
        // remainder = "abc", pos = 0. The dict has "abcdef" (6 bytes)
        // which starts with our remainder but extends past it. UNSAFE.
        assert!(!is_safe_position(&d, &idx, b"abc", 0));
    }
}
