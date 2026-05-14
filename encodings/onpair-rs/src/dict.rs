// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/core/dictionary.h` and `dictionary_view.h`.
//
// The dictionary maps `Token` -> byte sequence with the Arrow binary layout:
// a flat `bytes` buffer plus an `offsets` array of length `num_tokens + 1`.
// Tokens are stored in lexicographic order so that prefix range lookups can
// use binary search.

use crate::types::{ByteSpan, MAX_TOKEN_SIZE, Token, TokenRange};

#[derive(Default, Debug, Clone)]
pub struct Dictionary {
    /// Flat concatenation of token bytes.
    ///
    /// `pad_for_decoder()` may extend this past `offsets.back()` with zeros
    /// so the decoder can issue a fixed `MAX_TOKEN_SIZE` byte over-copy from
    /// any token offset without reading out of bounds. `bytes_used()` always
    /// reports the logical size derived from `offsets`.
    pub bytes: Vec<u8>,

    /// `offsets[i]..offsets[i+1]` = byte range of token `i` in `bytes`.
    /// Invariants: `offsets[0] == 0`, `offsets.len() == num_tokens + 1`.
    pub offsets: Vec<u32>,
}

impl Dictionary {
    #[inline]
    pub fn num_tokens(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1
        }
    }

    /// Logical byte cost (true token bytes + offsets array). Unaffected by
    /// any padding `pad_for_decoder()` appended.
    #[inline]
    pub fn bytes_used(&self) -> usize {
        let true_bytes = self.offsets.last().copied().unwrap_or(0) as usize;
        let offsets_bytes = self.offsets.len() * size_of::<u32>();
        true_bytes + offsets_bytes
    }

    /// Append zero bytes so the decoder may safely over-copy `MAX_TOKEN_SIZE`
    /// bytes from any token offset. Idempotent.
    pub fn pad_for_decoder(&mut self) {
        if self.offsets.len() < 2 {
            return;
        }
        let last_off = *self.offsets.last().unwrap() as usize;
        if self.bytes.len() > last_off {
            return; // already padded
        }
        let last_start = self.offsets[self.offsets.len() - 2] as usize;
        let last_len = last_off - last_start;
        self.bytes.resize(self.bytes.len() + (MAX_TOKEN_SIZE - last_len), 0);
    }

    // ── DictionaryView equivalents (free fns on Dictionary; no view type) ──

    #[inline]
    pub fn span(&self, id: Token) -> ByteSpan {
        ByteSpan {
            begin: self.offsets[id as usize],
            end: self.offsets[id as usize + 1],
        }
    }

    #[inline]
    pub fn data(&self, id: Token) -> &[u8] {
        let s = self.span(id);
        &self.bytes[s.begin as usize..s.end as usize]
    }

    #[inline]
    pub fn token_size(&self, id: Token) -> usize {
        let s = self.span(id);
        s.size() as usize
    }

    /// Inclusive `[lo, hi]` token-id range whose byte sequences begin with
    /// `prefix`. Mirrors `DictionaryView::prefix_range` in onpair_cpp.
    pub fn prefix_range(&self, prefix: &[u8]) -> TokenRange {
        if prefix.len() > MAX_TOKEN_SIZE {
            return TokenRange::default();
        }

        let n = self.num_tokens() as u32;

        // Find the first token whose bytes >= `target`, starting at `start`.
        let lower_bound = |target: &[u8], start: u32| -> u32 {
            let mut lo = start;
            let mut hi = n;
            while lo < hi {
                let mid = lo + ((hi - lo) >> 1);
                let m_off = self.offsets[mid as usize] as usize;
                let m_end = self.offsets[mid as usize + 1] as usize;
                let m_len = m_end - m_off;
                let cmp_len = m_len.min(target.len());
                let cmp = self.bytes[m_off..m_off + cmp_len].cmp(&target[..cmp_len]);
                // token[mid] < target iff cmp == Less, or cmp == Equal AND token shorter
                if cmp == std::cmp::Ordering::Less
                    || (cmp == std::cmp::Ordering::Equal && m_len < target.len())
                {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };

        let lo = lower_bound(prefix, 0);

        // Compute the next lexicographic prefix by incrementing the last
        // non-0xFF byte, trimming trailing 0xFF bytes first.
        let mut buf = [0u8; MAX_TOKEN_SIZE];
        let mut ulen = prefix.len();
        let mut overflow = true;
        while ulen > 0 {
            if prefix[ulen - 1] < 0xFF {
                buf[..ulen].copy_from_slice(&prefix[..ulen]);
                buf[ulen - 1] = buf[ulen - 1].wrapping_add(1);
                overflow = false;
                break;
            }
            ulen -= 1;
        }

        let hi = if overflow { n } else { lower_bound(&buf[..ulen], lo) };

        if lo < hi {
            TokenRange { begin: lo as Token, last: (hi - 1) as Token }
        } else {
            TokenRange::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Dictionary basics ──────────────────────────────────────────────────

    #[test]
    fn num_tokens_zero_when_offsets_empty() {
        let d = Dictionary::default();
        assert_eq!(d.num_tokens(), 0);
    }

    #[test]
    fn num_tokens_is_offsets_len_minus_one() {
        let d = Dictionary { bytes: vec![], offsets: vec![0, 3, 5, 8] };
        assert_eq!(d.num_tokens(), 3);
    }

    #[test]
    fn num_tokens_single_entry_minus_one() {
        let d = Dictionary { bytes: vec![], offsets: vec![0, 7] };
        assert_eq!(d.num_tokens(), 1);
    }

    #[test]
    fn bytes_used_accounts_for_both_vectors() {
        let d = Dictionary {
            bytes: vec![0x00, 0x01, 0x02],
            offsets: vec![0, 1, 2, 3],
        };
        let expected = 3 + 4 * size_of::<u32>();
        assert_eq!(d.bytes_used(), expected);
    }

    #[test]
    fn bytes_used_zero_when_empty() {
        let d = Dictionary::default();
        assert_eq!(d.bytes_used(), 0);
    }

    // ─── pad_for_decoder ────────────────────────────────────────────────────

    #[test]
    fn pad_for_decoder_adds_trailing_zeros() {
        let mut d = Dictionary {
            bytes: b"hello".to_vec(),
            offsets: vec![0, 5],
        };
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), MAX_TOKEN_SIZE);
    }

    #[test]
    fn pad_for_decoder_is_idempotent() {
        let mut d = Dictionary { bytes: b"ab".to_vec(), offsets: vec![0, 2] };
        d.pad_for_decoder();
        let size1 = d.bytes.len();
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), size1);
    }

    #[test]
    fn pad_for_decoder_noop_for_max_token_size() {
        let mut d = Dictionary {
            bytes: vec![b'x'; MAX_TOKEN_SIZE],
            offsets: vec![0, MAX_TOKEN_SIZE as u32],
        };
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), MAX_TOKEN_SIZE);
    }

    #[test]
    fn pad_for_decoder_uses_last_token_only() {
        // tokens: "ab" (2 bytes), "cde" (3 bytes); pad based on last (3).
        let mut d = Dictionary {
            bytes: b"abcde".to_vec(),
            offsets: vec![0, 2, 5],
        };
        d.pad_for_decoder();
        assert_eq!(d.bytes.len(), 5 + (MAX_TOKEN_SIZE - 3));
    }

    #[test]
    fn bytes_used_unchanged_after_padding() {
        let mut d = Dictionary { bytes: b"hello".to_vec(), offsets: vec![0, 5] };
        let before = d.bytes_used();
        d.pad_for_decoder();
        assert_eq!(d.bytes_used(), before);
        assert!(d.bytes.len() > *d.offsets.last().unwrap() as usize);
    }

    #[test]
    fn pad_for_decoder_noop_when_fewer_than_two_offsets() {
        let mut d = Dictionary { bytes: vec![], offsets: vec![0] };
        d.pad_for_decoder();
        assert!(d.bytes.is_empty());
    }

    #[test]
    fn pad_for_decoder_padding_bytes_are_zero() {
        let mut d = Dictionary { bytes: b"xy".to_vec(), offsets: vec![0, 2] };
        d.pad_for_decoder();
        for (i, b) in d.bytes.iter().enumerate().skip(2) {
            assert_eq!(*b, 0, "non-zero padding at index {i}");
        }
    }

    // ─── DictionaryView equivalents ─────────────────────────────────────────

    fn make_abc() -> Dictionary {
        Dictionary { bytes: vec![b'a', b'b', b'c'], offsets: vec![0, 1, 2, 3] }
    }

    fn make_varying() -> Dictionary {
        // "a", "bc", "def"
        Dictionary {
            bytes: b"abcdef".to_vec(),
            offsets: vec![0, 1, 3, 6],
        }
    }

    fn make_prefix_dict() -> Dictionary {
        // sorted: "a", "ab", "b"
        Dictionary {
            bytes: vec![b'a', b'a', b'b', b'b'],
            offsets: vec![0, 1, 3, 4],
        }
    }

    #[test]
    fn span_returns_correct_range() {
        let d = make_varying();
        assert_eq!(d.span(0), ByteSpan { begin: 0, end: 1 });
        assert_eq!(d.span(1), ByteSpan { begin: 1, end: 3 });
        assert_eq!(d.span(2), ByteSpan { begin: 3, end: 6 });
    }

    #[test]
    fn data_points_to_correct_byte() {
        let d = make_varying();
        assert_eq!(d.data(0)[0], b'a');
        assert_eq!(d.data(1)[0], b'b');
        assert_eq!(d.data(2)[0], b'd');
    }

    #[test]
    fn token_size_consistent_with_span() {
        let d = make_varying();
        for t in 0u16..3 {
            assert_eq!(d.token_size(t), d.span(t).size() as usize);
        }
    }

    #[test]
    fn token_sizes() {
        let d = make_varying();
        assert_eq!(d.token_size(0), 1);
        assert_eq!(d.token_size(1), 2);
        assert_eq!(d.token_size(2), 3);
    }

    #[test]
    fn num_tokens_matches_storage() {
        let d = make_varying();
        assert_eq!(d.num_tokens(), 3);
    }

    #[test]
    fn accepts_empty_dictionary() {
        let d = Dictionary::default();
        assert_eq!(d.num_tokens(), 0);
    }

    #[test]
    fn bytes_used_unaffected_by_padding() {
        let mut d = make_varying();
        let before_pad = d.bytes_used();
        d.pad_for_decoder();
        assert_eq!(d.bytes_used(), before_pad);
    }

    // ─── prefix_range ───────────────────────────────────────────────────────

    #[test]
    fn empty_dict_returns_empty_range() {
        let d = Dictionary::default();
        assert!(d.prefix_range(b"a").empty());
    }

    #[test]
    fn exact_single_token_match() {
        let d = make_abc();
        let r = d.prefix_range(b"b");
        assert!(!r.empty());
        assert_eq!(r.size(), 1);
        assert_eq!(r.begin, 1);
        assert_eq!(r.last, 1);
    }

    #[test]
    fn prefix_matches_multiple_tokens() {
        let d = make_prefix_dict();
        let r = d.prefix_range(b"a");
        assert!(!r.empty());
        assert_eq!(r.begin, 0);
        assert_eq!(r.last, 1);
        assert_eq!(r.size(), 2);
    }

    #[test]
    fn no_match_returns_empty() {
        let d = make_abc();
        assert!(d.prefix_range(b"z").empty());
    }

    #[test]
    fn prefix_longer_than_max_returns_empty() {
        let d = make_abc();
        let buf = [0u8; MAX_TOKEN_SIZE + 1];
        assert!(d.prefix_range(&buf).empty());
    }

    #[test]
    fn all_ff_bytes_prefix() {
        // tokens {0xFF}, {0xFF, 0xFF}
        let d = Dictionary { bytes: vec![0xFF, 0xFF, 0xFF], offsets: vec![0, 1, 3] };
        let r = d.prefix_range(&[0xFF]);
        assert_eq!(r.begin, 0);
        assert_eq!(r.last, 1);
    }

    #[test]
    fn exact_length_match_first_and_only_token() {
        let d = Dictionary { bytes: b"hello".to_vec(), offsets: vec![0, 5] };
        let r = d.prefix_range(b"hello");
        assert!(!r.empty());
        assert_eq!(r.begin, 0);
        assert_eq!(r.last, 0);
    }

    #[test]
    fn contains_returns_true_for_all_in_range() {
        let d = make_prefix_dict();
        let r = d.prefix_range(b"a");
        assert!(r.contains(0));
        assert!(r.contains(1));
        assert!(!r.contains(2));
    }

    #[test]
    fn empty_pattern_matches_all_tokens() {
        let d = make_abc();
        let r = d.prefix_range(&[]);
        assert_eq!(r.size(), 3);
        assert_eq!(r.begin, 0);
        assert_eq!(r.last, 2);
    }

    #[test]
    fn empty_pattern_on_single_token_dict() {
        let d = Dictionary { bytes: vec![b'x'], offsets: vec![0, 1] };
        let r = d.prefix_range(&[]);
        assert!(!r.empty());
        assert_eq!(r.size(), 1);
    }

    #[test]
    fn pattern_exactly_max_token_size_can_match() {
        let d = Dictionary {
            bytes: vec![b'z'; MAX_TOKEN_SIZE],
            offsets: vec![0, MAX_TOKEN_SIZE as u32],
        };
        let needle = vec![b'z'; MAX_TOKEN_SIZE];
        let r = d.prefix_range(&needle);
        assert!(!r.empty());
        assert_eq!(r.size(), 1);
        assert_eq!(r.begin, 0);
    }

    #[test]
    fn all_ff_multi_byte_prefix() {
        // tokens: {0xFF} and {0xFF, 0xFF}; prefix {0xFF, 0xFF} matches only the
        // second.
        let d = Dictionary { bytes: vec![0xFF, 0xFF, 0xFF], offsets: vec![0, 1, 3] };
        let r = d.prefix_range(&[0xFF, 0xFF]);
        assert!(!r.empty());
        assert_eq!(r.size(), 1);
        assert_eq!(r.begin, 1);
        assert_eq!(r.last, 1);
    }

    #[test]
    fn all_ff_prefix_beyond_all_tokens() {
        let d = Dictionary { bytes: vec![0xFE], offsets: vec![0, 1] };
        let r = d.prefix_range(&[0xFF, 0xFF]);
        assert!(r.empty());
    }

    #[test]
    fn single_token_dict_matching_prefix() {
        let d = Dictionary { bytes: b"hello".to_vec(), offsets: vec![0, 5] };
        let r = d.prefix_range(b"he");
        assert!(!r.empty());
        assert_eq!(r.size(), 1);
        assert_eq!(r.begin, 0);
    }

    #[test]
    fn single_token_dict_non_matching_prefix() {
        let d = Dictionary { bytes: b"hello".to_vec(), offsets: vec![0, 5] };
        assert!(d.prefix_range(b"x").empty());
    }

    #[test]
    fn overlapping_prefixes_deep_nesting() {
        // "a", "aa", "aaa", "b"
        let d = Dictionary {
            bytes: vec![b'a', b'a', b'a', b'a', b'a', b'a', b'b'],
            offsets: vec![0, 1, 3, 6, 7],
        };
        let r_a = d.prefix_range(b"a");
        assert_eq!(r_a.size(), 3);
        assert_eq!(r_a.begin, 0);
        assert_eq!(r_a.last, 2);

        let r_aa = d.prefix_range(b"aa");
        assert_eq!(r_aa.size(), 2);
        assert_eq!(r_aa.begin, 1);
        assert_eq!(r_aa.last, 2);

        let r_aaa = d.prefix_range(b"aaa");
        assert_eq!(r_aaa.size(), 1);
        assert_eq!(r_aaa.begin, 2);

        let r_b = d.prefix_range(b"b");
        assert_eq!(r_b.size(), 1);
        assert_eq!(r_b.begin, 3);
    }

    #[test]
    fn contiguous_range_bounds_no_spillover() {
        // "apple", "apt", "b"
        let d = Dictionary {
            bytes: b"appleaptb".to_vec(),
            offsets: vec![0, 5, 8, 9],
        };
        let r = d.prefix_range(b"ap");
        assert_eq!(r.size(), 2);
        assert!(!r.contains(2));
    }

    #[test]
    fn prefix_equals_full_token_content() {
        // "ab", "abc", "abd", "b"
        let d = Dictionary {
            bytes: b"ababcabdb".to_vec(),
            offsets: vec![0, 2, 5, 8, 9],
        };
        let r = d.prefix_range(b"ab");
        assert_eq!(r.size(), 3);
        assert_eq!(r.begin, 0);
        assert_eq!(r.last, 2);
        assert!(!r.contains(3));
    }
}
