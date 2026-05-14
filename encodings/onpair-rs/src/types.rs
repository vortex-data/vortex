// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/core/types.h`.

/// Number of bits per packed token. Legal range: 9..=16.
pub type BitWidth = u8;

/// Token identifier within a dictionary. Capped at `2^bits` per column.
pub type Token = u16;

/// Maximum byte length of any dictionary token.
pub const MAX_TOKEN_SIZE: usize = 16;

/// Byte range `[begin, end)` inside the dictionary buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ByteSpan {
    pub begin: u32,
    pub end: u32,
}

impl ByteSpan {
    #[inline]
    pub const fn size(self) -> u32 {
        self.end - self.begin
    }
}

/// Token-stream index range `[begin, end)` inside the packed store.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StreamSpan {
    pub begin: u32,
    pub end: u32,
}

impl StreamSpan {
    #[inline]
    pub const fn size(self) -> u32 {
        self.end - self.begin
    }
}

/// Closed range `[begin, last]` of token IDs. `begin > last` denotes empty.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TokenRange {
    pub begin: Token,
    pub last: Token,
}

impl Default for TokenRange {
    fn default() -> Self {
        // Canonical empty range: begin > last.
        Self { begin: 1, last: 0 }
    }
}

impl TokenRange {
    #[inline]
    pub const fn empty(self) -> bool {
        self.begin > self.last
    }

    #[inline]
    pub const fn size(self) -> u32 {
        if self.empty() {
            0
        } else {
            (self.last as u32) - (self.begin as u32) + 1
        }
    }

    #[inline]
    pub const fn contains(self, t: Token) -> bool {
        t >= self.begin && t <= self.last
    }
}

/// Maximum dictionary size given a bit width.
#[inline]
pub const fn max_dict_size(bits: BitWidth) -> usize {
    1usize << bits
}

/// Whether `bits` is in the legal range 9..=16.
#[inline]
pub const fn is_valid_bits(bits: BitWidth) -> bool {
    bits >= 9 && bits <= 16
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ByteSpan ────────────────────────────────────────────────────────────
    #[test]
    fn byte_span_size_is_end_minus_begin() {
        assert_eq!(ByteSpan { begin: 0, end: 0 }.size(), 0);
        assert_eq!(ByteSpan { begin: 0, end: 1 }.size(), 1);
        assert_eq!(ByteSpan { begin: 5, end: 10 }.size(), 5);
        assert_eq!(ByteSpan { begin: 100, end: 100 }.size(), 0);
    }

    // ── StreamSpan ──────────────────────────────────────────────────────────
    #[test]
    fn stream_span_size_is_end_minus_begin() {
        assert_eq!(StreamSpan { begin: 0, end: 0 }.size(), 0);
        assert_eq!(StreamSpan { begin: 0, end: 1 }.size(), 1);
        assert_eq!(StreamSpan { begin: 3, end: 7 }.size(), 4);
    }

    // ── TokenRange ──────────────────────────────────────────────────────────
    #[test]
    fn token_range_default_is_empty() {
        let r = TokenRange::default();
        assert!(r.empty());
        assert_eq!(r.size(), 0);
    }

    #[test]
    fn token_range_empty_when_begin_gt_last() {
        assert!(TokenRange { begin: 5, last: 4 }.empty());
        assert!(TokenRange { begin: 1, last: 0 }.empty());
    }

    #[test]
    fn token_range_not_empty_when_begin_leq_last() {
        assert!(!TokenRange { begin: 0, last: 0 }.empty());
        assert!(!TokenRange { begin: 5, last: 5 }.empty());
        assert!(!TokenRange { begin: 0, last: 100 }.empty());
    }

    #[test]
    fn token_range_size_zero_for_empty() {
        assert_eq!(TokenRange { begin: 10, last: 5 }.size(), 0);
    }

    #[test]
    fn token_range_size_one_for_single() {
        assert_eq!(TokenRange { begin: 5, last: 5 }.size(), 1);
        assert_eq!(TokenRange { begin: 0, last: 0 }.size(), 1);
    }

    #[test]
    fn token_range_size_last_minus_begin_plus_one() {
        assert_eq!(TokenRange { begin: 3, last: 7 }.size(), 5);
        assert_eq!(TokenRange { begin: 0, last: 255 }.size(), 256);
    }

    #[test]
    fn token_range_contains_boundary_tokens() {
        let r = TokenRange { begin: 10, last: 20 };
        assert!(r.contains(10));
        assert!(r.contains(20));
        assert!(r.contains(15));
    }

    #[test]
    fn token_range_contains_false_outside() {
        let r = TokenRange { begin: 10, last: 20 };
        assert!(!r.contains(9));
        assert!(!r.contains(21));
    }

    #[test]
    fn token_range_contains_false_for_empty() {
        let r = TokenRange::default();
        assert!(!r.contains(0));
        assert!(!r.contains(1));
    }

    // ── max_dict_size ───────────────────────────────────────────────────────
    #[test]
    fn max_dict_size_12_is_4096() {
        assert_eq!(max_dict_size(12), 4096);
    }

    #[test]
    fn max_dict_size_16_is_65536() {
        assert_eq!(max_dict_size(16), 65536);
    }

    #[test]
    fn max_dict_size_is_pow2_for_all_valid_widths() {
        for b in 9u8..=16 {
            assert_eq!(max_dict_size(b), 1usize << b);
        }
    }

    // ── is_valid_bits ───────────────────────────────────────────────────────
    #[test]
    fn is_valid_bits_accepts_9_to_16() {
        for b in 9u8..=16 {
            assert!(is_valid_bits(b), "expected true for bits={b}");
        }
    }

    #[test]
    fn is_valid_bits_rejects_8_and_17() {
        assert!(!is_valid_bits(8));
        assert!(!is_valid_bits(17));
    }

    #[test]
    fn is_valid_bits_rejects_0_and_255() {
        assert!(!is_valid_bits(0));
        assert!(!is_valid_bits(255));
    }

    // ── MAX_TOKEN_SIZE ──────────────────────────────────────────────────────
    #[test]
    fn max_token_size_is_16() {
        assert_eq!(MAX_TOKEN_SIZE, 16);
    }
}
