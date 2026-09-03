// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fundamental types shared across the crate.
//!
//! A *token* is a dictionary entry — a byte string of `1..=MAX_TOKEN_SIZE`
//! bytes. A *code* is a [`Token`] in a column's code stream; decoding looks the
//! code up in the dictionary and copies the token's bytes. A column holds at
//! most `2^16` tokens, so a fixed-width `u16` addresses every code regardless of
//! dictionary size.

/// Identifies a dictionary token: a token id, equivalently a *code* in a
/// column's code stream. Named `Token` for simplicity.
pub type Token = u16;

/// Maximum byte length of any dictionary token, and the fixed width the decoder
/// reads per token.
pub const MAX_TOKEN_SIZE: usize = 16;

/// Inclusive token-id range `[begin, last]` over the sorted dictionary.
///
/// A contiguous run of token ids in the dictionary. The canonical empty range
/// is [`TokenRange::EMPTY`] (`begin > last`).
///
/// A dedicated type (rather than [`std::ops::Range`]) because the search layer
/// needs `Copy`, an explicit empty case, and inclusive-end semantics that a
/// half-open range models awkwardly.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TokenRange {
    /// First token id (inclusive).
    pub begin: Token,
    /// Last token id (inclusive). `begin > last` denotes an empty range.
    pub last: Token,
}

impl TokenRange {
    /// The canonical empty range (`begin > last`).
    pub const EMPTY: Self = Self { begin: 1, last: 0 };

    /// Whether the range contains no token ids.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.begin > self.last
    }

    /// Number of token ids in the range.
    #[inline]
    pub const fn len(self) -> usize {
        if self.is_empty() {
            0
        } else {
            (self.last as usize) - (self.begin as usize) + 1
        }
    }

    /// Whether `t` lies within the range.
    #[inline]
    pub const fn contains(self, t: Token) -> bool {
        t >= self.begin && t <= self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_token_size_is_16() {
        assert_eq!(MAX_TOKEN_SIZE, 16);
    }

    #[test]
    fn token_range_empty_is_empty() {
        assert!(TokenRange::EMPTY.is_empty());
        assert_eq!(TokenRange::EMPTY.len(), 0);
    }

    #[test]
    fn token_range_inclusive_len_and_contains() {
        let r = TokenRange { begin: 3, last: 5 };
        assert!(!r.is_empty());
        assert_eq!(r.len(), 3);
        // Endpoints are inclusive on both sides.
        assert!(r.contains(3));
        assert!(r.contains(4));
        assert!(r.contains(5));
        // Just outside either end is excluded.
        assert!(!r.contains(2));
        assert!(!r.contains(6));
    }
}
