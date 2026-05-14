// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/core/store.h` and `store_view.h`.
//
// The packed token store holds the bit-packed token stream produced by the
// parser. All tokens in a column share the same fixed bit width (9..=16).
//
// Per-string boundaries are stored in token-stream indices (Arrow-style):
// `boundaries.len() == num_strings + 1`, with `boundaries[i]` the
// inclusive start and `boundaries[num_strings]` the total token count.

use crate::types::{BitWidth, StreamSpan};

#[derive(Default, Debug, Clone)]
pub struct Store {
    /// Immutable after first write (9..=16).
    pub bit_width: BitWidth,
    /// LSB-first bit-packed token stream.
    pub packed: Vec<u64>,
    /// `boundaries[i]` is the token-index start of string `i`;
    /// `boundaries.last()` is the total token count.
    pub boundaries: Vec<u32>,
}

impl Store {
    #[inline]
    pub fn num_strings(&self) -> usize {
        if self.boundaries.is_empty() {
            0
        } else {
            self.boundaries.len() - 1
        }
    }

    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.boundaries.last().copied().unwrap_or(0) as usize
    }

    #[inline]
    pub fn bytes_used(&self) -> usize {
        if self.boundaries.is_empty() {
            return 0;
        }
        let total_bits = self.num_tokens() * self.bit_width as usize;
        let packed_bytes = total_bits.div_ceil(8);
        packed_bytes + self.boundaries.len() * size_of::<u32>()
    }

    /// Token-stream range `[begin, end)` for string at position `idx`.
    /// Precondition: `idx < num_strings()`.
    #[inline]
    pub fn string_span(&self, idx: usize) -> StreamSpan {
        StreamSpan {
            begin: self.boundaries[idx],
            end: self.boundaries[idx + 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Store ───────────────────────────────────────────────────────────────

    #[test]
    fn num_strings_zero_when_boundaries_empty() {
        let s = Store { bit_width: 16, ..Default::default() };
        assert_eq!(s.num_strings(), 0);
    }

    #[test]
    fn num_strings_zero_when_only_sentinel() {
        let s = Store { bit_width: 16, boundaries: vec![0], ..Default::default() };
        assert_eq!(s.num_strings(), 0);
    }

    #[test]
    fn num_strings_is_correct() {
        let s = Store {
            bit_width: 16,
            boundaries: vec![0, 3, 5, 8],
            ..Default::default()
        };
        assert_eq!(s.num_strings(), 3);
    }

    #[test]
    fn num_tokens_zero_when_boundaries_empty() {
        let s = Store { bit_width: 16, ..Default::default() };
        assert_eq!(s.num_tokens(), 0);
    }

    #[test]
    fn num_tokens_is_last_boundary() {
        let s = Store {
            bit_width: 16,
            boundaries: vec![0, 4, 7],
            ..Default::default()
        };
        assert_eq!(s.num_tokens(), 7);
    }

    #[test]
    fn bytes_used_counts_packed_bits_and_boundaries() {
        let s = Store {
            bit_width: 16,
            packed: vec![0xDEAD, 0xBEEF],
            boundaries: vec![0, 2, 4],
        };
        // total_bits = 4*16 = 64; packed_bytes = 8; boundaries = 3 * 4 = 12.
        assert_eq!(s.bytes_used(), 8 + 3 * size_of::<u32>());
    }

    #[test]
    fn bytes_used_with_different_bit_width() {
        let s = Store {
            bit_width: 13,
            packed: vec![0xDEAD, 0xBEEF],
            boundaries: vec![0, 2, 4],
        };
        // total_bits = 4*13 = 52; packed_bytes = 7.
        assert_eq!(s.bytes_used(), 7 + 3 * size_of::<u32>());
    }

    // ── StoreView equivalents (string_span + raw access) ────────────────────

    #[test]
    fn inherits_metadata() {
        let s = Store {
            bit_width: 14,
            packed: vec![1, 2],
            boundaries: vec![0, 5, 10],
        };
        assert_eq!(s.bit_width, 14);
        assert_eq!(s.num_strings(), 2);
        assert_eq!(s.num_tokens(), 10);
        assert_eq!(s.bytes_used(), s.bytes_used());
    }

    #[test]
    fn string_span_returns_correct_range() {
        let s = Store {
            bit_width: 16,
            boundaries: vec![0, 3, 7, 10],
            ..Default::default()
        };
        assert_eq!(s.string_span(0), StreamSpan { begin: 0, end: 3 });
        assert_eq!(s.string_span(1), StreamSpan { begin: 3, end: 7 });
        assert_eq!(s.string_span(2), StreamSpan { begin: 7, end: 10 });
    }

    #[test]
    fn empty_store_view() {
        let s = Store { bit_width: 12, ..Default::default() };
        assert_eq!(s.num_strings(), 0);
        assert_eq!(s.num_tokens(), 0);
        assert_eq!(s.bytes_used(), 0);
    }
}
