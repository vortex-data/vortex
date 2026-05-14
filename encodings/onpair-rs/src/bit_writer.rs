// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/encoding/parsing/bit_writer.h`.
//
// LSB-first bit-packing into a `Store`. Tokens straddling a 64-bit word
// boundary are split: low bits go into the current word and the remaining
// bits open the next word. Always appends a zero sentinel word after the
// final partial word so readers can safely do a 4-byte look-ahead at the
// last token.

use crate::store::Store;
use crate::types::Token;

pub struct BitWriter<'a> {
    store: &'a mut Store,
    bits: u8,
    mask: u64,
    buf: u64,
    shift: u32,
    count: usize,
    flushed: bool,
}

impl<'a> BitWriter<'a> {
    pub fn new(store: &'a mut Store) -> Self {
        let bits = store.bit_width;
        let mask = (1u64 << bits) - 1;
        store.packed.clear();
        store.packed.reserve(256);
        Self {
            store,
            bits,
            mask,
            buf: 0,
            shift: 0,
            count: 0,
            flushed: false,
        }
    }

    /// Append one token into the packed stream.
    #[inline]
    pub fn write(&mut self, token: Token) {
        let value = (token as u64) & self.mask;
        self.buf |= value << self.shift;
        self.shift += self.bits as u32;
        if self.shift >= 64 {
            self.store.packed.push(self.buf);
            self.shift -= 64;
            // Spill the bits that crossed the word boundary. When the token
            // exactly fills the word (shift == 0 after subtraction) the spill
            // must be zero; an unconditional right-shift of `bits` would be
            // a no-op only if `bits == 64`, so guard it.
            self.buf = if self.shift == 0 {
                0
            } else {
                value >> (self.bits as u32 - self.shift)
            };
        }
        self.count += 1;
    }

    /// Flush the in-progress word (zero-padded), then append one zero
    /// sentinel so readers can safely over-read. Idempotent.
    pub fn flush(&mut self) {
        if self.flushed {
            return;
        }
        if self.shift > 0 {
            self.store.packed.push(self.buf);
            self.buf = 0;
            self.shift = 0;
        }
        if self.count > 0 {
            self.store.packed.push(0); // sentinel
        }
        self.flushed = true;
    }

    #[inline]
    pub fn tokens_written(&self) -> usize {
        self.count
    }
}

impl Drop for BitWriter<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_unpack::read_bits_lsb;
    use crate::types::BitWidth;

    fn roundtrip(bits: BitWidth, tokens: &[Token]) -> Vec<Token> {
        let mut store = Store { bit_width: bits, ..Default::default() };
        {
            let mut w = BitWriter::new(&mut store);
            for &t in tokens {
                w.write(t);
            }
        }
        let mut out = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            out.push(read_bits_lsb(&store.packed, i * bits as usize, bits as u32));
        }
        out
    }

    fn max_token(bits: BitWidth) -> Token {
        ((1u32 << bits) - 1) as Token
    }

    fn group_size(bits: u32) -> usize {
        let mut lcm = bits;
        while !lcm.is_multiple_of(64) {
            lcm += bits;
        }
        (lcm / bits) as usize
    }

    fn expected_packed_words(n: usize, bits: BitWidth) -> usize {
        (n * bits as usize).div_ceil(64)
    }

    const WIDTHS: &[BitWidth] = &[9, 10, 11, 12, 13, 14, 15, 16];

    // ── Degenerate inputs ───────────────────────────────────────────────────

    #[test]
    fn zero_tokens_produce_empty_packed() {
        for &bw in WIDTHS {
            let mut s = Store { bit_width: bw, ..Default::default() };
            {
                let w = BitWriter::new(&mut s);
                assert_eq!(w.tokens_written(), 0);
            }
            assert!(s.packed.is_empty(), "bits={bw}");
        }
    }

    // ── Structural invariants ───────────────────────────────────────────────

    #[test]
    fn packed_size_consistent_with_token_count() {
        for &bw in WIDTHS {
            let n = group_size(bw as u32) + 3;
            let mut s = Store { bit_width: bw, ..Default::default() };
            {
                let mut w = BitWriter::new(&mut s);
                for _ in 0..n {
                    w.write(1);
                }
            }
            assert_eq!(s.packed.len(), expected_packed_words(n, bw) + 1, "bits={bw}");
        }
    }

    #[test]
    fn tokens_written_count_equals_tokens_written() {
        for &bw in WIDTHS {
            let mut s = Store { bit_width: bw, ..Default::default() };
            let mut w = BitWriter::new(&mut s);
            for i in 0u16..10 {
                assert_eq!(w.tokens_written(), i as usize);
                w.write(i);
            }
            assert_eq!(w.tokens_written(), 10);
            w.flush();
            assert_eq!(w.tokens_written(), 10);
        }
    }

    // ── Round-trip correctness ──────────────────────────────────────────────

    #[test]
    fn single_zero_token_roundtrip() {
        for &bw in WIDTHS {
            let r = roundtrip(bw, &[0]);
            assert_eq!(r.len(), 1);
            assert_eq!(r[0], 0);
        }
    }

    #[test]
    fn single_max_token_roundtrip() {
        for &bw in WIDTHS {
            let mx = max_token(bw);
            let r = roundtrip(bw, &[mx]);
            assert_eq!(r.len(), 1);
            assert_eq!(r[0], mx);
        }
    }

    #[test]
    fn mixed_zero_and_max_roundtrip() {
        for &bw in WIDTHS {
            let mx = max_token(bw);
            let tokens: Vec<Token> =
                (0..30).map(|i| if i % 2 == 0 { 0 } else { mx }).collect();
            assert_eq!(roundtrip(bw, &tokens), tokens, "bits={bw}");
        }
    }

    #[test]
    fn incrementing_tokens_roundtrip() {
        for &bw in WIDTHS {
            let range = (max_token(bw) as u32) + 1;
            let tokens: Vec<Token> = (0..200u32).map(|i| (i % range) as Token).collect();
            assert_eq!(roundtrip(bw, &tokens), tokens, "bits={bw}");
        }
    }

    // ── Word-boundary cases ─────────────────────────────────────────────────

    #[test]
    fn group_boundary_token_counts() {
        for &bw in WIDTHS {
            let gs = group_size(bw as u32);
            for &count in &[gs - 1, gs, gs + 1] {
                let tokens = vec![bw as Token; count];
                assert_eq!(roundtrip(bw, &tokens), tokens, "bits={bw} count={count}");
            }
        }
    }

    // ── Non-parameterized manual cases ──────────────────────────────────────

    #[test]
    fn implicit_flush_via_drop() {
        let mut s = Store { bit_width: 16, ..Default::default() };
        {
            let mut w = BitWriter::new(&mut s);
            w.write(0xABCD);
        }
        assert_eq!(s.packed.len(), 2); // data + sentinel
        assert_eq!(s.packed[0] & 0xFFFF, 0xABCD);
    }

    #[test]
    fn explicit_flush_is_idempotent() {
        let mut s = Store { bit_width: 16, ..Default::default() };
        {
            let mut w = BitWriter::new(&mut s);
            w.write(0xABCD);
            w.flush();
        }
        assert_eq!(s.packed.len(), 2);
        assert_eq!(s.packed[0] & 0xFFFF, 0xABCD);
    }

    #[test]
    fn constructor_clears_previous_data() {
        let mut s = Store { bit_width: 16, ..Default::default() };
        {
            let mut w = BitWriter::new(&mut s);
            w.write(0xAAAA);
        }
        assert_eq!(s.packed.len(), 2);
        {
            let mut w = BitWriter::new(&mut s);
            w.write(0xBBBB);
        }
        assert_eq!(s.packed.len(), 2);
        assert_eq!(s.packed[0] & 0xFFFF, 0xBBBB);
    }

    #[test]
    fn straddling_bit_layout_at_12_bits() {
        // Token sequence chosen so the 6th token straddles word 0/1.
        let tokens: [Token; 6] = [0xABC, 0, 0, 0, 0, 0x123];
        let mut s = Store { bit_width: 12, ..Default::default() };
        {
            let mut w = BitWriter::new(&mut s);
            for &t in &tokens {
                w.write(t);
            }
        }
        assert_eq!(s.packed.len(), 3); // 2 data + sentinel
        assert_eq!(s.packed[0], 0x3000000000000ABC);
        assert_eq!(s.packed[1], 0x0000000000000012);
        assert_eq!(roundtrip(12, &tokens), tokens);
    }
}
