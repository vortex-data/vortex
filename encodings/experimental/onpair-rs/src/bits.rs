// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Bit-packing primitives: writer, reader, monomorphic cursor.
//
// Ports of:
//   * `encoding/parsing/bit_writer.h`     -> [`BitWriter`]
//   * (sys helper)                        -> [`read_bits_lsb`], [`unpack_codes_to_u16`]
//   * `decoding/token_cursor.h`           -> [`TokenCursor`] + [`dispatch_bits!`]
//
// All tokens are packed LSB-first across consecutive u64 words. `BitWriter`
// always appends one zero sentinel word after flushing so readers can safely
// over-read 8 bytes past the last real token.

use std::marker::PhantomData;
use std::ptr;

use crate::store::Store;
use crate::types::StreamSpan;
use crate::types::Token;

// ─────────────────────────────────────────────────────────────────────────────
// Bit reader — runtime bit width.
// ─────────────────────────────────────────────────────────────────────────────

/// Read `bits` (1..=16) bits from `packed` starting at LSB-first bit position
/// `bit_pos`. Matches OnPair's `BitWriter` layout exactly.
#[inline]
pub fn read_bits_lsb(packed: &[u64], bit_pos: usize, bits: u32) -> u16 {
    debug_assert!((1..=16).contains(&bits));
    let word_idx = bit_pos / 64;
    let bit_off = (bit_pos % 64) as u32;
    let mask: u64 = (1u64 << bits) - 1;
    let low = packed[word_idx] >> bit_off;
    let combined = if bit_off + bits <= 64 {
        low & mask
    } else {
        let high = packed[word_idx + 1] << (64 - bit_off);
        (low | high) & mask
    };
    combined as u16
}

/// Decompress an LSB-first bit-packed token stream into a flat `Vec<u16>`.
pub fn unpack_codes_to_u16(packed: &[u64], total_tokens: usize, bits: u32) -> Vec<u16> {
    assert!((9..=16).contains(&bits), "bits must be in [9, 16]");
    let mut out = Vec::with_capacity(total_tokens);
    for t in 0..total_tokens {
        out.push(read_bits_lsb(packed, t * bits as usize, bits));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Monomorphic cursor — compile-time bit width.
// ─────────────────────────────────────────────────────────────────────────────

/// Pull-model cursor over a bit-packed token stream, monomorphised on `BITS`
/// so every shift / mask folds to a literal. Use [`dispatch_bits!`] to lift
/// a runtime `BitWidth` (9..=16) into the const-generic parameter.
pub struct TokenCursor<'a, const BITS: u32> {
    base: *const u8,
    bit_pos: u32,
    bit_end: u32,
    _marker: PhantomData<&'a [u64]>,
}

impl<'a, const BITS: u32> TokenCursor<'a, BITS> {
    /// Bind to `packed` and select `span`.
    #[inline]
    pub fn new(packed: &'a [u64], span: StreamSpan) -> Self {
        Self {
            base: packed.as_ptr() as *const u8,
            bit_pos: span.begin * BITS,
            bit_end: span.end * BITS,
            _marker: PhantomData,
        }
    }

    /// Bind without selecting a span yet; call [`Self::reset_to`] before reading.
    #[inline]
    pub fn new_unbound(packed: &'a [u64]) -> Self {
        Self {
            base: packed.as_ptr() as *const u8,
            bit_pos: 0,
            bit_end: 0,
            _marker: PhantomData,
        }
    }

    /// Reset to a new span inside the same packed buffer.
    #[inline]
    pub fn reset_to(&mut self, span: StreamSpan) {
        self.bit_pos = span.begin * BITS;
        self.bit_end = span.end * BITS;
    }

    #[inline]
    pub fn has_more(&self) -> bool {
        self.bit_pos < self.bit_end
    }

    #[inline]
    pub fn remaining(&self) -> u32 {
        (self.bit_end - self.bit_pos) / BITS
    }

    /// Decode and return the next token, advancing the cursor.
    ///
    /// # Safety
    /// Caller must guarantee `has_more()` and that `packed` has the
    /// `BitWriter`-emitted trailing zero-sentinel (8 bytes of safe
    /// over-read).
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Token {
        // SAFETY: byte offset is bit_pos / 8 < bit_end / 8; the 4-byte load
        // extends at most 3 bytes past the last real token byte, within the
        // BitWriter sentinel pad.
        unsafe {
            let off = (self.bit_pos >> 3) as usize;
            let raw = ptr::read_unaligned(self.base.add(off) as *const u32);
            let mask: u32 = (1u32 << BITS) - 1;
            let t = ((raw >> (self.bit_pos & 7)) & mask) as Token;
            self.bit_pos += BITS;
            t
        }
    }
}

/// Dispatch a runtime `BitWidth` (9..=16) to a body where a `const BITS: u32`
/// is in scope. Equivalent to the C++ `dispatch_bits(bw, fn)` template.
///
/// # Safety
/// The default arm is `unreachable_unchecked`. `Column::compress` validates
/// bits ∈ 9..=16, so any column-derived dispatch is sound.
#[macro_export]
macro_rules! dispatch_bits {
    ($bits:expr, | $bits_const:ident | $body:expr) => {
        match $bits {
            9 => {
                const $bits_const: u32 = 9;
                $body
            }
            10 => {
                const $bits_const: u32 = 10;
                $body
            }
            11 => {
                const $bits_const: u32 = 11;
                $body
            }
            12 => {
                const $bits_const: u32 = 12;
                $body
            }
            13 => {
                const $bits_const: u32 = 13;
                $body
            }
            14 => {
                const $bits_const: u32 = 14;
                $body
            }
            15 => {
                const $bits_const: u32 = 15;
                $body
            }
            16 => {
                const $bits_const: u32 = 16;
                $body
            }
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// BitWriter — runtime bit width, RAII flush.
// ─────────────────────────────────────────────────────────────────────────────

/// LSB-first bit-packing into a [`Store`]. Tokens straddling a 64-bit word
/// boundary are split across the next word. The destructor flushes any
/// partial word and appends a zero sentinel.
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

    /// Append one token to the packed stream.
    #[inline]
    pub fn write(&mut self, token: Token) {
        let value = (token as u64) & self.mask;
        self.buf |= value << self.shift;
        self.shift += self.bits as u32;
        if self.shift >= 64 {
            self.store.packed.push(self.buf);
            self.shift -= 64;
            // When the token exactly fills the word the spill must be zero;
            // an unconditional right-shift by `bits` would only be a no-op
            // at `bits == 64`.
            self.buf = if self.shift == 0 {
                0
            } else {
                value >> (self.bits as u32 - self.shift)
            };
        }
        self.count += 1;
    }

    /// Flush the partial word and append the zero sentinel. Idempotent.
    /// Called automatically on drop.
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
            self.store.packed.push(0);
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
    use crate::types::BitWidth;

    // ── read_bits_lsb / unpack_codes_to_u16 ────────────────────────────────

    #[test]
    fn unpack_roundtrips_simple_pattern() {
        let bits = 12u32;
        let a = 0xABC_u64;
        let b = 0xDEF_u64;
        let c = 0x123_u64;
        let word = a | (b << 12) | (c << 24);
        let packed = vec![word, 0];
        assert_eq!(read_bits_lsb(&packed, 0, bits), 0xABC);
        assert_eq!(read_bits_lsb(&packed, 12, bits), 0xDEF);
        assert_eq!(read_bits_lsb(&packed, 24, bits), 0x123);
        assert_eq!(
            unpack_codes_to_u16(&packed, 3, bits),
            vec![0xABC, 0xDEF, 0x123]
        );
    }

    // ── BitWriter ──────────────────────────────────────────────────────────

    fn roundtrip(bits: BitWidth, tokens: &[Token]) -> Vec<Token> {
        let mut store = Store {
            bit_width: bits,
            ..Default::default()
        };
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

    #[test]
    fn zero_tokens_produce_empty_packed() {
        for &bw in WIDTHS {
            let mut s = Store {
                bit_width: bw,
                ..Default::default()
            };
            {
                let w = BitWriter::new(&mut s);
                assert_eq!(w.tokens_written(), 0);
            }
            assert!(s.packed.is_empty(), "bits={bw}");
        }
    }

    #[test]
    fn packed_size_consistent_with_token_count() {
        for &bw in WIDTHS {
            let n = group_size(bw as u32) + 3;
            let mut s = Store {
                bit_width: bw,
                ..Default::default()
            };
            {
                let mut w = BitWriter::new(&mut s);
                for _ in 0..n {
                    w.write(1);
                }
            }
            assert_eq!(
                s.packed.len(),
                expected_packed_words(n, bw) + 1,
                "bits={bw}"
            );
        }
    }

    #[test]
    fn tokens_written_count_increases_per_write() {
        for &bw in WIDTHS {
            let mut s = Store {
                bit_width: bw,
                ..Default::default()
            };
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

    #[test]
    fn single_zero_token_roundtrip() {
        for &bw in WIDTHS {
            let r = roundtrip(bw, &[0]);
            assert_eq!(r, vec![0]);
        }
    }

    #[test]
    fn single_max_token_roundtrip() {
        for &bw in WIDTHS {
            let mx = max_token(bw);
            assert_eq!(roundtrip(bw, &[mx]), vec![mx]);
        }
    }

    #[test]
    fn mixed_zero_and_max_roundtrip() {
        for &bw in WIDTHS {
            let mx = max_token(bw);
            let tokens: Vec<Token> = (0..30).map(|i| if i % 2 == 0 { 0 } else { mx }).collect();
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

    #[test]
    fn implicit_flush_via_drop() {
        let mut s = Store {
            bit_width: 16,
            ..Default::default()
        };
        {
            let mut w = BitWriter::new(&mut s);
            w.write(0xABCD);
        }
        assert_eq!(s.packed.len(), 2);
        assert_eq!(s.packed[0] & 0xFFFF, 0xABCD);
    }

    #[test]
    fn explicit_flush_is_idempotent() {
        let mut s = Store {
            bit_width: 16,
            ..Default::default()
        };
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
        let mut s = Store {
            bit_width: 16,
            ..Default::default()
        };
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
        let tokens: [Token; 6] = [0xABC, 0, 0, 0, 0, 0x123];
        let mut s = Store {
            bit_width: 12,
            ..Default::default()
        };
        {
            let mut w = BitWriter::new(&mut s);
            for &t in &tokens {
                w.write(t);
            }
        }
        assert_eq!(s.packed.len(), 3);
        assert_eq!(s.packed[0], 0x3000000000000ABC);
        assert_eq!(s.packed[1], 0x0000000000000012);
        assert_eq!(roundtrip(12, &tokens), tokens);
    }

    // ── TokenCursor ────────────────────────────────────────────────────────

    fn collect_with<const BITS: u32>(s: &Store, n: usize) -> Vec<Token> {
        let mut c = TokenCursor::<BITS>::new(
            &s.packed,
            StreamSpan {
                begin: 0,
                end: n as u32,
            },
        );
        let mut out = Vec::with_capacity(n);
        while c.has_more() {
            out.push(c.next());
        }
        out
    }

    fn pack(bits: BitWidth, tokens: &[Token]) -> Store {
        let mut s = Store {
            bit_width: bits,
            ..Default::default()
        };
        {
            let mut w = BitWriter::new(&mut s);
            for &t in tokens {
                w.write(t);
            }
        }
        s
    }

    #[test]
    fn cursor_roundtrip_all_widths() {
        for bits in 9u8..=16 {
            let max = ((1u32 << bits) - 1) as Token;
            let tokens: Vec<Token> = (0..100u32)
                .map(|i| ((i as Token).wrapping_mul(7)) & max)
                .collect();
            let s = pack(bits, &tokens);
            let out = match bits {
                9 => collect_with::<9>(&s, tokens.len()),
                10 => collect_with::<10>(&s, tokens.len()),
                11 => collect_with::<11>(&s, tokens.len()),
                12 => collect_with::<12>(&s, tokens.len()),
                13 => collect_with::<13>(&s, tokens.len()),
                14 => collect_with::<14>(&s, tokens.len()),
                15 => collect_with::<15>(&s, tokens.len()),
                16 => collect_with::<16>(&s, tokens.len()),
                _ => unreachable!(),
            };
            assert_eq!(out, tokens, "bits={bits}");
        }
    }

    #[test]
    fn cursor_remaining_decrements() {
        let s = pack(12, &[1, 2, 3, 4, 5]);
        let mut c = TokenCursor::<12>::new(&s.packed, StreamSpan { begin: 0, end: 5 });
        assert_eq!(c.remaining(), 5);
        c.next();
        assert_eq!(c.remaining(), 4);
    }

    #[test]
    fn cursor_reset_to_works() {
        let s = pack(12, &[10, 20, 30, 40, 50]);
        let mut c = TokenCursor::<12>::new_unbound(&s.packed);
        c.reset_to(StreamSpan { begin: 2, end: 5 });
        assert_eq!(c.next(), 30);
        assert_eq!(c.next(), 40);
        assert_eq!(c.next(), 50);
        assert!(!c.has_more());
    }

    #[test]
    fn dispatch_bits_routes_to_correct_arm() {
        for bits in 9u8..=16 {
            let result = dispatch_bits!(bits, |B| B);
            assert_eq!(result, bits as u32);
        }
    }
}
