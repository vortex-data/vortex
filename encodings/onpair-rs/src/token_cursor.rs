// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Port of `include/onpair/decoding/token_cursor.h`.
//
// Pull-model iterator over the bit-packed token stream, monomorphised on
// `BITS` so every shift / mask folds to a literal at compile time. Use
// [`crate::dispatch_bits`] to lift a runtime `BitWidth` (9..=16) into the
// const-generic parameter once at column open.
//
// Layout notes (mirrors the C++ implementation exactly):
//   * `base` is a raw `*const u8` into the packed buffer; advances are in
//     bit units.
//   * `next()` reads 4 unaligned bytes and shifts/masks to extract the
//     next `BITS` bits. For `BITS ∈ 9..=16` and any `bit_off ∈ 0..8`,
//     `bit_off + BITS ≤ 24 ≤ 32`, so a single 32-bit load always suffices.
//   * `BitWriter::flush` emits a trailing zero-sentinel `u64` word, giving
//     us at least 8 bytes of safe over-read past the last real token.

use std::marker::PhantomData;
use std::ptr;

use crate::types::{StreamSpan, Token};

pub struct TokenCursor<'a, const BITS: u32> {
    /// Raw byte pointer into the packed `u64` buffer.
    base: *const u8,
    /// Current bit offset.
    bit_pos: u32,
    /// One-past-the-last bit offset.
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

    /// Bind to `packed` without selecting a span yet. Call [`Self::reset_to`]
    /// before reading.
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
        // SAFETY: byte offset is `bit_pos / 8`, which is `< bit_end / 8`. The
        // unaligned 4-byte load extends at most 3 bytes past the last real
        // token byte, well within the BitWriter sentinel pad.
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

/// Dispatch a runtime [`crate::types::BitWidth`] (9..=16) to a body that
/// uses a `BITS` const-generic in scope.
///
/// Equivalent to the C++ `dispatch_bits(bw, fn)` template helper. The
/// macro guarantees there is exactly one match arm per legal width, so all
/// downstream functions stamped with the const get monomorphised at the
/// call site — shifts, masks and cursor advances fold to literals.
///
/// # Safety
/// The default arm is `unreachable_unchecked`. Callers must ensure
/// `bits ∈ 9..=16` (e.g. via [`crate::is_valid_bits`] at column open).
#[macro_export]
macro_rules! dispatch_bits {
    ($bits:expr, |$bits_const:ident| $body:expr) => {
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
            // SAFETY: every public Column is constructed via
            // Column::compress, which validates bits ∈ 9..=16.
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_writer::BitWriter;
    use crate::store::Store;

    fn pack(bits: u8, tokens: &[Token]) -> Store {
        let mut s = Store { bit_width: bits, ..Default::default() };
        {
            let mut w = BitWriter::new(&mut s);
            for &t in tokens {
                w.write(t);
            }
        }
        s
    }

    fn collect_with<const BITS: u32>(s: &Store, n: usize) -> Vec<Token> {
        let mut c = TokenCursor::<BITS>::new(&s.packed, StreamSpan { begin: 0, end: n as u32 });
        let mut out = Vec::with_capacity(n);
        while c.has_more() {
            out.push(c.next());
        }
        out
    }

    #[test]
    fn cursor_roundtrip_single_token() {
        for &t in &[0u16, 1, 0x100, 0x1FF, 0xFFFF] {
            let max = (1u32 << 16) - 1;
            if (t as u32) > max {
                continue;
            }
            let s = pack(16, &[t]);
            let out = collect_with::<16>(&s, 1);
            assert_eq!(out, vec![t]);
        }
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
        c.next();
        c.next();
        assert_eq!(c.remaining(), 2);
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
