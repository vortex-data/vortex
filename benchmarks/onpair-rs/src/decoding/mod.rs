// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token decode.
//!
//! Decoding is a gather-copy: each code names a dictionary token; the output is
//! those tokens concatenated. [`decode_into`] copies a fixed 16 bytes per token
//! (one `copy16` store) whatever the token's true length — no per-token length
//! branch. The cost is two paddings to keep that over-read in bounds: the
//! dictionary's read-padding backs the source over-read, and the output buffer's
//! [`DECODE_PADDING`] backs the destination over-store. It is generic over the
//! dictionary representation ([`DictionaryView`]) — a load-free
//! [`WideDictionaryView`](crate::core::dictionary::WideDictionaryView) or a
//! [`CompactDictionaryView`](crate::core::dictionary::CompactDictionaryView).
//! Codes are bounds-checked in the loop; [`ColumnView`](crate::ColumnView) wraps
//! it as the buffer-sized [`decompress_into`](crate::ColumnView::decompress_into).

use std::mem::MaybeUninit;

use crate::core::dictionary::DictionaryView;
use crate::core::types::{MAX_TOKEN_SIZE, Token};
use crate::core::validate::{InvalidColumn, panic_malformed};

mod copy;

/// Trailing bytes a [`decode_into`] output buffer needs **beyond** the decoded
/// length. The decoder over-stores a fixed [`MAX_TOKEN_SIZE`]-byte chunk for the
/// final token — up to `MAX_TOKEN_SIZE - 1` bytes past the logical end — so an
/// output buffer is sized as `decoded_len + DECODE_PADDING`.
pub const DECODE_PADDING: usize = MAX_TOKEN_SIZE;

/// Exact decoded byte length of `codes` against `dict` (the sum of token
/// lengths) — sizes a decode buffer.
///
/// Bounds-checks each code, so a malformed code stream panics with
/// [`InvalidColumn::CodeOutOfRange`] rather than reading out of bounds. The sum
/// is only meaningful for a dictionary satisfying [`DictionaryView`]'s
/// structural invariants.
#[inline]
pub fn decoded_len<V: DictionaryView>(codes: &[Token], dict: V) -> usize {
    let n = dict.num_tokens();
    let mut sum = 0usize;
    for &c in codes {
        if (c as usize) >= n {
            panic_malformed(InvalidColumn::CodeOutOfRange);
        }
        // SAFETY: c < num_tokens.
        let len = unsafe { dict.token_len_unchecked(c) };
        sum = sum
            .checked_add(len)
            .unwrap_or_else(|| panic_malformed(InvalidColumn::DecodedLenOverflow));
    }
    sum
}

/// Decode `codes` against `dict` into `out`, returning the bytes written. It
/// over-reads a fixed 16 bytes per token, relying on the structural invariants
/// guaranteed by [`DictionaryView`]. Each code is bounds-checked in the loop (a
/// near-free, predicted-not-taken branch); an out-of-range code panics with
/// [`InvalidColumn::CodeOutOfRange`].
///
/// # Safety
/// `out.len() >= decoded_len(codes, dict) + DECODE_PADDING`.
pub unsafe fn decode_into<V: DictionaryView>(
    codes: &[Token],
    dict: V,
    out: &mut [MaybeUninit<u8>],
) -> usize {
    let ntok = dict.num_tokens();
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut w = 0usize;
    for &code in codes {
        if code as usize >= ntok {
            panic_malformed(InvalidColumn::CodeOutOfRange);
        }
        // SAFETY: code in range ⇒ token_ptr is readable for 16 bytes (the dict is
        // read-padded); the trailing DECODE_PADDING keeps the last token's
        // over-store within `out`.
        unsafe {
            let src = dict.token_ptr(code);
            let len = dict.token_len_unchecked(code);
            copy::copy16(src, dst.add(w));
            w += len;
        }
    }
    w
}

/// [`try_decode_into`]'s `out` could not hold the decoded bytes. Nothing was
/// written past the end of `out`; size it from [`decoded_len`] and retry (the
/// checked decode needs no [`DECODE_PADDING`] slack — see [`try_decode_into`]).
///
/// This is a *buffer-capacity* outcome, deliberately distinct from
/// [`InvalidColumn`]: a too-small caller buffer is not a
/// malformation of the compressed data, so it does not belong to the validation
/// vocabulary. An out-of-range code *is* a malformation and is handled like the
/// other decoders — a panic at the point of use — not a variant here.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OutputTooSmall;

impl std::fmt::Display for OutputTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("output buffer too small for decoded bytes")
    }
}

impl std::error::Error for OutputTooSmall {}

/// Decode `codes` against `dict` into `out` — the **safe,
/// buffer-checked** counterpart of [`decode_into`]. Returns `Ok(bytes_written)`, or
/// `Err(`[`OutputTooSmall`]`)` if `out` cannot hold the result (in which case
/// nothing is written past the end of `out`). Never reads or writes out of bounds.
///
/// Like [`decode_into`] it over-copies a fixed [`MAX_TOKEN_SIZE`] bytes per token,
/// but it keeps that over-store in bounds *without* a per-token output check or a
/// [`decoded_len`] pre-pass. Each round it issues `(out.len() - written) /
/// MAX_TOKEN_SIZE` over-copies: since every token is at most [`MAX_TOKEN_SIZE`],
/// that many 16-byte stores from the current cursor cannot reach the end of `out`,
/// so the whole batch needs no per-store check; then it recomputes. Once fewer than
/// [`MAX_TOKEN_SIZE`] bytes remain, the tail is finished with exact per-token copies
/// — so `out` needs **no** [`DECODE_PADDING`] slack: a buffer of exactly
/// [`decoded_len`] suffices (and an over-store-padded one stays on the fast path
/// throughout).
///
/// Bounds-checking codes (the obligation [`decode_into`] leaves to the dictionary's
/// trust) is necessary but **not** sufficient for memory safety: in-range codes can
/// address [`MAX_TOKEN_SIZE`]-byte tokens, so a buffer sized from any figure other
/// than this column's own decoded length can still overflow. This routine derives
/// the bound from `out` itself, so it is sound for any `(codes, out)` pair.
///
/// The [`DictionaryView`] contract guarantees that token reads and the fixed
/// 16-byte over-read are in bounds.
///
/// # Panics
/// With [`InvalidColumn::CodeOutOfRange`] on a code that does not index the
/// dictionary — the same malformation backstop as [`decode_into`] / [`decoded_len`]
/// (never UB). Buffer fit is this function's recoverable concern.
pub fn try_decode_into<V: DictionaryView>(
    codes: &[Token],
    dict: V,
    out: &mut [MaybeUninit<u8>],
) -> Result<usize, OutputTooSmall> {
    let ntok = dict.num_tokens();
    let cap = out.len();
    let dst = out.as_mut_ptr().cast::<u8>();
    let mut w = 0usize;
    let mut i = 0usize;

    // Fast path: batches of fixed 16-byte over-copies. `batch` is the largest count
    // provably within `out` from the current cursor, so the inner loop needs no
    // per-store output check.
    while i < codes.len() {
        let batch = (cap - w) / MAX_TOKEN_SIZE;
        if batch == 0 {
            break;
        }
        let end = (i + batch).min(codes.len());
        for &code in &codes[i..end] {
            if code as usize >= ntok {
                panic_malformed(InvalidColumn::CodeOutOfRange);
            }
            // SAFETY: code < ntok ⇒ `token_ptr` is readable for MAX_TOKEN_SIZE (the
            // dict is read-padded). With `w0` the cursor when this round took `batch
            // = (cap - w0) / MAX_TOKEN_SIZE`, this is store number `< batch`, and each
            // preceding token advanced the cursor by at most MAX_TOKEN_SIZE, so the
            // cursor is `< w0 + MAX_TOKEN_SIZE * batch <= cap`; the 16-byte store
            // therefore ends at `<= cap`, in bounds.
            unsafe {
                let len = dict.token_len_unchecked(code);
                copy::copy16(dict.token_ptr(code), dst.add(w));
                w += len;
            }
        }
        i = end;
    }

    // Tail: fewer than MAX_TOKEN_SIZE bytes remain but codes are left. Exact copies
    // (no over-store), failing the moment a token would not fit.
    for &code in &codes[i..] {
        if code as usize >= ntok {
            panic_malformed(InvalidColumn::CodeOutOfRange);
        }
        // SAFETY: `code < ntok` (just checked) — the precondition for both unchecked
        // accessors here, matching the fast loop rather than re-checking the bound.
        let len = unsafe { dict.token_len_unchecked(code) };
        if w + len > cap {
            return Err(OutputTooSmall);
        }
        // SAFETY: `token_ptr` is readable for MAX_TOKEN_SIZE >= len, and just checked
        // `w + len <= cap`; `copy_token_bytes` touches only `[ptr, ptr + len)` on
        // both sides, so it stays in bounds — no over-store into the (absent) slack.
        unsafe {
            copy::copy_token_bytes(dict.token_ptr(code), dst.add(w), len);
        }
        w += len;
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::{CompactDictionaryView, Dictionary};

    /// Build read-padded compact `(bytes, offsets)` from tokens.
    fn padded(tokens: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut offsets = vec![0u32];
        for t in tokens {
            bytes.extend_from_slice(t);
            offsets.push(bytes.len() as u32);
        }
        bytes.resize(bytes.len() + MAX_TOKEN_SIZE, 0); // worst-case read padding
        (bytes, offsets)
    }

    fn expected(tokens: &[&[u8]], codes: &[Token]) -> Vec<u8> {
        codes
            .iter()
            .flat_map(|&c| tokens[c as usize].iter().copied())
            .collect()
    }

    /// Decode into a padded buffer and return the initialized bytes.
    fn vec_decode<V: DictionaryView>(codes: &[Token], dict: V) -> Vec<u8> {
        let n = decoded_len(codes, dict);
        let mut out = Vec::with_capacity(n + DECODE_PADDING);
        // SAFETY: read-padded dict, in-range codes, buffer + DECODE_PADDING.
        let w = unsafe { decode_into(codes, dict, out.spare_capacity_mut()) };
        unsafe { out.set_len(w) };
        out
    }

    /// Decode through both representations and compare. Cheap enough to run under
    /// Miri, which then proves the over-copy has no UB.
    fn check(tokens: &[&[u8]], codes: &[Token]) {
        let (bytes, offsets) = padded(tokens);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let want = expected(tokens, codes);

        assert_eq!(decoded_len(codes, view), want.len());
        assert_eq!(vec_decode(codes, view), want, "compact");

        let wide = view.to_wide();
        assert_eq!(vec_decode(codes, wide.as_view()), want, "wide");
    }

    #[test]
    fn decodes_mixed_length_tokens() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        let codes: Vec<Token> = (0..40).map(|i| (i % 4) as Token).collect();
        check(tokens, &codes);
    }

    #[test]
    fn decodes_full_width_last_token() {
        let full = vec![b'z'; MAX_TOKEN_SIZE];
        let tokens: &[&[u8]] = &[b"x", &full];
        let codes: Vec<Token> = (0..40).map(|i| (i % 2) as Token).collect();
        check(tokens, &codes);
    }

    #[test]
    fn decodes_short_final_token() {
        // Ends in a 1-byte token: the last over-store must land in the
        // DECODE_PADDING room. Miri checks the bound.
        let tokens: &[&[u8]] = &[b"a", b"bcde"];
        let mut codes: Vec<Token> = (0..40).map(|i| (i % 2) as Token).collect();
        *codes.last_mut().unwrap() = 0;
        check(tokens, &codes);
    }

    #[test]
    fn decodes_all_tail_length_buckets() {
        let t = [
            vec![b'a'; 1],
            vec![b'b'; 3],
            vec![b'c'; 5],
            vec![b'd'; 11],
            vec![b'e'; 15],
            vec![b'f'; 16],
        ];
        let tokens: Vec<&[u8]> = t.iter().map(Vec::as_slice).collect();
        let codes: Vec<Token> = (0..40).map(|i| (i % t.len()) as Token).collect();
        check(&tokens, &codes);
    }

    #[test]
    fn decodes_empty_code_stream() {
        check(&[b"a", b"b"], &[]);
    }

    #[test]
    fn single_token_decode() {
        let tokens: &[&[u8]] = &[b"hello", b"world"];
        check(tokens, &[0]);
        check(tokens, &[1]);
    }

    #[test]
    #[should_panic(expected = "code index out of range")]
    fn decode_into_panics_on_out_of_range_code() {
        let (bytes, offsets) = padded(&[b"a", b"b"]);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let mut out = vec![MaybeUninit::uninit(); 64];
        // SAFETY: read-padded dict + generous buffer; the bad code must panic.
        unsafe { decode_into(&[0, 5], view, &mut out) };
    }

    #[test]
    #[should_panic(expected = "code index out of range")]
    fn decoded_len_panics_on_out_of_range_code() {
        let (bytes, offsets) = padded(&[b"a", b"b"]);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let _ = decoded_len(&[0, 5], view);
    }

    /// `try_decode_into` agrees with `decode_into` both on a tight (exactly
    /// `decoded_len`, no padding) buffer — exercising the exact-copy tail — and on a
    /// padded one that stays on the all-over-copy fast path.
    #[test]
    fn try_decode_matches_decode_into() {
        let tokens: &[&[u8]] = &[b"a", b"bc", b"def", b"ghij"];
        let codes: Vec<Token> = (0..40).map(|i| (i % 4) as Token).collect();
        let (bytes, offsets) = padded(tokens);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        let want = expected(tokens, &codes);

        for cap in [want.len(), want.len() + DECODE_PADDING] {
            let mut out = vec![MaybeUninit::uninit(); cap];
            let w = try_decode_into(&codes, view, &mut out).unwrap();
            assert_eq!(w, want.len());
            let got = unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<u8>(), w) };
            assert_eq!(got, want.as_slice(), "cap {cap}");
        }
    }

    #[test]
    fn try_decode_fits_exact_and_rejects_one_short() {
        let tokens: &[&[u8]] = &[b"a", b"bcde"];
        let codes: Vec<Token> = vec![1, 1, 1]; // 12 decoded bytes
        let (bytes, offsets) = padded(tokens);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);

        // Exactly enough succeeds; one byte short is rejected, never overflowing.
        assert_eq!(
            try_decode_into(&codes, view, &mut [MaybeUninit::uninit(); 12]),
            Ok(12)
        );
        assert_eq!(
            try_decode_into(&codes, view, &mut [MaybeUninit::uninit(); 11]),
            Err(OutputTooSmall)
        );
    }

    #[test]
    #[should_panic(expected = "code index out of range")]
    fn try_decode_panics_on_out_of_range_code() {
        let (bytes, offsets) = padded(&[b"a", b"b"]);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        // Generous buffer, so the bad code is hit on the fast path, not the tail.
        let mut out = vec![MaybeUninit::uninit(); 64];
        // A malformed code panics like the other decoders — recover via `validate`.
        let _ = try_decode_into(&[0, 5], view, &mut out);
    }

    #[test]
    fn try_decode_empty_codes_into_empty_buffer() {
        let (bytes, offsets) = padded(&[b"a", b"b"]);
        let view = CompactDictionaryView::from_raw(&bytes, &offsets);
        assert_eq!(try_decode_into(&[], view, &mut []), Ok(0));
    }

    /// The exact-copy tail (`copy_token_bytes`) must reproduce a token of every
    /// length bucket. Each token is decoded into an exactly-sized buffer, so for
    /// `len < MAX_TOKEN_SIZE` the whole decode goes through the tail (`batch == 0`),
    /// hitting the 1, 2|3, 4..7 and 8..15 arms.
    #[test]
    fn try_decode_tail_reproduces_every_length_bucket() {
        for len in [1usize, 2, 3, 4, 5, 7, 8, 11, 15] {
            let tok = vec![b'a' + (len as u8); len];
            let (bytes, offsets) = padded(&[tok.as_slice()]);
            let view = CompactDictionaryView::from_raw(&bytes, &offsets);
            let mut out = vec![MaybeUninit::uninit(); len]; // exact ⇒ tail-only
            let w = try_decode_into(&[0], view, &mut out).unwrap();
            assert_eq!(w, len, "len {len}");
            let got = unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<u8>(), w) };
            assert_eq!(got, tok.as_slice(), "len {len}");
        }
    }
}
