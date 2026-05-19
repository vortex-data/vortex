// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Owning compressed column. API-compatible with the subset of
// `vortex-onpair-sys::Column` that `vortex-onpair` actually consumes:
// `compress`, `len`, `bits`, `dict_size`, `parts`. The shim accepts
// `&[u64]` row offsets so callers don't need to truncate to u32; internally
// we sanity-check and downcast.

use aho_corasick::AhoCorasick;

use crate::automaton::TokenAutomaton;
use crate::bits::TokenCursor;
use crate::config::{Error, OnPairTrainingConfig};
use crate::dict::Dictionary;
use crate::dispatch_bits;
use crate::parser::parse;
use crate::store::Store;
use crate::trainer::{TrainResult, train};
use crate::types::{StreamSpan, is_valid_bits};

// ─────────────────────────────────────────────────────────────────────────────
// Bitmap helpers (private — `Column::*_bitmap` are the public entry points).
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn empty_bitmap(n: usize) -> Vec<u8> {
    vec![0u8; n.div_ceil(8)]
}

#[inline]
fn set_bit(bits: &mut [u8], i: usize) {
    bits[i / 8] |= 1u8 << (i % 8);
}

#[inline]
fn fill_bitmap(n: usize) -> Vec<u8> {
    let mut bits = empty_bitmap(n);
    for i in 0..n {
        set_bit(&mut bits, i);
    }
    bits
}

// ─────────────────────────────────────────────────────────────────────────────
// Const-generic decode and scan inner loops. Each is monomorphised per
// `BITS ∈ 9..=16` by `dispatch_bits!`, which lets the compiler fold every
// shift / mask in the cursor to a literal — same effect as the C++
// `scan_impl<Bits>` template after `dispatch_bits()` resolves it.
// ─────────────────────────────────────────────────────────────────────────────

/// Sum decoded-byte lengths for a token-id span in one pass over the cursor.
#[inline]
fn span_decoded_len<const BITS: u32>(
    dict_table: &[u64],
    packed: &[u64],
    span: StreamSpan,
) -> usize {
    let mut total = 0usize;
    let mut cursor = TokenCursor::<BITS>::new(packed, span);
    while cursor.has_more() {
        let code = cursor.next() as usize;
        // SAFETY: every code is < dict_size, validated at compress time.
        total += unsafe { (*dict_table.get_unchecked(code) & 0xffff) as usize };
    }
    total
}

/// Decode a token span into `out`. Uses a fixed 16-byte over-copy per token
/// (the trainer pads `dict_bytes` with MAX_TOKEN_SIZE trailing zeros so this
/// never reads past the end) and advances the cursor by the token's true
/// length. The compiler lowers `copy_nonoverlapping` of MAX_TOKEN_SIZE to a
/// single unaligned SIMD store on x86_64 / aarch64.
///
/// `out` must have at least `decoded_len + MAX_TOKEN_SIZE` reserved
/// capacity at call time; we always set the final length to the *true* total
/// (no over-copy bytes are visible).
#[inline]
unsafe fn decode_span_unchecked<const BITS: u32>(
    dict_bytes: *const u8,
    dict_table: &[u64],
    packed: &[u64],
    span: StreamSpan,
    dst: *mut u8,
) -> usize {
    let mut cursor = TokenCursor::<BITS>::new(packed, span);
    let mut cur = dst;
    // SAFETY: caller invariants — dict_table indices are bounded by
    // dict_size; dst has decoded_len + MAX_TOKEN_SIZE capacity.
    unsafe {
        while cursor.has_more() {
            let code = cursor.next() as usize;
            let entry = *dict_table.get_unchecked(code);
            let off = (entry >> 16) as usize;
            let len = (entry & 0xffff) as usize;
            std::ptr::copy_nonoverlapping(dict_bytes.add(off), cur, crate::MAX_TOKEN_SIZE);
            cur = cur.add(len);
        }
        cur.offset_from(dst) as usize
    }
}

#[inline]
fn scan_with_bits<const BITS: u32, A, F>(
    packed: &[u64],
    boundaries: &[u32],
    num_rows: usize,
    aut: &mut A,
    on_match: &mut F,
) where
    A: TokenAutomaton,
    F: FnMut(usize),
{
    let mut cursor = TokenCursor::<BITS>::new_unbound(packed);
    for row in 0..num_rows {
        aut.reset();
        cursor.reset_to(StreamSpan { begin: boundaries[row], end: boundaries[row + 1] });
        while cursor.has_more() {
            let t = cursor.next();
            aut.step(t);
            if aut.is_dead() {
                break;
            }
        }
        if aut.is_accepted() {
            on_match(row);
        }
    }
}

/// Pack `Dictionary` into a per-token `(offset << 16) | length` table.
/// Token length is bounded by `MAX_TOKEN_SIZE = 16`, so 16 bits suffice.
fn build_dict_table(dict: &Dictionary) -> Vec<u64> {
    let n = dict.num_tokens();
    let mut table = Vec::with_capacity(n);
    for i in 0..n {
        let off = dict.offsets[i] as u64;
        let len = (dict.offsets[i + 1] - dict.offsets[i]) as u64;
        debug_assert!(len <= crate::MAX_TOKEN_SIZE as u64);
        table.push((off << 16) | len);
    }
    table
}

/// Owning compressed column. Built by [`Column::compress`].
#[derive(Debug, Clone)]
pub struct Column {
    dict: Dictionary,
    store: Store,
    num_rows: usize,
    /// Per-token `(offset << 16) | length` packed into a `u64`. Built once
    /// at compress / from_parts time so the decode and predicate hot loops
    /// do one indexed load per token instead of two indexed offset reads.
    /// Length = `dict.num_tokens()`.
    dict_table: Vec<u64>,
}

/// Borrowed raw arrays of a column. Mirrors `vortex-onpair-sys::Parts`.
#[derive(Copy, Clone)]
pub struct Parts<'a> {
    /// Concatenated dictionary entry bytes. The C++ shim's caller pads this
    /// with `MAX_TOKEN_SIZE` zeros before handing to the decoder; we expose
    /// the unpadded logical slice (length `dict_offsets.last()`).
    pub dict_bytes: &'a [u8],
    /// Length `dict_size + 1`.
    pub dict_offsets: &'a [u32],
    /// LSB-first bit-packed token stream.
    pub codes_packed: &'a [u64],
    /// Length `num_rows + 1`.
    pub codes_boundaries: &'a [u32],
    /// Bits per token (9..=16).
    pub bits: u32,
    pub num_rows: usize,
}

impl Column {
    /// Compress `n` byte strings described by a flat `bytes` blob and an
    /// `offsets` array of length `n + 1`. Matches
    /// `vortex-onpair-sys::Column::compress`.
    pub fn compress(
        bytes: &[u8],
        offsets: &[u64],
        config: OnPairTrainingConfig,
    ) -> Result<Self, Error> {
        if offsets.is_empty() {
            return Err(Error::InvalidArg);
        }
        if !is_valid_bits(config.bits as u8) {
            return Err(Error::InvalidArg);
        }
        let n = offsets.len() - 1;

        // Downcast u64 offsets to u32. Bail on overflow rather than wrap.
        let mut off32 = Vec::with_capacity(offsets.len());
        for &o in offsets {
            if o > u32::MAX as u64 {
                return Err(Error::InvalidArg);
            }
            off32.push(o as u32);
        }
        if (off32[n] as usize) > bytes.len() {
            return Err(Error::InvalidArg);
        }

        let cfg = config.into();
        let TrainResult { dict, lpm } = train(bytes, &off32, n, &cfg);
        let mut store = Store::default();
        parse(bytes, &off32, n, &lpm, config.bits as u8, &mut store);
        let dict_table = build_dict_table(&dict);

        Ok(Self { dict, store, num_rows: n, dict_table })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_rows
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    #[inline]
    pub fn bits(&self) -> u32 {
        self.store.bit_width as u32
    }

    #[inline]
    pub fn dict_size(&self) -> usize {
        self.dict.num_tokens()
    }

    /// Decompress row `row_id` into `out`, clearing it first. The hot path
    /// uses the const-generic [`TokenCursor`] dispatched on `bits` plus a
    /// fixed-width 16-byte over-copy per token (`MAX_TOKEN_SIZE`). LLVM
    /// lowers each copy to one unaligned 128-bit SIMD store on x86_64 and
    /// aarch64.
    pub fn decompress_row(&self, row_id: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        if row_id >= self.num_rows {
            return Err(Error::OutOfRange);
        }
        out.clear();
        let span = self.store.string_span(row_id);
        // SAFETY of dispatch: bit_width is validated to 9..=16 in compress().
        dispatch_bits!(self.store.bit_width as u32, |B| {
            self.decode_one_span_into::<B>(span, out);
        });
        Ok(())
    }

    /// Decompress every row into a flat byte buffer + `n + 1` offsets.
    pub fn decode_all(&self) -> (Vec<u8>, Vec<u32>) {
        let n = self.num_rows;
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u32);
        if n == 0 {
            return (Vec::new(), offsets);
        }
        let bytes = dispatch_bits!(self.store.bit_width as u32, |B| {
            self.decode_all_inner::<B>(&mut offsets)
        });
        (bytes, offsets)
    }

    /// Inner monomorphic body of [`Self::decode_all`].
    #[inline]
    fn decode_all_inner<const BITS: u32>(&self, offsets: &mut Vec<u32>) -> Vec<u8> {
        // Pre-compute the total decoded length: one cursor pass over every
        // row in token-id space, summing each token's length from
        // `dict_table[code] & 0xffff`. Lets us reserve `bytes` once and
        // skip per-row capacity grow checks in the hot loop.
        let last = *self.store.boundaries.last().unwrap_or(&0);
        let total = span_decoded_len::<BITS>(
            &self.dict_table,
            &self.store.packed,
            StreamSpan { begin: 0, end: last },
        );
        let mut bytes: Vec<u8> = Vec::with_capacity(total + crate::MAX_TOKEN_SIZE);
        let dst = bytes.as_mut_ptr();
        let dict_bytes = self.dict.bytes.as_ptr();
        let dict_table = self.dict_table.as_slice();
        let mut cur_off = 0usize;
        for row in 0..self.num_rows {
            let span = self.store.string_span(row);
            // SAFETY: `dst` has `total + MAX_TOKEN_SIZE` reserved
            // capacity; each token's over-copy stays within `dict_bytes`
            // (padded by `pad_for_decoder`).
            let written = unsafe {
                decode_span_unchecked::<BITS>(
                    dict_bytes,
                    dict_table,
                    &self.store.packed,
                    span,
                    dst.add(cur_off),
                )
            };
            cur_off += written;
            offsets.push(cur_off as u32);
        }
        // SAFETY: cur_off == total ≤ reserved capacity.
        unsafe { bytes.set_len(cur_off); }
        bytes
    }

    /// Internal helper: decode one span into `out` with the SIMD-friendly
    /// over-copy loop. Used by `decompress_row` and `run_byte_predicate`.
    #[inline]
    fn decode_one_span_into<const BITS: u32>(&self, span: StreamSpan, out: &mut Vec<u8>) {
        let len = span_decoded_len::<BITS>(&self.dict_table, &self.store.packed, span);
        let start = out.len();
        out.reserve(len + crate::MAX_TOKEN_SIZE);
        // SAFETY: capacity reserved above; over-copy stays within dict pad.
        unsafe {
            let written = decode_span_unchecked::<BITS>(
                self.dict.bytes.as_ptr(),
                &self.dict_table,
                &self.store.packed,
                span,
                out.as_mut_ptr().add(start),
            );
            debug_assert_eq!(written, len);
            out.set_len(start + written);
        }
    }

    /// `WHERE col = needle` as an LSB-first packed bitmap of length `(n + 7) / 8`.
    ///
    /// Decompress-then-match implementation. For very large columns prefer the
    /// compressed-domain [`crate::EqAutomaton`] via [`Self::scan_bitmap`].
    pub fn equals_bitmap(&self, needle: &[u8]) -> Vec<u8> {
        self.run_byte_predicate(|row| row == needle)
    }

    /// `col LIKE 'needle%'` as an LSB-first packed bitmap.
    pub fn starts_with_bitmap(&self, needle: &[u8]) -> Vec<u8> {
        self.run_byte_predicate(|row| row.starts_with(needle))
    }

    /// `col LIKE '%needle%'` as an LSB-first packed bitmap. Uses `memchr::memmem`.
    pub fn contains_bitmap(&self, needle: &[u8]) -> Vec<u8> {
        if needle.is_empty() {
            return fill_bitmap(self.num_rows);
        }
        let finder = memchr::memmem::Finder::new(needle);
        self.run_byte_predicate(|row| finder.find(row).is_some())
    }

    /// `LIKE '%a%' OR '%b%' OR ...` via Aho-Corasick. Empty `needles` →
    /// all-zero bitmap.
    pub fn multi_pattern_bitmap(&self, needles: &[&[u8]]) -> Vec<u8> {
        if needles.is_empty() {
            return empty_bitmap(self.num_rows);
        }
        let ac = AhoCorasick::new(needles).expect("aho-corasick: build");
        self.run_byte_predicate(|row| ac.is_match(row))
    }

    /// Decompress every row and apply `pred`. Shared backend for the
    /// `*_bitmap` methods.
    fn run_byte_predicate<F: FnMut(&[u8]) -> bool>(&self, mut pred: F) -> Vec<u8> {
        let mut bits = empty_bitmap(self.num_rows);
        // One reusable scratch buffer; the over-copy in
        // `decode_one_span_into` extends the spare capacity by
        // MAX_TOKEN_SIZE each call.
        let mut buf: Vec<u8> = Vec::with_capacity(128);
        // SAFETY of dispatch: bit_width validated 9..=16 in compress().
        dispatch_bits!(self.store.bit_width as u32, |B| {
            for i in 0..self.num_rows {
                buf.clear();
                let span = self.store.string_span(i);
                self.decode_one_span_into::<B>(span, &mut buf);
                if pred(&buf) {
                    set_bit(&mut bits, i);
                }
            }
        });
        bits
    }

    /// Run a [`TokenAutomaton`] over every row's compressed token stream
    /// and collect matching row ids. The automaton is reset at the start of
    /// each row and stepped on every token; the loop breaks early when
    /// `is_dead()` returns true.
    pub fn scan<A: TokenAutomaton>(&self, mut aut: A) -> Vec<usize> {
        let mut out = Vec::new();
        self.scan_with(&mut aut, |i| out.push(i));
        out
    }

    /// Callback form of [`Self::scan`] — no `Vec<usize>` allocation.
    /// Hot path runs through a monomorphic [`TokenCursor<BITS>`] selected
    /// once via [`dispatch_bits!`], identical structure to the C++
    /// `scan_impl<Bits>` template.
    pub fn scan_with<A: TokenAutomaton, F: FnMut(usize)>(&self, mut aut: A, mut on_match: F) {
        // SAFETY of dispatch: bit_width is validated to 9..=16 in compress().
        dispatch_bits!(self.store.bit_width as u32, |B| {
            scan_with_bits::<B, _, _>(
                &self.store.packed,
                &self.store.boundaries,
                self.num_rows,
                &mut aut,
                &mut on_match,
            );
        });
    }

    /// Run a [`TokenAutomaton`] and collect matches as an LSB-first packed
    /// bitmap. Same shape as the byte-level `*_bitmap` APIs.
    pub fn scan_bitmap<A: TokenAutomaton>(&self, aut: A) -> Vec<u8> {
        let mut bits = empty_bitmap(self.num_rows);
        self.scan_with(aut, |i| bits[i / 8] |= 1u8 << (i % 8));
        bits
    }

    /// Access the column's dictionary. Required to construct any
    /// `*Automaton` (they take `&Dictionary`).
    pub fn dictionary(&self) -> &Dictionary {
        &self.dict
    }

    /// Total byte cost of the compressed column. Sum of the dictionary
    /// bytes, dictionary offsets, packed code stream, and code boundaries.
    /// Useful when comparing configurations on the same input.
    pub fn compressed_size(&self) -> usize {
        let dict_bytes = *self.dict.offsets.last().unwrap_or(&0) as usize;
        let dict_offsets_bytes = self.dict.offsets.len() * size_of::<u32>();
        let codes_packed_bytes = if self.store.packed.is_empty() {
            0
        } else {
            // Exclude the trailing zero sentinel; consumers don't see it.
            (self.store.packed.len() - 1) * size_of::<u64>()
        };
        let codes_boundaries_bytes = self.store.boundaries.len() * size_of::<u32>();
        dict_bytes + dict_offsets_bytes + codes_packed_bytes + codes_boundaries_bytes
    }

    /// Try `configs` and return the column with the smallest
    /// `compressed_size()`. Errors out only when *every* config errors —
    /// otherwise individual failures are skipped.
    ///
    /// Useful for input where the best `(bits, threshold, seed)` is not
    /// known up-front: on URL-shaped data the right bit width can change
    /// compressed size by 8 % or more. Cost scales linearly with `configs.len()`.
    pub fn compress_search(
        bytes: &[u8],
        offsets: &[u64],
        configs: &[OnPairTrainingConfig],
    ) -> Result<Self, Error> {
        if configs.is_empty() {
            return Err(Error::InvalidArg);
        }
        let mut best: Option<(usize, Self)> = None;
        let mut last_err = Error::InvalidArg;
        for &cfg in configs {
            match Self::compress(bytes, offsets, cfg) {
                Ok(col) => {
                    let sz = col.compressed_size();
                    match &best {
                        Some((bs, _)) if *bs <= sz => {}
                        _ => best = Some((sz, col)),
                    }
                }
                Err(e) => last_err = e,
            }
        }
        best.map(|(_, c)| c).ok_or(last_err)
    }

    /// Compress with a small `(bit-width, threshold)` sweep and return the
    /// smallest result. Trades CPU for compression ratio — runs the trainer
    /// `bits.len() × thresholds.len()` times. The default sweep covers the
    /// configurations that dominate the Pareto frontier on real text/URL
    /// data.
    ///
    /// Verified-best configurations from sweeps over 9..=16 on real data:
    /// * synthetic URLs   → bits=10, threshold=0.5
    /// * TPCH `l_comment` → bits=11, threshold=1.0
    /// * ClickBench URL   → bits=14, threshold=1.0
    /// * Wikipedia text   → bits=16, threshold=0.5
    ///
    /// The sweep below covers all four with fourteen trainings (~10× the
    /// single-shot cost). For best-compression workloads only — single-shot
    /// `compress` is faster when the right configuration is known.
    pub fn compress_auto(bytes: &[u8], offsets: &[u64]) -> Result<Self, Error> {
        const BITS_SWEEP: &[u32] = &[10, 11, 12, 13, 14, 15, 16];
        const THR_SWEEP: &[f64] = &[0.5, 1.0];
        let mut cfgs: Vec<OnPairTrainingConfig> = Vec::with_capacity(
            BITS_SWEEP.len() * THR_SWEEP.len(),
        );
        for &b in BITS_SWEEP {
            for &t in THR_SWEEP {
                cfgs.push(OnPairTrainingConfig { bits: b, threshold: t, seed: 42 });
            }
        }
        Self::compress_search(bytes, offsets, &cfgs)
    }

    /// Borrow the column's raw arrays for downstream consumers (decode loop,
    /// predicate kernels). Mirrors `vortex-onpair-sys::Column::parts`.
    pub fn parts(&self) -> Result<Parts<'_>, Error> {
        // dict_bytes: logical-size slice, not including decoder padding. The
        // C++ shim returns the same thing — `dict_bytes_len` is the byte
        // count from offsets.back(), and `vortex-onpair`'s compress.rs adds
        // MAX_TOKEN_SIZE of trailing zero padding itself.
        let true_dict_bytes = *self.dict.offsets.last().unwrap_or(&0) as usize;
        // Skip the trailing zero sentinel `BitWriter::flush` appended.
        let codes_packed = if self.store.packed.is_empty() {
            &self.store.packed[..]
        } else {
            &self.store.packed[..self.store.packed.len() - 1]
        };
        Ok(Parts {
            dict_bytes: &self.dict.bytes[..true_dict_bytes],
            dict_offsets: &self.dict.offsets,
            codes_packed,
            codes_boundaries: &self.store.boundaries,
            bits: self.store.bit_width as u32,
            num_rows: self.num_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::unpack_codes_to_u16;
    use crate::config::DEFAULT_DICT12_CONFIG;
    use crate::types::Token;

    fn pack_inputs(strings: &[&str]) -> (Vec<u8>, Vec<u64>) {
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(strings.len() + 1);
        offsets.push(0u64);
        for s in strings {
            bytes.extend_from_slice(s.as_bytes());
            offsets.push(bytes.len() as u64);
        }
        (bytes, offsets)
    }

    fn decode_row(parts: &Parts<'_>, dict_bytes: &[u8], row: usize) -> Vec<u8> {
        let begin = parts.codes_boundaries[row] as usize;
        let end = parts.codes_boundaries[row + 1] as usize;
        let codes = unpack_codes_to_u16(parts.codes_packed, end, parts.bits);
        let mut out = Vec::new();
        for &c in &codes[begin..end] {
            let s = parts.dict_offsets[c as usize] as usize;
            let e = parts.dict_offsets[c as usize + 1] as usize;
            out.extend_from_slice(&dict_bytes[s..e]);
        }
        out
    }

    #[test]
    fn empty_offsets_returns_invalid_arg() {
        let r = Column::compress(&[], &[], DEFAULT_DICT12_CONFIG);
        assert_eq!(r.err(), Some(Error::InvalidArg));
    }

    #[test]
    fn invalid_bits_returns_invalid_arg() {
        let cfg = OnPairTrainingConfig { bits: 8, threshold: 0.5, seed: 0 };
        let r = Column::compress(&[], &[0], cfg);
        assert_eq!(r.err(), Some(Error::InvalidArg));
    }

    #[test]
    fn zero_rows_compress_succeeds() {
        let col = Column::compress(&[], &[0], DEFAULT_DICT12_CONFIG).unwrap();
        assert_eq!(col.len(), 0);
        let parts = col.parts().unwrap();
        assert_eq!(parts.num_rows, 0);
        assert_eq!(parts.codes_boundaries, &[0u32]);
        assert!(parts.codes_packed.is_empty());
    }

    #[test]
    fn roundtrip_simple_strings() {
        let strings = ["user_000001", "user_000002", "admin_001", "user_000003", "guest_001"];
        let (bytes, offsets) = pack_inputs(&strings);
        let cfg = OnPairTrainingConfig { bits: 12, threshold: 0.5, seed: 7 };
        let col = Column::compress(&bytes, &offsets, cfg).unwrap();
        assert_eq!(col.len(), 5);
        assert_eq!(col.bits(), 12);
        let parts = col.parts().unwrap();
        let dict_bytes_padded = {
            let mut v = parts.dict_bytes.to_vec();
            v.extend(std::iter::repeat_n(0u8, crate::MAX_TOKEN_SIZE));
            v
        };
        for (i, &s) in strings.iter().enumerate() {
            let decoded = decode_row(&parts, &dict_bytes_padded, i);
            assert_eq!(decoded, s.as_bytes(), "row {i}");
        }
    }

    #[test]
    fn roundtrip_with_binary_data_and_all_bit_widths() {
        let strings: Vec<Vec<u8>> = (0..30u8)
            .map(|i| {
                let mut v = Vec::with_capacity(20);
                for j in 0..20u8 {
                    v.push(i.wrapping_add(j));
                }
                v
            })
            .collect();
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(strings.len() + 1);
        offsets.push(0u64);
        for s in &strings {
            bytes.extend_from_slice(s);
            offsets.push(bytes.len() as u64);
        }
        for bw in 9u32..=16 {
            let cfg = OnPairTrainingConfig { bits: bw, threshold: 0.5, seed: 99 };
            let col = Column::compress(&bytes, &offsets, cfg).unwrap();
            assert_eq!(col.bits(), bw);
            let parts = col.parts().unwrap();
            let dict_bytes_padded = {
                let mut v = parts.dict_bytes.to_vec();
                v.extend(std::iter::repeat_n(0u8, crate::MAX_TOKEN_SIZE));
                v
            };
            for (i, s) in strings.iter().enumerate() {
                let decoded = decode_row(&parts, &dict_bytes_padded, i);
                assert_eq!(decoded, *s, "bits={bw} row={i}");
            }
        }
    }

    #[test]
    fn dict_first_256_tokens_cover_all_bytes_after_sort() {
        let strings = ["hello world", "another row"];
        let (bytes, offsets) = pack_inputs(&strings);
        let col = Column::compress(&bytes, &offsets, DEFAULT_DICT12_CONFIG).unwrap();
        let parts = col.parts().unwrap();
        // Every byte value 0..=255 must appear as a single-byte token somewhere
        // in the dictionary.
        let mut found = [false; 256];
        for i in 0..parts.dict_offsets.len() - 1 {
            let s = parts.dict_offsets[i] as usize;
            let e = parts.dict_offsets[i + 1] as usize;
            if e - s == 1 {
                found[parts.dict_bytes[s] as usize] = true;
            }
        }
        for (i, &f) in found.iter().enumerate() {
            assert!(f, "byte {i} missing");
        }
    }

    #[test]
    fn parts_codes_packed_excludes_sentinel() {
        let strings = ["x", "y", "z"];
        let (bytes, offsets) = pack_inputs(&strings);
        let col = Column::compress(&bytes, &offsets, DEFAULT_DICT12_CONFIG).unwrap();
        let parts = col.parts().unwrap();
        // Number of token bits actually used.
        let total_tokens = *parts.codes_boundaries.last().unwrap() as usize;
        let needed_words = (total_tokens * parts.bits as usize).div_ceil(64);
        assert_eq!(parts.codes_packed.len(), needed_words);
        // unpack should still yield exactly total_tokens valid codes.
        let codes = unpack_codes_to_u16(parts.codes_packed, total_tokens, parts.bits);
        assert_eq!(codes.len(), total_tokens);
        // All codes must be valid dict indices.
        let dict_size = col.dict_size() as Token;
        for &c in &codes {
            assert!(c < dict_size);
        }
    }
}
