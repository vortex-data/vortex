// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-parallel Shift-Or / Bitap matcher for short FSST-LIKE contains needles.
//!
//! ## Background
//!
//! For a needle of length `L` (1 ≤ L ≤ 8), classic Shift-Or maintains a
//! single `u64` state where bit `i` is 0 iff the most recent `i + 1` input
//! bytes match `needle[0..=i]`. Initialize `state = !0u64`. For each input
//! byte `b`, update `state = (state << 1) | B[b]`, where the 256-entry byte
//! mask `B` has bit `i` cleared at every `b` that matches needle position
//! `i`. The match condition is `state & (1 << (L - 1)) == 0`.
//!
//! ## FSST extension
//!
//! Vortex's FSST encoding produces a stream of codes 0..255, where codes
//! `0..n_symbols` expand to 1..8 decompressed bytes via the symbol table
//! and `255 == ESCAPE_CODE` consumes the next byte as a literal. To match
//! on the compressed stream we precompute one transition per symbol:
//!
//! - `shift_bits[c]`: number of state shifts (= symbol length, 1..8).
//! - `or_mask[c]`: `(B[b0] << (L_c-1)) | (B[b1] << (L_c-2)) | ... | B[b_{L_c-1}]`.
//! - `state_accept_mask[c]`: bitmask over input-state bits indicating which
//!   bits of `state`, if cleared on entry, would have triggered an accept
//!   during *some* intermediate position within the symbol's expansion.
//!
//! Per code, the update is `state = (state << shift_bits[c]) | or_mask[c]`.
//! Per code, the intermediate accept check is
//! `(!state) & state_accept_mask[c] != 0`. The final accept check is
//! `state & accept_bit == 0`, which is equivalent to the k=L_c-1 case of
//! the intermediate check when symbol length ≤ needle length.
//!
//! ## Why no SSA codes
//!
//! A "single-step accept" code is one whose decompressed expansion itself
//! contains the needle — `state` is irrelevant. Handling SSA correctly
//! requires an unconditional per-code accept flag plus a per-step
//! re-evaluation. We could support it, but the brief restricts this matcher
//! to the easy case: needles short enough that ShiftOr wins, AND no symbol
//! contains the needle outright. Bigger needles + SSA already have good
//! coverage via Teddy-2/3 + FoldedContainsDfa.
//!
//! ## Wildcards and case-insensitive
//!
//! - `_` at position `i` clears bit `i` of every `B[b]` (matches any byte).
//! - `ci`: for each ASCII letter byte in the needle, also clear the same bit
//!   for its case-flipped counterpart.

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::WILDCARD;
use super::ascii_to_lower;
use super::scan_to_bitbuf_with;

/// Maximum supported needle length for [`ShiftOrDfa`].
///
/// A state bit per needle position means the accept bit is `1 << (L - 1)`;
/// per-symbol-step composition can shift state by up to 8 (max symbol
/// length) per code, so we cap `L ≤ 8` to keep the shift composition
/// within `u64`.
pub(super) const MAX_NEEDLE_LEN: usize = 8;

/// Bit-parallel Shift-Or contains matcher over FSST-compressed codes.
///
/// Selected by [`super::FsstMatcher::try_new_with`] when the needle is
/// 1..=8 bytes, the pattern is `%needle%`, and no symbol's expansion
/// contains the needle. For everything else the matcher falls back to
/// [`super::folded_contains::FoldedContainsDfa`].
#[cfg_attr(any(test, feature = "_test-harness"), allow(unreachable_pub))]
pub struct ShiftOrDfa {
    /// Per-code `(or_mask, shift_bits, state_accept_mask)` triples, indexed
    /// by code byte 0..=255. ESCAPE_CODE entry is irrelevant — the scan
    /// handles ESCAPE inline. Unmapped codes (≥ n_symbols, ≠ ESCAPE_CODE)
    /// are filled with no-op transitions (shift=0, or_mask=!0, accept=0)
    /// to keep the inner loop branchless.
    code_table: Box<[CodeEntry; 256]>,
    /// Byte mask table over decompressed bytes — used by the ESCAPE
    /// branch (`code = ESCAPE_CODE`, next code byte is read as a literal).
    /// Bit `i` of `byte_mask[b]` is 0 iff input byte `b` matches needle
    /// position `i`.
    byte_mask: Box<[u64; 256]>,
    /// `1 << (needle_len - 1)`. State bit cleared on accept.
    accept_bit: u64,
    /// Set of state-0 progressing codes: codes that, when consumed from
    /// `state = !0u64`, clear at least one of the low `needle_len` bits
    /// of state. Empty when every potentially-matching byte is
    /// represented only by ESCAPE pairs (in which case ESCAPE_CODE
    /// itself is in the set). Used by [`Self::scan_to_bitbuf`] to drive
    /// an anchor-bitset prefilter — without it, the scan reverts to a
    /// per-byte inner loop and regresses 30× on sparse-needle workloads
    /// vs the [`super::folded_contains::FoldedContainsDfa`] anchor-scan
    /// path.
    progressing_codes: Option<Vec<u8>>,
}

#[derive(Copy, Clone)]
struct CodeEntry {
    /// State left-shift in bits (= symbol length for normal codes, 0 for
    /// unmapped codes, 0 for ESCAPE_CODE since the ESCAPE path is handled
    /// inline by the scan).
    shift_bits: u8,
    /// `or_mask` applied after the shift.
    or_mask: u64,
    /// Bitmask over the input state: bit `j` is set iff input state having
    /// bit `j` clear could trigger an intermediate accept anywhere within
    /// this symbol's expansion.
    state_accept_mask: u64,
}

impl ShiftOrDfa {
    /// Build a Shift-Or matcher for `needle`.
    ///
    /// Returns `Err` if:
    /// - `needle.is_empty()` (caller should handle MatchAll separately).
    /// - `needle.len() > MAX_NEEDLE_LEN`.
    /// - Any symbol's expansion contains the needle as a substring (SSA).
    ///   In that case the caller should fall back to FoldedContainsDfa,
    ///   which has its own SSA handling.
    pub(super) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
        case_insensitive: bool,
    ) -> VortexResult<Self> {
        if needle.is_empty() {
            vortex_bail!("ShiftOrDfa: empty needle is not supported (use MatchAll)");
        }
        if needle.len() > MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds ShiftOrDfa max {}",
                needle.len(),
                MAX_NEEDLE_LEN
            );
        }
        debug_assert!(symbol_lengths.len() >= symbols.len());

        let needle_len = needle.len();
        let accept_bit = 1u64 << (needle_len - 1);

        // Build the decompressed-byte mask `B[b]`. Bit `i` is 0 iff byte
        // `b` matches needle position `i`.
        let mut byte_mask = [!0u64; 256];
        for (i, &nb) in needle.iter().enumerate() {
            let clear_bit = 1u64 << i;
            if nb == WILDCARD {
                // Wildcard: clear bit i in every byte mask entry.
                for slot in byte_mask.iter_mut() {
                    *slot &= !clear_bit;
                }
            } else if case_insensitive && nb.is_ascii_alphabetic() {
                let lo = ascii_to_lower(nb);
                let hi = lo ^ 0x20;
                byte_mask[usize::from(lo)] &= !clear_bit;
                byte_mask[usize::from(hi)] &= !clear_bit;
            } else {
                byte_mask[usize::from(nb)] &= !clear_bit;
            }
        }

        // Reject when any symbol's expansion contains the needle outright
        // (SSA): the symbol-step composition cannot represent the
        // unconditional accept correctly without a per-code SSA flag, and
        // the brief restricts ShiftOr to the no-SSA regime.
        for (sym, &len) in symbols.iter().zip(symbol_lengths.iter()) {
            let bytes = sym.to_u64().to_le_bytes();
            let sym_len = usize::from(len).min(8);
            if sym_len < needle_len {
                continue;
            }
            if symbol_expansion_contains_needle(
                &bytes[..sym_len],
                needle,
                case_insensitive,
                &byte_mask,
                needle_len,
            ) {
                vortex_bail!("ShiftOrDfa: needle is contained in symbol expansion (SSA)");
            }
        }

        // Build per-symbol `(or_mask, shift_bits, state_accept_mask)`.
        let no_op = CodeEntry {
            shift_bits: 0,
            or_mask: !0u64,
            state_accept_mask: 0,
        };
        let mut code_table = Box::new([no_op; 256]);

        for (code, (sym, &len)) in symbols.iter().zip(symbol_lengths.iter()).enumerate() {
            if code == usize::from(ESCAPE_CODE) {
                // Defensive: shouldn't occur — FSST reserves 255. Keep
                // the no-op so we don't corrupt the ESCAPE inline path.
                continue;
            }
            let bytes = sym.to_u64().to_le_bytes();
            let sym_len = usize::from(len).min(8);
            // shift_bits ≤ 8 by construction (max symbol length).
            let shift_bits = u8::try_from(sym_len).vortex_expect("sym_len ≤ 8 fits in u8");
            // or_mask = (B[b0] << (sym_len-1)) | ... | B[b_{sym_len-1}].
            // Shifts ≥ 64 are UB; clamp by skipping bytes whose shift
            // would exceed 63.
            let mut or_mask = 0u64;
            for (i, &b) in bytes[..sym_len].iter().enumerate() {
                let shift = sym_len - 1 - i;
                if shift < 64 {
                    or_mask |= byte_mask[usize::from(b)] << shift;
                }
            }
            // state_accept_mask: bit `j` is set iff there is some step k
            // in 0..sym_len with j == needle_len - 1 - (k+1) + k_internal
            // ... derived in the module doc-comment. We compute it
            // empirically by simulating the symbol on every possible
            // input-state bit.
            let state_accept_mask =
                compute_state_accept_mask(&bytes[..sym_len], &byte_mask, needle_len, accept_bit);
            code_table[code] = CodeEntry {
                shift_bits,
                or_mask,
                state_accept_mask,
            };
        }
        // ESCAPE_CODE entry: keep no-op. The scan path handles ESCAPE
        // inline by reading the next code byte and applying `byte_mask`
        // directly.

        // Collect state-0 progressing codes: codes whose consumption
        // from `state = !0u64` clears at least one of the low
        // `needle_len` bits. Plus ESCAPE_CODE, since any escape pair
        // whose literal byte matches needle[0] would progress.
        let progress_mask: u64 = (1u64 << needle_len) - 1;
        let mut progressing: Vec<u8> = Vec::new();
        for (c, entry) in code_table.iter().enumerate() {
            if c == usize::from(ESCAPE_CODE) {
                continue;
            }
            // After consuming code c from state !0u64:
            //   new_state = (!0u64 << shift) | or_mask
            // Bit i of new_state is 0 iff (shift > i ? bit i of or_mask : 1) is 0,
            // i.e. only if shift > i AND bit i of or_mask is 0.
            let shift = u32::from(entry.shift_bits);
            if shift == 0 {
                continue;
            }
            let upper_ones: u64 = if shift >= 64 { 0 } else { !0u64 << shift };
            let new_state = upper_ones | entry.or_mask;
            if (new_state & progress_mask) != progress_mask {
                progressing.push(u8::try_from(c).vortex_expect("c < 256"));
            }
        }
        // ESCAPE pairs can progress whenever any literal byte clears bit
        // 0 of state. Equivalently, ESCAPE_CODE belongs to the
        // progressing set whenever at least one byte b has bit 0 of
        // byte_mask[b] cleared (i.e. matches needle[0]).
        let escape_progresses = byte_mask.iter().any(|m| m & 1 == 0);
        if escape_progresses {
            progressing.push(ESCAPE_CODE);
        }
        let progressing_codes = if progressing.is_empty() {
            None
        } else {
            Some(progressing)
        };

        Ok(Self {
            code_table,
            byte_mask: Box::new(byte_mask),
            accept_bit,
            progressing_codes,
        })
    }

    /// Run the matcher over a single FSST-compressed code sequence.
    /// Returns `true` iff the needle appears anywhere in the decompressed
    /// expansion.
    #[inline(always)]
    pub(super) fn matches(&self, codes: &[u8]) -> bool {
        let accept_bit = self.accept_bit;
        let byte_mask = &self.byte_mask;
        let code_table = &self.code_table;

        let mut state: u64 = !0u64;
        let mut i = 0usize;
        let len = codes.len();
        while i < len {
            // SAFETY: i < len ≤ codes.len().
            let c = unsafe { *codes.get_unchecked(i) };
            i += 1;
            if c == ESCAPE_CODE {
                if i >= len {
                    // Trailing ESCAPE with no literal — no accept possible
                    // from this position.
                    return false;
                }
                // SAFETY: i < len.
                let literal = unsafe { *codes.get_unchecked(i) };
                i += 1;
                state = (state << 1) | byte_mask[usize::from(literal)];
                if state & accept_bit == 0 {
                    return true;
                }
            } else {
                // SAFETY: c is a u8, code_table has 256 entries.
                let entry = unsafe { code_table.get_unchecked(usize::from(c)) };
                // Intermediate-accept check: input state bits that, if
                // clear, would have triggered an accept inside this
                // symbol's expansion.
                if (!state) & entry.state_accept_mask != 0 {
                    return true;
                }
                let shift = u32::from(entry.shift_bits);
                // shift is 0..=8 by construction.
                state = if shift >= 64 {
                    entry.or_mask
                } else {
                    (state << shift) | entry.or_mask
                };
                if state & accept_bit == 0 {
                    return true;
                }
            }
        }
        false
    }

    /// Specialized scan over `n` strings, returning a `BitBuffer` of accept
    /// results (XOR `negated`). Mirrors the API of
    /// [`super::folded_contains::FoldedContainsDfa::scan_to_bitbuf`] so the
    /// matcher enum can dispatch uniformly.
    ///
    /// When a state-0 progressing-code set is available, a single AVX2
    /// PSHUFB-Mula pass over `all_bytes` builds a candidate-position
    /// bitset; rows with no candidate bytes return `false` after one
    /// `u64` load, and rows with candidates skip non-candidate bytes
    /// via `tzcnt`. This matches the FoldedContainsDfa anchor-scan
    /// throughput and avoids the 30× regression observed when the
    /// fallback row-by-row loop scans every byte.
    #[inline]
    pub(super) fn scan_to_bitbuf<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        if let Some(codes) = self.progressing_codes.as_deref() {
            let bitset = super::anchor_scan::build_progressing_bitset_unbounded(all_bytes, codes);
            return self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
        }
        scan_to_bitbuf_with(n, offsets, all_bytes, negated, |codes| self.matches(codes))
    }

    /// Driven by a precomputed progressing-code bitset over `all_bytes`,
    /// evaluate `matches_with_bitset` per row and collect into a
    /// `BitBuffer`. Mirrors
    /// [`super::folded_contains::FoldedContainsDfa::scan_with_anchor_bitset`].
    #[inline]
    fn scan_with_anchor_bitset<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        bitset: &[u64],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        debug_assert!(offsets.len() > n);
        // SAFETY: caller guarantees `offsets.len() > n`.
        let mut start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
        BitBuffer::collect_bool(n, |i| {
            // SAFETY: `i < n` and `offsets.len() >= n + 1`.
            let end: usize = unsafe { *offsets.get_unchecked(i + 1) }.as_();
            debug_assert!(start <= end && end <= all_bytes.len());
            let result = self.matches_with_bitset(all_bytes, bitset, start, end) != negated;
            start = end;
            result
        })
    }

    /// Variant of [`Self::matches`] that uses a precomputed
    /// progressing-code bitset over `all_bytes` to fast-skip
    /// non-candidate code positions. When the bitset is fully unset for
    /// a row's range, this returns `false` after a single `u64` load.
    /// Otherwise it lands on each candidate via `tzcnt` and runs the
    /// shift-or inner loop until state collapses back to "no match in
    /// progress" (all low bits of state set), at which point we resume
    /// the next `tzcnt` jump.
    #[inline]
    fn matches_with_bitset(
        &self,
        all_bytes: &[u8],
        bitset: &[u64],
        abs_start: usize,
        abs_end: usize,
    ) -> bool {
        let accept_bit = self.accept_bit;
        let byte_mask = &self.byte_mask;
        let code_table = &self.code_table;
        // The "match in progress" mask covers the low `needle_len` bits;
        // when state ANDed with this equals the mask, no progress has
        // been made and we can resume the anchor-bitset skip.
        let progress_mask: u64 = ((accept_bit << 1).wrapping_sub(1)) | accept_bit;

        let mut pos = abs_start;
        loop {
            match super::anchor_scan::next_set_in_range(bitset, pos, abs_end) {
                Some(p) => pos = p,
                None => return false,
            }
            // Run the standard inner loop starting at `pos`, restarting
            // from the "fresh" state. Any progressing-code bitset
            // position is a valid restart point because the matcher is
            // search-anywhere.
            let mut state: u64 = !0u64;
            while pos < abs_end {
                // SAFETY: pos < abs_end ≤ all_bytes.len().
                let c = unsafe { *all_bytes.get_unchecked(pos) };
                pos += 1;
                if c == ESCAPE_CODE {
                    if pos >= abs_end {
                        return false;
                    }
                    // SAFETY: pos < abs_end.
                    let literal = unsafe { *all_bytes.get_unchecked(pos) };
                    pos += 1;
                    state = (state << 1) | byte_mask[usize::from(literal)];
                    if state & accept_bit == 0 {
                        return true;
                    }
                } else {
                    // SAFETY: c is a u8.
                    let entry = unsafe { code_table.get_unchecked(usize::from(c)) };
                    if (!state) & entry.state_accept_mask != 0 {
                        return true;
                    }
                    let shift = u32::from(entry.shift_bits);
                    state = if shift >= 64 {
                        entry.or_mask
                    } else {
                        (state << shift) | entry.or_mask
                    };
                    if state & accept_bit == 0 {
                        return true;
                    }
                }
                // If no match is in progress (all low bits set), resume
                // the outer anchor-bitset skip from `pos`.
                if (state & progress_mask) == progress_mask {
                    break;
                }
            }
        }
    }
}

/// Compute the per-symbol `state_accept_mask`.
///
/// For each input-state bit `j` (0 ≤ j < 64), determine whether
/// `state = (!0u64) & !(1 << j)` (a state with bit `j` clear and all
/// others set) would, when fed through the symbol's expansion byte-by-byte
/// using `byte_mask`, cause the accept bit to be cleared at any
/// intermediate position.
///
/// Returns a 64-bit mask: bit `j` set iff such a state triggers
/// intermediate acceptance during this symbol.
fn compute_state_accept_mask(
    sym_bytes: &[u8],
    byte_mask: &[u64; 256],
    needle_len: usize,
    accept_bit: u64,
) -> u64 {
    let mut mask = 0u64;
    debug_assert!((1..=8).contains(&needle_len));
    // We only need to consider `j` positions that could ever be relevant:
    // `j < needle_len`. Beyond `needle_len`, clearing a high state bit
    // can only shift OUT of the accept window during the symbol — but
    // actually j can still matter if it ends up reaching accept_bit
    // through the shift. We iterate j over a safe range, capped at
    // `needle_len + 8` to cover symbol-length-8 transitions.
    let probe_range = (needle_len + sym_bytes.len()).min(64);
    for j in 0..probe_range {
        // Probe state: all 1s except bit j cleared.
        let mut state = !0u64;
        state &= !(1u64 << j);
        let mut hit = false;
        for &b in sym_bytes {
            state = (state << 1) | byte_mask[usize::from(b)];
            if state & accept_bit == 0 {
                hit = true;
                break;
            }
        }
        if hit && j < 64 {
            mask |= 1u64 << j;
        }
    }
    // The "all 1s" state should never accept (no bytes matched yet
    // means no state bits are clear). If the symbol alone accepts
    // (i.e. SSA), the caller would have rejected. Defensive check:
    // verify the all-1s state path doesn't produce an unconditional
    // accept. If it does, the result here is incorrect, but the
    // `new()` SSA pre-check should have caught it.
    #[cfg(debug_assertions)]
    debug_assert!(!unconditional_accept_under_symbol(
        sym_bytes, byte_mask, accept_bit
    ));
    mask
}

/// Returns `true` iff feeding `sym_bytes` through `byte_mask` starting
/// from the all-1s state clears the accept bit (i.e. the symbol's
/// expansion alone contains the needle — SSA). Pure function — safe to
/// call from `debug_assert!`.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
fn unconditional_accept_under_symbol(
    sym_bytes: &[u8],
    byte_mask: &[u64; 256],
    accept_bit: u64,
) -> bool {
    let mut s: u64 = !0u64;
    for &b in sym_bytes {
        s = (s << 1) | byte_mask[usize::from(b)];
        if s & accept_bit == 0 {
            return true;
        }
    }
    false
}

/// Check whether `sym_bytes` (an FSST symbol's decompressed expansion)
/// contains `needle` as a substring under the same matching semantics
/// (wildcard via `_`, case-insensitive folding).
///
/// Used to detect SSA codes during `ShiftOrDfa::new` so the caller can
/// fall back to FoldedContainsDfa cleanly.
fn symbol_expansion_contains_needle(
    sym_bytes: &[u8],
    needle: &[u8],
    case_insensitive: bool,
    byte_mask: &[u64; 256],
    needle_len: usize,
) -> bool {
    let _ = needle;
    let _ = case_insensitive;
    if sym_bytes.len() < needle_len {
        return false;
    }
    // Reuse the shift-or check: feed `sym_bytes` through `byte_mask`
    // starting from `state = !0`. If the accept bit is ever clear, the
    // expansion contains the needle.
    let mut state: u64 = !0u64;
    let accept_bit = 1u64 << (needle_len - 1);
    for &b in sym_bytes {
        state = (state << 1) | byte_mask[usize::from(b)];
        if state & accept_bit == 0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::dfa::folded_contains::FoldedContainsDfa;

    /// Helper: make a Symbol from a byte string (≤ 8 bytes, zero-padded).
    fn sym(bytes: &[u8]) -> Symbol {
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        Symbol::from_slice(&buf)
    }

    /// Helper: encode `bytes` as a sequence of `[ESCAPE_CODE, b]` pairs.
    fn escaped(bytes: &[u8]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(bytes.len() * 2);
        for &b in bytes {
            codes.push(ESCAPE_CODE);
            codes.push(b);
        }
        codes
    }

    /// 1-byte needle without symbols: every needle byte must appear.
    #[test]
    fn matches_single_byte_needle_no_symbols() -> VortexResult<()> {
        let dfa = ShiftOrDfa::new(&[], &[], b"x", false)?;
        assert!(dfa.matches(&escaped(b"x")));
        assert!(dfa.matches(&escaped(b"abxcd")));
        assert!(!dfa.matches(&escaped(b"abcd")));
        assert!(!dfa.matches(&[]));
        Ok(())
    }

    /// 2-byte needle: match must be contiguous in the decompressed stream.
    #[test]
    fn matches_two_byte_needle_no_symbols() -> VortexResult<()> {
        let dfa = ShiftOrDfa::new(&[], &[], b"ab", false)?;
        assert!(dfa.matches(&escaped(b"ab")));
        assert!(dfa.matches(&escaped(b"xxabxx")));
        assert!(!dfa.matches(&escaped(b"a")));
        assert!(!dfa.matches(&escaped(b"b")));
        // Crossing a non-needle byte breaks the match.
        assert!(!dfa.matches(&escaped(b"axb")));
        Ok(())
    }

    /// 8-byte needle (the maximum length).
    #[test]
    fn matches_eight_byte_needle_no_symbols() -> VortexResult<()> {
        let dfa = ShiftOrDfa::new(&[], &[], b"abcdefgh", false)?;
        assert!(dfa.matches(&escaped(b"abcdefgh")));
        assert!(dfa.matches(&escaped(b"___abcdefgh___")));
        assert!(!dfa.matches(&escaped(b"abcdefg")));
        assert!(!dfa.matches(&escaped(b"abcdxfgh")));
        Ok(())
    }

    /// Wildcard `_` matches any single byte at that position.
    #[test]
    fn matches_with_wildcard() -> VortexResult<()> {
        let dfa = ShiftOrDfa::new(&[], &[], b"a_c", false)?;
        assert!(dfa.matches(&escaped(b"abc")));
        assert!(dfa.matches(&escaped(b"aXc")));
        assert!(dfa.matches(&escaped(b"xxa1cxx")));
        assert!(!dfa.matches(&escaped(b"ab")));
        assert!(!dfa.matches(&escaped(b"ac")));
        Ok(())
    }

    /// Case-insensitive ASCII matching.
    #[test]
    fn matches_case_insensitive() -> VortexResult<()> {
        let dfa = ShiftOrDfa::new(&[], &[], b"Ab", true)?;
        assert!(dfa.matches(&escaped(b"AB")));
        assert!(dfa.matches(&escaped(b"ab")));
        assert!(dfa.matches(&escaped(b"aB")));
        assert!(dfa.matches(&escaped(b"xxxabyyy")));
        assert!(!dfa.matches(&escaped(b"acab")[..4]));
        Ok(())
    }

    /// FSST symbols: the matcher composes correctly through multi-byte
    /// symbol expansions.
    #[test]
    fn matches_with_fsst_symbols() -> VortexResult<()> {
        // Symbol 0 = "abc" (3 bytes), symbol 1 = "xy" (2 bytes), symbol 2 = "d" (1 byte).
        let symbols = [sym(b"abc"), sym(b"xy"), sym(b"d")];
        let lengths = [3u8, 2, 1];

        // Needle "bcd" — symbol 0 = "abc" ends in "bc", symbol 2 = "d"
        // starts with "d". A two-code sequence [0, 2] decompresses to
        // "abcd" which contains "bcd".
        let dfa = ShiftOrDfa::new(&symbols, &lengths, b"bcd", false)?;
        assert!(dfa.matches(&[0u8, 2]));
        // Just symbol 0 = "abc": no "bcd".
        assert!(!dfa.matches(&[0u8]));
        // [0, 1, 2] = "abc" + "xy" + "d" = "abcxyd": no "bcd".
        assert!(!dfa.matches(&[0u8, 1, 2]));

        Ok(())
    }

    /// Constructor rejects when a symbol's expansion contains the needle
    /// (SSA), so the caller can fall back to FoldedContainsDfa.
    #[test]
    fn rejects_when_symbol_contains_needle() {
        // Symbol "abc" contains needle "bc".
        let symbols = [sym(b"abc")];
        let lengths = [3u8];
        assert!(ShiftOrDfa::new(&symbols, &lengths, b"bc", false).is_err());
    }

    /// Constructor rejects empty needles and oversized needles.
    #[test]
    fn rejects_bad_lengths() {
        assert!(ShiftOrDfa::new(&[], &[], b"", false).is_err());
        assert!(ShiftOrDfa::new(&[], &[], b"abcdefghi", false).is_err());
    }

    /// Property test: on random short needles + random code streams,
    /// ShiftOrDfa agrees with FoldedContainsDfa byte-for-byte.
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    #[case(8)]
    fn agrees_with_folded_contains(#[case] needle_len: usize) -> VortexResult<()> {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::prelude::StdRng;

        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64.wrapping_add(needle_len as u64));

        // Build a small symbol table that does NOT contain the needle.
        let symbols = [sym(b"xy"), sym(b"zz"), sym(b"q")];
        let lengths = [2u8, 2, 1];

        // Pool of pattern bytes — wildcards are intentionally rare so most
        // needles exercise the literal-byte path.
        const POOL: &[u8] = b"abcdef";

        for _ in 0..32 {
            let needle: Vec<u8> = (0..needle_len)
                .map(|_| POOL[rng.random_range(0..POOL.len())])
                .collect();
            // Skip if any symbol expansion contains the needle (SSA);
            // ShiftOrDfa rejects in that regime and the matcher falls
            // back to FoldedContainsDfa.
            let folded = FoldedContainsDfa::new(&symbols, &lengths, &needle, false)?;
            let shift_or = match ShiftOrDfa::new(&symbols, &lengths, &needle, false) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Generate 64 random compressed code streams. Codes 0..=2 are
            // valid symbols; ESCAPE (255) consumes a literal byte.
            for _ in 0..64 {
                let len = rng.random_range(0..32usize);
                let mut codes = Vec::with_capacity(len);
                while codes.len() < len {
                    let r = rng.random_range(0..6u32);
                    match r {
                        0..=2 => codes.push(u8::try_from(r).expect("r < 6")),
                        3 if codes.len() + 1 < len => {
                            codes.push(ESCAPE_CODE);
                            codes.push(POOL[rng.random_range(0..POOL.len())]);
                        }
                        _ => codes.push(u8::try_from(rng.random_range(0..3u32)).expect("< 3")),
                    }
                }

                let want = folded.matches(&codes);
                let got = shift_or.matches(&codes);
                assert_eq!(
                    got,
                    want,
                    "needle={:?} codes={:?}",
                    String::from_utf8_lossy(&needle),
                    codes
                );
            }
        }

        Ok(())
    }

    /// Reuse the property scaffolding with case-insensitive matching.
    #[test]
    fn agrees_with_folded_contains_case_insensitive() -> VortexResult<()> {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::prelude::StdRng;

        let symbols = [sym(b"xY"), sym(b"q")];
        let lengths = [2u8, 1];

        let needles: &[&[u8]] = &[b"AB", b"ab", b"AbC", b"x"];
        for needle in needles {
            let folded = FoldedContainsDfa::new(&symbols, &lengths, needle, true)?;
            let Ok(shift_or) = ShiftOrDfa::new(&symbols, &lengths, needle, true) else {
                continue;
            };
            let mut rng = StdRng::seed_from_u64(0xBABE);
            const POOL: &[u8] = b"AaBbCcDd";
            for _ in 0..64 {
                let len = rng.random_range(0..24usize);
                let mut codes = Vec::with_capacity(len);
                while codes.len() < len {
                    let r = rng.random_range(0..5u32);
                    match r {
                        0..=1 => codes.push(u8::try_from(r).expect("r < 5")),
                        2 if codes.len() + 1 < len => {
                            codes.push(ESCAPE_CODE);
                            codes.push(POOL[rng.random_range(0..POOL.len())]);
                        }
                        _ => codes.push(u8::try_from(rng.random_range(0..2u32)).expect("< 2")),
                    }
                }
                let want = folded.matches(&codes);
                let got = shift_or.matches(&codes);
                assert_eq!(
                    got,
                    want,
                    "ci: needle={:?} codes={:?}",
                    String::from_utf8_lossy(needle),
                    codes
                );
            }
        }
        Ok(())
    }
}
