// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Suffix matching (`LIKE '%suffix'`) on FSST-compressed strings via a
//! backward-scanning DFA.
//!
//! The DFA processes codes from the END of each compressed string. States
//! track how many suffix bytes have been confirmed from the right. A single
//! table lookup per code gives the next state, enabling early-exit on mismatch
//! just like the prefix DFA exits on fail.
//!
//! ```text
//! Suffix: "bar" (states: 0=nothing matched, 1=matched "r", 2=matched "ar", 3=ACCEPT)
//!
//! Scanning backward: last code first.
//! If last symbol expands to "bar" → state goes 0 → 3 (accept).
//! If last symbol expands to "ar"  → state goes 0 → 2. Next code must end with "b".
//! If last symbol expands to "x"   → state goes 0 → FAIL.
//! ```
//!
//! ## Escape handling
//!
//! Uses the same sentinel strategy as prefix/contains: ESCAPE_CODE in the
//! transition table maps to a sentinel, triggering a byte-level lookup.
//! Since we scan backward, we detect escapes by checking `codes[pos-2] == 255`.

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::scan_to_bitbuf_with;

/// Backward-scanning DFA for suffix matching on FSST codes.
///
/// States 0..suffix_len track confirmed suffix bytes from the right.
/// State suffix_len is ACCEPT. A FAIL state enables early exit.
/// The DFA is scanned from the end of the code stream toward the beginning.
pub(crate) struct SuffixMatcher {
    /// `transitions[state * 256 + code]` -> next state (scanning backward).
    transitions: Vec<u8>,
    /// `escape_transitions[state * 256 + byte]` -> next state for escaped literal bytes.
    escape_transitions: Vec<u8>,
    accept_state: u8,
    fail_state: u8,
    sentinel: u8,
}

impl SuffixMatcher {
    pub(crate) const MAX_SUFFIX_LEN: usize = (u8::MAX - 2) as usize; // same as prefix

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        suffix: &[u8],
        case_insensitive: bool,
    ) -> VortexResult<Self> {
        if suffix.len() > Self::MAX_SUFFIX_LEN {
            vortex_bail!(
                "suffix length {} exceeds maximum {} for suffix DFA",
                suffix.len(),
                Self::MAX_SUFFIX_LEN
            );
        }

        let accept_state = u8::try_from(suffix.len()).vortex_expect("suffix fits in u8");
        let fail_state = accept_state + 1;
        let n_states = fail_state + 1;
        let sentinel = fail_state + 1;

        // Build byte-level backward transition table.
        // State `s` means "we have confirmed suffix[suf_len-s..suf_len]".
        // Processing a byte from the right: if it matches suffix[suf_len-s-1], advance.
        // Otherwise, fail (no KMP fallback needed for suffix — a mismatch from the
        // end means the string doesn't end with the suffix).
        //
        // Wait — this is wrong for multi-byte symbols. A symbol might partially match.
        // For example, suffix "bar", symbol "foobar". Scanning backward, the symbol
        // decodes to "foobar". From state 0, we process bytes right-to-left: 'r' matches
        // suffix[2], 'a' matches suffix[1], 'b' matches suffix[0] → accept.
        // That's correct. But what about partial symbols that span the suffix boundary?
        //
        // Actually for the byte-level table, we process ONE byte at a time from the right.
        // State s means s bytes confirmed. For byte b at the next position from the right:
        // - If s < suf_len and b == suffix[suf_len - 1 - s] → advance to s+1
        // - Otherwise → FAIL (the string doesn't end with our suffix)
        let byte_table =
            build_suffix_byte_table(suffix, accept_state, fail_state, case_insensitive);

        // Build symbol-level transitions: for each (state, symbol), simulate feeding
        // the symbol's bytes through the byte table IN REVERSE ORDER (since we're
        // scanning the code stream backward, and each symbol's bytes decode left-to-right,
        // but we encounter them right-to-left).
        let n_symbols = symbols.len();
        let mut sym_trans = vec![0u8; n_states as usize * n_symbols];
        for state in 0..n_states {
            for code in 0..n_symbols {
                if state == accept_state {
                    sym_trans[state as usize * n_symbols + code] = accept_state;
                    continue;
                }
                if state == fail_state {
                    sym_trans[state as usize * n_symbols + code] = fail_state;
                    continue;
                }
                let sym_bytes = symbols[code].to_u64().to_le_bytes();
                let sym_len = usize::from(symbol_lengths[code]);
                // Process bytes right-to-left (backward within the symbol)
                let mut s = state;
                for i in (0..sym_len).rev() {
                    if s == fail_state || s == accept_state {
                        break;
                    }
                    s = byte_table[s as usize * 256 + usize::from(sym_bytes[i])];
                }
                sym_trans[state as usize * n_symbols + code] = s;
            }
        }

        // Build fused 256-wide table (same layout as prefix/contains)
        let mut fused = vec![fail_state; usize::from(n_states) * 256];
        for state in 0..n_states {
            let s = usize::from(state);
            for code in 0..n_symbols {
                fused[s * 256 + code] = sym_trans[s * n_symbols + code];
            }
            fused[s * 256 + usize::from(ESCAPE_CODE)] = sentinel;
        }

        Ok(Self {
            transitions: fused,
            escape_transitions: byte_table,
            accept_state,
            fail_state,
            sentinel,
        })
    }

    /// Check if codes decode to a string ending with the suffix.
    /// Scans codes from the END toward the beginning.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = codes.len();

        while pos > 0 {
            // Detect escape: if codes[pos-2] == ESCAPE_CODE, then codes[pos-1] is a literal
            if pos >= 2 && codes[pos - 2] == ESCAPE_CODE {
                let b = codes[pos - 1];
                state = self.escape_transitions[usize::from(state) * 256 + usize::from(b)];
                pos -= 2;
            } else {
                let code = codes[pos - 1];
                let next = self.transitions[usize::from(state) * 256 + usize::from(code)];
                if next == self.sentinel {
                    // This shouldn't happen in backward scan (escapes detected above),
                    // but handle gracefully.
                    return false;
                }
                state = next;
                pos -= 1;
            }

            if state == self.accept_state {
                return true;
            }
            if state == self.fail_state {
                return false;
            }
        }

        state == self.accept_state
    }

    /// Specialized scan over `n` strings, returning a `BitBuffer` of accept
    /// results (XOR `negated`). The `matches` body is monomorphized into the
    /// bit-packing loop.
    #[inline]
    pub(crate) fn scan_to_bitbuf<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        scan_to_bitbuf_with(n, offsets, all_bytes, negated, |codes| self.matches(codes))
    }
}

/// Build a byte-level backward transition table for suffix matching.
///
/// State `s` means "confirmed `s` bytes from the right end of the suffix".
/// For each state, only the correct next byte (going leftward in the suffix)
/// advances; everything else goes to fail.
fn build_suffix_byte_table(
    suffix: &[u8],
    accept_state: u8,
    fail_state: u8,
    case_insensitive: bool,
) -> Vec<u8> {
    let n_states = fail_state + 1;
    let suf_len = suffix.len();
    let mut table = vec![fail_state; usize::from(n_states) * 256];

    for state in 0..n_states {
        let s = usize::from(state);
        if state == accept_state {
            // Accept is sticky (once confirmed, stays confirmed)
            for byte in 0..256 {
                table[s * 256 + byte] = accept_state;
            }
        } else if state != fail_state {
            // State s: confirmed s bytes from the right. Next byte must be
            // suffix[suf_len - 1 - s] to advance — or any byte if that
            // pattern position is the `_` wildcard.
            let expected = suffix[suf_len - 1 - s];
            let next_state = if s + 1 >= suf_len {
                accept_state
            } else {
                state + 1
            };
            if expected == super::WILDCARD {
                for byte in 0..256 {
                    table[s * 256 + byte] = next_state;
                }
            } else {
                super::set_advance(&mut table, s * 256, expected, next_state, case_insensitive);
            }
        }
        // fail_state stays fail for all bytes (default)
    }
    table
}
