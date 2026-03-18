// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DFA for prefix matching (`LIKE 'prefix%'`).
//!
//! TODO(joe): support longer prefixes (14–253 bytes) via a flat `Vec<u8>` table
//! with escape sentinel, similar to `FlatContainsDfa`. The construction is simpler
//! than contains (no KMP — mismatches go to a sticky fail state). Would need states
//! 0..N (progress) + accept + fail + sentinel, so N+3 ≤ 256 → max prefix = 253.

use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::pack_shift_table;
use super::shift_extract;

/// Precomputed shift-based DFA for prefix matching on FSST codes.
///
/// States 0..prefix_len track match progress, plus ACCEPT and FAIL.
/// Uses the same shift-based approach as the contains DFA: all state
/// transitions packed into a `u64` per code byte. For prefixes longer
/// than 13 characters, pushdown is disabled and LIKE falls back.
///
/// ```text
/// Prefix: "http"  (4 progress states + accept + fail)
///
///          Input byte
/// State    'h'    't'    'p'    other
/// ─────    ────   ────   ────   ─────
///   0       1      F      F      F     ← want 'h'
///   1       F      2      F      F     ← want 't'
///   2       F      3      F      F     ← want 't'
///   3       F      F      4✓     F     ← want 'p'
///   4✓      4✓     4✓     4✓     4✓    ← accept (sticky)
///   F       F      F      F      F     ← fail (sticky)
///
/// Escape handling: code 255 → sentinel → read next literal byte → byte table
/// ```
pub(crate) struct FsstPrefixDfa {
    /// Packed transitions: `(table[code] >> (state * 4)) & 0xF` gives next state.
    transitions: [u64; 256],
    /// Packed escape transitions for literal bytes.
    escape_transitions: [u64; 256],
    accept_state: u8,
    fail_state: u8,
}

impl FsstPrefixDfa {
    pub(crate) const BITS: u32 = 4;
    pub(crate) const MAX_PREFIX_LEN: usize = (1 << Self::BITS) as usize - 3;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        prefix: &[u8],
    ) -> VortexResult<Self> {
        if prefix.len() > Self::MAX_PREFIX_LEN {
            vortex_bail!(
                "prefix length {} exceeds maximum {} for shift-packed prefix DFA",
                prefix.len(),
                Self::MAX_PREFIX_LEN
            );
        }

        let accept_state = u8::try_from(prefix.len()).vortex_expect("prefix fits in u8");
        let fail_state = accept_state + 1;
        let n_states = fail_state + 1;

        // Prefix matching uses a simpler transition rule than KMP: on mismatch
        // we go to fail_state (no fallback). Build the byte table inline.
        let byte_table = Self::build_prefix_byte_table(prefix, accept_state, fail_state);

        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);

        // Override fail_state rows: fail is sticky.
        let escape_sentinel = fail_state + 1;
        let mut fused = build_fused_table(
            &sym_trans,
            symbols.len(),
            n_states,
            |_| escape_sentinel,
            fail_state,
        );

        // Accept and fail states are sticky for all inputs.
        let accept_row = usize::from(accept_state) * 256;
        fused[accept_row..accept_row + 256].fill(accept_state);
        let fail_row = usize::from(fail_state) * 256;
        fused[fail_row..fail_row + 256].fill(fail_state);

        let transitions = pack_shift_table(&fused, n_states, Self::BITS);

        // Escape transitions: for an escaped literal byte, use the byte-level transition.
        let escape_transitions = pack_shift_table(&byte_table, n_states, Self::BITS);

        Ok(Self {
            transitions,
            escape_transitions,
            accept_state,
            fail_state,
        })
    }

    /// Build a byte-level transition table for prefix matching (no KMP fallback).
    fn build_prefix_byte_table(prefix: &[u8], accept_state: u8, fail_state: u8) -> Vec<u8> {
        let n_states = fail_state + 1;
        let mut table = vec![fail_state; usize::from(n_states) * 256];

        for state in 0..n_states {
            let s = usize::from(state);
            if state == accept_state {
                for byte in 0..256 {
                    table[s * 256 + byte] = accept_state;
                }
            } else if state != fail_state {
                // Only the correct next byte advances; everything else fails.
                let next_byte = prefix[s];
                let next_state = if s + 1 >= prefix.len() {
                    accept_state
                } else {
                    state + 1
                };
                table[s * 256 + usize::from(next_byte)] = next_state;
            }
        }
        table
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let packed = self.transitions[usize::from(code)];
            // Masked to BITS (4) bits, result ≤ 15, fits in u8
            let next = shift_extract(packed, state, Self::BITS);
            if next == self.fail_state + 1 {
                // Escape sentinel: read literal byte.
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                let esc_packed = self.escape_transitions[usize::from(b)];
                state = shift_extract(esc_packed, state, Self::BITS);
            } else {
                state = next;
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
}
