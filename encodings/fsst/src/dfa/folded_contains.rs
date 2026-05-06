// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Escape-folded flat `u8` transition table DFA for contains matching
//! (`LIKE '%needle%'`).
//!
//! ## Why escape-fold?
//!
//! The plain [`super::flat_contains::FlatContainsDfa`] keeps a sentinel branch
//! in its inner loop: when the current code is `ESCAPE_CODE`, the table maps to
//! a sentinel value, the scanner detects it, and a second table lookup (in a
//! separate byte table) consumes the following literal byte. That's a hard-to-
//! predict branch on every code byte.
//!
//! The escape-folded DFA encodes "we just saw an `ESCAPE_CODE`, expecting a
//! literal byte" directly into the state space. With needle length `N`, where
//! `N <= 127`:
//!
//! - **Normal states** `0..=N`: regular KMP-style match progress; `N` is the
//!   accept state (sticky).
//! - **Escape states** `N+1..=2N`: "in-escape from base normal state
//!   `s = state - (N + 1)`" for `s` in `0..=N-1`. A read here is interpreted
//!   as a literal byte, advancing per the byte-level transition table for `s`.
//!
//! Total states: `2N + 1 <= 255`, so the state id fits in `u8`.
//!
//! The transition table is a flat `Vec<u8>` of size `(2N + 1) * 256`. For
//! normal states, the entry on `ESCAPE_CODE` goes to the matching escape
//! state `s + N + 1`. For escape states, all 256 entries are read as literal
//! bytes and dispatched through the byte table for the base state. There is
//! no sentinel branch in the inner loop -- every code byte produces exactly
//! one table lookup.
//!
//! The state-0 skip strategy (`memchr` / bitmap) still applies in the same way
//! as the plain DFA: when in state 0 we jump to the next code that could
//! progress the match.

use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ESCAPE_CODE;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
use super::skip::SkipStrategy;

/// Escape-folded flat `u8` transition table DFA for contains matching.
///
/// Supports needles up to [`Self::MAX_NEEDLE_LEN`] bytes (so the state count
/// `2N + 1` fits in `u8`).
pub(crate) struct FoldedContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    ///
    /// Layout: rows `0..=N` are normal states (regular byte/symbol dispatch);
    /// rows `N+1..=2N` are escape states whose 256 entries are literal-byte
    /// dispatches via the underlying byte table.
    transitions: Vec<u8>,
    accept_state: u8,
    /// State-0 skip strategy.
    skip: SkipStrategy,
}

impl FoldedContainsDfa {
    /// Maximum needle length: `2N + 1 <= 255` so `N <= 127`.
    pub(crate) const MAX_NEEDLE_LEN: usize = 127;

    /// Build a folded contains DFA for `needle`.
    ///
    /// Returns `Err` if `needle.len() > `[`Self::MAX_NEEDLE_LEN`].
    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for folded contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }
        // Empty needles are handled at a higher level (MatchAll), but we still
        // accept them here defensively (N=0 -> only the accept state).
        let accept_state =
            u8::try_from(needle.len()).vortex_expect("FoldedContainsDfa: accept state fits in u8");
        let n_normal = accept_state + 1; // states 0..=N
        // Total states: 2N+1 (normal 0..=N, escape N+1..=2N for base 0..=N-1).
        let n_states_usize = 2 * usize::from(accept_state) + 1;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_normal, accept_state);

        // Build the folded fused table: (2N+1) * 256.
        let n_symbols = symbols.len();
        let mut transitions = vec![0u8; n_states_usize * 256];

        // Rows 0..=N: normal states.
        for s in 0..n_normal {
            let row = usize::from(s) * 256;
            // Symbol codes 0..n_symbols.
            for code in 0..n_symbols {
                transitions[row + code] = sym_trans[usize::from(s) * n_symbols + code];
            }
            // ESCAPE_CODE: go to the matching escape state, except for accept
            // (which is sticky -- all transitions remain at accept).
            let escape_target = if s == accept_state {
                accept_state
            } else {
                // Escape state for base s = N + 1 + s.
                accept_state + 1 + s
            };
            transitions[row + usize::from(ESCAPE_CODE)] = escape_target;
            // Other code bytes (n_symbols..255 except ESCAPE_CODE) default to 0,
            // matching the plain `FlatContainsDfa` semantics.
        }

        // Rows N+1..=2N: escape states. For escape state e = N + 1 + s where
        // s in 0..=N-1, all 256 entries dispatch the next byte as a literal
        // through `byte_table[s * 256 + b]`.
        for s in 0..accept_state {
            let escape_state = accept_state + 1 + s;
            let row = usize::from(escape_state) * 256;
            let byte_row = usize::from(s) * 256;
            transitions[row..row + 256].copy_from_slice(&byte_table[byte_row..byte_row + 256]);
        }

        // Build the skip strategy from row 0 of the transitions (the first 256
        // entries). State 0 is reached either initially or by KMP fallback,
        // and we want to skip codes that leave us at 0.
        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        Ok(Self {
            transitions,
            accept_state,
            skip,
        })
    }

    /// Run the matcher over `codes`. Returns `true` iff the needle appears.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let transitions = self.transitions.as_slice();
        let accept = self.accept_state;
        let mut pos: usize = 0;
        let len = codes.len();

        // Outer loop: SIMD-skip in state 0 to the next progressing code, then
        // run a tight inner loop while state != 0. The inner loop is uniform:
        // one table lookup per code byte, no sentinel branch. We only return
        // to the outer loop when the DFA falls back to state 0 (KMP failure).
        loop {
            match self.skip.find_next_progressing(codes, pos) {
                Some(next) => pos = next,
                None => return false,
            }

            // We're at a progressing code: step once.
            let code = codes[pos];
            pos += 1;
            let mut state = transitions[usize::from(code)];
            if state == accept {
                return true;
            }

            // Inner loop while state != 0.
            while state != 0 && pos < len {
                let c = codes[pos];
                pos += 1;
                state = transitions[usize::from(state) * 256 + usize::from(c)];
                if state == accept {
                    return true;
                }
            }
            if pos >= len {
                return false;
            }
        }
    }
}
