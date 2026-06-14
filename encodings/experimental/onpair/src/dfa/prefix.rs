// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat per-code transition table DFA for prefix matching (`LIKE 'prefix%'`).
//!
//! Supports prefixes up to 253 bytes (states: 0..N progress + accept + fail ≤
//! 256). OnPair has no escape code, so — unlike the FSST prefix DFA — there is
//! no escape sentinel or escape table.

use vortex_error::VortexExpect;

use super::build_code_transitions;
use super::byte_mask;
use super::n_codes;

/// Flat per-code transition table DFA for prefix matching on OnPair codes.
///
/// States `0..prefix_len` track match progress, plus ACCEPT and FAIL.
/// Transitions are stored in a flat `Vec<u8>` indexed as
/// `[state * n_codes + code]`.
///
/// ```text
/// Prefix: "http"  (4 progress states + accept + fail)
///
///          Token bytes feed the byte table; the lifted code table records the
///          resulting state for each dictionary token.
/// State 0 wants 'h', 1 wants 't', 2 wants 't', 3 wants 'p', 4 = accept (sticky),
/// 5 = fail (sticky).
/// ```
pub(crate) struct FlatPrefixDfa {
    /// `transitions[state * n_codes + code]` -> next state.
    transitions: Vec<u8>,
    n_codes: usize,
    accept_state: u8,
    fail_state: u8,
}

impl FlatPrefixDfa {
    /// Maximum prefix length: need progress + accept + fail to fit in `u8`.
    pub(crate) const MAX_PREFIX_LEN: usize = (u8::MAX - 2) as usize;

    pub(crate) fn new(dict_bytes: &[u8], dict_offsets: &[u32], prefix: &[u8]) -> Self {
        debug_assert!(prefix.len() <= Self::MAX_PREFIX_LEN);

        let accept_state = u8::try_from(prefix.len()).vortex_expect("prefix length fits in u8");
        let fail_state = accept_state + 1;
        let n_states = usize::from(fail_state) + 1;

        let byte_table = build_prefix_byte_table(prefix, accept_state, fail_state);
        // A byte not in the prefix fails from every live state, so tokens with no
        // prefix byte have an all-fail column.
        let transitions = build_code_transitions(
            dict_bytes,
            dict_offsets,
            &byte_table,
            n_states,
            fail_state,
            &byte_mask(prefix),
        );

        Self {
            transitions,
            n_codes: n_codes(dict_offsets),
            accept_state,
            fail_state,
        }
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u16]) -> bool {
        let mut state = 0u8;
        for &code in codes {
            state = self.transitions[usize::from(state) * self.n_codes + usize::from(code)];
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

/// Build a byte-level transition table for prefix matching.
///
/// For each state, only the correct next byte advances; everything else goes to
/// the fail state. ACCEPT and FAIL are both sticky.
fn build_prefix_byte_table(prefix: &[u8], accept_state: u8, fail_state: u8) -> Vec<u8> {
    let n_states = usize::from(fail_state) + 1;
    let mut table = vec![fail_state; n_states * 256];

    for state in 0..n_states {
        if state == usize::from(accept_state) {
            for byte in 0..256 {
                table[state * 256 + byte] = accept_state;
            }
        } else if state != usize::from(fail_state) {
            // Only the correct next byte advances; everything else fails.
            let next_byte = prefix[state];
            let next_state = if state + 1 >= prefix.len() {
                accept_state
            } else {
                u8::try_from(state + 1).vortex_expect("progress state fits in u8")
            };
            table[state * 256 + usize::from(next_byte)] = next_state;
        }
    }
    table
}
