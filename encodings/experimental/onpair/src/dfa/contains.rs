// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat per-code transition table DFA for contains matching (`LIKE '%needle%'`).
//!
//! Built from the standard KMP failure function lifted to the token level. As
//! OnPair has no escape code, there is no escape sentinel: every code maps
//! directly to its byte-derived transition.

use vortex_error::VortexExpect;

use super::build_code_transitions;
use super::byte_mask;
use super::kmp_byte_transitions;
use super::n_codes;

/// Flat per-code transition table DFA for contains matching on OnPair codes.
///
/// States `0..needle_len` track match progress, `needle_len` is the (sticky)
/// accept state. Transitions are stored in a flat `Vec<u8>` indexed as
/// `[state * n_codes + code]`.
pub(crate) struct FlatContainsDfa {
    /// `transitions[state * n_codes + code]` -> next state.
    transitions: Vec<u8>,
    n_codes: usize,
    accept_state: u8,
}

impl FlatContainsDfa {
    /// Maximum needle length: the accept state must fit in `u8`.
    pub(crate) const MAX_NEEDLE_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(dict_bytes: &[u8], dict_offsets: &[u32], needle: &[u8]) -> Self {
        debug_assert!(needle.len() <= Self::MAX_NEEDLE_LEN);

        let accept_state = u8::try_from(needle.len()).vortex_expect("needle length fits in u8");
        let n_states = usize::from(accept_state) + 1;

        let byte_table = kmp_byte_transitions(needle);
        // A non-needle byte resets the KMP automaton to state 0 from any live
        // state, so tokens with no needle byte have an all-zero column.
        let transitions = build_code_transitions(
            dict_bytes,
            dict_offsets,
            &byte_table,
            n_states,
            0,
            &byte_mask(needle),
        );

        Self {
            transitions,
            n_codes: n_codes(dict_offsets),
            accept_state,
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
        }
        false
    }
}
