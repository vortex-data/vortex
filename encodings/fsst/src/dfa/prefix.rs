// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for prefix matching (`LIKE 'prefix%'`).
//!
//! ## Escape-folded state machine
//!
//! For prefixes ≤ 126 bytes we use an escape-folded DFA. The state space is:
//!
//! - Progress states `0..=N` (N+1 states). State `N` is accept (sticky).
//! - Fail state `N+1` (sticky).
//! - "In-escape" states `N+2..=2N+1` (N states), one per progress state that
//!   can enter an escape.
//!
//! Total states: `2N+2 ≤ 255`, so max prefix = 126 bytes when folded.
//!
//! The scanner is a uniform single-lookup loop with an early-exit on fail:
//!
//! ```text
//! state = transitions[state * 256 + code];
//! if state == accept { return true; }
//! if state == fail   { return false; }
//! ```
//!
//! For prefixes 127..253 bytes we fall back to the two-table sentinel scan.

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;

/// Flat `u8` transition table DFA for prefix matching on FSST codes.
pub(crate) struct FlatPrefixDfa {
    inner: PrefixInner,
}

enum PrefixInner {
    /// Escape-folded DFA (2N+2 states, N ≤ 126).
    Folded {
        transitions: Vec<u8>,
        accept_state: u8,
        fail_state: u8,
    },
    /// Sentinel-based DFA for longer prefixes (up to 253 bytes).
    Sentinel {
        transitions: Vec<u8>,
        escape_transitions: Vec<u8>,
        accept_state: u8,
        fail_state: u8,
        sentinel: u8,
    },
}

impl FlatPrefixDfa {
    pub(crate) const MAX_PREFIX_LEN: usize = (u8::MAX - 2) as usize;
    const MAX_FOLDED_PREFIX_LEN: usize = 126;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        prefix: &[u8],
    ) -> VortexResult<Self> {
        if prefix.len() > Self::MAX_PREFIX_LEN {
            vortex_bail!(
                "prefix length {} exceeds maximum {} for flat prefix DFA",
                prefix.len(),
                Self::MAX_PREFIX_LEN
            );
        }

        if prefix.len() <= Self::MAX_FOLDED_PREFIX_LEN {
            Self::new_folded(symbols, symbol_lengths, prefix)
        } else {
            Self::new_sentinel(symbols, symbol_lengths, prefix)
        }
    }

    fn new_folded(symbols: &[Symbol], symbol_lengths: &[u8], prefix: &[u8]) -> VortexResult<Self> {
        let n = prefix.len();
        let accept_state = u8::try_from(n).vortex_expect("folded prefix: accept fits in u8");
        let fail_state = accept_state + 1;
        let n_progress = fail_state as usize + 1; // progress states 0..=fail

        let byte_table = build_prefix_byte_table(prefix, accept_state, fail_state);
        let sym_trans = build_symbol_transitions(
            symbols,
            symbol_lengths,
            &byte_table,
            n_progress as u8,
            accept_state,
        );

        // In-escape states for progress states 0..N-1 (not accept, not fail).
        let n_in_escape = accept_state as usize;
        let n_total = n_progress + n_in_escape;

        let mut transitions = vec![fail_state; n_total * 256];
        let n_symbols = symbols.len();

        // Progress & accept & fail rows.
        for state in 0..n_progress {
            let row = state * 256;
            // Default is already fail_state.
            for code in 0..n_symbols {
                transitions[row + code] = sym_trans[state * n_symbols + code];
            }
            if state == accept_state as usize {
                // Accept sticky on all bytes.
                for b in 0..256 {
                    transitions[row + b] = accept_state;
                }
            } else if state == fail_state as usize {
                // Fail sticky on all bytes.
                for b in 0..256 {
                    transitions[row + b] = fail_state;
                }
            } else {
                // Progress state: ESCAPE_CODE → in-escape.
                let in_escape = (n_progress + state) as u8;
                transitions[row + ESCAPE_CODE as usize] = in_escape;
            }
        }

        // In-escape states for progress 0..N-1.
        for s in 0..n_in_escape {
            let in_esc_row = (n_progress + s) * 256;
            let byte_row = s * 256;
            transitions[in_esc_row..in_esc_row + 256]
                .copy_from_slice(&byte_table[byte_row..byte_row + 256]);
        }

        Ok(Self {
            inner: PrefixInner::Folded {
                transitions,
                accept_state,
                fail_state,
            },
        })
    }

    fn new_sentinel(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        prefix: &[u8],
    ) -> VortexResult<Self> {
        let accept_state = u8::try_from(prefix.len()).vortex_expect("prefix fits in u8");
        let fail_state = accept_state + 1;
        let n_states = fail_state + 1;
        let sentinel = fail_state + 1;

        let byte_table = build_prefix_byte_table(prefix, accept_state, fail_state);

        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);

        let transitions = build_fused_table(
            &sym_trans,
            symbols.len(),
            n_states,
            |_| sentinel,
            fail_state,
        );

        Ok(Self {
            inner: PrefixInner::Sentinel {
                transitions,
                escape_transitions: byte_table,
                accept_state,
                fail_state,
                sentinel,
            },
        })
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            PrefixInner::Folded {
                transitions,
                accept_state,
                fail_state,
            } => matches_folded(transitions, *accept_state, *fail_state, codes),
            PrefixInner::Sentinel {
                transitions,
                escape_transitions,
                accept_state,
                fail_state,
                sentinel,
            } => matches_sentinel(
                transitions,
                escape_transitions,
                *accept_state,
                *fail_state,
                *sentinel,
                codes,
            ),
        }
    }
}

#[inline(always)]
fn matches_folded(transitions: &[u8], accept_state: u8, fail_state: u8, codes: &[u8]) -> bool {
    let mut state = 0u8;
    let len = codes.len();
    let mut pos = 0;
    while pos < len {
        // SAFETY: pos < len; state < n_total = 2N+2, transitions has n_total*256 entries.
        let code = unsafe { *codes.get_unchecked(pos) };
        pos += 1;
        state = unsafe { *transitions.get_unchecked(usize::from(state) * 256 + usize::from(code)) };
        if state == accept_state {
            return true;
        }
        if state == fail_state {
            return false;
        }
    }
    state == accept_state
}

#[inline(always)]
fn matches_sentinel(
    transitions: &[u8],
    escape_transitions: &[u8],
    accept_state: u8,
    fail_state: u8,
    sentinel: u8,
    codes: &[u8],
) -> bool {
    let mut state = 0u8;
    let mut pos = 0;
    while pos < codes.len() {
        let code = codes[pos];
        pos += 1;
        let next = transitions[usize::from(state) * 256 + usize::from(code)];
        if next == sentinel {
            if pos >= codes.len() {
                return false;
            }
            let b = codes[pos];
            pos += 1;
            state = escape_transitions[usize::from(state) * 256 + usize::from(b)];
        } else {
            state = next;
        }
        if state == accept_state {
            return true;
        }
        if state == fail_state {
            return false;
        }
    }
    state == accept_state
}

/// Build a byte-level transition table for prefix matching.
///
/// For each state, only the correct next byte advances; everything else goes
/// to the fail state.
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
