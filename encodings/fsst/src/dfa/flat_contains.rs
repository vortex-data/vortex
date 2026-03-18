// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for contains matching (`LIKE '%needle%'`, needle 8-254).

use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_escape_folded_table;
use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;

/// Flat `u8` transition table DFA for contains matching (needles 8-254 bytes).
///
/// Uses two escape strategies depending on needle length:
/// - **Escape-folded** (needle ≤ 127): escape handling is folded into the state
///   space (2N+1 states), making the scan loop branchless.
/// - **Escape sentinel** (needle 128-254): escape code maps to a sentinel state
///   with a separate byte-level escape table. Required because 2N+1 > 255 won't
///   fit in `u8`.
pub(crate) struct FlatContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    transitions: Vec<u8>,
    accept_state: u8,
    escape: EscapeStrategy,
}

/// How the flat DFA handles the FSST escape code.
enum EscapeStrategy {
    /// Escape states folded into the transition table (branchless scan).
    Folded,
    /// Escape code maps to a sentinel; next byte uses a separate table.
    Sentinel {
        escape_transitions: Vec<u8>,
        sentinel: u8,
    },
}

impl FlatContainsDfa {
    /// Maximum needle for escape-folded mode: 2N+1 ≤ 255, so N ≤ 127.
    const MAX_FOLDED_LEN: usize = 127;
    /// Maximum needle overall: need accept + sentinel to fit in u8.
    pub(crate) const MAX_NEEDLE_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for flat contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }

        let accept_state = u8::try_from(needle.len())
            .vortex_expect("FlatContainsDfa: accept state must fit into u8");

        if needle.len() <= Self::MAX_FOLDED_LEN {
            let transitions = build_escape_folded_table(symbols, symbol_lengths, needle);
            Ok(Self {
                transitions,
                accept_state,
                escape: EscapeStrategy::Folded,
            })
        } else {
            let n_states = accept_state + 1;
            let sentinel = n_states;

            let byte_table = kmp_byte_transitions(needle);
            let sym_trans = build_symbol_transitions(
                symbols,
                symbol_lengths,
                &byte_table,
                n_states,
                accept_state,
            );
            let transitions =
                build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

            let escape_transitions = byte_table;

            Ok(Self {
                transitions,
                accept_state,
                escape: EscapeStrategy::Sentinel {
                    escape_transitions,
                    sentinel,
                },
            })
        }
    }

    #[inline(never)]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.escape {
            EscapeStrategy::Folded => self.matches_folded(codes),
            EscapeStrategy::Sentinel {
                escape_transitions,
                sentinel,
            } => Self::matches_sentinel(
                codes,
                &self.transitions,
                escape_transitions,
                self.accept_state,
                *sentinel,
            ),
        }
    }

    /// Branchless scan: escape handling is folded into the state space.
    #[inline(always)]
    fn matches_folded(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        for &byte in codes {
            state = self.transitions[usize::from(state) * 256 + usize::from(byte)];
        }
        state == self.accept_state
    }

    /// Sentinel scan: escape code triggers a separate table lookup.
    #[inline(always)]
    fn matches_sentinel(
        codes: &[u8],
        transitions: &[u8],
        escape_transitions: &[u8],
        accept_state: u8,
        sentinel: u8,
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
        }
        false
    }
}
