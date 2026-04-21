// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for multi-wildcard contains matching
//! (`LIKE '%seg1%seg2%...%segN%'`).
//!
//! Chains multiple KMP automata into a single linear state space. Each segment's
//! states are concatenated: phase k's accept state IS phase k+1's start state.
//! The final segment's accept is the global accept (sticky).
//!
//! ## State Layout
//!
//! ```text
//! Pattern: %abc%def%
//! Segments: ["abc", "def"]
//!
//! Global states:
//!   0: 0 of "abc" matched   (phase 0 start)
//!   1: 1 of "abc" matched
//!   2: 2 of "abc" matched
//!   3: all of "abc" matched = 0 of "def" matched  (phase 1 start)
//!   4: 1 of "def" matched
//!   5: 2 of "def" matched
//!   6: ACCEPT (all of "def" matched)
//! ```
//!
//! Each phase uses its own independent KMP failure function for backtracking.
//! The `%` between segments is implicit: once phase k accepts, phase k+1
//! searches for its needle anywhere in the remaining input.
//!
//! Uses the same escape-sentinel strategy as [`super::flat_contains::FlatContainsDfa`].

use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_failure_table;

/// Flat `u8` transition table DFA for multi-wildcard contains matching.
///
/// Supports patterns like `%abc%def%ghi%` by chaining per-segment KMP
/// automata into one linear state space. The scanner is identical to
/// [`super::flat_contains::FlatContainsDfa`]: a single forward pass with
/// escape-sentinel handling.
pub(crate) struct MultiContainsDfa {
    /// `transitions[state * 256 + code_byte]` -> next state.
    transitions: Vec<u8>,
    /// `escape_transitions[state * 256 + literal_byte]` -> next state for escaped bytes.
    escape_transitions: Vec<u8>,
    accept_state: u8,
    sentinel: u8,
}

impl MultiContainsDfa {
    /// Maximum total needle length (sum of all segments): need accept + sentinel in u8.
    pub(crate) const MAX_TOTAL_LEN: usize = u8::MAX as usize - 1;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        segments: &[&[u8]],
    ) -> VortexResult<Self> {
        let total_len: usize = segments.iter().map(|s| s.len()).sum();
        if total_len > Self::MAX_TOTAL_LEN {
            vortex_bail!(
                "total segment length {} exceeds maximum {} for multi-contains DFA",
                total_len,
                Self::MAX_TOTAL_LEN
            );
        }

        let accept_state = u8::try_from(total_len)
            .vortex_expect("MultiContainsDfa: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = chained_kmp_byte_transitions(segments, accept_state);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        Ok(Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            sentinel,
        })
    }

    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let next = self.transitions[usize::from(state) * 256 + usize::from(code)];
            if next == self.sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[usize::from(state) * 256 + usize::from(b)];
            } else {
                state = next;
            }
            if state == self.accept_state {
                return true;
            }
        }
        false
    }
}

/// Build a chained KMP byte-level transition table for multiple segments.
///
/// States are the concatenation of each segment's progress states:
/// - Phase k occupies global states `offsets[k]..offsets[k] + segments[k].len()`
/// - Phase k's accept (= `offsets[k+1]`) is phase k+1's start state
/// - The final phase's accept is the global accept state (sticky)
///
/// Each phase has its own KMP failure function for intra-segment backtracking.
fn chained_kmp_byte_transitions(segments: &[&[u8]], accept_state: u8) -> Vec<u8> {
    let n_states = accept_state + 1;
    let mut table = vec![0u8; usize::from(n_states) * 256];

    // Phase offsets: offsets[k] = global state index for phase k's start
    let mut offsets = Vec::with_capacity(segments.len() + 1);
    let mut off = 0usize;
    for seg in segments {
        offsets.push(off);
        off += seg.len();
    }
    offsets.push(off); // = total_len = accept_state

    for (k, segment) in segments.iter().enumerate() {
        let base = offsets[k];
        let failure = kmp_failure_table(segment);

        for local_s in 0..segment.len() {
            let global_s = base + local_s;
            for byte in 0..256usize {
                let mut s = local_s;
                loop {
                    if byte == usize::from(segment[s]) {
                        s += 1;
                        break;
                    }
                    if s == 0 {
                        break;
                    }
                    s = usize::from(failure[s - 1]);
                }
                // If s == segment.len(), this maps to offsets[k+1] =
                // phase k+1's start (or the final accept for the last phase).
                table[global_s * 256 + byte] =
                    u8::try_from(base + s).vortex_expect("chained KMP state must fit in u8");
            }
        }
    }

    // Final accept state: sticky
    let acc = usize::from(accept_state);
    for byte in 0..256 {
        table[acc * 256 + byte] = accept_state;
    }

    table
}
