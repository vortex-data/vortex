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
//! Uses the same escape-folded strategy as [`super::flat_contains::FlatContainsDfa`]:
//! 2N+1 states for a total-length-N pattern when N ≤ 127, single-lookup scan;
//! otherwise falls back to the sentinel-based DFA.

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_failure_table;

/// Flat `u8` transition table DFA for multi-wildcard contains matching.
pub(crate) struct MultiContainsDfa {
    inner: MultiInner,
    /// 256-byte lookup table: `state0_adv[b] != 0` iff code byte `b`
    /// transitions state 0 (start of first segment) to a non-zero state.
    /// Allows memchr-style skip when the scanner is in state 0 exactly
    /// like the single-needle contains DFA does.
    state0_adv: Box<[u8; 256]>,
}

enum MultiInner {
    /// Escape-folded DFA (2N+1 states, total_len ≤ 127). Single-lookup scan.
    Folded {
        transitions: Vec<u8>,
        accept_state: u8,
    },
    /// Sentinel-based DFA for longer total lengths (≤ 254).
    Sentinel {
        transitions: Vec<u8>,
        escape_transitions: Vec<u8>,
        accept_state: u8,
        sentinel: u8,
    },
}

impl MultiContainsDfa {
    /// Maximum total needle length (sum of all segments): need accept + sentinel in u8.
    pub(crate) const MAX_TOTAL_LEN: usize = u8::MAX as usize - 1;
    /// Maximum total length that can use the escape-folded DFA.
    const MAX_FOLDED_TOTAL_LEN: usize = 127;

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

        if total_len <= Self::MAX_FOLDED_TOTAL_LEN {
            Self::new_folded(symbols, symbol_lengths, segments, total_len)
        } else {
            Self::new_sentinel(symbols, symbol_lengths, segments, total_len)
        }
    }

    fn new_folded(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        segments: &[&[u8]],
        total_len: usize,
    ) -> VortexResult<Self> {
        let accept_state =
            u8::try_from(total_len).vortex_expect("folded multi-contains: accept fits in u8");
        let n_progress = accept_state as usize + 1; // states 0..=accept

        let byte_table = chained_kmp_byte_transitions(segments, accept_state);
        let sym_trans = build_symbol_transitions(
            symbols,
            symbol_lengths,
            &byte_table,
            n_progress as u8,
            accept_state,
        );

        let n_in_escape = accept_state as usize; // one per non-accept progress state
        let n_total = n_progress + n_in_escape;

        let mut transitions = vec![0u8; n_total * 256];
        let n_symbols = symbols.len();

        for state in 0..n_progress {
            let row = state * 256;
            for code in 0..n_symbols {
                transitions[row + code] = sym_trans[state * n_symbols + code];
            }
            if state == accept_state as usize {
                transitions[row + ESCAPE_CODE as usize] = accept_state;
            } else {
                let in_escape = (n_progress + state) as u8;
                transitions[row + ESCAPE_CODE as usize] = in_escape;
            }
        }

        for s in 0..n_in_escape {
            let in_esc_row = (n_progress + s) * 256;
            let byte_row = s * 256;
            transitions[in_esc_row..in_esc_row + 256]
                .copy_from_slice(&byte_table[byte_row..byte_row + 256]);
        }

        let state0_adv = build_state0_adv_folded(&transitions);

        Ok(Self {
            inner: MultiInner::Folded {
                transitions,
                accept_state,
            },
            state0_adv,
        })
    }

    fn new_sentinel(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        segments: &[&[u8]],
        total_len: usize,
    ) -> VortexResult<Self> {
        let accept_state = u8::try_from(total_len)
            .vortex_expect("MultiContainsDfa: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = chained_kmp_byte_transitions(segments, accept_state);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        let state0_adv = build_state0_adv_sentinel(&transitions, sentinel);

        Ok(Self {
            inner: MultiInner::Sentinel {
                transitions,
                escape_transitions: byte_table,
                accept_state,
                sentinel,
            },
            state0_adv,
        })
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            MultiInner::Folded {
                transitions,
                accept_state,
            } => matches_folded(transitions, *accept_state, &self.state0_adv, codes),
            MultiInner::Sentinel {
                transitions,
                escape_transitions,
                accept_state,
                sentinel,
            } => matches_sentinel(
                transitions,
                escape_transitions,
                *accept_state,
                *sentinel,
                &self.state0_adv,
                codes,
            ),
        }
    }
}

/// Build a 256-byte advancing-code table from the first row of a folded
/// multi-contains transition table. `adv[b] != 0` iff byte `b` in state 0
/// transitions to a state other than 0 (i.e. it's worth running the DFA
/// for this byte).
fn build_state0_adv_folded(transitions: &[u8]) -> Box<[u8; 256]> {
    let mut adv = [0u8; 256];
    for b in 0..256usize {
        if transitions[b] != 0 {
            adv[b] = 1;
        }
    }
    Box::new(adv)
}

/// Build the state-0 advancing table for the sentinel variant. The sentinel
/// (ESCAPE_CODE row) is always advancing since an escape can advance if the
/// escaped byte is the needle's first byte.
fn build_state0_adv_sentinel(transitions: &[u8], sentinel: u8) -> Box<[u8; 256]> {
    let mut adv = [0u8; 256];
    for b in 0..256usize {
        let t = transitions[b];
        // `t != 0` means advance; `t == sentinel` means "escape" — always
        // treat escapes as advancing so the scanner drops into the DFA to
        // handle the literal byte properly.
        if t != 0 || t == sentinel {
            adv[b] = 1;
        }
    }
    // Also force ESCAPE_CODE row to be advancing in case the transitions
    // table stores a 0 there (shouldn't happen, but defensive).
    adv[ESCAPE_CODE as usize] = 1;
    Box::new(adv)
}

/// Scan for the escape-folded multi-contains DFA with state-0 skip.
///
/// While in state 0 we byte-skip through the 256-byte advancing-code table
/// until we find a code that transitions out of state 0. The stateful
/// scan is a single table lookup per code. (Experiments with 2× unroll
/// here regressed benchmarks with dense first-segment matches, where the
/// extra accept-check work hurt more than the loop-overhead win helped.)
#[inline(always)]
fn matches_folded(
    transitions: &[u8],
    accept_state: u8,
    state0_adv: &[u8; 256],
    codes: &[u8],
) -> bool {
    let mut state = 0u8;
    let mut pos = 0;
    let len = codes.len();
    while pos < len {
        if state == 0 {
            // Skip non-advancing bytes.
            while pos < len {
                // SAFETY: pos < len; state0_adv is 256 bytes.
                let b = unsafe { *codes.get_unchecked(pos) };
                if unsafe { *state0_adv.get_unchecked(b as usize) } != 0 {
                    break;
                }
                pos += 1;
            }
            if pos >= len {
                return false;
            }
        }
        // SAFETY: pos < len; state < 2N+1.
        let code = unsafe { *codes.get_unchecked(pos) };
        pos += 1;
        state = unsafe { *transitions.get_unchecked(usize::from(state) * 256 + usize::from(code)) };
        if state == accept_state {
            return true;
        }
    }
    false
}

#[inline(always)]
fn matches_sentinel(
    transitions: &[u8],
    escape_transitions: &[u8],
    accept_state: u8,
    sentinel: u8,
    state0_adv: &[u8; 256],
    codes: &[u8],
) -> bool {
    let mut state = 0u8;
    let mut pos = 0;
    while pos < codes.len() {
        if state == 0 {
            // Skip non-advancing bytes using the 256-byte skip table.
            while pos < codes.len() {
                let b = codes[pos];
                if state0_adv[b as usize] != 0 {
                    break;
                }
                pos += 1;
            }
            if pos >= codes.len() {
                return false;
            }
        }
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
