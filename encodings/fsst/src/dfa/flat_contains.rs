// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for contains matching (`LIKE '%needle%'`).
//!
//! ## Escape-folded state machine
//!
//! To eliminate the per-byte "is this the escape sentinel?" branch, we fold the
//! escape handling directly into the state space. For a needle of N bytes
//! (N ≤ 127):
//!
//! - Progress states `0..=N` (N+1 states) track match progress. `N` is accept.
//! - "In-escape" states `N+1..=2N` (N states) mark "we just saw ESCAPE_CODE
//!   from progress state k". The next code byte is interpreted as a literal
//!   through the byte-level transition table.
//!
//! Total states: `2N+1 ≤ 255`, so max needle = 127 bytes.
//!
//! The scanner becomes a uniform single-lookup loop:
//!
//! ```text
//! state = transitions[state * 256 + code];
//! if state == accept { return true; }
//! ```
//!
//! There is no sentinel check and no second table lookup. All escape semantics
//! live in the single 256-wide transition table.
//!
//! For needles of length 128..254, we fall back to the sentinel-based DFA
//! which uses two tables and an explicit sentinel branch.
//!
//! ## State-0 skip strategies
//!
//! While in state 0, the DFA is a sequential dependency chain on the inner
//! loop. We break it with a SIMD skip:
//!
//! - **memchr skip** (1-3 advancing codes): use `memchr`/`memchr2`/`memchr3`
//!   inline in the DFA loop. SIMD-accelerated, 32+ bytes/cycle.
//!
//! - **bitmap skip** (4+ advancing codes): packed `[u64; 4]` bitmap check.
//!   1 cache line, branchless per code.

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;

/// How to skip non-progressing codes when in state 0.
enum SkipStrategy {
    /// 1 advancing code — `memchr::memchr` (SIMD, 32+ bytes/cycle).
    Memchr1(u8),
    /// 2 advancing codes — `memchr::memchr2` (SIMD).
    Memchr2(u8, u8),
    /// 3 advancing codes — `memchr::memchr3` (SIMD).
    Memchr3(u8, u8, u8),
    /// 4+ advancing codes — 256-byte lookup table (`table[b] != 0` iff `b` is
    /// in the advancing set). 4 cache lines; small enough to stay hot in L1,
    /// and the branchless probe autovectorizes nicely.
    Table(Box<[u8; 256]>),
}

/// Flat `u8` transition table DFA for contains matching.
pub(crate) struct FlatContainsDfa {
    inner: ContainsInner,
}

enum ContainsInner {
    /// Escape-folded DFA (2N+1 states, N ≤ 127). Single-lookup scan.
    Folded {
        transitions: Vec<u8>,
        accept_state: u8,
        skip: SkipStrategy,
    },
    /// Sentinel-based DFA for long needles (N ≤ 254). Two-table scan.
    Sentinel {
        transitions: Vec<u8>,
        escape_transitions: Vec<u8>,
        accept_state: u8,
        sentinel: u8,
        skip: SkipStrategy,
    },
}

impl FlatContainsDfa {
    /// Maximum needle length: fall-back sentinel DFA supports up to 254.
    pub(crate) const MAX_NEEDLE_LEN: usize = u8::MAX as usize - 1;
    /// Maximum needle length that can use the escape-folded DFA.
    const MAX_FOLDED_NEEDLE_LEN: usize = 127;

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

        if needle.len() <= Self::MAX_FOLDED_NEEDLE_LEN {
            Self::new_folded(symbols, symbol_lengths, needle)
        } else {
            Self::new_sentinel(symbols, symbol_lengths, needle)
        }
    }

    fn new_folded(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> VortexResult<Self> {
        let n = needle.len();
        let accept_state = u8::try_from(n).vortex_expect("folded contains: accept fits in u8");
        let n_progress = accept_state as usize + 1; // states 0..=accept

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans = build_symbol_transitions(
            symbols,
            symbol_lengths,
            &byte_table,
            n_progress as u8,
            accept_state,
        );

        let n_in_escape = accept_state as usize; // one per non-accept progress state
        let n_total = n_progress + n_in_escape; // 2N+1

        let mut transitions = vec![0u8; n_total * 256];
        let n_symbols = symbols.len();

        // Progress states 0..=accept
        for state in 0..n_progress {
            let row = state * 256;
            for code in 0..n_symbols {
                transitions[row + code] = sym_trans[state * n_symbols + code];
            }
            if state == accept_state as usize {
                // Accept is sticky for ESCAPE_CODE too.
                transitions[row + ESCAPE_CODE as usize] = accept_state;
            } else {
                // Escape: enter the in-escape state for this progress state.
                let in_escape = (n_progress + state) as u8;
                transitions[row + ESCAPE_CODE as usize] = in_escape;
            }
        }

        // In-escape states: the next byte is a literal. Re-use the byte table
        // for progress state `s` (states 0..N-1; we don't need one for accept).
        for s in 0..n_in_escape {
            let in_esc_row = (n_progress + s) * 256;
            let byte_row = s * 256;
            transitions[in_esc_row..in_esc_row + 256]
                .copy_from_slice(&byte_table[byte_row..byte_row + 256]);
        }

        // Collect advancing code bytes from state 0 (those that move past state 0).
        // The folded table is what the scanner actually consults, so read it directly.
        let mut adv: Vec<u8> = Vec::new();
        for code in 0..=255u8 {
            if transitions[usize::from(code)] != 0 {
                adv.push(code);
            }
        }
        let skip = build_skip(&adv);

        Ok(Self {
            inner: ContainsInner::Folded {
                transitions,
                accept_state,
                skip,
            },
        })
    }

    fn new_sentinel(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        let accept_state = u8::try_from(needle.len())
            .vortex_expect("FlatContainsDfa: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        let mut adv: Vec<u8> = Vec::new();
        for code in 0..=255u8 {
            if transitions[usize::from(code)] != 0 || code == ESCAPE_CODE {
                adv.push(code);
            }
        }
        let skip = build_skip(&adv);

        Ok(Self {
            inner: ContainsInner::Sentinel {
                transitions,
                escape_transitions: byte_table,
                accept_state,
                sentinel,
                skip,
            },
        })
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            ContainsInner::Folded {
                transitions,
                accept_state,
                skip,
            } => matches_folded(transitions, *accept_state, skip, codes),
            ContainsInner::Sentinel {
                transitions,
                escape_transitions,
                accept_state,
                sentinel,
                skip,
            } => matches_sentinel(
                transitions,
                escape_transitions,
                *accept_state,
                *sentinel,
                skip,
                codes,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Inner scan loops — extracted as separate functions so each one can be
// inlined and monomorphized by itself. The enum dispatch happens once per
// `matches()` call, outside the hot loop.
// ---------------------------------------------------------------------------

fn build_skip(adv: &[u8]) -> SkipStrategy {
    match adv.len() {
        // Empty set: build a 256-byte table of zeros (matches nothing).
        0 => SkipStrategy::Table(Box::new([0u8; 256])),
        1 => SkipStrategy::Memchr1(adv[0]),
        2 => SkipStrategy::Memchr2(adv[0], adv[1]),
        3 => SkipStrategy::Memchr3(adv[0], adv[1], adv[2]),
        _ => {
            let mut table = [0u8; 256];
            for &c in adv {
                table[c as usize] = 1;
            }
            SkipStrategy::Table(Box::new(table))
        }
    }
}

#[inline(always)]
fn skip_state0(rest: &[u8], skip: &SkipStrategy) -> Option<usize> {
    match skip {
        SkipStrategy::Memchr1(a) => memchr::memchr(*a, rest),
        SkipStrategy::Memchr2(a, b) => memchr::memchr2(*a, *b, rest),
        SkipStrategy::Memchr3(a, b, c) => memchr::memchr3(*a, *b, *c, rest),
        SkipStrategy::Table(t) => {
            let n = rest.len();
            let mut i = 0;
            while i < n {
                // SAFETY: i < n; t is 256 bytes, byte indexes always in bounds.
                let b = unsafe { *rest.get_unchecked(i) };
                if unsafe { *t.get_unchecked(b as usize) } != 0 {
                    return Some(i);
                }
                i += 1;
            }
            None
        }
    }
}

/// Uniform single-lookup scan for the escape-folded DFA.
///
/// Uses `get_unchecked` to remove bounds checks from the hot load:
/// - `state` is always < 2N+1 ≤ 255, and `transitions.len() == (2N+1) * 256`,
///   so `state * 256 + code` is always < `transitions.len()` for any u8 code.
/// - `pos` is checked before each access to `codes`.
#[inline(always)]
fn matches_folded(transitions: &[u8], accept_state: u8, skip: &SkipStrategy, codes: &[u8]) -> bool {
    let mut state = 0u8;
    let mut pos = 0;
    let len = codes.len();
    while pos < len {
        if state == 0 {
            // SAFETY: pos < len.
            let rest = unsafe { codes.get_unchecked(pos..) };
            match skip_state0(rest, skip) {
                Some(offset) => pos += offset,
                None => return false,
            }
        }
        // SAFETY: pos < len; state < 2N+1; transitions has (2N+1)*256 entries.
        let code = unsafe { *codes.get_unchecked(pos) };
        pos += 1;
        state = unsafe { *transitions.get_unchecked(usize::from(state) * 256 + usize::from(code)) };
        if state == accept_state {
            return true;
        }
    }
    false
}

/// Two-table scan for needles > 127 bytes.
#[inline(always)]
fn matches_sentinel(
    transitions: &[u8],
    escape_transitions: &[u8],
    accept_state: u8,
    sentinel: u8,
    skip: &SkipStrategy,
    codes: &[u8],
) -> bool {
    let mut state = 0u8;
    let mut pos = 0;
    while pos < codes.len() {
        if state == 0 {
            let rest = &codes[pos..];
            match skip_state0(rest, skip) {
                Some(offset) => pos += offset,
                None => return false,
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
