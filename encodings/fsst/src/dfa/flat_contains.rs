// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for contains matching (`LIKE '%needle%'`).
//!
//! ## State-0 skip strategies
//!
//! The DFA is a sequential dependency chain. We break it while in state 0:
//!
//! - **memchr skip** (1-3 advancing codes): use `memchr`/`memchr2`/`memchr3`
//!   inline in the DFA loop. SIMD-accelerated, 32+ bytes/cycle. Only fires
//!   when the DFA drops back to state 0, so no overhead for high-match patterns
//!   where the DFA rarely returns to state 0.
//!
//! - **bitmap skip** (4+ advancing codes): packed `[u64; 4]` bitmap check.
//!   1 cache line, branchless per code.
//!
//! Additionally, a **memchr anchor prefilter** uses the longest FSST symbol
//! whose expansion is a substring of the needle. If that code byte is absent
//! from the compressed string, the needle can't match.
//!
//! ## Per-state shufti skip (Variant A)
//!
//! `FlatContainsDfa` generalises the state-0 skip to ALL states using a
//! Hyperscan-style "shufti" classifier (2× `PSHUFB` + `AND`, classifies 16
//! bytes per shuffle). At any state `s`, we skip 16-byte chunks of the code
//! stream until we hit an "interesting" code (one that would change state or
//! is an escape), then take a single scalar DFA step.
//!
//! `FlatContainsDfaBaseline` preserves the original state-0-only skip for
//! side-by-side benchmarking.
//!
//! TODO(joe): for short needles (≤7 bytes), a branchless escape-folded DFA
//! with hierarchical 4-byte composition is ~2x faster. For needles ≤127 bytes,
//! an escape-folded flat DFA (2N+1 states) avoids the sentinel branch.
//! See commit 7faf9f36f for those implementations.

use fsst::Symbol;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_fused_table;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
use super::shufti::ShuftiMask;
use super::skip::SkipStrategy;

// ---------------------------------------------------------------------------
// Optional skip-fire instrumentation (feature = "shufti-counters")
// ---------------------------------------------------------------------------

/// Total number of per-state shufti `find_next` calls across all DFA invocations.
#[cfg(feature = "shufti-counters")]
pub static SHUFTI_SKIP_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Number of calls where `find_next` actually skipped at least one code (returned
/// `Some(next)` with `next > pos`).
#[cfg(feature = "shufti-counters")]
pub static SHUFTI_SKIP_FIRED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Total codes skipped across all fired skip calls.
#[cfg(feature = "shufti-counters")]
pub static SHUFTI_CODES_SKIPPED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Baseline (state-0 skip only) — preserved for benchmarking
// ---------------------------------------------------------------------------

/// Flat `u8` transition table DFA for contains matching — baseline implementation.
///
/// Uses a state-0-only skip strategy (memchr or bitmap). Preserved for side-by-side
/// benchmarking against the shufti variant.
pub(crate) struct FlatContainsDfaBaseline {
    transitions: Vec<u8>,
    escape_transitions: Vec<u8>,
    accept_state: u8,
    sentinel: u8,
    skip: SkipStrategy,
}

impl FlatContainsDfaBaseline {
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
            .vortex_expect("FlatContainsDfaBaseline: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        Ok(Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            sentinel,
            skip,
        })
    }

    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            if state == 0 {
                match self.skip.find_next_progressing(codes, pos) {
                    Some(next) => pos = next,
                    None => return false,
                }
            }

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

// ---------------------------------------------------------------------------
// Shufti variant (per-state skip)
// ---------------------------------------------------------------------------

/// Flat `u8` transition table DFA for contains matching — shufti-accelerated variant.
///
/// Generalises the state-0 skip to ALL DFA states: at any state `s`, we use a
/// Hyperscan-style shufti classifier to skip 16-byte chunks of boring codes with
/// two `PSHUFB` instructions (SSSE3), then take a single scalar DFA step on each
/// interesting code.
pub(crate) struct FlatContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    transitions: Vec<u8>,
    /// `escape_transitions[state * 256 + byte]` -> next state for escaped bytes.
    escape_transitions: Vec<u8>,
    accept_state: u8,
    sentinel: u8,
    /// Per-state shufti masks. `shufti[s]` classifies codes at state `s`.
    shufti: Vec<ShuftiMask>,
    /// Number of DFA states (excluding accept, which is sticky and never needs a skip).
    n_states: u8,
}

impl FlatContainsDfa {
    /// Maximum needle length: need accept + sentinel to fit in u8.
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
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        // Build per-state shufti masks for all non-accept states.
        let mut shufti = Vec::with_capacity(usize::from(n_states));
        for state in 0..n_states {
            let row_start = usize::from(state) * 256;
            shufti.push(ShuftiMask::from_transition_row(
                &transitions[row_start..row_start + 256],
                state,
            ));
        }

        Ok(Self {
            transitions,
            escape_transitions: byte_table,
            accept_state,
            sentinel,
            shufti,
            n_states,
        })
    }

    /// Run the DFA with per-state shufti skip on the given code stream.
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            // Per-state shufti skip: jump over codes that leave `state` unchanged.
            // The accept state is sticky and is checked immediately after each step,
            // so we only need skips for states < accept_state.
            if state < self.n_states {
                #[cfg(feature = "shufti-counters")]
                SHUFTI_SKIP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                match self.shufti[usize::from(state)].find_next(codes, pos) {
                    Some(next) => {
                        #[cfg(feature = "shufti-counters")]
                        if next > pos {
                            SHUFTI_SKIP_FIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            SHUFTI_CODES_SKIPPED.fetch_add(
                                (next - pos) as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        pos = next;
                    }
                    None => return false,
                }
            }

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

/// Find the best "anchor" symbol for the memchr prefilter.
///
/// Scans all symbols to find one whose expansion is the longest substring of
/// the needle. Returns `None` if no multi-byte symbol matches.
#[allow(dead_code)]
fn find_anchor_symbol(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Option<u8> {
    if needle.is_empty() {
        return None;
    }

    let n_symbols = symbols.len();
    let mut best_code: Option<u8> = None;
    let mut best_len: usize = 0;

    for code in 0..n_symbols {
        let sym_bytes = symbols[code].to_u64().to_le_bytes();
        let sym_len = usize::from(symbol_lengths[code]);
        if sym_len == 0 || sym_len > 8 || sym_len <= best_len || sym_len > needle.len() {
            continue;
        }
        let expansion = &sym_bytes[..sym_len];

        for start in 0..=needle.len() - sym_len {
            if &needle[start..start + sym_len] == expansion {
                best_len = sym_len;
                best_code = u8::try_from(code).ok();
                break;
            }
        }
    }

    if best_len >= 2 { best_code } else { None }
}
