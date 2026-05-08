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
//! ## Byte-class minimization (Variant B)
//!
//! `FlatContainsDfaClasses` collapses equivalent code bytes into classes,
//! shrinking the transition table from `n_states × 256` to `n_states × n_classes`.
//! Two code bytes are equivalent iff they produce the same successor in **every**
//! state. The inner loop adds one indirection (`code_to_class[code]`) but the
//! transition table is typically 5–15× smaller — so for long needles the table
//! fits in L1 even when the baseline doesn't.
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

// ---------------------------------------------------------------------------
// Variant E: escape-folded flat DFA (no sentinel branch in the inner loop)
// ---------------------------------------------------------------------------

/// Flat transition table DFA where the escape sentinel is folded into the
/// state space, eliminating the per-step escape branch from the inner loop.
///
/// State layout (`N` = needle length, only valid when `2N + 1 ≤ 256`):
///
/// - States `0..N-1`: progress states (no escape pending, like the baseline)
/// - State `N`: accept (sticky)
/// - States `N+1..2N`: post-escape companions for progress states `0..N-1`.
///   State `N+1+s` means "we just consumed an escape code from progress state
///   `s`; the next code byte is a literal".
///
/// Transitions:
/// - Progress state `s`, code `0..ESCAPE_CODE`: same as the baseline symbol-level
///   transition (advance the byte-DFA over the symbol's bytes).
/// - Progress state `s`, `ESCAPE_CODE`: → post-escape state `N+1+s`.
/// - Accept state `N`, any code: → `N` (sticky).
/// - Post-escape state `N+1+s`, any byte `b`: → `byte_table[s][b]`. This is
///   the byte-level DFA target after reading one literal byte from progress
///   state `s` — including byte 255 which here is a literal, not an escape.
///
/// Inner loop becomes one lookup + accept check, no sentinel branch:
/// ```text
/// state = transitions[state * 256 + code];
/// if state == accept { return true; }
/// ```
pub(crate) struct FlatContainsDfaEscapeFolded {
    transitions: Vec<u8>,
    accept_state: u8,
    n_states: u16,
    skip: SkipStrategy,
}

impl FlatContainsDfaEscapeFolded {
    /// Maximum needle length: 2N + 1 must fit in u8, so N ≤ 127.
    pub(crate) const MAX_NEEDLE_LEN: usize = 127;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for escape-folded contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }

        let n = needle.len();
        let accept_state = u8::try_from(n)
            .vortex_expect("FlatContainsDfaEscapeFolded: accept state must fit into u8");
        let total_states = 2 * n + 1;
        let n_states = u16::try_from(total_states)
            .vortex_expect("FlatContainsDfaEscapeFolded: 2N+1 must fit in u16");

        let byte_table = kmp_byte_transitions(needle);

        // Build the standard fused 256-wide table for progress states 0..n
        // (n+1 states: progress 0..n-1 + accept n). Code 255 maps to the
        // post-escape state for that progress state via escape_value_fn.
        let progress_states = accept_state + 1;
        let sym_trans = build_symbol_transitions(
            symbols,
            symbol_lengths,
            &byte_table,
            progress_states,
            accept_state,
        );

        let mut transitions = vec![0u8; total_states * 256];

        // Fill progress states 0..accept_state (inclusive of accept).
        // The accept row stays sticky in build_fused_table because the symbol
        // pass maps every symbol to accept once we're there.
        let escape_target = |state: u8| -> u8 {
            if state == accept_state {
                accept_state
            } else {
                u8::try_from(usize::from(accept_state) + 1 + usize::from(state))
                    .vortex_expect("post-escape state fits in u8 (N ≤ 127)")
            }
        };
        let progress_table = build_fused_table(
            &sym_trans,
            symbols.len(),
            progress_states,
            escape_target,
            0,
        );
        transitions[..progress_table.len()].copy_from_slice(&progress_table);

        // Post-escape states: post_s = accept_state + 1 + s (for s in 0..accept_state).
        // For any byte b, transition to byte_table[s][b].
        for s in 0..accept_state {
            let post = usize::from(accept_state) + 1 + usize::from(s);
            for b in 0..256usize {
                transitions[post * 256 + b] = byte_table[usize::from(s) * 256 + b];
            }
        }

        // State-0 skip is on the raw code alphabet, same as baseline.
        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        Ok(Self {
            transitions,
            accept_state,
            n_states,
            skip,
        })
    }

    /// Total number of DFA states (`2N + 1`).
    pub(crate) fn n_states(&self) -> u16 {
        self.n_states
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
            state = self.transitions[usize::from(state) * 256 + usize::from(code)];
            if state == self.accept_state {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Variant B: byte-class minimization
// ---------------------------------------------------------------------------

/// Flat transition table DFA for contains matching with byte-class minimization.
///
/// Collapses code bytes that produce the same successor in every state into
/// equivalence classes, shrinking the transition table from `n_states * 256` to
/// `n_states * n_classes`. Inner loop adds a `code_to_class` indirection.
pub(crate) struct FlatContainsDfaClasses {
    /// Compact transitions: `class_trans[state * n_classes + class]` -> next state.
    class_trans: Vec<u8>,
    /// Escape-byte transition table; uses raw byte alphabet (no class compression).
    escape_transitions: Vec<u8>,
    /// `code_to_class[code]` -> class id.
    code_to_class: [u8; 256],
    accept_state: u8,
    sentinel: u8,
    n_classes: u16,
    /// State-0 skip strategy on the raw code alphabet (same as baseline).
    skip: SkipStrategy,
}

impl FlatContainsDfaClasses {
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
            .vortex_expect("FlatContainsDfaClasses: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);

        let (code_to_class, n_classes, class_trans) = compute_byte_classes(&transitions, n_states);

        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        Ok(Self {
            class_trans,
            escape_transitions: byte_table,
            code_to_class,
            accept_state,
            sentinel,
            n_classes,
            skip,
        })
    }

    /// Number of equivalence classes (1..=256).
    pub(crate) fn n_classes(&self) -> u16 {
        self.n_classes
    }

    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        let n_classes = usize::from(self.n_classes);
        while pos < codes.len() {
            if state == 0 {
                match self.skip.find_next_progressing(codes, pos) {
                    Some(next) => pos = next,
                    None => return false,
                }
            }

            let code = codes[pos];
            pos += 1;
            let class = self.code_to_class[usize::from(code)];
            let next = self.class_trans[usize::from(state) * n_classes + usize::from(class)];
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
// Variant C: byte-class minimization + bulk pre-classify
// ---------------------------------------------------------------------------

/// Like [`FlatContainsDfaClasses`] but the inner DFA loop reads pre-classified
/// bytes from a parallel buffer instead of looking up `code_to_class[code]` per
/// step. The classification pass is run once over `all_bytes` ahead of the
/// per-string scan, so the cost is amortized across all strings sharing the
/// same code buffer.
///
/// Trade-off: the bulk pre-classify pass touches every byte in `all_bytes`,
/// even ones the per-string state-0 skip would otherwise jump over. It pays
/// off only when the DFA inner loop is the bottleneck (dense partial matches),
/// not when the corpus is dominated by skip-able codes.
pub(crate) struct FlatContainsDfaClassesPre {
    class_trans: Vec<u8>,
    escape_transitions: Vec<u8>,
    code_to_class: [u8; 256],
    accept_state: u8,
    sentinel: u8,
    n_classes: u16,
    skip: SkipStrategy,
}

impl FlatContainsDfaClassesPre {
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
            .vortex_expect("FlatContainsDfaClassesPre: accept state must fit into u8");
        let n_states = accept_state + 1;
        let sentinel = n_states;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);
        let transitions = build_fused_table(&sym_trans, symbols.len(), n_states, |_| sentinel, 0);
        let (code_to_class, n_classes, class_trans) = compute_byte_classes(&transitions, n_states);
        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        Ok(Self {
            class_trans,
            escape_transitions: byte_table,
            code_to_class,
            accept_state,
            sentinel,
            n_classes,
            skip,
        })
    }

    /// Bulk-classify a raw byte buffer into a parallel class stream.
    ///
    /// `out[i] = code_to_class[raw_bytes[i]]`. Auto-vectorized by rustc into
    /// PSHUFB-style gathers in release builds.
    pub(crate) fn classify_bulk(&self, raw_bytes: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; raw_bytes.len()];
        for (o, &b) in out.iter_mut().zip(raw_bytes.iter()) {
            *o = self.code_to_class[usize::from(b)];
        }
        out
    }

    /// Run the DFA on a slice of pre-classified bytes alongside the matching
    /// slice of raw bytes (raw bytes only consulted on escape).
    pub(crate) fn matches_pre(&self, classified: &[u8], raw_bytes: &[u8]) -> bool {
        debug_assert_eq!(classified.len(), raw_bytes.len());
        let mut state = 0u8;
        let mut pos = 0;
        let n_classes = usize::from(self.n_classes);
        while pos < classified.len() {
            if state == 0 {
                match self.skip.find_next_progressing(raw_bytes, pos) {
                    Some(next) => pos = next,
                    None => return false,
                }
            }

            let class = classified[pos];
            pos += 1;
            let next = self.class_trans[usize::from(state) * n_classes + usize::from(class)];
            if next == self.sentinel {
                if pos >= raw_bytes.len() {
                    return false;
                }
                let b = raw_bytes[pos];
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

/// Compute byte equivalence classes from a 256-wide transition table.
///
/// Two code bytes are equivalent if for every state they map to the same
/// successor state. Returns `(code_to_class, n_classes, class_trans)` where
/// `class_trans` is laid out as `[state * n_classes + class]`.
fn compute_byte_classes(transitions: &[u8], n_states: u8) -> ([u8; 256], u16, Vec<u8>) {
    use std::collections::HashMap;

    let n = usize::from(n_states);
    let mut code_to_class = [0u8; 256];
    let mut class_columns: Vec<Vec<u8>> = Vec::new();
    let mut map: HashMap<Vec<u8>, u8> = HashMap::new();

    for code in 0..256usize {
        let column: Vec<u8> = (0..n).map(|s| transitions[s * 256 + code]).collect();
        let class_id = if let Some(&existing) = map.get(&column) {
            existing
        } else {
            let id = u8::try_from(class_columns.len())
                .vortex_expect("byte-class id fits in u8 (at most 256 classes)");
            map.insert(column.clone(), id);
            class_columns.push(column);
            id
        };
        code_to_class[code] = class_id;
    }

    let n_classes = u16::try_from(class_columns.len()).vortex_expect("n_classes ≤ 256");
    let n_classes_usize = class_columns.len();
    let mut class_trans = vec![0u8; n * n_classes_usize];
    for (class_idx, col) in class_columns.iter().enumerate() {
        for (s, &v) in col.iter().enumerate() {
            class_trans[s * n_classes_usize + class_idx] = v;
        }
    }
    (code_to_class, n_classes, class_trans)
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
