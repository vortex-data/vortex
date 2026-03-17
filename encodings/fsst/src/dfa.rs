// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # FSST LIKE Pushdown via DFA Construction
//!
//! This module implements DFA-based pattern matching directly on FSST-compressed
//! strings, without decompressing them. It handles two pattern shapes:
//!
//! - **Prefix**: `'prefix%'`  — matches strings starting with a literal prefix.
//! - **Contains**: `'%needle%'` — matches strings containing a literal substring.
//!
//! Pushdown is intentionally conservative. If the pattern shape is unsupported,
//! or if the pattern exceeds the DFA's representable state space, construction
//! returns `None` and the caller must fall back to ordinary decompression-based
//! LIKE evaluation.
//!
//! TODO(joe): suffix (`'%suffix'`) pushdown. Two approaches:
//! - **Forward DFA**: use a non-sticky accept state with KMP fallback transitions,
//!   check `state == accept` after processing all codes. Branchless and vectorizable.
//! - **Backward scan**: walk the compressed code stream in reverse, comparing symbol
//!   bytes from the end. Simpler, no DFA construction, but requires reverse parsing
//!   of the FSST escape mechanism.
//!
//! ## Background: FSST Encoding
//!
//! [FSST](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) compresses strings by
//! replacing frequent byte sequences with single-byte **symbol codes** (0–254). Code
//! byte 255 is reserved as the **escape code**: the next byte is a literal (uncompressed)
//! byte. So a compressed string is a stream of:
//!
//! ```text
//! [symbol_code] ... [symbol_code] [ESCAPE literal_byte] [symbol_code] ...
//! ```
//!
//! A single symbol can expand to 1–8 bytes. Matching on compressed codes requires
//! the DFA to handle multi-byte symbol expansions and the escape mechanism.
//!
//! ## The Algorithm: KMP → Byte Table → Symbol Table → Packed DFA
//!
//! Construction proceeds through four stages:
//!
//! ### Stage 1: KMP Failure Function
//!
//! We compute the standard [KMP](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)
//! failure function for the needle bytes. This tells us, on a mismatch at
//! position `i`, the longest proper prefix of `needle[0..i]` that is also a
//! suffix — i.e., where to resume matching instead of starting over.
//!
//! ```text
//! Needle: "abcabd"
//! Failure: [0, 0, 0, 1, 2, 0]
//!                      ^  ^
//!                      At position 3 ('a'), the prefix "a" matches suffix "a"
//!                      At position 4 ('b'), the prefix "ab" matches suffix "ab"
//! ```
//!
//! ### Stage 2: Byte-Level Transition Table
//!
//! From the failure function, we build a full `(state × byte) → state` transition
//! table. State `i` means "we have matched `needle[0..i]`". State `n` (= needle
//! length) is the **accept** state.
//!
//! ```text
//! Needle: "aba"  (3 states + accept)
//!
//!         Input byte
//! State   'a'    'b'    other
//! ─────   ────   ────   ─────
//!   0      1      0      0      ← looking for first 'a'
//!   1      1      2      0      ← matched "a", want 'b'
//!   2      3✓     0      0      ← matched "ab", want 'a'
//!   3✓     3✓     3✓     3✓     ← accept (sticky)
//! ```
//!
//! For prefix matching, a mismatch at any state goes to a **fail** state (no
//! fallback). For contains matching, mismatches follow KMP fallback transitions
//! so we can find the needle anywhere in the string.
//!
//! ### Stage 3: Symbol-Level Transition Table
//!
//! FSST symbols can be multi-byte. To compute the transition for symbol code `c`
//! in state `s`, we simulate feeding each byte of the symbol through the byte
//! table:
//!
//! ```text
//! Symbol #42 = "the" (3 bytes)
//! State 0 + 't' → 0, + 'h' → 0, + 'e' → 0  ⟹ sym_trans[0][42] = 0
//!
//! If needle = "them":
//! State 0 + 't' → 1, + 'h' → 2, + 'e' → 3  ⟹ sym_trans[0][42] = 3
//! ```
//!
//! We then build a **fused 256-wide table**: for code bytes 0–254, use the
//! symbol transition; for code byte 255 (ESCAPE_CODE), transition to a
//! special sentinel that tells the scanner to read the next literal byte.
//!
//! ### Stage 4: Packing into the Final Representation
//!
//! The fused table can be stored in different layouts depending on the number
//! of states:
//!
//! - **Shift-packed `u64`** (≤16 states): Each state needs 4 bits. All state
//!   transitions for one input byte fit in a single `u64`. Lookup:
//!   `next = (table[byte] >> (state * 4)) & 0xF`. One cache line per lookup.
//!
//! - **Flat `u8` table** (≤255 states): `transitions[state * 256 + byte]`.
//!   Larger, but still bounded by the `u8` state representation.
//!
//! ## State-Space Limits
//!
//! The public behavior is shaped by two implementation limits, both measured in
//! pattern **bytes** rather than Unicode scalar values:
//!
//! - `prefix%` pushdown is limited to **13 bytes**. The packed prefix DFA uses
//!   4-bit state ids and needs room for normal prefix-progress states, an
//!   accept state, a fail state, and one escape sentinel for FSST literals.
//! - `%needle%` pushdown is limited to **254 bytes**. The long-needle DFA stores
//!   states in `u8`, so it needs room for every match-progress state plus both
//!   the accept state and the escape sentinel.
//!
//! Patterns beyond those limits are still valid LIKE patterns; they simply do
//! not use FSST pushdown and must be evaluated through the fallback path.
//!
//! ## DFA Variants and When Each Is Used
//!
//! ```text
//! ┌───────────────┬──────────────────────────────────────────────────────┐
//! │ Pattern       │ Needle length → DFA variant                        │
//! ├───────────────┼──────────────────────────────────────────────────────┤
//! │ prefix%       │ 0–13 → FsstPrefixDfa (shift-packed, no KMP)        │
//! ├───────────────┼──────────────────────────────────────────────────────┤
//! │ %needle%      │ 1–7    → BranchlessShiftDfa (hierarchical 4-byte)  │
//! │               │ 8–14   → FlatBranchlessDfa (flat u8, escape-folded)│
//! │               │ 15–254 → FusedDfa (escape sentinel)                │
//! └───────────────┴──────────────────────────────────────────────────────┘
//! ```
//!
//! ## Escape Handling Strategies
//!
//! There are two ways to handle the FSST escape code in the DFA:
//!
//! **Escape sentinel** (used by `ShiftDfa`, `FusedDfa`, `FsstPrefixDfa`):
//! The escape code maps to a sentinel state. The scanner checks for it and
//! reads the next byte from a separate escape transition table.
//!
//! ```text
//! loop:
//!   state = transitions[byte]       // might be sentinel
//!   if state == SENTINEL:
//!     state = escape_transitions[next_byte]  // branch
//! ```
//!
//! **Escape folding** (used by `BranchlessShiftDfa`, `FlatBranchlessDfa`):
//! Escape states are folded into the state space. State `s+N+1` means "was in
//! state `s`, just consumed ESCAPE_CODE". The next byte's transition from an
//! escape state uses the byte-level table. No branch needed in the scanner.
//!
//! ```text
//! States: [0..N-1: normal] [N: accept] [N+1..2N: escape shadows]
//! Total: 2N+1 states. With 4-bit packing, max N=7.
//!
//! loop:
//!   state = transitions[state][byte]   // branchless!
//! ```

#![allow(clippy::cast_possible_truncation)]

use fsst::ESCAPE_CODE;
use fsst::Symbol;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

// ---------------------------------------------------------------------------
// FsstMatcher — unified public API
// ---------------------------------------------------------------------------

/// A compiled matcher for LIKE patterns on FSST-compressed strings.
///
/// Encapsulates pattern parsing and DFA variant selection. Returns `None` from
/// [`try_new`](Self::try_new) for patterns that cannot be evaluated without
/// decompression (e.g., `_` wildcards, multiple `%` in non-standard positions,
/// or patterns that exceed the DFA's representable byte-length limits).
pub(crate) struct FsstMatcher {
    inner: MatcherInner,
}

enum MatcherInner {
    MatchAll,
    Prefix(Box<FsstPrefixDfa>),
    ContainsBranchless(Box<BranchlessShiftDfa>),
    ContainsFlat(FlatBranchlessDfa),
    Contains(FsstContainsDfa),
}

impl FsstMatcher {
    /// Try to build a matcher for the given LIKE pattern.
    ///
    /// Returns `Ok(None)` if the pattern shape is not supported for pushdown
    /// (e.g. `_` wildcards, multiple non-bookend `%`, `prefix%` longer than
    /// 13 bytes, or `%needle%` longer than 254 bytes).
    pub(crate) fn try_new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        pattern: &str,
    ) -> VortexResult<Option<Self>> {
        let Some(like_kind) = LikeKind::parse(pattern) else {
            return Ok(None);
        };

        let inner = match like_kind {
            LikeKind::Prefix("") => MatcherInner::MatchAll,
            LikeKind::Prefix(prefix) => {
                let prefix = prefix.as_bytes();
                if prefix.len() > FsstPrefixDfa::MAX_PREFIX_LEN {
                    return Ok(None);
                }
                MatcherInner::Prefix(Box::new(FsstPrefixDfa::new(
                    symbols,
                    symbol_lengths,
                    prefix,
                )))
            }
            LikeKind::Contains(needle) => {
                let needle = needle.as_bytes();
                if needle.len() > FusedDfa::MAX_NEEDLE_LEN {
                    return Ok(None);
                }
                if needle.len() <= BranchlessShiftDfa::MAX_NEEDLE_LEN {
                    MatcherInner::ContainsBranchless(Box::new(BranchlessShiftDfa::new(
                        symbols,
                        symbol_lengths,
                        needle,
                    )))
                } else if needle.len() <= FlatBranchlessDfa::MAX_NEEDLE_LEN {
                    MatcherInner::ContainsFlat(FlatBranchlessDfa::new(
                        symbols,
                        symbol_lengths,
                        needle,
                    ))
                } else {
                    MatcherInner::Contains(FsstContainsDfa::new(symbols, symbol_lengths, needle))
                }
            }
        };

        Ok(Some(Self { inner }))
    }

    /// Run the matcher on a single FSST-compressed code sequence.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match &self.inner {
            MatcherInner::MatchAll => true,
            MatcherInner::Prefix(dfa) => dfa.matches(codes),
            MatcherInner::ContainsBranchless(dfa) => dfa.matches(codes),
            MatcherInner::ContainsFlat(dfa) => dfa.matches(codes),
            MatcherInner::Contains(dfa) => dfa.matches(codes),
        }
    }
}

/// The subset of LIKE patterns we can handle without decompression.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(&'a str),
    /// `%needle%`
    Contains(&'a str),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a str) -> Option<Self> {
        if pattern == "%" {
            return Some(LikeKind::Prefix(""));
        }

        // Find first wildcard.
        let first_wild = pattern.find(['%', '_'])?;

        // `_` as first wildcard means we can't handle it.
        if pattern.as_bytes()[first_wild] == b'_' {
            return None;
        }

        // `prefix%` — single trailing %
        if first_wild > 0 && &pattern[first_wild..] == "%" {
            return Some(LikeKind::Prefix(&pattern[..first_wild]));
        }

        // `%needle%` — leading and trailing %, no inner wildcards
        if first_wild == 0
            && pattern.len() > 2
            && pattern.as_bytes()[pattern.len() - 1] == b'%'
            && !pattern[1..pattern.len() - 1].contains(['%', '_'])
        {
            return Some(LikeKind::Contains(&pattern[1..pattern.len() - 1]));
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Scan helper
// ---------------------------------------------------------------------------

/// Scan all strings through a DFA matcher, packing results directly into a
/// `BitBuffer` one u64 word (64 strings) at a time. This avoids the overhead
/// of `BitBufferMut::collect_bool`'s cross-crate closure indirection and
/// guarantees the compiler can see the full loop body for optimization.
// TODO: add N-way ILP overrun scan for higher throughput on short strings.
#[inline]
pub(crate) fn dfa_scan_to_bitbuf<T, F>(
    n: usize,
    offsets: &[T],
    all_bytes: &[u8],
    negated: bool,
    matcher: F,
) -> BitBuffer
where
    T: vortex_array::dtype::IntegerPType,
    F: Fn(&[u8]) -> bool,
{
    let n_words = n / 64;
    let remainder = n % 64;
    let mut words: BufferMut<u64> = BufferMut::with_capacity(n.div_ceil(64));

    for chunk in 0..n_words {
        let base = chunk * 64;
        let mut word = 0u64;
        let mut start: usize = offsets[base].as_();
        for bit in 0..64 {
            let end: usize = offsets[base + bit + 1].as_();
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
            start = end;
        }
        // SAFETY: we allocated capacity for n.div_ceil(64) words.
        unsafe { words.push_unchecked(word) };
    }

    if remainder != 0 {
        let base = n_words * 64;
        let mut word = 0u64;
        let mut start: usize = offsets[base].as_();
        for bit in 0..remainder {
            let end: usize = offsets[base + bit + 1].as_();
            word |= ((matcher(&all_bytes[start..end]) != negated) as u64) << bit;
            start = end;
        }
        unsafe { words.push_unchecked(word) };
    }

    BitBuffer::new(words.into_byte_buffer().freeze(), n)
}

// ---------------------------------------------------------------------------
// Shared DFA construction helpers
// ---------------------------------------------------------------------------

/// Builds the per-symbol transition table for FSST symbols.
///
/// For each `(state, symbol_code)` pair, simulates feeding the symbol's bytes
/// through the byte-level transition table to compute the resulting state.
///
/// Returns a flat `Vec<u8>` indexed as `[state * n_symbols + code]`.
fn build_symbol_transitions(
    symbols: &[Symbol],
    symbol_lengths: &[u8],
    byte_table: &[u16],
    n_states: usize,
    accept_state: u8,
) -> Vec<u8> {
    let n_symbols = symbols.len();
    let mut sym_trans = vec![0u8; n_states * n_symbols];
    for state in 0..n_states {
        for code in 0..n_symbols {
            if state as u8 == accept_state {
                sym_trans[state * n_symbols + code] = accept_state;
                continue;
            }
            let sym = symbols[code].to_u64().to_le_bytes();
            let sym_len = symbol_lengths[code] as usize;
            let mut s = state as u16;
            for &b in &sym[..sym_len] {
                if s == accept_state as u16 {
                    break;
                }
                s = byte_table[s as usize * 256 + b as usize];
            }
            sym_trans[state * n_symbols + code] = s as u8;
        }
    }
    sym_trans
}

/// Builds a fused 256-wide transition table from symbol transitions.
///
/// For each `(state, code_byte)`:
/// - Code bytes `0..n_symbols`: use the symbol transition
/// - `ESCAPE_CODE`: maps to `escape_value` (either a sentinel or escape state)
/// - All others: use `default` (typically 0 for contains, fail_state for prefix)
///
/// Returns a flat `Vec<u8>` indexed as `[state * 256 + code_byte]`.
fn build_fused_table(
    sym_trans: &[u8],
    n_symbols: usize,
    n_states: usize,
    escape_value_fn: impl Fn(usize) -> u8,
    default: u8,
) -> Vec<u8> {
    let mut fused = vec![default; n_states * 256];
    for state in 0..n_states {
        for code in 0..n_symbols {
            fused[state * 256 + code] = sym_trans[state * n_symbols + code];
        }
        fused[state * 256 + ESCAPE_CODE as usize] = escape_value_fn(state);
    }
    fused
}

/// Packs a fused table into shift-encoded `u64` arrays.
///
/// Each `u64` encodes transitions for ALL states for one input byte.
/// Lookup: `next = (table[byte] >> (state * BITS)) & MASK`.
fn pack_shift_table(fused: &[u8], n_states: usize, bits: u32) -> [u64; 256] {
    let mut packed = [0u64; 256];
    for code_byte in 0..256usize {
        let mut val = 0u64;
        for state in 0..n_states {
            val |= (fused[state * 256 + code_byte] as u64) << (state as u32 * bits);
        }
        packed[code_byte] = val;
    }
    packed
}

/// Packs a byte-level KMP table into shift-encoded `u64` arrays for escape handling.
fn pack_escape_shift_table(byte_table: &[u16], n_states: usize, bits: u32) -> [u64; 256] {
    let mut packed = [0u64; 256];
    for byte_val in 0..256usize {
        let mut val = 0u64;
        for state in 0..n_states {
            let next = byte_table[state * 256 + byte_val] as u8;
            val |= (next as u64) << (state as u32 * bits);
        }
        packed[byte_val] = val;
    }
    packed
}

// ---------------------------------------------------------------------------
// DFA for prefix matching (LIKE 'prefix%')
// ---------------------------------------------------------------------------

/// Precomputed shift-based DFA for prefix matching on FSST codes.
///
/// States 0..prefix_len track match progress, plus ACCEPT and FAIL.
/// Uses the same shift-based approach as the contains DFA: all state
/// transitions packed into a `u64` per code byte. For prefixes longer
/// than 13 characters, pushdown is disabled and LIKE falls back.
struct FsstPrefixDfa {
    /// Packed transitions: `(table[code] >> (state * 4)) & 0xF` gives next state.
    transitions: [u64; 256],
    /// Packed escape transitions for literal bytes.
    escape_transitions: [u64; 256],
    accept_state: u8,
    fail_state: u8,
}

impl FsstPrefixDfa {
    pub(crate) const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;
    const MAX_PREFIX_LEN: usize = (1 << Self::BITS) as usize - 3;

    pub(crate) fn new(symbols: &[Symbol], symbol_lengths: &[u8], prefix: &[u8]) -> Self {
        // Need room for states 0..prefix_len, accept, fail, and an escape sentinel.
        debug_assert!(prefix.len() <= Self::MAX_PREFIX_LEN);

        let accept_state = prefix.len() as u8;
        let fail_state = prefix.len() as u8 + 1;
        let n_states = prefix.len() + 2;

        // Prefix matching uses a simpler transition rule than KMP: on mismatch
        // we go to fail_state (no fallback). Build the byte table inline.
        let byte_table = Self::build_prefix_byte_table(prefix, accept_state, fail_state);

        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);

        // Override fail_state rows: fail is sticky.
        let escape_sentinel = fail_state + 1;
        let mut fused = build_fused_table(
            &sym_trans,
            symbols.len(),
            n_states,
            |_| escape_sentinel,
            fail_state,
        );

        // Accept state is sticky for all inputs.
        for code_byte in 0..256usize {
            fused[accept_state as usize * 256 + code_byte] = accept_state;
        }
        // Fail state is sticky for all inputs.
        for code_byte in 0..256usize {
            fused[fail_state as usize * 256 + code_byte] = fail_state;
        }

        let transitions = pack_shift_table(&fused, n_states, Self::BITS);

        // Build escape transitions from the byte table.
        let mut esc_trans = vec![fail_state; n_states * 256];
        for state in 0..n_states {
            if state as u8 == accept_state {
                for b in 0..256 {
                    esc_trans[state * 256 + b] = accept_state;
                }
            } else if state as u8 != fail_state {
                for b in 0..256usize {
                    if b as u8 == prefix[state] {
                        let next = state + 1;
                        esc_trans[state * 256 + b] = if next >= prefix.len() {
                            accept_state
                        } else {
                            next as u8
                        };
                    }
                }
            }
        }
        let escape_transitions = pack_shift_table(&esc_trans, n_states, Self::BITS);

        Self {
            transitions,
            escape_transitions,
            accept_state,
            fail_state,
        }
    }

    /// Build a byte-level transition table for prefix matching (no KMP fallback).
    fn build_prefix_byte_table(prefix: &[u8], accept_state: u8, fail_state: u8) -> Vec<u16> {
        let n_states = prefix.len() + 2;
        let mut table = vec![fail_state as u16; n_states * 256];

        for state in 0..n_states {
            if state as u8 == accept_state {
                for byte in 0..256 {
                    table[state * 256 + byte] = accept_state as u16;
                }
            } else if state as u8 != fail_state {
                // Only the correct next byte advances; everything else fails.
                let next_byte = prefix[state];
                let next_state = if state + 1 >= prefix.len() {
                    accept_state as u16
                } else {
                    (state + 1) as u16
                };
                table[state * 256 + next_byte as usize] = next_state;
            }
        }
        table
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let packed = self.transitions[code as usize];
            let next = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if next == self.fail_state + 1 {
                // Escape sentinel: read literal byte.
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                let esc_packed = self.escape_transitions[b as usize];
                state = ((esc_packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            } else {
                state = next;
            }
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

// ---------------------------------------------------------------------------
// DFA for contains matching (LIKE '%needle%')
// ---------------------------------------------------------------------------

/// Contains DFA dispatch for long needles (>14 bytes). Short needles (len <= 7)
/// are handled by `BranchlessShiftDfa`, medium needles (8-14) by
/// `FlatBranchlessDfa`, and longer supported needles (15-254) by `FusedDfa`.
enum FsstContainsDfa {
    /// Retained internal alternative; not currently selected by `FsstMatcher`.
    Shift(Box<ShiftDfa>),
    /// Fused u8 table DFA for long needles (15-254 bytes).
    Fused(FusedDfa),
}

impl FsstContainsDfa {
    pub(crate) fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        if needle.len() <= ShiftDfa::MAX_NEEDLE_LEN {
            FsstContainsDfa::Shift(Box::new(ShiftDfa::new(symbols, symbol_lengths, needle)))
        } else {
            FsstContainsDfa::Fused(FusedDfa::new(symbols, symbol_lengths, needle))
        }
    }

    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        match self {
            FsstContainsDfa::Shift(dfa) => dfa.matches(codes),
            FsstContainsDfa::Fused(dfa) => dfa.matches(codes),
        }
    }
}

/// Branchless escape-folded DFA for short needles (len <= 7).
///
/// Folds escape handling into the state space so that `matches()` is
/// completely branchless (except for loop control). The state layout is:
/// - States 0..N-1: normal match-progress states
/// - State N: accept (sticky for all inputs)
/// - States N+1..2N: escape states (state `s+N+1` means "was in state `s`,
///   just consumed ESCAPE_CODE")
///
/// Total states: 2N+1. With 4-bit packing, max N=7.
///
/// Uses a decomposed hierarchical lookup that processes 4 code bytes per
/// loop iteration with only ~3 KB of tables:
///
/// 1. **Equivalence class table** (256 B): maps each code byte to a class
///    id. Bytes with identical transition u64s share a class -- typically
///    only ~6-10 classes exist (needle chars + escape + "miss-all").
/// 2. **Pair-compose table** (~N^2 B): maps `(class0, class1)` to a 2-byte
///    palette index. Typically ~36 entries.
/// 3. **4-byte compose table** (~M^2 x 8 B): maps `(palette0, palette1)` to
///    the composed packed u64 for all 4 bytes. Typically ~81 entries = 648 B.
///
/// Each loop iteration: 4 class lookups (parallel, 256 B table) -> 2
/// pair-compose lookups (parallel, ~36 B table) -> 1 compose lookup
/// (~648 B table) -> 1 shift+mask. All tables fit in L1 cache.
struct BranchlessShiftDfa {
    /// Maps each code byte to its equivalence class. Bytes with the same
    /// packed transition u64 share a class. (256 bytes)
    eq_class: [u8; 256],
    /// Maps `(class0 * n_classes + class1)` -> 2-byte palette index.
    pair_compose: Vec<u8>,
    /// Number of equivalence classes (stride for pair_compose).
    n_classes: usize,
    /// Maps `(palette0 * n_palette + palette1)` -> composed packed u64
    /// for 4 bytes.
    compose_4b: Vec<u64>,
    /// Number of unique 2-byte palette entries (stride for compose_4b).
    n_palette: usize,
    /// 1-byte fallback transitions for trailing bytes.
    transitions_1b: [u64; 256],
    /// 2-byte palette for the remainder path (2-3 trailing bytes).
    palette_2b: Vec<u64>,
    accept_state: u8,
}

impl BranchlessShiftDfa {
    const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;
    /// Maximum needle length: need 2N+1 states to fit in 16 slots (4 bits).
    /// 2*7+1 = 15 <= 16, so max N = 7.
    pub(crate) const MAX_NEEDLE_LEN: usize = 7;

    pub(crate) fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n = needle.len();
        debug_assert!(n <= Self::MAX_NEEDLE_LEN);

        let accept_state = n as u8;
        let total_states = 2 * n + 1;
        debug_assert!(total_states <= (1 << Self::BITS));

        let transitions_1b =
            Self::build_escape_folded_transitions(symbols, symbol_lengths, needle, total_states);

        // Build equivalence classes: group bytes with identical transition u64.
        let mut eq_class = [0u8; 256];
        let mut class_representatives: Vec<u64> = Vec::new();
        for byte_val in 0..256usize {
            let t = transitions_1b[byte_val];
            let cls = class_representatives
                .iter()
                .position(|&v| v == t)
                .unwrap_or_else(|| {
                    class_representatives.push(t);
                    class_representatives.len() - 1
                });
            eq_class[byte_val] = cls as u8;
        }
        let n_classes = class_representatives.len();

        // Build pair-compose: for each (class0, class1), compose the two
        // 1-byte transitions and deduplicate into a 2-byte palette.
        let (pair_compose, palette_2b) =
            Self::build_pair_compose(&class_representatives, n_classes, total_states);

        // Build 4-byte composition: compose_4b[p0 * n + p1] gives the packed
        // u64 for applying palette_2b[p0] then palette_2b[p1] in sequence.
        let n_palette = palette_2b.len();
        let compose_4b = Self::build_compose_4b(&palette_2b, total_states);

        Self {
            eq_class,
            pair_compose,
            n_classes,
            compose_4b,
            n_palette,
            transitions_1b,
            palette_2b,
            accept_state,
        }
    }

    /// Build the 1-byte packed transition table with escape handling folded
    /// into the state space (no branch needed in the scanner).
    fn build_escape_folded_transitions(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
        total_states: usize,
    ) -> [u64; 256] {
        let n = needle.len();
        let n_normal_states = n + 1;
        let accept_state = n as u8;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans = build_symbol_transitions(
            symbols,
            symbol_lengths,
            &byte_table,
            n_normal_states,
            accept_state,
        );

        // Build fused transition table with escape folding.
        let n_symbols = symbols.len();
        let mut fused = vec![0u8; total_states * 256];
        for code_byte in 0..256usize {
            for s in 0..n {
                if code_byte == ESCAPE_CODE as usize {
                    fused[s * 256 + code_byte] = (s + n + 1) as u8;
                } else if code_byte < n_symbols {
                    fused[s * 256 + code_byte] = sym_trans[s * n_symbols + code_byte];
                }
            }
            fused[n * 256 + code_byte] = accept_state;
            for s in 0..n {
                let esc_state = s + n + 1;
                let next = byte_table[s * 256 + code_byte] as u8;
                fused[esc_state * 256 + code_byte] = next;
            }
        }

        // Pack into u64 shift table.
        pack_shift_table(&fused, total_states, Self::BITS)
    }

    /// Build the pair-compose table and 2-byte palette from equivalence
    /// class representatives.
    fn build_pair_compose(
        class_reps: &[u64],
        n_classes: usize,
        total_states: usize,
    ) -> (Vec<u8>, Vec<u64>) {
        let mut pair_compose = vec![0u8; n_classes * n_classes];
        let mut palette_2b: Vec<u64> = Vec::new();

        for c0 in 0..n_classes {
            for c1 in 0..n_classes {
                let t0 = class_reps[c0];
                let t1 = class_reps[c1];
                let mut packed = 0u64;
                for state in 0..total_states {
                    let mid = ((t0 >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
                    let final_s = ((t1 >> (mid as u32 * Self::BITS)) & Self::MASK) as u8;
                    packed |= (final_s as u64) << (state as u32 * Self::BITS);
                }
                let idx = palette_2b
                    .iter()
                    .position(|&v| v == packed)
                    .unwrap_or_else(|| {
                        palette_2b.push(packed);
                        palette_2b.len() - 1
                    });
                pair_compose[c0 * n_classes + c1] = idx as u8;
            }
        }
        (pair_compose, palette_2b)
    }

    /// Compose pairs of 2-byte palette entries into a 4-byte lookup table.
    fn build_compose_4b(palette_2b: &[u64], total_states: usize) -> Vec<u64> {
        let n = palette_2b.len();
        let mut compose = vec![0u64; n * n];
        for p0 in 0..n {
            for p1 in 0..n {
                let mut packed = 0u64;
                for state in 0..total_states {
                    let mid = ((palette_2b[p0] >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
                    let final_s =
                        ((palette_2b[p1] >> (mid as u32 * Self::BITS)) & Self::MASK) as u8;
                    packed |= (final_s as u64) << (state as u32 * Self::BITS);
                }
                compose[p0 * n + p1] = packed;
            }
        }
        compose
    }

    /// Process remaining bytes after the interleaved common prefix.
    #[inline]
    fn finish_tail(&self, mut state: u8, codes: &[u8]) -> u8 {
        let chunks = codes.chunks_exact(4);
        let rem = chunks.remainder();

        for chunk in chunks {
            let ec0 = unsafe { *self.eq_class.get_unchecked(chunk[0] as usize) } as usize;
            let ec1 = unsafe { *self.eq_class.get_unchecked(chunk[1] as usize) } as usize;
            let ec2 = unsafe { *self.eq_class.get_unchecked(chunk[2] as usize) } as usize;
            let ec3 = unsafe { *self.eq_class.get_unchecked(chunk[3] as usize) } as usize;
            let p0 =
                unsafe { *self.pair_compose.get_unchecked(ec0 * self.n_classes + ec1) } as usize;
            let p1 =
                unsafe { *self.pair_compose.get_unchecked(ec2 * self.n_classes + ec3) } as usize;
            let packed = unsafe { *self.compose_4b.get_unchecked(p0 * self.n_palette + p1) };
            state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
        }

        if rem.len() >= 2 {
            let ec0 = self.eq_class[rem[0] as usize] as usize;
            let ec1 = self.eq_class[rem[1] as usize] as usize;
            let p = self.pair_compose[ec0 * self.n_classes + ec1] as usize;
            let packed = self.palette_2b[p];
            state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if rem.len() == 3 {
                let packed = self.transitions_1b[rem[2] as usize];
                state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            }
        } else if rem.len() == 1 {
            let packed = self.transitions_1b[rem[0] as usize];
            state = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
        }

        state
    }

    /// Branchless matching processing four code bytes per iteration.
    #[inline(never)]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        self.finish_tail(0, codes) == self.accept_state
    }
}

/// Flat u8 escape-folded DFA for medium needles (8-14 chars).
///
/// Like `BranchlessShiftDfa`, folds escape handling into the state space
/// (2N+1 states), but uses a flat `u8` transition table instead of
/// shift-packed `u64`. Supports up to 14-char needles (2*14+1 = 29 states).
/// Table size: 29 * 256 = 7,424 bytes, fits in L1.
struct FlatBranchlessDfa {
    /// transitions[state * 256 + byte] -> next state
    transitions: Vec<u8>,
    accept_state: u8,
}

impl FlatBranchlessDfa {
    pub(crate) const MAX_NEEDLE_LEN: usize = 14;

    pub(crate) fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        let n = needle.len();
        debug_assert!(n <= Self::MAX_NEEDLE_LEN);

        let accept_state = n as u8;
        let total_states = 2 * n + 1;
        let n_symbols = symbols.len();

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n + 1, accept_state);

        // Build fused transition table with escape folding.
        let mut transitions = vec![0u8; total_states * 256];
        for code_byte in 0..256usize {
            // Normal states 0..n
            for s in 0..n {
                if code_byte == ESCAPE_CODE as usize {
                    transitions[s * 256 + code_byte] = (s + n + 1) as u8;
                } else if code_byte < n_symbols {
                    transitions[s * 256 + code_byte] = sym_trans[s * n_symbols + code_byte];
                }
            }
            // Accept state (sticky)
            transitions[n * 256 + code_byte] = accept_state;
            // Escape states n+1..2n
            for s in 0..n {
                let esc_state = s + n + 1;
                let next = byte_table[s * 256 + code_byte] as u8;
                transitions[esc_state * 256 + code_byte] = next;
            }
        }

        Self {
            transitions,
            accept_state,
        }
    }

    #[inline(never)]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        for &byte in codes {
            state = self.transitions[state as usize * 256 + byte as usize];
        }
        state == self.accept_state
    }
}

/// Shift-based DFA: packs all state transitions into a `u64` per input byte.
///
/// For a DFA with S states (S <= 16, using 4 bits each), we store transitions
/// for ALL states in one `u64`. Transition: `next = (table[code] >> (state * 4)) & 0xF`.
///
/// Supports needles up to 14 characters (needle.len() + 2 <= 16 to fit escape
/// sentinel). This covers virtually all practical LIKE patterns.
pub(crate) struct ShiftDfa {
    /// For each code byte (0..255): a `u64` packing all state transitions.
    /// Bits `[state*4 .. state*4+4)` encode the next state for that input.
    transitions: [u64; 256],
    /// Same layout for escape byte transitions.
    escape_transitions: [u64; 256],
    accept_state: u8,
    escape_sentinel: u8,
}

impl ShiftDfa {
    const BITS: u32 = 4;
    const MASK: u64 = (1 << Self::BITS) - 1;
    /// Maximum needle length: 2^BITS - 2 (need room for accept + sentinel).
    const MAX_NEEDLE_LEN: usize = (1 << Self::BITS) - 2;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        debug_assert!(needle.len() <= Self::MAX_NEEDLE_LEN);

        let n_states = needle.len() + 1;
        let accept_state = needle.len() as u8;
        let escape_sentinel = needle.len() as u8 + 1;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);

        let fused = build_fused_table(&sym_trans, symbols.len(), n_states, |_| escape_sentinel, 0);

        let transitions = pack_shift_table(&fused, n_states, Self::BITS);
        let escape_transitions = pack_escape_shift_table(&byte_table, n_states, Self::BITS);

        Self {
            transitions,
            escape_transitions,
            accept_state,
            escape_sentinel,
        }
    }

    /// Match with iterator-based traversal.
    ///
    /// Using `iter.next()` instead of manual index + bounds check helps the
    /// compiler eliminate redundant bounds checks.
    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut iter = codes.iter();
        while let Some(&code) = iter.next() {
            let packed = self.transitions[code as usize];
            let next = ((packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            if next == self.escape_sentinel {
                let Some(&b) = iter.next() else {
                    return false;
                };
                let esc_packed = self.escape_transitions[b as usize];
                state = ((esc_packed >> (state as u32 * Self::BITS)) & Self::MASK) as u8;
            } else {
                state = next;
            }
        }
        state == self.accept_state
    }
}

/// Fused 256-entry u8 table DFA for contains needles in the 15-254 byte range.
///
/// This representation stores state ids in `u8`, so it cannot represent
/// needles longer than 254 bytes once the accept state and escape sentinel are
/// included.
pub(crate) struct FusedDfa {
    transitions: Vec<u8>,
    escape_transitions: Vec<u8>,
    accept_state: u8,
    escape_sentinel: u8,
}

impl FusedDfa {
    const MAX_NEEDLE_LEN: usize = u8::MAX as usize - 1;

    fn new(symbols: &[Symbol], symbol_lengths: &[u8], needle: &[u8]) -> Self {
        debug_assert!(needle.len() <= Self::MAX_NEEDLE_LEN);

        let n_states = needle.len() + 1;
        let accept_state = needle.len() as u8;
        let escape_sentinel = needle.len() as u8 + 1;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_states, accept_state);

        let transitions =
            build_fused_table(&sym_trans, symbols.len(), n_states, |_| escape_sentinel, 0);

        let escape_transitions: Vec<u8> = byte_table.iter().map(|&v| v as u8).collect();

        Self {
            transitions,
            escape_transitions,
            accept_state,
            escape_sentinel,
        }
    }

    #[inline]
    fn matches(&self, codes: &[u8]) -> bool {
        let mut state = 0u8;
        let mut pos = 0;
        while pos < codes.len() {
            let code = codes[pos];
            pos += 1;
            let next = self.transitions[state as usize * 256 + code as usize];
            if next == self.escape_sentinel {
                if pos >= codes.len() {
                    return false;
                }
                let b = codes[pos];
                pos += 1;
                state = self.escape_transitions[state as usize * 256 + b as usize];
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
// KMP helpers
// ---------------------------------------------------------------------------

fn kmp_byte_transitions(needle: &[u8]) -> Vec<u16> {
    let n_states = needle.len() + 1;
    let accept = needle.len() as u16;
    let failure = kmp_failure_table(needle);

    let mut table = vec![0u16; n_states * 256];
    for state in 0..n_states {
        for byte in 0..256u16 {
            if state == needle.len() {
                table[state * 256 + byte as usize] = accept;
                continue;
            }
            let mut s = state;
            loop {
                if byte as u8 == needle[s] {
                    s += 1;
                    break;
                }
                if s == 0 {
                    break;
                }
                s = failure[s - 1];
            }
            table[state * 256 + byte as usize] = s as u16;
        }
    }
    table
}

fn kmp_failure_table(needle: &[u8]) -> Vec<usize> {
    let mut failure = vec![0usize; needle.len()];
    let mut k = 0;
    for i in 1..needle.len() {
        while k > 0 && needle[k] != needle[i] {
            k = failure[k - 1];
        }
        if needle[k] == needle[i] {
            k += 1;
        }
        failure[i] = k;
    }
    failure
}

#[cfg(test)]
mod tests {
    use fsst::ESCAPE_CODE;

    use super::FusedDfa;
    use super::FsstMatcher;
    use super::FsstPrefixDfa;
    use super::LikeKind;

    fn escaped(bytes: &[u8]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(bytes.len() * 2);
        for &b in bytes {
            codes.push(ESCAPE_CODE);
            codes.push(b);
        }
        codes
    }

    #[test]
    fn test_like_kind_parse() {
        assert!(matches!(
            LikeKind::parse("http%"),
            Some(LikeKind::Prefix("http"))
        ));
        assert!(matches!(
            LikeKind::parse("%needle%"),
            Some(LikeKind::Contains("needle"))
        ));
        assert!(matches!(LikeKind::parse("%"), Some(LikeKind::Prefix(""))));
        // Suffix and underscore patterns are not supported.
        assert!(LikeKind::parse("%suffix").is_none());
        assert!(LikeKind::parse("a_c").is_none());
    }

    #[test]
    fn test_prefix_pushdown_len_13_with_escapes() {
        let matcher = FsstMatcher::try_new(&[], &[], "abcdefghijklm%")
            .unwrap()
            .unwrap();

        assert!(matcher.matches(&escaped(b"abcdefghijklm")));
        assert!(!matcher.matches(&escaped(b"abcdefghijklx")));
    }

    #[test]
    fn test_prefix_pushdown_rejects_len_14() {
        debug_assert_eq!(FsstPrefixDfa::MAX_PREFIX_LEN, 13);
        assert!(
            FsstMatcher::try_new(&[], &[], "abcdefghijklmn%")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_contains_pushdown_len_254_with_escapes() {
        let needle = "a".repeat(FusedDfa::MAX_NEEDLE_LEN);
        let pattern = format!("%{needle}%");
        let matcher = FsstMatcher::try_new(&[], &[], &pattern).unwrap().unwrap();

        assert!(matcher.matches(&escaped(needle.as_bytes())));

        let mut mismatch = needle.into_bytes();
        mismatch[FusedDfa::MAX_NEEDLE_LEN - 1] = b'b';
        assert!(!matcher.matches(&escaped(&mismatch)));
    }

    #[test]
    fn test_contains_pushdown_rejects_len_255() {
        let needle = "a".repeat(FusedDfa::MAX_NEEDLE_LEN + 1);
        let pattern = format!("%{needle}%");
        assert!(FsstMatcher::try_new(&[], &[], &pattern).unwrap().is_none());
    }
}
