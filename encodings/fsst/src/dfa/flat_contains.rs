// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flat `u8` transition table DFA for contains matching (`LIKE '%needle%'`).
//!
//! Uses an escape-sentinel strategy: the FSST escape code maps to a sentinel
//! state, and the next literal byte is looked up in a separate byte-level
//! transition table.
//! This is to support needles up to u8::MAX long.
//!
//! ## Construction (needle = `"aba"`, symbols = `[0:"ab", 1:"ba"]`)
//!
//! ### Step 1: KMP (Knuth–Morris–Pratt) byte-level transition table
//!
//! See: <https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm>
//!
//! Build a `(state × byte) → state` table using the KMP failure function.
//! States 0..2 track match progress, state 3 is accept (sticky).
//!
//! ```text
//!         Input byte
//! State   'a'    'b'    other
//! ─────   ────   ────   ─────
//!   0      1      0      0      ← want 'a'
//!   1      1      2      0      ← matched "a", want 'b' (KMP: 'a'→stay at 1)
//!   2      3✓     0      0      ← matched "ab", want 'a'
//!   3✓     3✓     3✓     3✓     ← accept (sticky)
//! ```
//!
//! ### Step 2: Symbol-level transitions
//!
//! For each `(state, symbol)` pair, simulate feeding the symbol's bytes
//! through the byte table:
//!
//! ```text
//! Symbol 0 = "ab" (2 bytes):
//!   state 0 + 'a' → 1, + 'b' → 2  ⟹ sym_trans[0][0] = 2
//!   state 1 + 'a' → 1, + 'b' → 2  ⟹ sym_trans[1][0] = 2
//!   state 2 + 'a' → 3✓             ⟹ sym_trans[2][0] = 3✓ (accept)
//!
//! Symbol 1 = "ba" (2 bytes):
//!   state 0 + 'b' → 0, + 'a' → 1  ⟹ sym_trans[0][1] = 1
//!   state 1 + 'b' → 2, + 'a' → 3✓ ⟹ sym_trans[1][1] = 3✓ (accept)
//!   state 2 + 'b' → 0, + 'a' → 1  ⟹ sym_trans[2][1] = 1
//! ```
//!
//! ### Step 3: Fused 256-wide table with escape sentinel
//!
//! Merge symbol transitions into a 256-wide table. Code bytes 0–1 use symbol
//! transitions, code 255 (ESCAPE_CODE) maps to the sentinel (4), and
//! unused code bytes default to 0:
//!
//! ```text
//!              Code byte
//! State   0("ab") 1("ba") 2..254  255(ESC)
//! ─────   ─────── ─────── ──────  ────────
//!   0       2       1       0       4(S)
//!   1       2       3✓      0       4(S)
//!   2       3✓      1       0       4(S)
//!   3✓      3✓      3✓      3✓      3✓
//! ```
//!
//! When the scanner sees sentinel (4), it reads the next byte and looks it
//! up in the byte-level escape table (from step 1).
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

/// Flat `u8` transition table DFA for contains matching.
///
/// The escape code maps to a sentinel state; the next literal byte is looked
/// up in a separate byte-level escape table.
pub(crate) struct FlatContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    transitions: Vec<u8>,
    /// `escape_transitions[state * 256 + byte]` -> next state for escaped bytes.
    escape_transitions: Vec<u8>,
    accept_state: u8,
    sentinel: u8,
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
