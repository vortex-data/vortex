// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared skip strategies for DFA state-0 (or phase-start) fast paths.
//!
//! When a DFA is in a "searching" state (state 0 for single-segment contains,
//! or a phase-start state for multi-segment), most code bytes leave the state
//! unchanged. A skip strategy accelerates the search by jumping directly to
//! the next code that could advance the DFA.

use fsst::ESCAPE_CODE;

/// Strategy for skipping non-progressing codes at a DFA start state.
///
/// Chosen at construction time based on how many code bytes "progress" the
/// DFA past the start state:
/// - 1–3 progressing codes → SIMD-accelerated `memchr` (32+ bytes/cycle)
/// - 4+ → packed `[u64; 4]` bitmap (branchless per-code check)
pub(super) enum SkipStrategy {
    /// Only 1 code byte progresses — use `memchr::memchr` (SIMD).
    Memchr1(u8),
    /// 2 code bytes progress — use `memchr::memchr2` (SIMD).
    Memchr2(u8, u8),
    /// 3 code bytes progress — use `memchr::memchr3` (SIMD).
    Memchr3(u8, u8, u8),
    /// More than 3 — use packed bitmap (1 cache line).
    Bitmap([u64; 4]),
}

impl SkipStrategy {
    /// Build a `SkipStrategy` from a 256-entry transition row.
    ///
    /// A code is "progressing" if `transition_row[code] != start_state`
    /// (it advances the DFA) or `code == ESCAPE_CODE` (the escaped literal
    /// might advance).
    pub(super) fn from_transition_row(transition_row: &[u8], start_state: u8) -> Self {
        debug_assert!(transition_row.len() >= 256);
        let mut prog_codes: Vec<u8> = Vec::new();
        for code in 0..=255u8 {
            if transition_row[usize::from(code)] != start_state || code == ESCAPE_CODE {
                prog_codes.push(code);
            }
        }

        match prog_codes.len() {
            0 => SkipStrategy::Bitmap([0u64; 4]),
            1 => SkipStrategy::Memchr1(prog_codes[0]),
            2 => SkipStrategy::Memchr2(prog_codes[0], prog_codes[1]),
            3 => SkipStrategy::Memchr3(prog_codes[0], prog_codes[1], prog_codes[2]),
            _ => {
                let mut bitmap = [0u64; 4];
                for &code in &prog_codes {
                    bitmap[usize::from(code >> 6)] |= 1u64 << (code & 63);
                }
                SkipStrategy::Bitmap(bitmap)
            }
        }
    }

    /// Find the next progressing code in `codes[start..]`.
    ///
    /// Returns the absolute index in `codes`, or `None` if no progressing code
    /// exists from `start` onward.
    #[inline]
    pub(super) fn find_next_progressing(&self, codes: &[u8], start: usize) -> Option<usize> {
        let slice = &codes[start..];
        match self {
            SkipStrategy::Memchr1(a) => memchr::memchr(*a, slice).map(|i| start + i),
            SkipStrategy::Memchr2(a, b) => memchr::memchr2(*a, *b, slice).map(|i| start + i),
            SkipStrategy::Memchr3(c0, c1, c2) => {
                memchr::memchr3(*c0, *c1, *c2, slice).map(|i| start + i)
            }
            SkipStrategy::Bitmap(bm) => {
                for (i, &code) in slice.iter().enumerate() {
                    if bm[usize::from(code >> 6)] & (1u64 << (code & 63)) != 0 {
                        return Some(start + i);
                    }
                }
                None
            }
        }
    }
}
