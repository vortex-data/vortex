// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Escape-folded flat `u8` transition table DFA for contains matching
//! (`LIKE '%needle%'`).
//!
//! ## Why escape-fold?
//!
//! The plain [`super::flat_contains::FlatContainsDfa`] keeps a sentinel branch
//! in its inner loop: when the current code is `ESCAPE_CODE`, the table maps to
//! a sentinel value, the scanner detects it, and a second table lookup (in a
//! separate byte table) consumes the following literal byte. That's a hard-to-
//! predict branch on every code byte.
//!
//! The escape-folded DFA encodes "we just saw an `ESCAPE_CODE`, expecting a
//! literal byte" directly into the state space. With needle length `N`, where
//! `N <= 127`:
//!
//! - **Normal states** `0..=N`: regular KMP-style match progress; `N` is the
//!   accept state (sticky).
//! - **Escape states** `N+1..=2N`: "in-escape from base normal state
//!   `s = state - (N + 1)`" for `s` in `0..=N-1`. A read here is interpreted
//!   as a literal byte, advancing per the byte-level transition table for `s`.
//!
//! Total states: `2N + 1 <= 255`, so the state id fits in `u8`.
//!
//! The transition table is a flat `Vec<u8>` of size `(2N + 1) * 256`. For
//! normal states, the entry on `ESCAPE_CODE` goes to the matching escape
//! state `s + N + 1`. For escape states, all 256 entries are read as literal
//! bytes and dispatched through the byte table for the base state. There is
//! no sentinel branch in the inner loop -- every code byte produces exactly
//! one table lookup.
//!
//! The state-0 skip strategy (`memchr` / bitmap) still applies in the same way
//! as the plain DFA: when in state 0 we jump to the next code that could
//! progress the match.

use fsst::Symbol;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ESCAPE_CODE;
use super::anchor_scan;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
use super::scan_to_bitbuf_with;
use super::skip::SkipStrategy;

/// Escape-folded flat `u8` transition table DFA for contains matching.
///
/// Supports needles up to [`Self::MAX_NEEDLE_LEN`] bytes (so the state count
/// `2N + 1` fits in `u8`).
pub(crate) struct FoldedContainsDfa {
    /// `transitions[state * 256 + byte]` -> next state.
    ///
    /// Layout: rows `0..=N` are normal states (regular byte/symbol dispatch);
    /// rows `N+1..=2N` are escape states whose 256 entries are literal-byte
    /// dispatches via the underlying byte table.
    transitions: Vec<u8>,
    accept_state: u8,
    /// State-0 skip strategy.
    skip: SkipStrategy,
    /// Optional set of state-0 progressing codes captured for the global
    /// anchor scan. When `Some`, the scan path builds a packed bitset over
    /// `all_bytes` and drives `tzcnt`-based state-0 jumps inside the DFA,
    /// avoiding byte-by-byte bitmap probing in the hot path. Sets larger
    /// than [`anchor_scan::MAX_SET_BYTES`] are scanned via a multi-pass
    /// PSHUFB-Mula OR-merge in
    /// [`anchor_scan::build_progressing_bitset_unbounded`], at the cost
    /// of `ceil(N / 8)` passes over `all_bytes`.
    progressing_codes: Option<Vec<u8>>,
    /// Pair-eligible (c1, c2) code sets for a 2-byte anchor scan. The
    /// first vec is the subset of `progressing_codes` whose one-step
    /// successor state is non-zero AND non-accept. The second is the
    /// union of strictly-advancing-or-escape c2 codes for those c1's.
    /// When both fit in [`anchor_scan::MAX_SET_BYTES`], the scan path
    /// builds a candidate-pair bitset (bit set iff `(all_bytes[i],
    /// all_bytes[i+1])` is a state-0-progressing pair). On real
    /// FSST-trained URL data this is typically 100–1000× sparser than
    /// the 1-byte progressing bitset because the trainer keeps the
    /// match-path bytes (e.g. `g`, `o`, `l` for `%google%`) as
    /// single-byte symbols.
    pair_codes: Option<(Vec<u8>, Vec<u8>)>,
    /// Subset of `progressing_codes` whose one-step state from state 0
    /// is `accept_state`. When the pair-bitset path fires we OR this
    /// set's 1-byte bitset into the pair bitset so single-step-accept
    /// matches aren't missed.
    single_step_accept_codes: Option<Vec<u8>>,
    /// Per-c1 buckets `(c1, c2_set)` for the bucketed-Cartesian Teddy
    /// scan. Strictly more selective than [`pair_codes`] when the
    /// distinct-c1 count fits in [`anchor_scan::MAX_TEDDY_BUCKETS`]:
    /// each bucket fires only when the c1 byte matches exactly one
    /// `c1` value (zero false positives on the c1 axis), and the
    /// nibble-table c2 check is no worse than the existing Cartesian
    /// path's c2 check. On real FSST tables where multiple progressing
    /// c1's exist with disjoint c2 sets, this eliminates the
    /// cross-product false-positive pairs `(c1_a, c2_b)` for `a ≠ b`,
    /// typically 3–10× sparser than the plain Cartesian bitset at the
    /// same SIMD cost.
    teddy_buckets: Option<Vec<(u8, Vec<u8>)>>,
}

impl FoldedContainsDfa {
    /// Maximum needle length: `2N + 1 <= 255` so `N <= 127`.
    pub(crate) const MAX_NEEDLE_LEN: usize = 127;

    /// Build a folded contains DFA for `needle`.
    ///
    /// Returns `Err` if `needle.len() > `[`Self::MAX_NEEDLE_LEN`].
    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        if needle.len() > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for folded contains DFA",
                needle.len(),
                Self::MAX_NEEDLE_LEN
            );
        }
        // Empty needles are handled at a higher level (MatchAll), but we still
        // accept them here defensively (N=0 -> only the accept state).
        let accept_state =
            u8::try_from(needle.len()).vortex_expect("FoldedContainsDfa: accept state fits in u8");
        let n_normal = accept_state + 1; // states 0..=N
        // Total states: 2N+1 (normal 0..=N, escape N+1..=2N for base 0..=N-1).
        let n_states_usize = 2 * usize::from(accept_state) + 1;

        let byte_table = kmp_byte_transitions(needle);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_normal, accept_state);

        // Build the folded fused table: (2N+1) * 256.
        let n_symbols = symbols.len();
        let mut transitions = vec![0u8; n_states_usize * 256];

        // Rows 0..=N: normal states.
        for s in 0..n_normal {
            let row = usize::from(s) * 256;
            // Symbol codes 0..n_symbols.
            for code in 0..n_symbols {
                transitions[row + code] = sym_trans[usize::from(s) * n_symbols + code];
            }
            // ESCAPE_CODE: go to the matching escape state, except for accept
            // (which is sticky -- all transitions remain at accept).
            let escape_target = if s == accept_state {
                accept_state
            } else {
                // Escape state for base s = N + 1 + s.
                accept_state + 1 + s
            };
            transitions[row + usize::from(ESCAPE_CODE)] = escape_target;
            // Other code bytes (n_symbols..255 except ESCAPE_CODE) default to 0,
            // matching the plain `FlatContainsDfa` semantics.
        }

        // Rows N+1..=2N: escape states. For escape state e = N + 1 + s where
        // s in 0..=N-1, all 256 entries dispatch the next byte as a literal
        // through `byte_table[s * 256 + b]`.
        for s in 0..accept_state {
            let escape_state = accept_state + 1 + s;
            let row = usize::from(escape_state) * 256;
            let byte_row = usize::from(s) * 256;
            transitions[row..row + 256].copy_from_slice(&byte_table[byte_row..byte_row + 256]);
        }

        // Build the skip strategy from row 0 of the transitions (the first 256
        // entries). State 0 is reached either initially or by KMP fallback,
        // and we want to skip codes that leave us at 0.
        let skip = SkipStrategy::from_transition_row(&transitions[0..256], 0);

        // Capture the state-0 progressing-code set for the global anchor
        // scan. The unbounded variant collects all progressing codes
        // regardless of count — sets larger than
        // [`anchor_scan::MAX_SET_BYTES`] are scanned via multi-pass
        // PSHUFB-Mula OR-merge in
        // [`anchor_scan::build_progressing_bitset_unbounded`]. We capture
        // for any non-empty set so that `scan_to_bitbuf` can take the
        // global-bitset path even when the per-string skip would have
        // been Memchr1/2/3 — replacing N per-string memchr scans (one
        // per state-0 visit) with a single AVX2 PSHUFB pass + tzcnt
        // jumps. On large corpora with sparse hits this trades ~1.5 ms
        // of one-shot bitset construction for 5–10 ms of avoided
        // per-string scanning.
        let codes = anchor_scan::collect_progressing_codes_unbounded(&transitions[0..256], 0);
        let progressing_codes = if codes.is_empty() { None } else { Some(codes) };

        // Pair-anchor support: when both pair-eligible c1 and advancing-
        // only c2 sets fit in `MAX_SET_BYTES`, `scan_to_bitbuf` prefers
        // the pair bitset over the 1-byte bitset — typically 100–1000×
        // sparser on real FSST-trained URL data, where the trainer
        // keeps `g`, `o`, `l` (the bytes on the `%google%` match path)
        // as single-byte symbols rather than packing them into
        // multi-byte codes.
        let pair_codes = progressing_codes
            .as_deref()
            .and_then(|c1| anchor_scan::collect_pair_codes(&transitions, c1, accept_state));

        // Single-step-accept set: codes whose one-step from state 0 is
        // accept. Excluded from `pair_codes.0` (so the pair bitset
        // doesn't include their positions); we OR a 1-byte bitset of
        // these into the pair bitset at scan time so SSA matches
        // aren't missed.
        let single_step_accept_codes = progressing_codes.as_deref().and_then(|c1| {
            let v: Vec<u8> = c1
                .iter()
                .copied()
                .filter(|&c| transitions[usize::from(c)] == accept_state)
                .collect();
            if v.is_empty() { None } else { Some(v) }
        });

        // Bucketed Teddy: one bucket per pair-eligible c1, each
        // carrying its strictly-advancing-or-escape c2 set. The
        // collection helper rejects sets larger than
        // [`anchor_scan::MAX_TEDDY_BUCKETS`]; in that case the scan
        // path falls back to the plain Cartesian or 1-byte path.
        let teddy_buckets = progressing_codes.as_deref().and_then(|c1| {
            anchor_scan::collect_pair_buckets_shared_c1(&transitions, c1, accept_state)
        });

        Ok(Self {
            transitions,
            accept_state,
            skip,
            progressing_codes,
            pair_codes,
            single_step_accept_codes,
            teddy_buckets,
        })
    }

    /// Run the matcher over `codes`. Returns `true` iff the needle appears.
    #[inline]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        let transitions = self.transitions.as_slice();
        let accept = self.accept_state;
        let mut pos: usize = 0;
        let len = codes.len();

        // Outer loop: SIMD-skip in state 0 to the next progressing code, then
        // run a tight inner loop while state != 0. The inner loop is uniform:
        // one table lookup per code byte, no sentinel branch. We only return
        // to the outer loop when the DFA falls back to state 0 (KMP failure).
        loop {
            match self.skip.find_next_progressing(codes, pos) {
                Some(next) => pos = next,
                None => return false,
            }

            // We're at a progressing code: step once.
            let code = codes[pos];
            pos += 1;
            let mut state = transitions[usize::from(code)];
            if state == accept {
                return true;
            }

            // Inner loop while state != 0.
            while state != 0 && pos < len {
                let c = codes[pos];
                pos += 1;
                state = transitions[usize::from(state) * 256 + usize::from(c)];
                if state == accept {
                    return true;
                }
            }
            if pos >= len {
                return false;
            }
        }
    }

    /// Specialized scan over `n` strings, returning a `BitBuffer` of accept
    /// results (XOR `negated`). The `matches` body is monomorphized into the
    /// bit-packing loop, eliminating the per-string enum dispatch in
    /// `FsstMatcher::matches`.
    ///
    /// Whenever the state-0 progressing-code set is non-empty, we take a
    /// global-anchor-scan fast path:
    ///
    /// 1. Stream `all_bytes` once with an AVX2 PSHUFB-Mula nibble check to
    ///    produce a `len(all_bytes)`-bit "candidate position" bitset
    ///    (~30 GB/s on Skylake-X-class parts). Sets larger than
    ///    [`anchor_scan::MAX_SET_BYTES`] are scanned via multi-pass
    ///    OR-merge — one PSHUFB pass per chunk of 8 codes.
    /// 2. For each string, run a DFA whose state-0 jump is driven by a single
    ///    `tzcnt` over the bitset rather than per-string `memchr` or
    ///    byte-by-byte bitmap probing. Strings with no candidate bytes
    ///    return `false` after a single word read.
    ///
    /// The materialized bitset moves the state-0 skip from per-string SIMD
    /// scans (one `memchr` call or per-byte bitmap probe per state-0 visit)
    /// to a single `tzcnt` per state-0 visit, while the AVX2 scan amortizes
    /// the membership check across all 32 input bytes per cycle. On
    /// large corpora with sparse hits the build cost (~1.5 ms per chunk
    /// per 36 MB of `all_bytes`) is repaid many times over.
    #[inline]
    pub(crate) fn scan_to_bitbuf<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        // Pre-filter ladder for state-0 jumps:
        //
        //   1. **Bucketed Cartesian Teddy** — one bucket per pair-eligible
        //      c1, each carrying that c1's `c2` set. Zero false positives
        //      on the c1 axis (each bucket is exact-byte) and no worse
        //      than the plain Cartesian path on the c2 axis. Eliminates
        //      cross-product pairs `(c1_a, c2_b)` for `a ≠ b` that the
        //      Cartesian path admits — typically 3–10× sparser bitset on
        //      real FSST-trained URL data, same SIMD cost (4 PSHUFBs +
        //      ANDs per 32 input bytes).
        //
        //   2. **Plain Cartesian pair bitset** — `c1_set ⨯ c2_set`
        //      independent membership, AND'd with a 1-position shift.
        //      Used when the bucketed path is unavailable (more than 8
        //      pair-eligible c1's) but the unioned c1/c2 sets still fit
        //      in `MAX_SET_BYTES`.
        //
        //   3. **1-byte progressing bitset (unbounded)** — universal
        //      fallback; multi-pass PSHUFB OR-merge for sets larger
        //      than `MAX_SET_BYTES`.
        //
        // Both pair-anchor paths require `single_step_accept_codes` to
        // be empty. SSA-present needles (e.g. `%https%` on synthetic
        // ClickBench where code 127 = "https://" already accepts in
        // one step) are already fast on the 1-byte path because each
        // progressing-position match completes in a single table
        // lookup, and either pair scheme would add a second PSHUFB
        // pass for the SSA-merge bitset that is pure overhead.
        if self.single_step_accept_codes.is_none() {
            if let Some(buckets) = self.teddy_buckets.as_deref()
                && let Some(bitset) = anchor_scan::build_pair_bitset_teddy(all_bytes, buckets)
            {
                return self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
            }
            if let Some((pc1, pc2)) = self.pair_codes.as_ref()
                && let Some(bitset) = anchor_scan::build_pair_bitset(all_bytes, pc1, pc2)
            {
                return self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
            }
        }

        // Pair path unavailable (or sets too large). Fall back to the
        // 1-byte unbounded path: still wins over per-string memchr on
        // large corpora.
        if let Some(codes) = self.progressing_codes.as_deref() {
            let bitset = anchor_scan::build_progressing_bitset_unbounded(all_bytes, codes);
            return self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
        }

        scan_to_bitbuf_with(n, offsets, all_bytes, negated, |codes| self.matches(codes))
    }

    /// Drive `BitBuffer::collect_bool` over `n` strings using a precomputed
    /// progressing-code bitset over `all_bytes`. For each string range,
    /// evaluate the DFA via [`Self::matches_with_bitset`] and bake the
    /// result into the per-bit closure.
    #[inline]
    fn scan_with_anchor_bitset<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        bitset: &[u64],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        debug_assert!(offsets.len() > n);
        // SAFETY: caller guarantees `offsets.len() > n`.
        let mut start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
        BitBuffer::collect_bool(n, |i| {
            // SAFETY: `i < n` and `offsets.len() >= n + 1`.
            let end: usize = unsafe { *offsets.get_unchecked(i + 1) }.as_();
            debug_assert!(start <= end && end <= all_bytes.len());
            let result = self.matches_with_bitset(all_bytes, bitset, start, end) != negated;
            start = end;
            result
        })
    }

    /// Variant of [`Self::matches`] that uses a precomputed progressing-code
    /// bitset over `all_bytes` for state-0 jumps. Equivalent to
    /// `self.matches(&all_bytes[abs_start..abs_end])` but ~5–10× faster on
    /// strings with sparse progressing codes because each "find next
    /// progressing position" reduces to one masked `u64` load + `tzcnt`
    /// rather than a byte-by-byte bitmap probe loop.
    #[inline]
    fn matches_with_bitset(
        &self,
        all_bytes: &[u8],
        bitset: &[u64],
        abs_start: usize,
        abs_end: usize,
    ) -> bool {
        let transitions = self.transitions.as_slice();
        let accept = self.accept_state;
        let mut pos = abs_start;

        loop {
            // Skip to next progressing code via the bitset.
            match anchor_scan::next_set_in_range(bitset, pos, abs_end) {
                Some(p) => pos = p,
                None => return false,
            }

            // SAFETY: `pos < abs_end <= all_bytes.len()`.
            let code = unsafe { *all_bytes.get_unchecked(pos) };
            pos += 1;
            let mut state = transitions[usize::from(code)];
            if state == accept {
                return true;
            }

            // Inner loop while state != 0.
            while state != 0 && pos < abs_end {
                // SAFETY: `pos < abs_end <= all_bytes.len()`.
                let c = unsafe { *all_bytes.get_unchecked(pos) };
                pos += 1;
                state = transitions[usize::from(state) * 256 + usize::from(c)];
                if state == accept {
                    return true;
                }
            }
            if pos >= abs_end {
                return false;
            }
        }
    }
}
