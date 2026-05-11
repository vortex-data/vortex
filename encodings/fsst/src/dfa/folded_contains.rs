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
#[cfg(any(test, feature = "_test-harness"))]
use super::scan_to_bitbuf_with as scan_to_bitbuf_with_for_bench;
use super::scan_to_bitbuf_with;
use super::skip::SkipStrategy;

/// Escape-folded flat `u8` transition table DFA for contains matching.
///
/// Supports needles up to [`Self::MAX_NEEDLE_LEN`] bytes (so the state count
/// `2N + 1` fits in `u8`).
#[cfg_attr(any(test, feature = "_test-harness"), allow(unreachable_pub))]
pub struct FoldedContainsDfa {
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
    /// Shared-c1 buckets for the bucketed Teddy 2-byte anchor scan. Each
    /// bucket is `(c1, c2_set)`: the bucket bit is set iff `all_bytes[i]
    /// == c1` AND `all_bytes[i+1] ∈ c2_set` (modulo a small
    /// nibble-cross within-bucket FP for diverse c2 sets). Compared to
    /// the prior Cartesian path's `c1_union × c2_union` flattening,
    /// this drops cross-bucket false positives — on real FSST-trained
    /// URL data with multiple `(c1, c2)` pairs sharing some but not
    /// all c1's, the bitset is 3–10× sparser at the same SIMD cost.
    /// Single-pass when `len ≤ MAX_SET_BYTES`, multi-pass OR-merge
    /// otherwise.
    bucketed_pair_codes: Option<anchor_scan::BucketedPairCodes>,
    /// Shared-c1 buckets for the bucketed Teddy-3 (3-byte fingerprint)
    /// anchor scan: each bucket is `(c1, c2_set, c3_set)`. The bucket
    /// bit is set at position `i` iff `all_bytes[i] == c1` AND
    /// `all_bytes[i+1] ∈ c2_set` AND `all_bytes[i+2] ∈ c3_set` (with
    /// the same within-bucket nibble-cross over-approximation as
    /// Teddy-2). Selectivity is roughly `|c1|·|c2|·|c3| / 256³` per
    /// position vs Teddy-2's `|c1|·|c2| / 256²` — on dense ASCII
    /// corpora the candidate count typically drops 100–1000×, at the
    /// cost of one extra PSHUFB-Mula pair plus one extra unaligned
    /// 32-byte load per chunk. Built only for needles with
    /// `accept_state ≥ 3` and progressing c1's that chain through two
    /// intermediate non-accept normal states; shorter or
    /// escape-anchored needles fall back to Teddy-2.
    bucketed_triple_codes: Option<anchor_scan::BucketedTripleCodes>,
    /// Subset of `progressing_codes` whose one-step state from state 0
    /// is `accept_state`. SSA-present needles skip the pair path: the
    /// pair scheme would miss SSA-anchored matches (their c1 lands in
    /// accept after a single step, so there's no advancing c2 to
    /// anchor on), and the 1-byte path is already fast on those data
    /// shapes.
    single_step_accept_codes: Option<Vec<u8>>,
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

        let trace = std::env::var_os("VORTEX_FSST_BUILD_TRACE")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let t0 = trace.then(std::time::Instant::now);
        let byte_table = kmp_byte_transitions(needle);
        let t_byte = trace.then(std::time::Instant::now);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_normal, accept_state);
        let t_sym = trace.then(std::time::Instant::now);

        // Build the folded fused table: (2N+1) * 256.
        let n_symbols = symbols.len();
        let mut transitions = vec![0u8; n_states_usize * 256];

        // Rows 0..=N: normal states. Bulk-copy the symbol row (`n_symbols`
        // entries) from `sym_trans` in one shot instead of a per-cell
        // loop with bounds checks, then patch the ESCAPE_CODE entry.
        for s in 0..n_normal {
            let row = usize::from(s) * 256;
            let sym_row = usize::from(s) * n_symbols;
            transitions[row..row + n_symbols]
                .copy_from_slice(&sym_trans[sym_row..sym_row + n_symbols]);
            // ESCAPE_CODE: go to the matching escape state, except for accept
            // (sticky — all transitions remain at accept).
            let escape_target = if s == accept_state {
                accept_state
            } else {
                accept_state + 1 + s
            };
            transitions[row + usize::from(ESCAPE_CODE)] = escape_target;
        }

        // Rows N+1..=2N: escape states. Each row mirrors a base normal
        // state's row in the byte table (post-escape bytes are read as
        // literals).
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

        // Bucketed-pair anchor support: per-c1 partition of the
        // strictly-advancing-or-escape c2 codes, used by the bucketed
        // Teddy scan path. The scan prefers this over the 1-byte
        // progressing bitset whenever non-empty; multi-pass OR-merge
        // handles `> MAX_SET_BYTES` buckets at a `ceil(buckets / 8)`-
        // pass cost over `all_bytes`.
        // Compute Teddy-3 first (preferred path). If it applies, skip the
        // Teddy-2 collection — `scan_to_bitbuf` ladder prefers triple.
        let bucketed_triple_codes = progressing_codes.as_deref().and_then(|c1| {
            anchor_scan::collect_bucketed_triple_codes(&transitions, c1, accept_state)
        });
        let bucketed_pair_codes = if bucketed_triple_codes.is_some() {
            None
        } else {
            progressing_codes.as_deref().and_then(|c1| {
                anchor_scan::collect_bucketed_pair_codes(&transitions, c1, accept_state)
            })
        };

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

        if let (Some(t0), Some(t_byte), Some(t_sym)) = (t0, t_byte, t_sym) {
            let t_end = std::time::Instant::now();
            let us = |a: std::time::Instant, b: std::time::Instant| {
                b.duration_since(a).as_secs_f64() * 1e6
            };
            eprintln!(
                "[fsst::build] N={} n_syms={} total={:.2}µs kmp={:.2} sym_trans={:.2} fused_table+collect={:.2}",
                accept_state,
                n_symbols,
                us(t0, t_end),
                us(t0, t_byte),
                us(t_byte, t_sym),
                us(t_sym, t_end),
            );
        }
        Ok(Self {
            transitions,
            accept_state,
            skip,
            progressing_codes,
            bucketed_pair_codes,
            bucketed_triple_codes,
            single_step_accept_codes,
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
        // Prefer the bucketed 2-byte pair anchor when buckets exist AND
        // there are no single-step-accept codes. This is the most
        // selective prefilter the folded DFA can run — eliminates
        // cross-bucket false-positive pairs that the legacy Cartesian
        // path admits, at the same single-pass SIMD cost when there
        // are ≤ MAX_SET_BYTES distinct c1's (multi-pass OR-merge for
        // more). On real ClickBench URL data on `%google%` with
        // single-byte 'g'/'o' anchors this is identical to Cartesian
        // (one bucket); on multi-anchor needles where several c1's
        // each have their own preferred c2 the bucketed path is
        // 3–10× sparser.
        //
        // SSA-present needles (e.g. `%https%` on synthetic ClickBench
        // where code 127 = "https://" already accepts in one step) skip
        // the pair path: they're already fast on the 1-byte path
        // because each progressing-position match completes in a single
        // table lookup, and the pair scheme would add a second
        // PSHUFB-Mula pass for the SSA-merge bitset that pure
        // overhead.
        if self.single_step_accept_codes.is_none() {
            // Streaming Teddy paths: no materialized bitset. Inline DFA
            // verify per AVX2 32-byte block; empty blocks short-circuit
            // at the movemask. This drops the
            // `Vec<u64>` allocation per chunk and folds the bitset walk
            // into the same pass that produced it.
            let trace = std::env::var_os("VORTEX_FSST_TEDDY_TRACE")
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            let force_pair = std::env::var_os("VORTEX_FSST_FORCE_TEDDY_PAIR")
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            let force_one_byte = std::env::var_os("VORTEX_FSST_FORCE_ONE_BYTE")
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            if !force_pair
                && !force_one_byte
                && let Some(triples) = self.bucketed_triple_codes.as_ref()
            {
                let t = trace.then(std::time::Instant::now);
                let result = anchor_scan::fused_teddy_triple_scan(
                    n,
                    offsets,
                    all_bytes,
                    triples,
                    negated,
                    |cand, end| self.verify_from_candidate(all_bytes, cand, end).0,
                );
                let total_us = t.map(|t| t.elapsed().as_secs_f64() * 1e6).unwrap_or_default();
                if trace {
                    eprintln!(
                        "[fsst::teddy] path=triple_streaming buckets={} bytes={} total_us={:.3}",
                        triples.len(),
                        all_bytes.len(),
                        total_us,
                    );
                }
                return result;
            }
            if !force_one_byte && let Some(buckets) = self.bucketed_pair_codes.as_ref() {
                let t = trace.then(std::time::Instant::now);
                let result = anchor_scan::fused_teddy_pair_scan(
                    n,
                    offsets,
                    all_bytes,
                    buckets,
                    negated,
                    |cand, end| self.verify_from_candidate(all_bytes, cand, end).0,
                );
                let total_us = t.map(|t| t.elapsed().as_secs_f64() * 1e6).unwrap_or_default();
                if trace {
                    eprintln!(
                        "[fsst::teddy] path=pair_streaming buckets={} bytes={} total_us={:.3}",
                        buckets.len(),
                        all_bytes.len(),
                        total_us,
                    );
                }
                return result;
            }
        }

        // Pair path unavailable (or sets too large). Fall back to the
        // 1-byte unbounded path: still wins over per-string memchr on
        // large corpora.
        if let Some(codes) = self.progressing_codes.as_deref() {
            let trace = std::env::var_os("VORTEX_FSST_TEDDY_TRACE")
                .map(|v| !v.is_empty())
                .unwrap_or(false);
            let t = trace.then(std::time::Instant::now);
            let bitset = anchor_scan::build_progressing_bitset_unbounded(all_bytes, codes);
            let build_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            let t = trace.then(std::time::Instant::now);
            let result = self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
            let scan_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            if trace {
                let bits = bitset.iter().map(|w| w.count_ones() as u64).sum::<u64>();
                eprintln!(
                    "[fsst::teddy] path=one_byte codes={} bytes={} bitset_bits={} build_us={:.3} scan_us={:.3}",
                    codes.len(),
                    all_bytes.len(),
                    bits,
                    build_us,
                    scan_us,
                );
            }
            return result;
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
        let trace = std::env::var_os("VORTEX_FSST_TEDDY_DEEP_TRACE")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        if trace {
            return self.scan_with_anchor_bitset_trace(n, offsets, all_bytes, bitset, negated);
        }
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

    #[inline]
    fn scan_with_anchor_bitset_trace<T>(
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
        let scan_t = std::time::Instant::now();
        let mut next_us = 0f64;
        let mut verify_us = 0f64;
        let mut candidate_positions = 0usize;
        let mut transition_steps = 0usize;
        let mut matches = 0usize;
        let mut empty_ranges = 0usize;
        let mut bits = if negated {
            vortex_buffer::BitBufferMut::new_set(n)
        } else {
            vortex_buffer::BitBufferMut::new_unset(n)
        };

        for i in 0..n {
            // SAFETY: `i < n` and `offsets.len() >= n + 1`.
            let start: usize = unsafe { *offsets.get_unchecked(i) }.as_();
            let end: usize = unsafe { *offsets.get_unchecked(i + 1) }.as_();
            let mut pos = start;
            let mut matched = false;
            loop {
                let t = std::time::Instant::now();
                let next = anchor_scan::next_set_in_range(bitset, pos, end);
                next_us += t.elapsed().as_secs_f64() * 1e6;
                let Some(p) = next else {
                    empty_ranges += 1;
                    break;
                };
                candidate_positions += 1;
                let t = std::time::Instant::now();
                let (accepted, next_pos, steps) = self.verify_from_candidate(all_bytes, p, end);
                verify_us += t.elapsed().as_secs_f64() * 1e6;
                transition_steps += steps;
                if accepted {
                    matched = true;
                    matches += 1;
                    break;
                }
                if next_pos <= p {
                    pos = p + 1;
                } else {
                    pos = next_pos;
                }
                if pos >= end {
                    break;
                }
            }
            if matched != negated {
                // SAFETY: i < n; bits has exactly `n` bits.
                unsafe { bits.set_unchecked(i) };
            } else if negated {
                unsafe { bits.unset_unchecked(i) };
            }
        }

        let total_us = scan_t.elapsed().as_secs_f64() * 1e6;
        eprintln!(
            "[fsst::teddy_scan] rows={} candidates={} matches={} empty_ranges={} transition_steps={} next_us={:.3} verify_us={:.3} other_us={:.3} total_us={:.3}",
            n,
            candidate_positions,
            matches,
            empty_ranges,
            transition_steps,
            next_us,
            verify_us,
            total_us - next_us - verify_us,
            total_us,
        );
        bits.freeze()
    }

    /// Build the bucketed-Teddy pair bitset for this DFA's bucket set
    /// over `all_bytes`. `None` when the bucketed pair path is not
    /// applicable (no buckets, or single-step-accept codes present).
    /// Exposed for benches that A/B the three prefilter variants on
    /// the same corpus.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn build_bucketed_bitset_for_bench(&self, all_bytes: &[u8]) -> Option<Vec<u64>> {
        if self.single_step_accept_codes.is_some() {
            return None;
        }
        let buckets = self.bucketed_pair_codes.as_ref()?;
        Some(anchor_scan::build_bucketed_pair_bitset(all_bytes, buckets))
    }

    /// Build the legacy Cartesian pair bitset for this DFA's pair codes
    /// over `all_bytes`. Reconstructs the unioned `(c1, c2)` sets from
    /// the bucketed representation. `None` when not applicable (no
    /// buckets, single-step-accept codes present, or either union
    /// exceeds `MAX_SET_BYTES`). Exposed for benches.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn build_cartesian_bitset_for_bench(&self, all_bytes: &[u8]) -> Option<Vec<u64>> {
        if self.single_step_accept_codes.is_some() {
            return None;
        }
        let buckets = self.bucketed_pair_codes.as_ref()?;
        let mut c1_set: Vec<u8> = Vec::new();
        let mut c2_set: Vec<u8> = Vec::new();
        let mut c1_seen = [false; 256];
        let mut c2_seen = [false; 256];
        for (c1, c2s) in buckets {
            if !c1_seen[usize::from(*c1)] {
                c1_seen[usize::from(*c1)] = true;
                c1_set.push(*c1);
            }
            for &c2 in c2s {
                if !c2_seen[usize::from(c2)] {
                    c2_seen[usize::from(c2)] = true;
                    c2_set.push(c2);
                }
            }
        }
        anchor_scan::build_pair_bitset(all_bytes, &c1_set, &c2_set)
    }

    /// Build the 1-byte progressing-code bitset for this DFA's
    /// progressing set over `all_bytes`. Multi-pass for sets larger
    /// than `MAX_SET_BYTES`. `None` when no progressing codes exist.
    /// Exposed for benches.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn build_one_byte_bitset_for_bench(&self, all_bytes: &[u8]) -> Option<Vec<u64>> {
        let codes = self.progressing_codes.as_deref()?;
        Some(anchor_scan::build_progressing_bitset_unbounded(all_bytes, codes))
    }

    /// Whether the bucketed pair path would fire for this DFA. Exposed
    /// for benches.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn bucketed_pair_applicable(&self) -> bool {
        self.single_step_accept_codes.is_none() && self.bucketed_pair_codes.is_some()
    }

    /// Build the bucketed Teddy-3 (3-byte fingerprint) bitset over
    /// `all_bytes` for this DFA's triple set. `None` when not applicable
    /// (no triples, single-step-accept codes present). Exposed for
    /// benches.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn build_triple_bitset_for_bench(&self, all_bytes: &[u8]) -> Option<Vec<u64>> {
        if self.single_step_accept_codes.is_some() {
            return None;
        }
        let triples = self.bucketed_triple_codes.as_ref()?;
        Some(anchor_scan::build_bucketed_triple_bitset(all_bytes, triples))
    }

    /// Whether the bucketed Teddy-3 path would fire for this DFA. Exposed
    /// for benches.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn bucketed_triple_applicable(&self) -> bool {
        self.single_step_accept_codes.is_none() && self.bucketed_triple_codes.is_some()
    }

    /// End-to-end scan forced through the 1-byte progressing bitset
    /// path, bypassing the bucketed/Cartesian pair routes. The "before"
    /// baseline for the bucketed-Teddy patch on data where the legacy
    /// Cartesian path is inapplicable (e.g. ClickBench URLs on
    /// `%google%`: c1∪c2 each exceed `MAX_SET_BYTES`, the legacy
    /// `build_pair_bitset` returned `None` and the production scan
    /// fell back to 1-byte). Exposed for benches only.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn scan_to_bitbuf_one_byte_only<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        if let Some(codes) = self.progressing_codes.as_deref() {
            let bitset = anchor_scan::build_progressing_bitset_unbounded(all_bytes, codes);
            return self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
        }
        scan_to_bitbuf_with_for_bench(n, offsets, all_bytes, negated, |codes| self.matches(codes))
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

    #[inline]
    fn verify_from_candidate(
        &self,
        all_bytes: &[u8],
        mut pos: usize,
        abs_end: usize,
    ) -> (bool, usize, usize) {
        let transitions = self.transitions.as_slice();
        let accept = self.accept_state;
        let mut steps = 0usize;

        // SAFETY: caller passes a candidate in range.
        let code = unsafe { *all_bytes.get_unchecked(pos) };
        pos += 1;
        steps += 1;
        let mut state = transitions[usize::from(code)];
        if state == accept {
            return (true, pos, steps);
        }

        while state != 0 && pos < abs_end {
            // SAFETY: `pos < abs_end <= all_bytes.len()`.
            let c = unsafe { *all_bytes.get_unchecked(pos) };
            pos += 1;
            steps += 1;
            state = transitions[usize::from(state) * 256 + usize::from(c)];
            if state == accept {
                return (true, pos, steps);
            }
        }
        (false, pos, steps)
    }
}
