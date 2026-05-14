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
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::ESCAPE_CODE;
use super::anchor_scan;
use super::build_escape_only_encoded_pattern;
use super::build_symbol_transitions;
use super::kmp_byte_transitions;
use super::needle_bytes_absent_from_all_symbols;
use super::planner::ArchProfile;
use super::planner::ScanContext;
use super::planner::ScanPlan;
use super::planner::ScanPlanner;
use super::planner::escape_pair_targets;
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
    /// Pair buckets that Teddy-3 does not cover and therefore must still be
    /// scanned after the triple prefilter. This preserves 2-code accepts and
    /// escape continuations without giving up Teddy-3 selectivity for the
    /// buckets it does model.
    bucketed_pair_fallback_codes: Option<anchor_scan::BucketedPairCodes>,
    /// Subset of `progressing_codes` whose one-step state from state 0
    /// is `accept_state`. SSA-present needles skip the pair path: the
    /// pair scheme would miss SSA-anchored matches (their c1 lands in
    /// accept after a single step, so there's no advancing c2 to
    /// anchor on), and the 1-byte path is already fast on those data
    /// shapes.
    single_step_accept_codes: Option<Vec<u8>>,
    /// Compressed `[ESCAPE, needle[0], ESCAPE, needle[1], …]` pattern,
    /// populated when no symbol's expansion contains any byte of the
    /// needle. In that regime, the only way the DFA can reach `accept`
    /// from state 0 is by consuming exactly this 2L-byte pattern, so the
    /// scan can prefilter with a single `memmem` whose pattern length
    /// equals the encoded needle — far more selective than the 2-byte
    /// `(ESCAPE, needle[0])` anchor the bucketed Teddy pair scan uses.
    /// Only set for needles of length `>= 2`, where the longer pattern
    /// strictly improves on the existing `escape_pair` 2-byte path.
    escape_only_pattern: Option<Vec<u8>>,
    /// Routing engine. Picks one of the `run_*` paths below per scan call.
    /// Architecture detection happens once at DFA construction time so
    /// hot-path calls do no CPUID work.
    planner: ScanPlanner,
}

/// Per-architecture suffix for the legacy `pair_streaming` / `triple_streaming`
/// plan names. Kept for trace/debug parity with the pre-planner output.
#[inline]
fn pair_streaming_suffix(arch: ArchProfile) -> &'static str {
    match arch {
        ArchProfile::Avx512 | ArchProfile::Avx2 => "pair_streaming_avx2",
        ArchProfile::Neon => "pair_streaming_neon",
        ArchProfile::Scalar => "pair_streaming_scalar",
    }
}

#[inline]
fn triple_streaming_suffix(arch: ArchProfile) -> &'static str {
    match arch {
        ArchProfile::Avx512 => "triple_streaming_avx512",
        ArchProfile::Avx2 => "triple_streaming_avx2",
        ArchProfile::Neon => "triple_streaming_neon",
        ArchProfile::Scalar => "triple_streaming_scalar",
    }
}

#[inline]
fn teddy_trace_enabled() -> bool {
    std::env::var_os("VORTEX_FSST_TEDDY_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Emit a planner-level trace under `VORTEX_FSST_PLAN_TRACE=1`: chosen
/// plan, inputs, estimated cost. Used to validate that the planner picks
/// the same path as the legacy cascade on every bench needle.
fn plan_trace(dfa: &FoldedContainsDfa, ctx: &ScanContext<'_>, plan: ScanPlan) {
    let on = std::env::var_os("VORTEX_FSST_PLAN_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    if !on {
        return;
    }
    eprintln!(
        "[fsst::planner] plan={} accept_state={} n={} bytes={} progressing={} triple={} pair={} pair_count={} ssa={} escape_only={} arch={} cost_ns={}",
        plan,
        dfa.accept_state,
        ctx.n,
        ctx.all_bytes.len(),
        ctx.has_progressing_codes,
        ctx.has_triple_buckets,
        ctx.has_pair_buckets,
        ctx.pair_bucket_count,
        ctx.ssa_codes.map(|s| s.len()).unwrap_or(0),
        ctx.has_escape_only_pattern,
        dfa.planner.arch(),
        dfa.planner.estimated_cost_ns(plan, ctx),
    );
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
        case_insensitive: bool,
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

        let byte_table = kmp_byte_transitions(needle, case_insensitive);
        let sym_trans =
            build_symbol_transitions(symbols, symbol_lengths, &byte_table, n_normal, accept_state);

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

        // Bucketed pair/triple anchors: Teddy-3 is preferred for selectivity,
        // but it intentionally omits some valid Teddy-2 cases (2-code accepts
        // and escape continuations). Keep the full pair buckets plus a
        // triple-subtracted remainder so `scan_to_bitbuf` can run Teddy-3
        // first and Teddy-2 only where triple has no coverage.
        let bucketed_pair_codes = progressing_codes.as_deref().and_then(|c1| {
            anchor_scan::collect_bucketed_pair_codes(&transitions, c1, accept_state)
        });
        let bucketed_triple_codes = progressing_codes.as_deref().and_then(|c1| {
            anchor_scan::collect_bucketed_triple_codes(&transitions, c1, accept_state)
        });
        let bucketed_pair_fallback_codes = match (
            bucketed_pair_codes.as_deref(),
            bucketed_triple_codes.as_deref(),
        ) {
            (Some(pairs), Some(triples)) => {
                anchor_scan::collect_pair_fallback_after_triple(pairs, triples)
            }
            _ => None,
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

        // Escape-only fast path: only when the needle is wildcard-free,
        // case-sensitive, and no symbol contains any literal needle byte.
        let escape_only_pattern = (!case_insensitive
            && needle.len() >= 2
            && super::needle_is_literal(needle)
            && needle_bytes_absent_from_all_symbols(symbols, symbol_lengths, needle))
        .then(|| build_escape_only_encoded_pattern(needle));

        Ok(Self {
            transitions,
            accept_state,
            skip,
            progressing_codes,
            bucketed_pair_codes,
            bucketed_triple_codes,
            bucketed_pair_fallback_codes,
            single_step_accept_codes,
            escape_only_pattern,
            planner: ScanPlanner::new(),
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

    /// The bucketed (c1, c2_set) pair buckets for this needle, if any.
    /// Exposed to the multi-needle Fat Teddy matcher so it can union
    /// per-needle c2 sets per bucketed c1 across needles.
    #[inline]
    pub(super) fn bucketed_pair_codes_slice(&self) -> Option<&[(u8, Vec<u8>)]> {
        self.bucketed_pair_codes.as_deref()
    }

    /// The single-step-accept c1 codes for this needle, if any. Exposed
    /// to the Fat Teddy matcher so it can fold SSA c1's into the
    /// bucket's c1 set (with the bucket's c2 union acting as a wildcard
    /// for SSA candidates).
    #[inline]
    pub(super) fn single_step_accept_codes_slice(&self) -> Option<&[u8]> {
        self.single_step_accept_codes.as_deref()
    }

    /// Static (data-independent) routing name. Returns the same plan
    /// label the legacy cascade produced ignoring runtime data — used
    /// by tests and for tracing in places where `all_bytes` isn't on hand.
    #[inline]
    pub(crate) fn scan_plan_name(&self) -> &'static str {
        let has_ssa = self.single_step_accept_codes.is_some();
        let plan = self.static_plan();
        match plan {
            ScanPlan::EscapeOnly => "escape_only_memmem",
            ScanPlan::OneByteSaturated => "ssa_saturated_one_byte",
            ScanPlan::TripleTeddy => {
                if has_ssa {
                    "triple_streaming+ssa_fused"
                } else {
                    triple_streaming_suffix(self.planner.arch())
                }
            }
            ScanPlan::PairTeddy => {
                if has_ssa {
                    "pair_streaming+ssa_fused"
                } else {
                    pair_streaming_suffix(self.planner.arch())
                }
            }
            ScanPlan::EscapePair => "escape_pair_streaming",
            ScanPlan::OneByteBitset => "one_byte_bitset",
            ScanPlan::RowLoop => "row_loop",
            // Reserved for Task A; not reachable from today's planner.
            ScanPlan::ShiftOr => "shift_or",
        }
    }

    /// Build a planner context using static metadata only (no
    /// `all_bytes`-dependent decisions). Used by `scan_plan_name`.
    fn build_static_context(&self) -> ScanContext<'_> {
        // Empty buffer: `ssa_saturated` returns false on it, so
        // `OneByteSaturated` is never picked from a static context —
        // matching `scan_plan_name`'s pre-planner semantics.
        static EMPTY: [u8; 0] = [];
        self.build_context(0, &EMPTY)
    }

    /// Compute the planner-selected scan plan from the DFA's static
    /// metadata. Exposed for tests and tracing.
    #[inline]
    pub(crate) fn static_plan(&self) -> ScanPlan {
        let ctx = self.build_static_context();
        self.planner.plan_folded(&ctx)
    }

    /// Build a [`ScanContext`] for `(n, all_bytes)`. The data-dependent
    /// branch (`ssa_saturated`) consults `all_bytes` inside the planner.
    #[inline]
    fn build_context<'a>(&'a self, n: usize, all_bytes: &'a [u8]) -> ScanContext<'a> {
        let ssa_codes = self.single_step_accept_codes.as_deref();
        let has_progressing_codes = self.progressing_codes.is_some();
        let has_escape_only_pattern = self.escape_only_pattern.is_some();
        let has_triple_buckets = self.bucketed_triple_codes.is_some();
        let pair_buckets_summary = self.bucketed_pair_codes.as_deref().map(|b| {
            let single = match b {
                [(c1, c2_set)] => Some((*c1, c2_set.len())),
                _ => None,
            };
            (b.len(), single)
        });
        ScanContext::new(
            n,
            all_bytes,
            ssa_codes,
            has_progressing_codes,
            has_escape_only_pattern,
            has_triple_buckets,
            pair_buckets_summary,
        )
    }

    /// Pick the scan plan for `(n, all_bytes)`. Exposed for the bench
    /// parity regression test.
    #[inline]
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn plan_for(&self, n: usize, all_bytes: &[u8]) -> ScanPlan {
        self.planner.plan_folded(&self.build_context(n, all_bytes))
    }

    /// Test-only accessors. Used by the bench-parity regression test
    /// that replicates the legacy cascade independently from the
    /// planner to assert plan equality.
    #[cfg(any(test, feature = "_test-harness"))]
    pub fn escape_only_pattern_for_test(&self) -> Option<&[u8]> {
        self.escape_only_pattern.as_deref()
    }

    #[cfg(any(test, feature = "_test-harness"))]
    pub fn progressing_codes_for_test(&self) -> Option<&[u8]> {
        self.progressing_codes.as_deref()
    }

    #[cfg(any(test, feature = "_test-harness"))]
    pub fn single_step_accept_codes_for_test(&self) -> Option<&[u8]> {
        self.single_step_accept_codes.as_deref()
    }

    #[cfg(any(test, feature = "_test-harness"))]
    pub fn bucketed_pair_codes_for_test(&self) -> Option<&[(u8, Vec<u8>)]> {
        self.bucketed_pair_codes.as_deref()
    }

    #[cfg(any(test, feature = "_test-harness"))]
    pub fn bucketed_triple_codes_for_test(&self) -> bool {
        self.bucketed_triple_codes.is_some()
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
        let ctx = self.build_context(n, all_bytes);
        let plan = self.planner.plan_folded(&ctx);
        plan_trace(self, &ctx, plan);
        let teddy_trace = teddy_trace_enabled();
        match plan {
            ScanPlan::EscapeOnly => {
                let pattern = self
                    .escape_only_pattern
                    .as_deref()
                    .vortex_expect("EscapeOnly plan requires escape_only_pattern");
                self.run_escape_only(n, offsets, all_bytes, pattern, negated)
            }
            ScanPlan::OneByteSaturated => {
                self.run_one_byte_saturated(n, offsets, all_bytes, negated, teddy_trace)
            }
            ScanPlan::TripleTeddy => {
                self.run_triple_teddy(n, offsets, all_bytes, negated, teddy_trace)
            }
            ScanPlan::EscapePair => {
                self.run_escape_pair(n, offsets, all_bytes, negated, teddy_trace)
            }
            ScanPlan::PairTeddy => self.run_pair_teddy(n, offsets, all_bytes, negated, teddy_trace),
            ScanPlan::OneByteBitset => {
                self.run_one_byte_bitset(n, offsets, all_bytes, negated, teddy_trace)
            }
            ScanPlan::RowLoop => self.run_row_loop(n, offsets, all_bytes, negated),
            // TODO Task A: dispatch to the Shift-Or matcher once Task A
            // wires it in. The planner reserves this variant but never
            // emits it today.
            ScanPlan::ShiftOr => {
                debug_assert!(false, "planner emitted ShiftOr before Task A landed");
                self.run_row_loop(n, offsets, all_bytes, negated)
            }
        }
    }

    /// `EscapeOnly`: prefilter rows with a single `memmem` for the
    /// precomputed escape-only encoded needle. The only DFA-accepting
    /// compressed sequence is the 2L-byte encoded pattern, so memmem is
    /// exact up to the rare literal-byte false positive — verified
    /// per-row by [`Self::matches`].
    #[inline]
    fn run_escape_only<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        pattern: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        self.scan_via_escape_only_memmem(n, offsets, all_bytes, pattern, negated)
    }

    /// `OneByteSaturated`: per-row `matches_with_bitset` short-circuit
    /// driven by a precomputed 1-byte progressing-code bitset. Selected
    /// when SSA codes saturate `all_bytes` (estimated candidate count
    /// above [`super::planner::SSA_CANDIDATE_BUDGET`]) so the
    /// fused-Teddy+SSA per-candidate verify dispatch would lose to the
    /// per-row short-circuit.
    #[inline]
    fn run_one_byte_saturated<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
        trace: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let progressing = self
            .progressing_codes
            .as_deref()
            .vortex_expect("OneByteSaturated plan requires progressing_codes");
        let t = trace.then(std::time::Instant::now);
        let bitset = anchor_scan::build_progressing_bitset_unbounded(all_bytes, progressing);
        let result = self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
        if trace {
            let total_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            eprintln!(
                "[fsst::teddy] path=ssa_saturated_one_byte progressing_codes={} bytes={} total_us={:.3}",
                progressing.len(),
                all_bytes.len(),
                total_us,
            );
        }
        result
    }

    /// `TripleTeddy`: streaming Teddy-3 fingerprint with inline DFA
    /// verify and optional fused SSA. When the triple set doesn't cover
    /// every pair bucket, run a second Teddy-2 pass over the leftover
    /// buckets and OR/AND-merge (depending on `negated`).
    #[inline]
    fn run_triple_teddy<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
        trace: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let triples = self
            .bucketed_triple_codes
            .as_ref()
            .vortex_expect("TripleTeddy plan requires triple buckets");
        let ssa_codes = self.single_step_accept_codes.as_deref();
        let t = trace.then(std::time::Instant::now);
        let triple = anchor_scan::fused_teddy_triple_scan(
            n,
            offsets,
            all_bytes,
            triples,
            ssa_codes,
            negated,
            |cand, end| self.verify_from_candidate(all_bytes, cand, end).0,
        );
        let pair_fallback_buckets = self.bucketed_pair_fallback_codes.as_ref();
        let result = if let Some(pairs) = pair_fallback_buckets {
            // SSA already folded into the triple pass; don't re-emit
            // the same candidates in the pair-fallback pass.
            let pair = anchor_scan::fused_teddy_pair_scan(
                n,
                offsets,
                all_bytes,
                pairs,
                None,
                negated,
                |cand, end| self.verify_from_candidate(all_bytes, cand, end).0,
            );
            if negated {
                &triple & &pair
            } else {
                &triple | &pair
            }
        } else {
            triple
        };
        if trace {
            let total_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            eprintln!(
                "[fsst::teddy] path=triple_streaming{}{} triple_buckets={} pair_fallback_buckets={} ssa_codes={} bytes={} total_us={:.3}",
                if pair_fallback_buckets.is_some() {
                    "+pair_fallback"
                } else {
                    ""
                },
                if ssa_codes.is_some() { "+ssa" } else { "" },
                triples.len(),
                pair_fallback_buckets.map_or(0, |pairs| pairs.len()),
                ssa_codes.map_or(0, |c| c.len()),
                all_bytes.len(),
                total_us,
            );
        }
        result
    }

    /// `EscapePair`: specialized memmem-style pass for the single-bucket
    /// (c1 = ESCAPE, ≤ 3 c2's, no SSA) shape.
    #[inline]
    fn run_escape_pair<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
        trace: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let buckets = self
            .bucketed_pair_codes
            .as_ref()
            .vortex_expect("EscapePair plan requires pair buckets");
        let c2_codes = escape_pair_targets(buckets)
            .vortex_expect("EscapePair plan requires escape-pair targets");
        let t = trace.then(std::time::Instant::now);
        let result = anchor_scan::fused_escape_pair_scan(
            n,
            offsets,
            all_bytes,
            c2_codes,
            negated,
            |cand, end| self.verify_from_candidate(all_bytes, cand, end).0,
        );
        if trace {
            let total_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            eprintln!(
                "[fsst::teddy] path=escape_pair_streaming c2_codes={} bytes={} total_us={:.3}",
                c2_codes.len(),
                all_bytes.len(),
                total_us,
            );
        }
        result
    }

    /// `PairTeddy`: streaming Teddy-2 with inline DFA verify and
    /// (optionally) fused SSA.
    #[inline]
    fn run_pair_teddy<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
        trace: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let buckets = self
            .bucketed_pair_codes
            .as_ref()
            .vortex_expect("PairTeddy plan requires pair buckets");
        let ssa_codes = self.single_step_accept_codes.as_deref();
        let t = trace.then(std::time::Instant::now);
        let result = anchor_scan::fused_teddy_pair_scan(
            n,
            offsets,
            all_bytes,
            buckets,
            ssa_codes,
            negated,
            |cand, end| self.verify_from_candidate(all_bytes, cand, end).0,
        );
        if trace {
            let total_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
            eprintln!(
                "[fsst::teddy] path=pair_streaming{} buckets={} ssa_codes={} bytes={} total_us={:.3}",
                if ssa_codes.is_some() { "+ssa" } else { "" },
                buckets.len(),
                ssa_codes.map_or(0, |c| c.len()),
                all_bytes.len(),
                total_us,
            );
        }
        result
    }

    /// `OneByteBitset`: streaming 1-byte progressing-code bitset over
    /// `all_bytes` with per-row `matches_with_bitset`.
    #[inline]
    fn run_one_byte_bitset<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        negated: bool,
        trace: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let codes = self
            .progressing_codes
            .as_deref()
            .vortex_expect("OneByteBitset plan requires progressing_codes");
        let t = trace.then(std::time::Instant::now);
        let bitset = anchor_scan::build_progressing_bitset_unbounded(all_bytes, codes);
        let build_us = t
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or_default();
        let t = trace.then(std::time::Instant::now);
        let result = self.scan_with_anchor_bitset(n, offsets, all_bytes, &bitset, negated);
        if trace {
            let scan_us = t
                .map(|t| t.elapsed().as_secs_f64() * 1e6)
                .unwrap_or_default();
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
        result
    }

    /// `RowLoop`: per-row enum-free DFA dispatch. Final fallback.
    #[inline]
    fn run_row_loop<T>(&self, n: usize, offsets: &[T], all_bytes: &[u8], negated: bool) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        scan_to_bitbuf_with(n, offsets, all_bytes, negated, |codes| self.matches(codes))
    }

    /// Scan via a single `memmem` pass for the precomputed escape-only
    /// encoded needle. Each `memmem` hit identifies the first row that
    /// covers the hit position; we verify the row once with the standard
    /// `matches` DFA (which is exact, including for the rare hits that
    /// land at a literal-byte position rather than a code position) and
    /// then skip remaining hits inside the same row.
    fn scan_via_escape_only_memmem<T>(
        &self,
        n: usize,
        offsets: &[T],
        all_bytes: &[u8],
        pattern: &[u8],
        negated: bool,
    ) -> BitBuffer
    where
        T: vortex_array::dtype::IntegerPType,
    {
        let mut bits = if negated {
            BitBufferMut::new_set(n)
        } else {
            BitBufferMut::new_unset(n)
        };
        if n == 0 || pattern.len() > all_bytes.len() {
            return bits.freeze();
        }
        debug_assert!(offsets.len() > n);

        // SAFETY: caller guarantees `offsets.len() > n`, i.e. at least
        // `n + 1` entries.
        let mut string_idx: usize = 0;
        let mut string_start: usize = unsafe { *offsets.get_unchecked(0) }.as_();
        let mut string_end: usize = unsafe { *offsets.get_unchecked(1) }.as_();
        let mut last_processed_row: Option<usize> = None;

        for hit in memchr::memmem::find_iter(all_bytes, pattern) {
            // A hit at position `hit` is only meaningful if the full
            // pattern fits inside a single row. Advance to the row that
            // would contain the start of the hit.
            while hit >= string_end {
                string_idx += 1;
                if string_idx >= n {
                    return bits.freeze();
                }
                // SAFETY: `string_idx < n` and `offsets.len() >= n + 1`.
                string_start = string_end;
                string_end = unsafe { *offsets.get_unchecked(string_idx + 1) }.as_();
            }

            if last_processed_row == Some(string_idx) {
                continue;
            }
            // The pattern must lie entirely within this row.
            if hit + pattern.len() > string_end {
                last_processed_row = Some(string_idx);
                continue;
            }

            // Verify with the full DFA on the row to handle the rare
            // literal-position false positive (where the candidate's
            // first byte is the literal `255` after an ESCAPE, not an
            // ESCAPE code at a code position).
            // SAFETY: `string_start <= string_end <= all_bytes.len()`.
            let row = unsafe { all_bytes.get_unchecked(string_start..string_end) };
            if self.matches(row) {
                // SAFETY: `string_idx < n`.
                unsafe {
                    if negated {
                        bits.unset_unchecked(string_idx);
                    } else {
                        bits.set_unchecked(string_idx);
                    }
                }
            }
            last_processed_row = Some(string_idx);
        }

        bits.freeze()
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
        Some(anchor_scan::build_progressing_bitset_unbounded(
            all_bytes, codes,
        ))
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
        Some(anchor_scan::build_bucketed_triple_bitset(
            all_bytes, triples,
        ))
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
