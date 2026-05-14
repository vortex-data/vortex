// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Engine planner / cost-model routing for the FSST contains-shape DFA scans.
//!
//! Replaces the hardcoded `if let Some(...) { ... } else if ...` cascade in
//! [`super::folded_contains::FoldedContainsDfa::scan_to_bitbuf`] (and the
//! parallel cascades in [`super::flat_contains::FlatContainsDfa`] /
//! [`super::multi_contains::MultiContainsDfa`]) with a single decision
//! function that picks a [`ScanPlan`] up-front. The scan body then dispatches
//! on the plan with one `match`.
//!
//! ## Routing decision table (folded contains)
//!
//! The planner encodes the legacy cascade, in priority order:
//!
//! | # | Condition | Plan |
//! |---|---|---|
//! | 1 | `escape_only_pattern` is available | [`ScanPlan::EscapeOnly`] |
//! | 2 | SSA codes present AND `ssa_saturated` AND progressing codes present | [`ScanPlan::OneByteSaturated`] |
//! | 3 | Triple buckets exist | [`ScanPlan::TripleTeddy`] |
//! | 4 | Pair buckets exist AND escape-pair specialization applies AND no SSA | [`ScanPlan::EscapePair`] |
//! | 5 | Pair buckets exist | [`ScanPlan::PairTeddy`] |
//! | 6 | Progressing codes exist | [`ScanPlan::OneByteBitset`] |
//! | 7 | Fallback | [`ScanPlan::RowLoop`] |
//!
//! [`FlatContainsDfa`](super::flat_contains::FlatContainsDfa) and
//! [`MultiContainsDfa`](super::multi_contains::MultiContainsDfa) skip
//! rows 2-6 (Teddy/SSA paths don't apply to those DFAs) and just choose
//! between [`ScanPlan::EscapeOnly`] and [`ScanPlan::RowLoop`].
//!
//! ## Cost model
//!
//! [`ScanPlanner::estimated_cost_ns`] returns approximate per-call cost in ns,
//! parameterized on the calibrated per-architecture throughput constants
//! tracked in [`ArchProfile`]. The constants come from the existing
//! `benches/fsst_like.rs` and `DESIGN.md` numbers reported on the
//! development machines:
//!
//! | Path | Throughput source |
//! |---|---|
//! | `escape_only_memmem` | Memory-bandwidth limited memmem (~25 GB/s on AVX-512). |
//! | Triple Teddy (AVX-512) | `DESIGN.md`: `teddy_triple_pass_avx512` at 4.28 GB/s aggregate (inline-verify path). |
//! | Triple Teddy (AVX-2) | `DESIGN.md`: AVX-2 triple ~2.74 GB/s aggregate. |
//! | Pair Teddy | Roughly the triple bandwidth but with ~2-4× higher candidate density on URL-shaped data. |
//! | 1-byte bitset | `anchor_scan::build_progressing_bitset_unbounded` at ~8 GB/s build + per-row `tzcnt` jumps. |
//! | Row loop | `scan_to_bitbuf_with` at ~150 ns per row enum-free dispatch. |
//!
//! The cost function is exposed for tracing (`VORTEX_FSST_PLAN_TRACE=1`)
//! and for tests asserting parity against the legacy cascade. The model
//! is intentionally simple: routing in [`ScanPlanner::plan_folded`] is
//! rules-based (matching the legacy cascade exactly), the cost is for
//! diagnostics and future comparison-based path choice.

use core::fmt;

use super::ESCAPE_CODE;

/// Maximum number of c2 codes for the `escape_pair` specialization to fire.
/// Mirrors the literal `c2_set.len() <= 3` check in the legacy cascade.
const ESCAPE_PAIR_MAX_C2: usize = 3;

/// Estimated total-candidate count above which fused-Teddy+SSA loses to
/// the 1-byte path. Calibrated empirically: on the bench corpora
/// `%ear%` (~10k candidates) and `%google%` (~14k) the fused path wins;
/// `%htt%` and `%https%` (~70k–80k candidates) regress. Threshold
/// chosen at 32k as the midpoint with comfortable margin.
///
/// Single source of truth for both [`ssa_saturated`] and the cost model.
pub(super) const SSA_CANDIDATE_BUDGET: usize = 32 * 1024;

/// 8 KiB sample used by [`ssa_saturated`] to estimate SSA density across
/// the full `all_bytes` buffer. Sized to cover a few hundred typical
/// URL-shaped rows in scalar time well under the routing overhead budget.
pub(super) const SSA_DENSITY_SAMPLE_BYTES: usize = 8 * 1024;

/// Cheap scalar sample of a prefix of `all_bytes` for SSA bytes, then
/// extrapolate to a full-buffer candidate estimate. Returns `true` when
/// fused-Teddy+SSA's per-candidate dispatch is likely to lose to the
/// 1-byte path's per-row `matches_with_bitset` short-circuit.
///
/// Same semantics as the previous inline `ssa_saturated` in
/// `folded_contains.rs`; lifted here so the planner can reuse it.
#[inline]
pub(super) fn ssa_saturated(all_bytes: &[u8], ssa_codes: &[u8]) -> bool {
    let sample = SSA_DENSITY_SAMPLE_BYTES.min(all_bytes.len());
    if sample == 0 || ssa_codes.is_empty() {
        return false;
    }
    let mut is_ssa = [false; 256];
    for &c in ssa_codes {
        is_ssa[usize::from(c)] = true;
    }
    // SAFETY: sample <= all_bytes.len().
    let head = unsafe { all_bytes.get_unchecked(..sample) };
    let hits = head.iter().filter(|&&b| is_ssa[usize::from(b)]).count();
    // Extrapolate to the full buffer.
    let estimated = hits.saturating_mul(all_bytes.len()) / sample;
    estimated >= SSA_CANDIDATE_BUDGET
}

/// Returns `Some(c2_set)` when `buckets` is exactly the
/// `escape_pair` specialization (single bucket, c1 = ESCAPE, small c2 set).
#[inline]
pub(super) fn escape_pair_targets(buckets: &[(u8, Vec<u8>)]) -> Option<&[u8]> {
    match buckets {
        [(c1, c2_set)] if *c1 == ESCAPE_CODE && c2_set.len() <= ESCAPE_PAIR_MAX_C2 => {
            Some(c2_set.as_slice())
        }
        _ => None,
    }
}

/// The architecture profile decided once at planner construction time.
///
/// `is_x86_feature_detected!` reads CPUID at runtime; we cache the result
/// in the DFA to avoid per-call detection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum ArchProfile {
    /// AVX-512BW available (preferred for Teddy-3 streaming).
    #[cfg_attr(
        not(target_arch = "x86_64"),
        allow(dead_code, reason = "Only constructable on x86_64.")
    )]
    Avx512,
    /// AVX2 available, no AVX-512BW.
    #[cfg_attr(
        not(target_arch = "x86_64"),
        allow(dead_code, reason = "Only constructable on x86_64.")
    )]
    Avx2,
    /// AArch64 NEON.
    #[cfg_attr(
        not(target_arch = "aarch64"),
        allow(dead_code, reason = "Only constructable on aarch64.")
    )]
    Neon,
    /// Scalar fallback.
    Scalar,
}

impl fmt::Display for ArchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Avx512 => "avx512",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Scalar => "scalar",
        };
        f.write_str(s)
    }
}

impl ArchProfile {
    /// Detect the current CPU's profile. Cheap on first call (CPUID +
    /// internal cache in `std::is_x86_feature_detected!`), then constant.
    #[inline]
    pub(super) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512bw") {
                Self::Avx512
            } else if std::is_x86_feature_detected!("avx2") {
                Self::Avx2
            } else {
                Self::Scalar
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Scalar
        }
    }
}

/// Selected scan path for a contains-shape FSST LIKE scan.
///
/// One variant per legacy cascade branch. The planner returns one; the
/// DFA's `scan_to_bitbuf` matches and calls the corresponding `run_*`
/// helper.
#[cfg_attr(any(test, feature = "_test-harness"), allow(unreachable_pub))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScanPlan {
    /// `memmem` over the precomputed escape-only encoded pattern.
    /// Selected when the needle's bytes don't appear in any FSST symbol
    /// and the pattern is wildcard-free + case-sensitive.
    EscapeOnly,
    /// 1-byte progressing bitset over `all_bytes` with per-row
    /// `matches_with_bitset` short-circuit. Selected when SSA codes are
    /// present and the SSA byte density on `all_bytes` exceeds
    /// [`SSA_CANDIDATE_BUDGET`].
    OneByteSaturated,
    /// TODO Task A: bit-parallel Shift-Or / Bitap matcher for needles
    /// ≤ 8 bytes without SSA. Slot reserved; never returned by today's
    /// planner. Once Task A lands, this gets emitted by `plan_folded`
    /// when `needle.len() ≤ 8` and no SSA codes are present.
    #[cfg_attr(
        not(any(test, feature = "_test-harness")),
        allow(dead_code, reason = "Reserved for Task A (Shift-Or).")
    )]
    ShiftOr,
    /// Streaming Teddy-3 pass with fused SSA verifier, optionally
    /// followed by a Teddy-2 fallback over buckets the triple
    /// fingerprint doesn't cover.
    TripleTeddy,
    /// Streaming Teddy-2 pass with fused SSA verifier.
    PairTeddy,
    /// Specialized `escape_pair` memmem over a single (ESCAPE, c2) bucket
    /// with `≤ 3` c2 codes and no SSA.
    EscapePair,
    /// 1-byte progressing bitset over `all_bytes`. Selected when no
    /// Teddy buckets are available but progressing codes exist.
    OneByteBitset,
    /// Per-row [`super::scan_to_bitbuf_with`] fallback.
    RowLoop,
}

impl ScanPlan {
    /// Stable, human-readable name for tracing / debugging.
    #[cfg_attr(any(test, feature = "_test-harness"), allow(unreachable_pub))]
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::EscapeOnly => "escape_only_memmem",
            Self::OneByteSaturated => "ssa_saturated_one_byte",
            Self::ShiftOr => "shift_or",
            Self::TripleTeddy => "triple_teddy",
            Self::PairTeddy => "pair_teddy",
            Self::EscapePair => "escape_pair",
            Self::OneByteBitset => "one_byte_bitset",
            Self::RowLoop => "row_loop",
        }
    }
}

impl fmt::Display for ScanPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Inputs available to the planner at scan time.
///
/// Built once per `scan_to_bitbuf` call. Every field is either a small
/// scalar or a borrowed slice; the planner does not own or allocate
/// anything.
#[derive(Copy, Clone)]
pub(super) struct ScanContext<'a> {
    /// Number of rows being scanned.
    pub(super) n: usize,
    /// Total compressed bytes (`all_bytes.len()`).
    pub(super) all_bytes: &'a [u8],
    /// SSA (single-step-accept) codes, if any.
    pub(super) ssa_codes: Option<&'a [u8]>,
    /// Whether a progressing-code set was captured.
    pub(super) has_progressing_codes: bool,
    /// Whether an escape-only encoded pattern is available.
    pub(super) has_escape_only_pattern: bool,
    /// Whether the bucketed Teddy-3 (triple) anchor set is available.
    pub(super) has_triple_buckets: bool,
    /// Whether the bucketed Teddy-2 (pair) anchor set is available.
    pub(super) has_pair_buckets: bool,
    /// Number of pair buckets, when available.
    pub(super) pair_bucket_count: usize,
    /// First pair bucket's c1 and c2-set length, when there's exactly
    /// one pair bucket. Used to recognize the `escape_pair` specialization
    /// without re-walking the bucket list.
    pub(super) single_pair_summary: Option<(u8, usize)>,
}

impl<'a> ScanContext<'a> {
    /// Build a planner context from the folded contains DFA's pre-built
    /// metadata + the per-call `all_bytes`.
    #[inline]
    pub(super) fn new(
        n: usize,
        all_bytes: &'a [u8],
        ssa_codes: Option<&'a [u8]>,
        has_progressing_codes: bool,
        has_escape_only_pattern: bool,
        has_triple_buckets: bool,
        pair_buckets_summary: Option<(usize, Option<(u8, usize)>)>,
    ) -> Self {
        let (has_pair_buckets, pair_bucket_count, single_pair_summary) = match pair_buckets_summary
        {
            Some((count, single)) => (true, count, single),
            None => (false, 0, None),
        };
        Self {
            n,
            all_bytes,
            ssa_codes,
            has_progressing_codes,
            has_escape_only_pattern,
            has_triple_buckets,
            has_pair_buckets,
            pair_bucket_count,
            single_pair_summary,
        }
    }

    /// Build a context for the simpler `flat`/`multi` DFAs, which only
    /// route between [`ScanPlan::EscapeOnly`] and [`ScanPlan::RowLoop`].
    #[inline]
    pub(super) fn for_flat_or_multi(
        n: usize,
        all_bytes: &'a [u8],
        has_escape_only_pattern: bool,
    ) -> Self {
        Self {
            n,
            all_bytes,
            ssa_codes: None,
            has_progressing_codes: false,
            has_escape_only_pattern,
            has_triple_buckets: false,
            has_pair_buckets: false,
            pair_bucket_count: 0,
            single_pair_summary: None,
        }
    }
}

/// Routing engine. Picks a [`ScanPlan`] from a [`ScanContext`].
///
/// Construction is free (just stores the arch profile). One planner
/// instance is owned per DFA and reused across calls; `plan_folded(&ctx)` is
/// pure / re-entrant and allocates nothing.
///
/// # Calibration note
///
/// Today the planner replicates the legacy cascade exactly — a
/// property test (`test_planner_matches_legacy_cascade`) asserts this
/// against every bench needle on each bench corpus. Future Shift-Or
/// (Task A) and Fat Teddy (Task B) additions will alter the decision in
/// ways that are data-driven on the cost model. The current cost values
/// exist mainly for tracing (`VORTEX_FSST_PLAN_TRACE=1`) and to make the
/// eventual transition mechanical.
#[derive(Copy, Clone, Debug)]
pub(super) struct ScanPlanner {
    arch: ArchProfile,
}

impl ScanPlanner {
    /// Build a planner. Runs CPU feature detection once.
    #[inline]
    pub(super) fn new() -> Self {
        Self {
            arch: ArchProfile::detect(),
        }
    }

    /// The detected architecture profile. Public for tracing.
    #[inline]
    pub(super) fn arch(&self) -> ArchProfile {
        self.arch
    }

    /// Pick a plan for the folded contains DFA's scan. Encodes the legacy
    /// cascade priority order:
    ///
    /// 1. `escape_only_pattern` available → `EscapeOnly`
    /// 2. SSA codes present + saturated + progressing codes → `OneByteSaturated`
    /// 3. triple buckets exist → `TripleTeddy`
    /// 4. pair buckets exist, escape-pair shape AND no SSA → `EscapePair`
    /// 5. pair buckets exist → `PairTeddy`
    /// 6. progressing codes exist → `OneByteBitset`
    /// 7. fallback → `RowLoop`
    #[inline]
    pub(super) fn plan_folded(&self, ctx: &ScanContext<'_>) -> ScanPlan {
        if ctx.has_escape_only_pattern {
            return ScanPlan::EscapeOnly;
        }
        if let Some(codes) = ctx.ssa_codes
            && ctx.has_progressing_codes
            && ssa_saturated(ctx.all_bytes, codes)
        {
            return ScanPlan::OneByteSaturated;
        }
        if ctx.has_triple_buckets {
            return ScanPlan::TripleTeddy;
        }
        if ctx.has_pair_buckets {
            // Escape-pair specialization: exactly one bucket, c1 ==
            // ESCAPE_CODE, |c2| ≤ 3, and no SSA codes (the specialization
            // doesn't fuse SSA today).
            if ctx.ssa_codes.is_none()
                && ctx.pair_bucket_count == 1
                && let Some((c1, c2_len)) = ctx.single_pair_summary
                && c1 == ESCAPE_CODE
                && c2_len <= ESCAPE_PAIR_MAX_C2
            {
                return ScanPlan::EscapePair;
            }
            return ScanPlan::PairTeddy;
        }
        if ctx.has_progressing_codes {
            return ScanPlan::OneByteBitset;
        }
        ScanPlan::RowLoop
    }

    /// Pick a plan for the flat / multi contains DFAs. They only route
    /// between [`ScanPlan::EscapeOnly`] and [`ScanPlan::RowLoop`].
    #[inline]
    pub(super) fn plan_flat_or_multi(&self, ctx: &ScanContext<'_>) -> ScanPlan {
        if ctx.has_escape_only_pattern {
            ScanPlan::EscapeOnly
        } else {
            ScanPlan::RowLoop
        }
    }

    /// Approximate per-call cost in nanoseconds. Used only for
    /// tracing today; the routing decision in [`Self::plan_folded`]
    /// is rules-based, not cost-driven. Exposed for future
    /// comparison-based selection and for `VORTEX_FSST_PLAN_TRACE`
    /// diagnostics.
    #[inline]
    pub(super) fn estimated_cost_ns(&self, plan: ScanPlan, ctx: &ScanContext<'_>) -> u64 {
        // Throughput constants in bytes per nanosecond (= GB/s).
        // Calibrated from `DESIGN.md` numbers + bench observations on
        // the dev machines (Xeon 6975P-C, Apple M-class).
        let teddy_triple_bps = match self.arch {
            ArchProfile::Avx512 => 4.28,
            ArchProfile::Avx2 => 2.74,
            ArchProfile::Neon => 2.50,
            ArchProfile::Scalar => 0.80,
        };
        let teddy_pair_bps = match self.arch {
            ArchProfile::Avx512 => 5.50,
            ArchProfile::Avx2 => 3.30,
            ArchProfile::Neon => 3.00,
            ArchProfile::Scalar => 1.00,
        };
        let one_byte_bps = match self.arch {
            ArchProfile::Avx512 => 12.0,
            ArchProfile::Avx2 => 8.0,
            ArchProfile::Neon => 7.0,
            ArchProfile::Scalar => 2.0,
        };
        // memmem is essentially memory-bandwidth limited on rare needles.
        let memmem_bps = 25.0;
        // Row loop is dominated by the DFA inner loop, not bandwidth;
        // bench-derived per-row constant.
        let row_loop_ns_per_row = 150u64;
        let setup_ns = 200u64;

        let bytes = ctx.all_bytes.len() as f64;
        // Diagnostic cost only — saturate to u64::MAX rather than panic
        // on the (impossible-in-practice) overflow of a multi-zettabyte
        // scan.
        let cost_bandwidth = |bps: f64| {
            let ns = bytes / bps;
            if ns.is_finite() && ns >= 0.0 && ns < u64::MAX as f64 {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "Bounds-checked above; diagnostic cost only."
                )]
                {
                    ns as u64
                }
            } else {
                u64::MAX
            }
        };

        match plan {
            ScanPlan::EscapeOnly => setup_ns + cost_bandwidth(memmem_bps),
            ScanPlan::OneByteSaturated => setup_ns + cost_bandwidth(one_byte_bps),
            ScanPlan::TripleTeddy => setup_ns + cost_bandwidth(teddy_triple_bps),
            ScanPlan::PairTeddy => setup_ns + cost_bandwidth(teddy_pair_bps),
            ScanPlan::EscapePair => setup_ns + cost_bandwidth(memmem_bps),
            ScanPlan::OneByteBitset => setup_ns + cost_bandwidth(one_byte_bps),
            ScanPlan::RowLoop => setup_ns + (ctx.n as u64).saturating_mul(row_loop_ns_per_row),
            // TODO Task A: calibrate when Shift-Or lands.
            ScanPlan::ShiftOr => setup_ns + cost_bandwidth(one_byte_bps * 1.5),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folded_ctx_minimal<'a>(all_bytes: &'a [u8]) -> ScanContext<'a> {
        ScanContext {
            n: 1,
            all_bytes,
            ssa_codes: None,
            has_progressing_codes: false,
            has_escape_only_pattern: false,
            has_triple_buckets: false,
            has_pair_buckets: false,
            pair_bucket_count: 0,
            single_pair_summary: None,
        }
    }

    #[test]
    fn test_planner_picks_escape_only_when_pattern_set() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_escape_only_pattern = true;
        // Even when everything else is set, escape-only wins.
        ctx.has_triple_buckets = true;
        ctx.has_pair_buckets = true;
        ctx.has_progressing_codes = true;
        assert_eq!(p.plan_folded(&ctx), ScanPlan::EscapeOnly);
    }

    #[test]
    fn test_planner_picks_one_byte_saturated_when_ssa_dense() {
        let p = ScanPlanner::new();
        // SSA byte = 7. Build `all_bytes` saturated with 7s so the sample
        // ratio extrapolates above SSA_CANDIDATE_BUDGET.
        let bytes = vec![7u8; 8 * SSA_CANDIDATE_BUDGET];
        let ssa = [7u8];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.ssa_codes = Some(&ssa);
        ctx.has_progressing_codes = true;
        ctx.has_pair_buckets = true; // would otherwise pick PairTeddy
        assert_eq!(p.plan_folded(&ctx), ScanPlan::OneByteSaturated);
    }

    #[test]
    fn test_planner_picks_triple_when_triple_buckets() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_triple_buckets = true;
        ctx.has_pair_buckets = true;
        ctx.has_progressing_codes = true;
        assert_eq!(p.plan_folded(&ctx), ScanPlan::TripleTeddy);
    }

    #[test]
    fn test_planner_picks_escape_pair_specialization() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_pair_buckets = true;
        ctx.pair_bucket_count = 1;
        ctx.single_pair_summary = Some((ESCAPE_CODE, 2));
        // No SSA codes.
        assert_eq!(p.plan_folded(&ctx), ScanPlan::EscapePair);
    }

    #[test]
    fn test_planner_disables_escape_pair_when_ssa_present() {
        let p = ScanPlanner::new();
        // No saturation: tiny buffer + ssa bytes absent from all_bytes.
        let bytes = vec![0u8; 64];
        let ssa = [99u8];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_pair_buckets = true;
        ctx.pair_bucket_count = 1;
        ctx.single_pair_summary = Some((ESCAPE_CODE, 2));
        ctx.ssa_codes = Some(&ssa);
        ctx.has_progressing_codes = true;
        // SSA codes present → escape_pair is disabled, falls through to PairTeddy.
        assert_eq!(p.plan_folded(&ctx), ScanPlan::PairTeddy);
    }

    #[test]
    fn test_planner_disables_escape_pair_when_c2_too_large() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_pair_buckets = true;
        ctx.pair_bucket_count = 1;
        ctx.single_pair_summary = Some((ESCAPE_CODE, 4));
        assert_eq!(p.plan_folded(&ctx), ScanPlan::PairTeddy);
    }

    #[test]
    fn test_planner_picks_pair_teddy_when_pair_buckets() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_pair_buckets = true;
        ctx.pair_bucket_count = 4;
        ctx.has_progressing_codes = true;
        assert_eq!(p.plan_folded(&ctx), ScanPlan::PairTeddy);
    }

    #[test]
    fn test_planner_picks_one_byte_bitset_when_only_progressing() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let mut ctx = folded_ctx_minimal(&bytes);
        ctx.has_progressing_codes = true;
        assert_eq!(p.plan_folded(&ctx), ScanPlan::OneByteBitset);
    }

    #[test]
    fn test_planner_picks_row_loop_fallback() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let ctx = folded_ctx_minimal(&bytes);
        assert_eq!(p.plan_folded(&ctx), ScanPlan::RowLoop);
    }

    #[test]
    fn test_planner_flat_or_multi_picks_escape_only() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let ctx = ScanContext::for_flat_or_multi(1, &bytes, true);
        assert_eq!(p.plan_flat_or_multi(&ctx), ScanPlan::EscapeOnly);
    }

    #[test]
    fn test_planner_flat_or_multi_picks_row_loop() {
        let p = ScanPlanner::new();
        let bytes = vec![0u8; 1024];
        let ctx = ScanContext::for_flat_or_multi(1, &bytes, false);
        assert_eq!(p.plan_flat_or_multi(&ctx), ScanPlan::RowLoop);
    }

    #[test]
    fn test_estimated_cost_is_monotonic_in_bytes() {
        let p = ScanPlanner::new();
        let small = vec![0u8; 1024];
        let large = vec![0u8; 1024 * 1024];
        let ctx_small = folded_ctx_minimal(&small);
        let ctx_large = folded_ctx_minimal(&large);
        for plan in [
            ScanPlan::EscapeOnly,
            ScanPlan::OneByteSaturated,
            ScanPlan::TripleTeddy,
            ScanPlan::PairTeddy,
            ScanPlan::EscapePair,
            ScanPlan::OneByteBitset,
            ScanPlan::ShiftOr,
        ] {
            assert!(
                p.estimated_cost_ns(plan, &ctx_large) >= p.estimated_cost_ns(plan, &ctx_small),
                "cost should be monotonic in bytes for {plan:?}"
            );
        }
    }

    #[test]
    fn test_scan_plan_names_unique() {
        let plans = [
            ScanPlan::EscapeOnly,
            ScanPlan::OneByteSaturated,
            ScanPlan::ShiftOr,
            ScanPlan::TripleTeddy,
            ScanPlan::PairTeddy,
            ScanPlan::EscapePair,
            ScanPlan::OneByteBitset,
            ScanPlan::RowLoop,
        ];
        let mut names: Vec<_> = plans.iter().map(|p| p.name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), plans.len());
    }
}
