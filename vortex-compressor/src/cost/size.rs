// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The default, size-only cost model.

use crate::candidate::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::estimate::EstimateScore;
use crate::stats::ArrayAndStats;

/// The default cost model: maximize estimated compression ratio.
///
/// `SizeCost` reproduces the compressor's historical ratio-argmax selection **bit-exactly**:
///
/// - A candidate is priced at `Cost(-ratio)` over the exact ratio signal the candidate
///   carries (an analytic estimate or a sample-measured `before / after`), so argmin cost is
///   argmax ratio with identical tie behavior. Pricing off the ratio rather than recomputing
///   bytes (`input / ratio`) matters: IEEE division could collapse strict ratio inequalities
///   into cost ties and silently flip winners via the registration-order tie-break.
/// - The canonical baseline is `Cost(-1.0)`, so `cost < canonical_cost` reproduces the
///   historical `ratio > 1.0` validity gate.
/// - Zero-byte sample results and non-finite or subnormal ratios are rejected (`None`),
///   reproducing the historical estimate-validity checks.
#[derive(Debug, Default, Clone, Copy)]
pub struct SizeCost;

impl CostModel for SizeCost {
    fn cost(&self, candidate: &Candidate) -> Option<Cost> {
        match candidate.score {
            EstimateScore::FiniteCompression(ratio) => {
                (ratio.is_finite() && !ratio.is_subnormal()).then(|| Cost::new(-ratio))
            }
            EstimateScore::ZeroBytes => None,
        }
    }

    fn canonical_cost(&self, _data: &ArrayAndStats, _n_values: u64) -> Cost {
        Cost::new(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::CascadingCompressor;
    use crate::ctx::CompressorContext;
    use crate::estimate::CompressionEstimate;
    use crate::estimate::EstimateVerdict;
    use crate::scheme::Scheme;
    use crate::stats::GenerateStatsOptions;

    #[derive(Debug)]
    struct TestScheme;

    impl Scheme for TestScheme {
        fn scheme_name(&self) -> &'static str {
            "test.size_cost"
        }

        fn matches(&self, _canonical: &Canonical) -> bool {
            true
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!("test helper should never be selected for compression")
        }
    }

    fn candidate(score: EstimateScore) -> Candidate {
        Candidate {
            scheme: &TestScheme,
            score,
            input_nbytes: 1024,
            n_values: 256,
            sampled: None,
            cascade: Vec::new(),
        }
    }

    fn canonical_cost() -> Cost {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let data = ArrayAndStats::new(array, GenerateStatsOptions::default());
        SizeCost.canonical_cost(&data, 4)
    }

    /// The historical estimate-validity rule that `SizeCost` must reproduce: finite,
    /// non-subnormal, and strictly better than canonical (`ratio > 1.0`).
    fn legacy_is_valid(ratio: f64) -> bool {
        ratio.is_finite() && !ratio.is_subnormal() && ratio > 1.0
    }

    /// Every ratio regime: clear winners, near-1.0 boundaries, exact ties, sub-1.0,
    /// non-positive, and non-finite/subnormal garbage.
    const RATIOS: &[f64] = &[
        100.0,
        3.0,
        2.0,
        1.0 + f64::EPSILON,
        1.0,
        1.0 - f64::EPSILON,
        0.5,
        0.0,
        -1.0,
        f64::MIN_POSITIVE / 2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    /// `cost < canonical_cost` must reproduce `legacy_is_valid` exactly, with non-priceable
    /// ratios rejected as `None`.
    #[test]
    fn validity_matches_ratio_gate() {
        let canonical = canonical_cost();
        for &ratio in RATIOS {
            let cost = SizeCost.cost(&candidate(EstimateScore::FiniteCompression(ratio)));
            let valid = cost.is_some_and(|cost| cost < canonical);
            assert_eq!(
                valid,
                legacy_is_valid(ratio),
                "validity mismatch for ratio {ratio}"
            );
        }
    }

    /// For every ordered pair of ratios, the cost ordering must equal the (reversed) ratio
    /// ordering the selector historically used: a challenger displaces the best iff its
    /// ratio is strictly greater. Equal ratios must produce equal costs (ties keep the
    /// incumbent via strict `<`).
    #[test]
    fn cost_ordering_matches_ratio_ordering() {
        let valid: Vec<f64> = RATIOS
            .iter()
            .copied()
            .filter(|&r| legacy_is_valid(r))
            .collect();
        for &best in &valid {
            for &challenger in &valid {
                let best_cost = SizeCost
                    .cost(&candidate(EstimateScore::FiniteCompression(best)))
                    .expect("valid ratio must be priceable");
                let challenger_cost = SizeCost
                    .cost(&candidate(EstimateScore::FiniteCompression(challenger)))
                    .expect("valid ratio must be priceable");

                let displaces = challenger_cost < best_cost;
                assert_eq!(
                    displaces,
                    challenger > best,
                    "ordering mismatch: challenger {challenger} vs best {best}"
                );
                assert_eq!(
                    challenger_cost == best_cost,
                    challenger == best,
                    "tie mismatch: challenger {challenger} vs best {best}"
                );
            }
        }
    }

    #[rstest]
    #[case::nan(f64::NAN)]
    #[case::infinite(f64::INFINITY)]
    #[case::neg_infinite(f64::NEG_INFINITY)]
    #[case::subnormal(f64::MIN_POSITIVE / 2.0)]
    fn unpriceable_ratios_are_rejected(#[case] ratio: f64) {
        assert!(
            SizeCost
                .cost(&candidate(EstimateScore::FiniteCompression(ratio)))
                .is_none()
        );
    }

    #[test]
    fn zero_bytes_is_rejected() {
        assert!(
            SizeCost
                .cost(&candidate(EstimateScore::ZeroBytes))
                .is_none()
        );
    }

    /// Under `SizeCost`, [`SkipThreshold::best_case_ratio_cannot_win`] must reduce to
    /// exactly the historical `max_ratio <= best_ratio` skip rule, including exact ties and
    /// one-ulp boundaries; unpriceable best cases must never skip.
    #[test]
    fn skip_threshold_reduces_to_ratio_comparison() {
        use std::sync::Arc;

        use crate::estimate::SkipThreshold;

        for &best_ratio in &[1.5, 2.0, 3.0, 60.8, 1024.0] {
            let threshold = SkipThreshold::new(
                Some((
                    Cost::new(-best_ratio),
                    EstimateScore::FiniteCompression(best_ratio),
                )),
                Arc::new(SizeCost),
                &TestScheme,
                1024,
                256,
                Vec::new(),
            );
            assert_eq!(threshold.best_ratio(), Some(best_ratio));

            let max_ratios = [
                0.5,
                1.0,
                best_ratio.next_down(),
                best_ratio,
                best_ratio.next_up(),
                best_ratio * 2.0,
            ];
            for max_ratio in max_ratios {
                assert_eq!(
                    threshold.best_case_ratio_cannot_win(max_ratio),
                    max_ratio <= best_ratio,
                    "mismatch for max_ratio {max_ratio} vs best {best_ratio}"
                );
            }

            // Best cases SizeCost cannot price never skip: the callback proceeds and its
            // real estimate is rejected later, exactly as before.
            assert!(!threshold.best_case_ratio_cannot_win(f64::NAN));
            assert!(!threshold.best_case_ratio_cannot_win(f64::INFINITY));
        }

        // Without a best candidate there is no threshold to lose to.
        let no_best =
            SkipThreshold::new(None, Arc::new(SizeCost), &TestScheme, 1024, 256, Vec::new());
        assert!(!no_best.best_case_ratio_cannot_win(1e9));
        assert_eq!(no_best.best_ratio(), None);
    }
}
