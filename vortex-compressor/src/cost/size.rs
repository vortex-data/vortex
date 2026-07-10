// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The default, size-only cost model.

use crate::candidate::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::stats::ArrayAndStats;

/// The default cost model: maximize estimated compression ratio.
///
/// `SizeCost` preserves the compressor's historical ratio-argmax ordering:
///
/// - A candidate's cost is `Cost(-ratio)` over the exact ratio signal the candidate carries
///   (an analytic estimate or a sample-measured `before / after`), so argmin cost is argmax
///   ratio with identical tie behavior. Computing cost from the ratio rather than recomputing
///   bytes (`input / ratio`) matters: IEEE division could collapse strict ratio inequalities
///   into cost ties and silently flip winners via the registration-order tie-break.
/// - The canonical baseline is `Cost(-1.0)`, so `cost < canonical_cost` reproduces the
///   historical `ratio > 1.0` validity gate.
/// - Zero-byte sample results and non-finite or subnormal ratios are rejected (`None`),
///   reproducing the historical estimate-validity checks.
#[derive(Debug, Default, Clone, Copy)]
pub struct SizeCost;

impl CostModel for SizeCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        let ratio = candidate.estimate().estimated_compression_ratio()?;
        (ratio.is_finite() && !ratio.is_subnormal()).then(|| Cost::new(-ratio))
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
    use crate::scheme::CandidateEstimate;
    use crate::scheme::CompressorContext;
    use crate::scheme::Scheme;
    use crate::scheme::SchemeEvaluation;
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

        fn evaluate(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> SchemeEvaluation {
            SchemeEvaluation::Skip
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

    fn test_data() -> ArrayAndStats {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        ArrayAndStats::new(array, GenerateStatsOptions::default())
    }

    fn cost(estimated_ratio: Option<f64>) -> Option<Cost> {
        let data = test_data();
        let estimate = estimated_ratio.map_or_else(
            CandidateEstimate::zero_bytes,
            CandidateEstimate::from_compression_ratio,
        );
        SizeCost.cost(&Candidate::new(&TestScheme, estimate, &data, None, &[]))
    }

    fn canonical_cost() -> Cost {
        let data = test_data();
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

    /// `cost < canonical_cost` must reproduce `legacy_is_valid` exactly, with invalid ratios
    /// producing no cost.
    #[test]
    fn validity_matches_ratio_gate() {
        let canonical = canonical_cost();
        for &ratio in RATIOS {
            let cost = cost(Some(ratio));
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
                let best_cost = cost(Some(best)).expect("valid ratio must have a cost");
                let challenger_cost = cost(Some(challenger)).expect("valid ratio must have a cost");

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
    fn invalid_ratios_have_no_cost(#[case] ratio: f64) {
        assert!(cost(Some(ratio)).is_none());
    }

    #[test]
    fn zero_bytes_is_rejected() {
        assert!(cost(None).is_none());
    }
}
