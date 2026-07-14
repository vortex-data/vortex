// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The default, size-only cost model.

use vortex_utils::aliases::hash_map::HashMap;

use crate::cost::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;

/// A per-scheme selection prior applied by [`SizeCost`]: **policy, not measurement**.
///
/// Priors let the model demand that a scheme win by a margin (`multiplier < 1.0`) or clear a
/// floor (`min_ratio`) before it is selected, without the scheme hardcoding that judgment
/// into its own estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchemePrior {
    /// Multiplier applied to the scheme's raw ratio signal before any comparison. Values
    /// below `1.0` handicap the scheme, demanding it win by a margin.
    pub multiplier: f64,

    /// The effective (post-multiplier) ratio the scheme must strictly exceed to be
    /// considered at all.
    pub min_ratio: f64,
}

impl Default for SchemePrior {
    fn default() -> Self {
        Self {
            multiplier: 1.0,
            min_ratio: 1.0,
        }
    }
}

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
///
/// Per-scheme [`SchemePrior`]s registered via [`with_scheme_prior`] additionally handicap a
/// scheme's raw ratio (`effective = raw * multiplier`) and gate it behind a floor
/// (`effective > min_ratio`). Schemes without a registered prior are priced off their raw
/// ratio unchanged.
///
/// [`with_scheme_prior`]: SizeCost::with_scheme_prior
#[derive(Debug, Default, Clone)]
pub struct SizeCost {
    /// Per-scheme selection priors; schemes without an entry are priced unmodified.
    priors: HashMap<SchemeId, SchemePrior>,
}

impl SizeCost {
    /// Registers a selection prior for `scheme`, replacing any existing entry.
    pub fn with_scheme_prior(mut self, scheme: SchemeId, prior: SchemePrior) -> Self {
        self.priors.insert(scheme, prior);
        self
    }
}

impl CostModel for SizeCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        let raw = candidate.estimate().estimated_compression_ratio()?;

        let ratio = match self.priors.get(&candidate.scheme_id()) {
            Some(prior) => {
                let effective = raw * prior.multiplier;
                // Non-finite `effective` falls through to the validity check below.
                if effective <= prior.min_ratio {
                    return None;
                }
                effective
            }
            None => raw,
        };

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
    use crate::scheme::SchemeExt;
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
        SizeCost::default().cost(&Candidate::new(&TestScheme, estimate, &data, None, &[]))
    }

    fn canonical_cost() -> Cost {
        let data = test_data();
        SizeCost::default().canonical_cost(&data, 4)
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

    /// A prior must reproduce the historical scheme-side policy arithmetic bit-for-bit:
    /// `effective = raw * multiplier` (the exact expression Delta used), gated at
    /// `effective > min_ratio`, with the effective ratio driving the cost.
    #[test]
    fn prior_reproduces_post_penalty_ratios_bit_for_bit() {
        let model = SizeCost::default().with_scheme_prior(
            TestScheme.id(),
            SchemePrior {
                multiplier: 0.95,
                min_ratio: 1.25,
            },
        );

        // Raw ratios spanning both sides of the floor, plus the exact boundary.
        let raws = [64.0, 64.0 / 9.0, 8.0, 2.0, 1.4, 1.25 / 0.95, 1.32, 1.3, 1.0];
        let data = test_data();
        for raw in raws {
            let expected = raw * 0.95;
            let candidate = Candidate::new(
                &TestScheme,
                CandidateEstimate::from_compression_ratio(raw),
                &data,
                None,
                &[],
            );
            let cost = model.cost(&candidate);
            if expected > 1.25 {
                let cost = cost.expect("above the floor must be priceable");
                // Bit-exact: the cost is the negated product, not a re-derived value.
                assert_eq!(cost.value(), -expected, "raw {raw}");
            } else {
                assert!(cost.is_none(), "raw {raw} must be gated by the floor");
            }
        }

        // Schemes without an entry are priced off the raw ratio unchanged.
        assert_eq!(cost(Some(1.1)).map(Cost::value), Some(-1.1));
    }
}
