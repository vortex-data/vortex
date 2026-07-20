// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic operation-aware workload costs.

use vortex_utils::aliases::hash_map::HashMap;

use crate::cost::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;

/// Expected operation counts per value for a static workload.
///
/// The weights need not sum to one. For example, a profile that expects one comparison and one
/// `LIKE` evaluation per value uses `compare = 1.0` and `like = 1.0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OperationWeights {
    /// Expected full materializations per value.
    pub full_decode: f64,
    /// Expected scalar comparisons per value.
    pub compare: f64,
    /// Expected string pattern matches per value.
    pub like: f64,
}

/// Estimated nanoseconds per value for operations on one representation.
#[derive(Debug, Clone, Copy, Default)]
pub struct OperationCosts {
    /// Nanoseconds per value to fully materialize the representation.
    pub full_decode: f64,
    /// Nanoseconds per value to compare the representation with a scalar.
    pub compare: f64,
    /// Nanoseconds per value to evaluate a string pattern against the representation.
    pub like: f64,
}

impl OperationCosts {
    /// Returns the weighted cost for one value.
    fn weighted(self, weights: OperationWeights) -> f64 {
        self.full_decode * weights.full_decode
            + self.compare * weights.compare
            + self.like * weights.like
    }

    /// Returns whether every cost is finite and non-negative.
    fn is_valid(self) -> bool {
        [self.full_decode, self.compare, self.like]
            .into_iter()
            .all(|cost| cost.is_finite() && cost >= 0.0)
    }
}

impl OperationWeights {
    /// Returns whether every weight is finite and non-negative.
    fn is_valid(self) -> bool {
        [self.full_decode, self.compare, self.like]
            .into_iter()
            .all(|weight| weight.is_finite() && weight >= 0.0)
    }
}

/// Prices compression candidates for a fixed operation mix.
///
/// A candidate is priced as:
///
/// `estimated_bytes / effective_bandwidth + values * weighted_operation_ns_per_value`
///
/// `effective_bandwidth` is bytes per nanosecond (numerically equal to decimal GB/s). Operation
/// costs are deterministic calibration data, never timings collected while writing a file.
/// Unknown schemes use the configured fallback costs.
///
/// This model prices the scheme currently being selected. Because the compressor invokes the
/// model recursively, the same workload policy also applies to descendant selection. It does not
/// yet search multiple completed encoding trees or model cross-column query dependencies.
#[derive(Debug, Clone)]
pub struct WorkloadCost {
    /// Assumed storage throughput in bytes per nanosecond.
    effective_bandwidth: f64,
    /// Expected operation counts per value.
    weights: OperationWeights,
    /// Operation costs for the canonical representation.
    canonical_costs: OperationCosts,
    /// Operation costs for schemes without a calibration.
    fallback_costs: OperationCosts,
    /// Per-scheme operation calibrations.
    scheme_costs: HashMap<SchemeId, OperationCosts>,
}

impl WorkloadCost {
    /// Creates an operation-aware cost model.
    ///
    /// # Panics
    ///
    /// Panics if bandwidth is not finite and positive, or if any weight or operation cost is
    /// non-finite or negative.
    pub fn new(
        effective_bandwidth: f64,
        weights: OperationWeights,
        canonical_costs: OperationCosts,
        fallback_costs: OperationCosts,
    ) -> Self {
        assert!(
            effective_bandwidth.is_finite() && effective_bandwidth > 0.0,
            "effective bandwidth must be finite and positive"
        );
        assert!(
            (u64::MAX as f64 / effective_bandwidth).is_finite(),
            "effective bandwidth is too small to produce finite costs"
        );
        assert!(
            weights.is_valid(),
            "operation weights must be finite and non-negative"
        );
        assert!(
            canonical_costs.is_valid(),
            "canonical operation costs must be finite and non-negative"
        );
        assert!(
            fallback_costs.is_valid(),
            "fallback operation costs must be finite and non-negative"
        );
        assert!(
            canonical_costs.weighted(weights).is_finite()
                && fallback_costs.weighted(weights).is_finite(),
            "weighted operation costs must be finite"
        );

        Self {
            effective_bandwidth,
            weights,
            canonical_costs,
            fallback_costs,
            scheme_costs: HashMap::default(),
        }
    }

    /// Sets the calibrated operation costs for one scheme.
    ///
    /// # Panics
    ///
    /// Panics if any cost is non-finite or negative.
    pub fn with_scheme_cost(mut self, scheme: SchemeId, costs: OperationCosts) -> Self {
        assert!(
            costs.is_valid(),
            "scheme operation costs must be finite and non-negative"
        );
        assert!(
            costs.weighted(self.weights).is_finite(),
            "weighted scheme operation cost must be finite"
        );
        self.scheme_costs.insert(scheme, costs);
        self
    }

    /// Returns the configured effective bandwidth in bytes per nanosecond.
    pub fn effective_bandwidth(&self) -> f64 {
        self.effective_bandwidth
    }

    /// Prices a candidate with operation costs derived from candidate-specific evidence.
    ///
    /// This is useful when a representation's execution cost depends on its sampled shape, such
    /// as dictionary predicates scaling with the number of dictionary values.
    ///
    /// # Panics
    ///
    /// Panics if any operation cost is non-finite or negative.
    pub fn cost_with_operations(
        &self,
        candidate: &Candidate<'_>,
        operation_costs: OperationCosts,
    ) -> Option<Cost> {
        assert!(
            operation_costs.is_valid(),
            "operation costs must be finite and non-negative"
        );
        let weighted_operation_cost = operation_costs.weighted(self.weights);
        assert!(
            weighted_operation_cost.is_finite(),
            "weighted operation cost must be finite"
        );

        let ratio = candidate.estimate().estimated_compression_ratio()?;
        if !ratio.is_finite() || ratio.is_subnormal() || ratio <= 1.0 {
            return None;
        }

        let estimated_nbytes = candidate.input_nbytes() as f64 / ratio;
        let cost = estimated_nbytes / self.effective_bandwidth
            + candidate.n_values() as f64 * weighted_operation_cost;
        cost.is_finite().then(|| Cost::new(cost))
    }
}

impl CostModel for WorkloadCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        let operation_costs = self
            .scheme_costs
            .get(&candidate.scheme_id())
            .copied()
            .unwrap_or(self.fallback_costs);
        self.cost_with_operations(candidate, operation_costs)
    }

    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost {
        Cost::new(
            data.array().nbytes() as f64 / self.effective_bandwidth
                + n_values as f64 * self.canonical_costs.weighted(self.weights),
        )
    }
}

#[cfg(test)]
mod tests {
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
            "test.workload_cost"
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
            unreachable!("test helper is never selected")
        }
    }

    fn test_data() -> ArrayAndStats {
        ArrayAndStats::new(
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array(),
            GenerateStatsOptions::default(),
        )
    }

    #[test]
    fn combines_io_and_weighted_operation_time() {
        let data = test_data();
        let model = WorkloadCost::new(
            4.0,
            OperationWeights {
                compare: 1.0,
                like: 0.5,
                ..Default::default()
            },
            OperationCosts::default(),
            OperationCosts::default(),
        )
        .with_scheme_cost(
            TestScheme.id(),
            OperationCosts {
                compare: 0.5,
                like: 1.0,
                ..Default::default()
            },
        );
        let candidate = Candidate::new(
            &TestScheme,
            CandidateEstimate::from_compression_ratio(2.0),
            &data,
            None,
            &[],
        );

        // 16 input bytes / ratio 2 / 4 bytes/ns + 4 values * (0.5 + 0.5 * 1.0).
        assert_eq!(model.cost(&candidate).map(Cost::value), Some(6.0));
    }

    #[test]
    fn canonical_includes_operation_time() {
        let data = test_data();
        let model = WorkloadCost::new(
            4.0,
            OperationWeights {
                like: 0.5,
                ..Default::default()
            },
            OperationCosts {
                like: 2.0,
                ..Default::default()
            },
            OperationCosts::default(),
        );

        // 16 bytes / 4 bytes/ns + 4 values * 0.5 expected LIKEs * 2 ns/value.
        assert_eq!(model.canonical_cost(&data, 4).value(), 8.0);
    }

    #[test]
    fn can_price_candidate_above_canonical() {
        let data = test_data();
        let model = WorkloadCost::new(
            4.0,
            OperationWeights {
                like: 1.0,
                ..Default::default()
            },
            OperationCosts::default(),
            OperationCosts {
                like: 1.0,
                ..Default::default()
            },
        );
        let candidate = Candidate::new(
            &TestScheme,
            CandidateEstimate::from_compression_ratio(2.0),
            &data,
            None,
            &[],
        );

        assert!(
            model
                .cost(&candidate)
                .is_some_and(|cost| cost >= model.canonical_cost(&data, candidate.n_values()))
        );
    }
}
