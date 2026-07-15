// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A deterministic read-time estimate for sequential scans.

use vortex_utils::aliases::hash_map::HashMap;

use crate::cost::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;

/// Estimates sequential read time as I/O time plus scheme decode time.
///
/// The model prices a candidate as:
///
/// `estimated_bytes / effective_bandwidth + values * decode_ns_per_value`
///
/// `effective_bandwidth` is expressed in bytes per nanosecond (numerically equal to decimal
/// GB/s), while scheme charges are expressed in nanoseconds per value. The coefficients are
/// configuration, never measurements taken on the write path, so selection remains deterministic.
///
/// This intentionally models only the scheme currently being selected. It does not yet price
/// operation pushdown, random access, or the complete produced encoding tree; those require the
/// capability and calibration artifacts described by the cost-model tracking plan.
#[derive(Debug, Clone)]
pub struct ReadTimeCost {
    /// Estimated bytes read per nanosecond.
    effective_bandwidth: f64,
    /// Decode charge used when a scheme has no explicit calibration.
    fallback_decode_ns_per_value: f64,
    /// Per-scheme decode charges in nanoseconds per value.
    decode_ns_per_value: HashMap<SchemeId, f64>,
}

impl ReadTimeCost {
    /// Creates a read-time model with a conservative charge for unregistered schemes.
    ///
    /// `effective_bandwidth` is bytes per nanosecond and `fallback_decode_ns_per_value` is
    /// nanoseconds per decoded value.
    ///
    /// # Panics
    ///
    /// Panics if either argument is non-finite, if bandwidth is not positive, or if the fallback
    /// decode charge is negative.
    pub fn new(effective_bandwidth: f64, fallback_decode_ns_per_value: f64) -> Self {
        assert!(
            effective_bandwidth.is_finite() && effective_bandwidth > 0.0,
            "effective bandwidth must be finite and positive"
        );
        assert!(
            (u64::MAX as f64 / effective_bandwidth).is_finite(),
            "effective bandwidth is too small to produce finite costs"
        );
        assert!(
            fallback_decode_ns_per_value.is_finite() && fallback_decode_ns_per_value >= 0.0,
            "fallback decode cost must be finite and non-negative"
        );
        Self {
            effective_bandwidth,
            fallback_decode_ns_per_value,
            decode_ns_per_value: HashMap::default(),
        }
    }

    /// Sets the decode charge for a scheme in nanoseconds per value.
    ///
    /// # Panics
    ///
    /// Panics if the charge is non-finite or negative.
    pub fn with_scheme_cost(mut self, scheme: SchemeId, decode_ns_per_value: f64) -> Self {
        assert!(
            decode_ns_per_value.is_finite() && decode_ns_per_value >= 0.0,
            "scheme decode cost must be finite and non-negative"
        );
        self.decode_ns_per_value.insert(scheme, decode_ns_per_value);
        self
    }

    /// Returns the configured effective bandwidth in bytes per nanosecond.
    pub fn effective_bandwidth(&self) -> f64 {
        self.effective_bandwidth
    }
}

impl CostModel for ReadTimeCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        let ratio = candidate.estimate().estimated_compression_ratio()?;
        // A larger representation cannot beat canonical because decode charges are non-negative.
        if !ratio.is_finite() || ratio.is_subnormal() || ratio <= 1.0 {
            return None;
        }

        let estimated_nbytes = candidate.input_nbytes() as f64 / ratio;
        let decode_ns_per_value = self
            .decode_ns_per_value
            .get(&candidate.scheme_id())
            .copied()
            .unwrap_or(self.fallback_decode_ns_per_value);
        let cost = estimated_nbytes / self.effective_bandwidth
            + candidate.n_values() as f64 * decode_ns_per_value;
        cost.is_finite().then(|| Cost::new(cost))
    }

    fn canonical_cost(&self, data: &ArrayAndStats, _n_values: u64) -> Cost {
        Cost::new(data.array().nbytes() as f64 / self.effective_bandwidth)
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
            "test.read_time_cost"
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
    fn combines_io_and_decode_time() {
        let data = test_data();
        let model = ReadTimeCost::new(4.0, 1.0).with_scheme_cost(TestScheme.id(), 0.5);
        let candidate = Candidate::new(
            &TestScheme,
            CandidateEstimate::from_compression_ratio(2.0),
            &data,
            None,
            &[],
        );

        // 16 input bytes / ratio 2 / 4 bytes/ns + 4 values * 0.5 ns/value.
        assert_eq!(model.cost(&candidate).map(Cost::value), Some(4.0));
        assert_eq!(model.canonical_cost(&data, 4).value(), 4.0);
    }

    #[test]
    fn decode_charge_can_reject_a_smaller_candidate() {
        let data = test_data();
        let model = ReadTimeCost::new(4.0, 1.0);
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
                .is_some_and(|cost| { cost >= model.canonical_cost(&data, candidate.n_values()) })
        );
    }
}
