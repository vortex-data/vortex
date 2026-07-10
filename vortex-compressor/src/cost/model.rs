// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of [`CostModel`] and [`Cost`].

use std::cmp::Ordering;
use std::fmt::Debug;

use crate::cost::candidate::Candidate;
use crate::stats::ArrayAndStats;

/// Computes costs for [`Candidate`]s during scheme selection.
///
/// Implementations must be deterministic pure functions of their inputs and cheap: `cost` is
/// called once per candidate per selection site, i.e. `O(schemes × cascade levels)`
/// times per compressed chunk.
///
/// Selection semantics: the winner is the candidate with the minimum cost, requiring strictly
/// lower cost both to displace the running best and to beat [`canonical_cost`]. Equal-cost ties
/// favor the candidate evaluated first: scheme registration order within each selection pass,
/// and immediate candidates over deferred candidates across passes. Returning `None` from [`cost`]
/// rejects the candidate outright.
///
/// See the [module docs](crate::cost) for the acceptance axioms that apply to all models.
///
/// [`canonical_cost`]: CostModel::canonical_cost
/// [`cost`]: CostModel::cost
pub trait CostModel: Debug + Send + Sync + 'static {
    /// Estimated cost of choosing this candidate. Lower is better; `None` rejects the
    /// candidate.
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost>;

    /// Cost of leaving the array canonical — the baseline every candidate must strictly
    /// beat to be selected.
    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost;
}

/// An opaque, totally ordered cost. Lower is better.
///
/// Units are defined by each [`CostModel`], not by the framework: costs are only ever
/// compared *within* one model on one compressor instance, so cross-model comparability is
/// intentionally not provided.
///
/// Values must be finite. `NaN` and infinite costs are not representable orderings; a model
/// that cannot compute a candidate's cost rejects it by returning `None` from [`CostModel::cost`].
#[derive(Debug, Clone, Copy)]
pub struct Cost(f64);

impl Cost {
    /// Creates a cost from a finite value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not finite.
    pub fn new(value: f64) -> Self {
        assert!(value.is_finite(), "Cost must be finite, got {value}");
        Self(if value == 0.0 { 0.0 } else { value })
    }

    /// Returns the raw cost value.
    pub fn value(self) -> f64 {
        self.0
    }
}

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::Cost;

    #[test]
    fn rejects_non_finite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| Cost::new(value)).is_err(),
                "non-finite cost {value} was accepted"
            );
        }
    }

    #[test]
    fn signed_zero_has_one_representation() {
        let negative = Cost::new(-0.0);
        let positive = Cost::new(0.0);

        assert_eq!(negative, positive);
        assert_eq!(negative.cmp(&positive), Ordering::Equal);
        assert_eq!(negative.value().to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn costs_have_a_total_order() {
        let mut costs = [Cost::new(3.0), Cost::new(-2.0), Cost::new(0.0)];
        costs.sort();

        assert_eq!(costs.map(Cost::value), [-2.0, 0.0, 3.0]);
    }
}
