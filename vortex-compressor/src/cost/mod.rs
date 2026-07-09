// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable cost models for scheme selection.
//!
//! A [`CostModel`] is the policy half of scheme selection: [`Scheme`]s produce mechanical
//! signals (estimated or sample-measured compression ratios, collected into a [`Candidate`]),
//! and the model prices each candidate. The compressor picks the candidate with the lowest
//! cost that (strictly) beats [`CostModel::canonical_cost`] — the price of leaving the array
//! in its canonical encoding.
//!
//! The default model is [`SizeCost`], which reproduces the compressor's historical
//! ratio-argmax selection bit-exactly.
//!
//! # What sits outside the model
//!
//! Two selection mechanisms are deliberately *not* routed through the cost model:
//!
//! - [`EstimateVerdict::AlwaysUse`] short-circuits selection entirely. It expresses semantic
//!   normalization (e.g. decimal byte-parts, temporal decomposition), not a priced trade-off.
//! - The byte-acceptance gate: after the winning scheme compresses the full array, the result
//!   is kept only if it is byte-wise smaller than its input. This is an axiom for **all**
//!   models — compression never grows bytes. A model that prefers canonical (e.g. for speed)
//!   expresses that by pricing every candidate at or above `canonical_cost`, so selection
//!   returns no winner and the array stays canonical; the gate never forces a *bad* encoding,
//!   only "no encoding". (The gate's `AnyScalarFn` carve-out is likewise semantic
//!   denormalization, not cost.)
//!
//! # Determinism
//!
//! Cost models must be pure functions of the candidate and their own configuration: no
//! timing measurements, no I/O, no global state. The compressor's output must be a
//! deterministic function of its input and configuration.
//!
//! [`Scheme`]: crate::scheme::Scheme
//! [`EstimateVerdict::AlwaysUse`]: crate::estimate::EstimateVerdict::AlwaysUse

mod size;

use std::fmt::Debug;

pub use size::SizeCost;

pub use crate::candidate::Candidate;
use crate::stats::ArrayAndStats;

/// An opaque, totally ordered cost. Lower is better.
///
/// Units are defined by each [`CostModel`], not by the framework: costs are only ever
/// compared *within* one model on one compressor instance, so cross-model comparability is
/// intentionally not provided.
///
/// Values must be finite. `NaN`/infinite prices are not representable orderings; a model
/// that cannot price a candidate rejects it by returning `None` from [`CostModel::cost`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Cost(f64);

impl Cost {
    /// Creates a cost from a finite value.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `value` is not finite.
    pub fn new(value: f64) -> Self {
        debug_assert!(value.is_finite(), "Cost must be finite, got {value}");
        Self(value)
    }

    /// Returns the raw cost value.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Prices [`Candidate`]s during scheme selection.
///
/// Implementations must be deterministic pure functions of their inputs and cheap: `cost` is
/// called once per scored candidate per selection site, i.e. `O(schemes × cascade levels)`
/// times per compressed chunk.
///
/// Selection semantics: the winner is the candidate with the minimum cost, requiring
/// strictly lower cost both to displace the running best (ties break by scheme registration
/// order) and to beat [`canonical_cost`]. Returning `None` from [`cost`] rejects the
/// candidate outright.
///
/// See the [module docs](self) for the acceptance axioms that apply to all models.
///
/// [`canonical_cost`]: CostModel::canonical_cost
/// [`cost`]: CostModel::cost
pub trait CostModel: Debug + Send + Sync + 'static {
    /// Estimated cost of choosing this candidate. Lower is better; `None` rejects the
    /// candidate.
    fn cost(&self, candidate: &Candidate) -> Option<Cost>;

    /// Cost of leaving the array canonical — the baseline every candidate must strictly
    /// beat to be selected.
    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost;
}
