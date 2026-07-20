// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A workload model for equality- and pattern-heavy string predicates.

use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_compressor::cost::Candidate;
use vortex_compressor::cost::Cost;
use vortex_compressor::cost::CostModel;
use vortex_compressor::cost::OperationCosts;
use vortex_compressor::cost::OperationWeights;
use vortex_compressor::cost::SizeCost;
use vortex_compressor::cost::WorkloadCost;
use vortex_compressor::stats::ArrayAndStats;

use crate::SchemeExt;
use crate::schemes::string::FSSTScheme;
use crate::schemes::string::StringDictScheme;

/// Experimental cost model for string-heavy filtering workloads.
///
/// The profile represents one scalar comparison, one `LIKE` evaluation, and a 10% chance of full
/// materialization per string value. Calibrations come from
/// `vortex-btrblocks/benches/predicate_strings.rs` and intentionally make FSST's specialized
/// pattern kernel compete with dictionary encoding's smaller representation.
///
/// Non-string candidates retain [`SizeCost`] behavior, so the experiment does not globally alter
/// integer, float, or descendant selection. UTF-8 candidates without calibrations are rejected,
/// limiting the experiment to canonical, dictionary, and FSST representations. The default
/// compressor also remains size-oriented; callers opt into this model with
/// [`crate::BtrBlocksCompressorBuilder::with_predicate_strings`].
#[derive(Debug, Clone)]
pub struct PredicateStringCost {
    string_cost: WorkloadCost,
}

impl Default for PredicateStringCost {
    fn default() -> Self {
        Self::new(OperationWeights {
            full_decode: 0.1,
            compare: 1.0,
            like: 1.0,
        })
    }
}

impl PredicateStringCost {
    /// Creates a calibrated UTF-8 cost model for the supplied operation mix.
    ///
    /// # Panics
    ///
    /// Panics if any operation weight is non-finite or negative.
    pub fn new(weights: OperationWeights) -> Self {
        let canonical = OperationCosts {
            full_decode: 0.0,
            compare: 2.0,
            like: 4.8,
        };

        Self {
            string_cost: WorkloadCost::new(20.0, weights, canonical, canonical)
                .with_scheme_cost(
                    FSSTScheme.id(),
                    OperationCosts {
                        full_decode: 5.7,
                        compare: 1.25,
                        like: 6.7,
                    },
                )
                .with_scheme_cost(
                    StringDictScheme.id(),
                    OperationCosts {
                        // Conservative fallback when a dictionary estimate has no sample.
                        full_decode: 14.3,
                        compare: 2.47,
                        like: 22.0,
                    },
                ),
        }
    }

    /// Creates a profile for scalar equality predicates.
    pub fn equality() -> Self {
        Self::new(OperationWeights {
            compare: 1.0,
            ..Default::default()
        })
    }

    /// Creates a profile for `LIKE` predicates followed by selective materialization.
    pub fn like() -> Self {
        Self::new(OperationWeights {
            full_decode: 0.1,
            like: 1.0,
            ..Default::default()
        })
    }
}

impl CostModel for PredicateStringCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        if !candidate.array().dtype().is_utf8() {
            return SizeCost.cost(candidate);
        }

        let scheme_id = candidate.scheme_id();
        if scheme_id != StringDictScheme.id() && scheme_id != FSSTScheme.id() {
            return None;
        }

        if scheme_id == StringDictScheme.id()
            && let Some(dict) = candidate
                .sampled()
                .and_then(|sample| sample.as_opt::<Dict>())
        {
            let distinct_fraction = dict.values().len() as f64 / dict.len() as f64;
            return self.string_cost.cost_with_operations(
                candidate,
                OperationCosts {
                    // Dictionary decode and predicates pay a fixed codes scan plus work that
                    // scales with the dictionary cardinality.
                    full_decode: 8.7 + 5.6 * distinct_fraction,
                    compare: 0.27 + 2.2 * distinct_fraction,
                    like: 13.4 + 8.6 * distinct_fraction,
                },
            );
        }

        self.string_cost.cost(candidate)
    }

    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost {
        if data.array().dtype().is_utf8() {
            self.string_cost.canonical_cost(data, n_values)
        } else {
            SizeCost.canonical_cost(data, n_values)
        }
    }
}
