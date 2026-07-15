// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A calibrated cost model for analytical scans.

use vortex_compressor::cost::Candidate;
use vortex_compressor::cost::Cost;
use vortex_compressor::cost::CostModel;
use vortex_compressor::cost::ReadTimeCost;
use vortex_compressor::stats::ArrayAndStats;

use crate::SchemeExt;
use crate::schemes::integer::RunEndScheme;

/// Estimates cached analytical scan time instead of minimizing encoded size alone.
///
/// The model combines estimated bytes with deterministic per-scheme decode charges. Its initial
/// calibration assumes 20 bytes/ns effective read bandwidth and charges RunEnd 0.12 ns/value.
/// This avoids selecting RunEnd for modest size wins when a faster stable encoding is available.
/// Schemes without a calibration currently receive no decode charge.
///
/// Use [`crate::SizeCost`] with
/// [`crate::BtrBlocksCompressorBuilder::with_cost_model`] when minimum encoded size is the desired
/// objective. Use [`ReadTimeCost`] directly to supply workload- or hardware-specific coefficients.
#[derive(Debug, Clone)]
pub struct ScanCost(ReadTimeCost);

impl Default for ScanCost {
    fn default() -> Self {
        Self(ReadTimeCost::new(20.0, 0.0).with_scheme_cost(RunEndScheme.id(), 0.12))
    }
}

impl CostModel for ScanCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        self.0.cost(candidate)
    }

    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost {
        self.0.canonical_cost(data, n_values)
    }
}
