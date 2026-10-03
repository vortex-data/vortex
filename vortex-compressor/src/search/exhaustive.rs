// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exhaustive plan search.

use std::sync::Arc;
use std::time::Duration;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use super::Candidate;
use super::CostModel;
use super::Measurement;
use super::Timings;
use super::search_input;
use super::time_decompress;
use crate::CascadingCompressor;
use crate::plan::Plan;
use crate::plan::PlanRecorder;
use crate::plan::Selection;
use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::trace;

/// Finds the cheapest plan for one array under a [`CostModel`] by trying every scheme at every
/// compression site.
///
/// At each site the search tries canonical and every eligible scheme. A scheme's children are
/// searched recursively before the scheme itself is measured, so each candidate carries the
/// cheapest plan for each of its children. Children are scored independently, which is exact
/// when costs add up across a cascade (sizes do, decompression times nearly do), and keeps the
/// search to roughly one compression per scheme per child site instead of one per cascade.
///
/// The search honors the compressor's cascade depth limit and exclusion rules, and skips schemes
/// whose estimate rules them out with an immediate [`Skip`](crate::scheme::EstimateVerdict::Skip)
/// verdict. Schemes that fail on the array are skipped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExhaustiveSearch {
    /// The cost plans are ranked by.
    pub cost: CostModel,
    /// How many times each timing is repeated. The fastest run is kept.
    pub iterations: usize,
}

impl ExhaustiveSearch {
    /// Creates a search that ranks plans by `cost`, timing each candidate three times.
    pub fn new(cost: CostModel) -> Self {
        Self {
            cost,
            iterations: 3,
        }
    }

    /// Sets how many times each timing is repeated.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Returns the cheapest plan for `array`, measured on every objective.
    ///
    /// # Errors
    ///
    /// Returns an error if `array` is not a leaf array, or if compression or decompression
    /// fails.
    pub fn run(
        &self,
        compressor: &CascadingCompressor,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Candidate> {
        let array = search_input(array, exec_ctx)?;

        let recorder = Arc::new(PlanRecorder::new(None));
        let compress_ctx = CompressorContext::new()
            .with_selection(Selection::Exhaustive {
                cost: self.cost,
                iterations: self.iterations,
            })
            .with_recorder(Some(Arc::clone(&recorder)));
        let canonical = array.clone().execute::<Canonical>(exec_ctx)?;
        compressor.compress_canonical(canonical, compress_ctx, exec_ctx)?;
        let plan = recorder.get(0).unwrap_or(Plan::Adaptive);

        let evaluation = compressor.evaluate_plan(
            &array,
            &plan,
            &CompressorContext::new(),
            Timings::ALL,
            self.iterations,
            exec_ctx,
        )?;
        Ok(Candidate {
            plan: evaluation.plan,
            measurement: evaluation.measurement,
        })
    }
}

/// The best candidate found so far at a compression site.
struct Best {
    /// The candidate's cost.
    cost: f64,
    /// The candidate's plan.
    plan: Plan,
    /// The array the candidate compressed to.
    array: ArrayRef,
}

impl CascadingCompressor {
    /// Compresses the array with the cheapest of canonical and every candidate scheme, each with
    /// the cheapest plan for its children, and records the winning plan.
    ///
    /// The context's selection must be [`Selection::Exhaustive`], so that the schemes' children
    /// are searched too.
    pub(crate) fn compress_exhaustively(
        &self,
        data: &ArrayAndStats,
        eligible_schemes: &[&'static dyn Scheme],
        compress_ctx: CompressorContext,
        cost: &CostModel,
        iterations: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let site_ctx = compress_ctx.clone().with_recorder(None);
        let timings = cost.timings();
        let canonical_nbytes = data.array().nbytes();

        let canonical = data.array().clone();
        let measurement = self.measure_candidate(
            &canonical,
            data.array(),
            &Plan::Canonical,
            &site_ctx,
            timings,
            iterations,
            exec_ctx,
        )?;
        let mut best = Best {
            cost: cost.cost(&measurement),
            plan: Plan::Canonical,
            array: canonical,
        };

        for scheme in self.candidate_schemes(eligible_schemes, data, &site_ctx, exec_ctx) {
            // The scheme's children record the plans their own searches settle on.
            let children = Arc::new(PlanRecorder::new(Some(scheme.id())));
            let scheme_ctx = site_ctx.clone().with_recorder(Some(Arc::clone(&children)));
            let compressed = match scheme.compress(self, data, scheme_ctx, exec_ctx) {
                Ok(compressed) => compressed,
                Err(err) => {
                    trace::candidate_failed(scheme.id(), &err);
                    continue;
                }
            };

            // The compressor keeps the canonical array when a scheme does not shrink it.
            if compressed.nbytes() >= canonical_nbytes {
                continue;
            }

            let plan = Plan::Scheme {
                scheme: scheme.id(),
                children: children.children(scheme.num_children()),
            };
            let measurement = self.measure_candidate(
                &compressed,
                data.array(),
                &plan,
                &site_ctx,
                timings,
                iterations,
                exec_ctx,
            )?;
            let candidate_cost = cost.cost(&measurement);
            if candidate_cost < best.cost {
                best = Best {
                    cost: candidate_cost,
                    plan,
                    array: compressed,
                };
            }
        }

        compress_ctx.record(best.plan);
        Ok(best.array)
    }

    /// Measures a candidate that compressed `input` into `compressed` with `plan`.
    ///
    /// Compression is timed by replaying `plan` on `input`, since the search's own compression
    /// includes the time spent searching the children.
    #[expect(
        clippy::too_many_arguments,
        reason = "internal helper threading search state"
    )]
    fn measure_candidate(
        &self,
        compressed: &ArrayRef,
        input: &ArrayRef,
        plan: &Plan,
        site_ctx: &CompressorContext,
        timings: Timings,
        iterations: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Measurement> {
        let compress_time = if timings.compress {
            let compress_only = Timings {
                compress: true,
                decompress: false,
            };
            self.evaluate_plan(input, plan, site_ctx, compress_only, iterations, exec_ctx)?
                .measurement
                .compress_time
        } else {
            Duration::ZERO
        };
        let decompress_time = if timings.decompress {
            time_decompress(compressed, iterations, exec_ctx)?
        } else {
            Duration::ZERO
        };
        Ok(Measurement {
            nbytes: compressed.nbytes(),
            compress_time,
            decompress_time,
        })
    }
}
