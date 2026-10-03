// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Offline searches for compression [`Plan`]s.
//!
//! The default compressor picks one scheme per compression site by estimated compression ratio.
//! The searches here instead measure whole cascades on a training sample, on three objectives:
//! compressed size, compression time and decompression time.
//!
//! - [`ExhaustiveSearch`] tries every scheme at every site and keeps the cheapest under one
//!   [`CostModel`]. Each child's subtree depends only on the child array, so the search scores
//!   each child independently instead of enumerating whole cascades.
//! - [`GeneticSearch`] runs NSGA-II over whole plans and returns the Pareto front: the plans that
//!   no other plan beats on all three objectives at once.
//!
//! Both return plans that [`CascadingCompressor::compress_with_plan`] can replay on new data.
//!
//! Searches apply to one leaf array (a column) at a time: booleans, primitives, decimals, strings
//! or binary.

mod exhaustive;
mod genetic;

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

pub use exhaustive::ExhaustiveSearch;
pub use genetic::GeneticSearch;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CascadingCompressor;
use crate::plan::Plan;
use crate::plan::PlanRecorder;
use crate::plan::Selection;
use crate::scheme::CompressorContext;

/// The measured cost of compressing one array with one plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measurement {
    /// Size of the compressed array's buffers, in bytes, as reported by [`ArrayRef::nbytes`].
    /// Serialization adds a little metadata per array on top.
    pub nbytes: u64,
    /// Fastest observed time to compress the array with the plan.
    pub compress_time: Duration,
    /// Fastest observed time to decompress the compressed array into its canonical form.
    pub decompress_time: Duration,
}

impl Measurement {
    /// Returns the objectives, all to be minimized: bytes, compression seconds and decompression
    /// seconds.
    pub fn objectives(&self) -> [f64; 3] {
        [
            self.nbytes as f64,
            self.compress_time.as_secs_f64(),
            self.decompress_time.as_secs_f64(),
        ]
    }

    /// Returns `true` if this measurement is no worse than `other` on every objective and
    /// strictly better on at least one.
    pub fn dominates(&self, other: &Self) -> bool {
        dominates(&self.objectives(), &other.objectives())
    }
}

/// Returns `true` if `a` is no worse than `b` on every objective and better on at least one.
fn dominates(a: &[f64; 3], b: &[f64; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

/// A plan together with its measured cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The plan, as applied to the training array.
    pub plan: Plan,
    /// The plan's cost on the training array.
    pub measurement: Measurement,
}

/// Collapses a [`Measurement`] into one cost, where lower is better.
///
/// The cost is a weighted sum:
///
/// ```text
/// cost = bytes_weight * nbytes + decompress_weight * decompress_seconds
///      + compress_weight * compress_seconds
/// ```
///
/// [`read_time`](Self::read_time) gives the weights a physical meaning: the seconds needed to
/// fetch the compressed bytes over a link of the given bandwidth and then decompress them. Slow
/// storage favors smaller plans, and fast storage favors plans that decompress quickly.
/// Sweeping the bandwidth traces the convex part of the Pareto front.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    /// Cost per compressed byte.
    pub bytes_weight: f64,
    /// Cost per second of decompression.
    pub decompress_weight: f64,
    /// Cost per second of compression.
    pub compress_weight: f64,
}

impl CostModel {
    /// Ranks plans by compressed size alone.
    pub fn size() -> Self {
        Self {
            bytes_weight: 1.0,
            decompress_weight: 0.0,
            compress_weight: 0.0,
        }
    }

    /// Ranks plans by the seconds needed to read the compressed bytes at `bytes_per_second` and
    /// then decompress them.
    pub fn read_time(bytes_per_second: f64) -> Self {
        Self {
            bytes_weight: 1.0 / bytes_per_second,
            decompress_weight: 1.0,
            compress_weight: 0.0,
        }
    }

    /// Adds the compression time, amortized over `reads` reads of each compressed array.
    pub fn with_reads_per_write(mut self, reads: f64) -> Self {
        self.compress_weight = 1.0 / reads;
        self
    }

    /// Returns the cost of `measurement`.
    pub fn cost(&self, measurement: &Measurement) -> f64 {
        self.bytes_weight * measurement.nbytes as f64
            + self.decompress_weight * measurement.decompress_time.as_secs_f64()
            + self.compress_weight * measurement.compress_time.as_secs_f64()
    }

    /// Returns the cheapest of `candidates`, such as a Pareto front from [`GeneticSearch`].
    pub fn cheapest<'a>(&self, candidates: &'a [Candidate]) -> Option<&'a Candidate> {
        candidates.iter().min_by(|a, b| {
            self.cost(&a.measurement)
                .total_cmp(&self.cost(&b.measurement))
        })
    }

    /// Returns which timings this model needs.
    fn timings(&self) -> Timings {
        Timings {
            compress: self.compress_weight != 0.0,
            decompress: self.decompress_weight != 0.0,
        }
    }
}

/// Which timings to take when measuring a plan.
#[derive(Debug, Clone, Copy)]
struct Timings {
    /// Whether to time compression.
    compress: bool,
    /// Whether to time decompression.
    decompress: bool,
}

impl Timings {
    /// Takes every timing.
    const ALL: Self = Self {
        compress: true,
        decompress: true,
    };
}

/// The plan that compressed an array, and its measured cost.
struct Evaluation {
    /// The plan as applied, with any fallback decisions filled in.
    plan: Plan,
    /// The measured cost.
    measurement: Measurement,
}

/// Canonicalizes a search input and checks that it is a leaf array.
fn search_input(array: &ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let canonical = CascadingCompressor::canonicalize_input(array, exec_ctx)?;
    match canonical {
        Canonical::Bool(_)
        | Canonical::Primitive(_)
        | Canonical::Decimal(_)
        | Canonical::VarBinView(_) => Ok(canonical.into()),
        _ => vortex_bail!(
            "plan search expects a boolean, primitive, decimal, string or binary array, got {}",
            array.dtype()
        ),
    }
}

/// Returns the fastest of `iterations` timed decompressions of `array`.
fn time_decompress(
    array: &ArrayRef,
    iterations: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Duration> {
    let mut fastest = Duration::MAX;
    for _ in 0..iterations.max(1) {
        let start = Instant::now();
        let decompressed = array.clone().execute::<CanonicalValidity>(exec_ctx)?;
        fastest = fastest.min(start.elapsed());
        drop(black_box(decompressed));
    }
    Ok(fastest)
}

impl CascadingCompressor {
    /// Compresses the canonical `array` at the compression site described by `site_ctx`,
    /// following `plan` and using `fallback` where the plan does not decide.
    ///
    /// Returns the compressed array, the plan as applied, and how long compression took.
    fn apply_plan(
        &self,
        array: &ArrayRef,
        plan: &Plan,
        site_ctx: &CompressorContext,
        fallback: Selection,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Plan, Duration)> {
        let (parent, child_index) = site_ctx.site();
        let recorder = Arc::new(PlanRecorder::new(parent));
        let compress_ctx = site_ctx
            .clone()
            .with_selection(Selection::follow(plan.clone(), fallback))
            .with_recorder(Some(Arc::clone(&recorder)));

        let canonical = array.clone().execute::<Canonical>(exec_ctx)?;
        let start = Instant::now();
        let compressed = self.compress_canonical(canonical, compress_ctx, exec_ctx)?;
        let elapsed = start.elapsed();

        let applied = recorder.get(child_index).unwrap_or(Plan::Adaptive);
        Ok((compressed, applied, elapsed))
    }

    /// Compresses and measures the canonical `array` with `plan`.
    ///
    /// Sites the plan does not decide are compressed by estimate-based selection. Each requested
    /// timing keeps the fastest of `iterations` runs. Timings that are not requested are zero.
    fn evaluate_plan(
        &self,
        array: &ArrayRef,
        plan: &Plan,
        site_ctx: &CompressorContext,
        timings: Timings,
        iterations: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Evaluation> {
        let runs = if timings.compress {
            iterations.max(1)
        } else {
            1
        };

        let mut fastest = Duration::MAX;
        let mut result = None;
        for _ in 0..runs {
            let (compressed, applied, elapsed) =
                self.apply_plan(array, plan, site_ctx, Selection::Estimate, exec_ctx)?;
            fastest = fastest.min(elapsed);
            result = Some((compressed, applied));
        }
        let (compressed, applied) = result.vortex_expect("at least one compression run");

        let decompress_time = if timings.decompress {
            time_decompress(&compressed, iterations, exec_ctx)?
        } else {
            Duration::ZERO
        };

        Ok(Evaluation {
            measurement: Measurement {
                nbytes: compressed.nbytes(),
                compress_time: if timings.compress {
                    fastest
                } else {
                    Duration::ZERO
                },
                decompress_time,
            },
            plan: applied,
        })
    }

    /// Measures `plan` on `array`, taking every timing.
    ///
    /// The plan is applied as in [`compress_with_plan`](Self::compress_with_plan). The returned
    /// candidate holds the plan as applied, with any fallback decisions filled in.
    ///
    /// # Errors
    ///
    /// Returns an error if `array` is not a leaf array, or if compression or decompression
    /// fails.
    pub fn measure_plan(
        &self,
        array: &ArrayRef,
        plan: &Plan,
        iterations: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Candidate> {
        let array = search_input(array, exec_ctx)?;
        let evaluation = self.evaluate_plan(
            &array,
            plan,
            &CompressorContext::new(),
            Timings::ALL,
            iterations,
            exec_ctx,
        )?;
        Ok(Candidate {
            plan: evaluation.plan,
            measurement: evaluation.measurement,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measurement(nbytes: u64, compress_micros: u64, decompress_micros: u64) -> Measurement {
        Measurement {
            nbytes,
            compress_time: Duration::from_micros(compress_micros),
            decompress_time: Duration::from_micros(decompress_micros),
        }
    }

    #[test]
    fn dominance() {
        let base = measurement(100, 10, 10);
        assert!(measurement(90, 10, 10).dominates(&base));
        assert!(!base.dominates(&base));
        assert!(!measurement(90, 20, 10).dominates(&base));
        assert!(!base.dominates(&measurement(90, 20, 10)));
    }

    #[test]
    fn read_time_trades_bytes_for_decompression() {
        let small_slow = Candidate {
            plan: Plan::Canonical,
            measurement: measurement(1_000, 0, 1_000),
        };
        let large_fast = Candidate {
            plan: Plan::Adaptive,
            measurement: measurement(100_000, 0, 10),
        };
        let front = [small_slow.clone(), large_fast.clone()];

        // At 1 MB/s, 99 kB more costs 99 ms, far more than the 1 ms decompression.
        let slow_link = CostModel::read_time(1e6);
        assert_eq!(slow_link.cheapest(&front), Some(&small_slow));

        // At 1 TB/s, the bytes are nearly free and decompression dominates.
        let fast_link = CostModel::read_time(1e12);
        assert_eq!(fast_link.cheapest(&front), Some(&large_fast));

        assert_eq!(CostModel::size().cheapest(&front), Some(&small_slow));
    }
}
