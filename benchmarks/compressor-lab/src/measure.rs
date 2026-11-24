// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Quality measures of candidate encodings: decode, pushdown operations, peak memory and
//! correctness, timed so that candidates of one chunk can be ranked reliably.
//!
//! The method, chosen to keep rankings stable rather than nanoseconds exact:
//!
//! - **Correctness first.** Each operation's result on a candidate is compared with the same
//!   operation on the canonical array before anything is timed. A wrong candidate is not timed.
//! - **Batching.** An operation is repeated inside one sample until the sample takes at least
//!   [`TimingConfig::min_sample_ns`], so timer resolution does not dominate small operations.
//! - **Interleaving.** Samples are taken round-robin across the candidates, so slow drift (thermal,
//!   frequency, noisy neighbours) affects every candidate alike.
//! - **Warm-up and adaptive stopping.** One discarded run, then samples until the 95% interval of
//!   the median is within [`TimingConfig::target_ci`] of it, or the sample or time cap is hit.
//! - **Raw samples kept.** Medians, minimums and intervals are derived at analysis time.
//!
//! The canonical array is always measured as the candidate named [`CANONICAL`], so
//! "decode then operate" is `decode + canonical op` at analysis time, and pushdown speedup follows.

use std::hint::black_box;
use std::time::Instant;

use anyhow::Context;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_arrow::ExecuteArrowExt;
use vortex_mask::Mask;

use crate::alloc::MemoryUse;
use crate::alloc::Region;

/// The name the canonical array is measured under.
pub const CANONICAL: &str = "canonical";

/// How much effort to spend timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfig {
    /// Samples before the stopping rule is checked.
    pub min_samples: usize,
    /// Samples per candidate and operation at most.
    pub max_samples: usize,
    /// The shortest sample, ns; shorter operations are batched.
    pub min_sample_ns: u64,
    /// Stop once the 95% interval of the median is within this fraction of it.
    pub target_ci: f64,
    /// Time budget per operation across all candidates, ns.
    pub budget_ns: u64,
    /// Random seed for masks, indices and probe positions.
    pub seed: u64,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            min_samples: 5,
            max_samples: 31,
            min_sample_ns: 50_000,
            target_ci: 0.05,
            budget_ns: 200_000_000,
            seed: 0x5eed,
        }
    }
}

/// One operation measured on every candidate.
#[derive(Debug, Clone)]
pub enum Op {
    /// Execute to canonical.
    Decode,
    /// Filter by a mask, then execute.
    Filter(Mask),
    /// Take indices, then execute.
    Take(ArrayRef),
    /// Compare with a constant, then execute.
    Compare(Scalar, CompareOperator),
    /// Read single rows.
    ScalarAt(Vec<usize>),
}

/// The operations of a chunk, by name.
#[derive(Debug, Clone)]
pub struct Workload {
    /// `(name, op)`, in measurement order.
    pub ops: Vec<(String, Op)>,
}

impl Workload {
    /// The standard workload for a canonical chunk.
    ///
    /// - `decode`
    /// - `filter.random_1pct`, `filter.random_50pct`, `filter.clustered_1pct` (one contiguous run)
    /// - `take.random_1k`, `take.sorted_1k`
    /// - `compare.eq_median`, `compare.lt_median` (median of up to 257 evenly spaced valid values;
    ///   omitted for dtypes without an order, or an all-null chunk)
    /// - `scalar_at.random_100`
    pub fn standard(
        canonical: &ArrayRef,
        seed: u64,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<Self> {
        let len = canonical.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ops = vec![("decode".to_string(), Op::Decode)];
        if len == 0 {
            return Ok(Self { ops });
        }
        let random_mask = |rng: &mut StdRng, p: f64| {
            Mask::from_indices(
                len,
                (0..len).filter(|_| rng.random_bool(p)).collect::<Vec<_>>(),
            )
        };
        ops.push((
            "filter.random_1pct".into(),
            Op::Filter(random_mask(&mut rng, 0.01)),
        ));
        ops.push((
            "filter.random_50pct".into(),
            Op::Filter(random_mask(&mut rng, 0.5)),
        ));
        let run = (len / 100).max(1);
        let start = rng.random_range(0..=len - run);
        ops.push((
            "filter.clustered_1pct".into(),
            Op::Filter(Mask::from_slices(len, vec![(start, start + run)])),
        ));
        let mut random: Vec<u64> = (0..1000).map(|_| rng.random_range(0..len as u64)).collect();
        ops.push((
            "take.random_1k".into(),
            Op::Take(PrimitiveArray::from_iter(random.clone()).into_array()),
        ));
        random.sort_unstable();
        ops.push((
            "take.sorted_1k".into(),
            Op::Take(PrimitiveArray::from_iter(random).into_array()),
        ));
        if let Some(median) = median(canonical, ctx)? {
            ops.push((
                "compare.eq_median".into(),
                Op::Compare(median.clone(), CompareOperator::Eq),
            ));
            ops.push((
                "compare.lt_median".into(),
                Op::Compare(median, CompareOperator::Lt),
            ));
        }
        ops.push((
            "scalar_at.random_100".into(),
            Op::ScalarAt((0..100).map(|_| rng.random_range(0..len)).collect()),
        ));
        Ok(Self { ops })
    }
}

/// The median of up to 257 evenly spaced valid values, or `None` when none are valid or they do
/// not have an order.
fn median(canonical: &ArrayRef, ctx: &mut ExecutionCtx) -> anyhow::Result<Option<Scalar>> {
    let len = canonical.len();
    let step = (len / 257).max(1);
    let mut values = Vec::new();
    for i in (0..len).step_by(step) {
        let s = canonical.execute_scalar(i, ctx)?;
        if !s.is_null() {
            values.push(s);
        }
    }
    if values.is_empty() || values[0].partial_cmp(&values[0]).is_none() {
        return Ok(None);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Some(values[values.len() / 2].clone()))
}

/// Runs an operation once and returns its result as a canonical array.
fn run(array: &ArrayRef, op: &Op, ctx: &mut ExecutionCtx) -> anyhow::Result<Option<ArrayRef>> {
    Ok(Some(match op {
        Op::Decode => array.clone().execute::<Canonical>(ctx)?.into_array(),
        Op::Filter(mask) => array
            .filter(mask.clone())?
            .execute::<Canonical>(ctx)?
            .into_array(),
        Op::Take(indices) => array
            .take(indices.clone())?
            .execute::<Canonical>(ctx)?
            .into_array(),
        Op::Compare(scalar, op) => array
            .compare(
                ConstantArray::new(scalar.clone(), array.len()).into_array(),
                *op,
            )?
            .execute::<Canonical>(ctx)?
            .into_array(),
        Op::ScalarAt(indices) => {
            for &i in indices {
                black_box(array.execute_scalar(i, ctx)?);
            }
            return Ok(None);
        }
    }))
}

/// Whether two canonical results are logically equal.
fn same(a: &ArrayRef, b: &ArrayRef, ctx: &mut ExecutionCtx) -> anyhow::Result<bool> {
    let a = a.clone().execute_arrow(None, ctx)?;
    let b = b.clone().execute_arrow(None, ctx)?;
    Ok(a.to_data() == b.to_data())
}

/// Whether scalar reads agree with the canonical array.
fn same_scalars(
    array: &ArrayRef,
    canonical: &ArrayRef,
    indices: &[usize],
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<bool> {
    for &i in indices {
        if array.execute_scalar(i, ctx)? != canonical.execute_scalar(i, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// One candidate's measures of one operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpMeasure {
    /// The operation name, e.g. `filter.random_1pct`.
    pub op: String,
    /// Whether the result matched the canonical array's. Unmatched results are not timed.
    pub correct: bool,
    /// The error, if the operation failed.
    pub error: Option<String>,
    /// Operations per sample.
    pub batch: u32,
    /// Time per operation in each sample, ns.
    pub samples_ns: Vec<f64>,
    /// Whether the stopping rule was met before a cap.
    pub converged: bool,
    /// Memory used by one run of the operation, when the counting allocator is installed.
    pub memory: Option<MemoryUse>,
}

/// Every operation's measures for one candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateMeasures {
    /// The candidate name, or [`CANONICAL`].
    pub name: String,
    /// One entry per workload operation.
    pub ops: Vec<OpMeasure>,
}

/// The 95% interval of the median, relative to it, from order statistics.
pub fn median_ci(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 3 {
        return f64::INFINITY;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let half = 1.96 * (n as f64).sqrt() / 2.0;
    let mid = n as f64 / 2.0;
    let lo = (mid - half).floor().max(0.0) as usize;
    let hi = ((mid + half).ceil() as usize).min(n - 1);
    let median = sorted[n / 2];
    if median <= 0.0 {
        return f64::INFINITY;
    }
    (sorted[hi] - sorted[lo]) / median
}

/// Measures every candidate on every operation of `workload`.
///
/// `candidates` are `(name, encoded array)`; the canonical array is added under [`CANONICAL`].
pub fn measure(
    canonical: &ArrayRef,
    candidates: &[(String, ArrayRef)],
    workload: &Workload,
    config: &TimingConfig,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Vec<CandidateMeasures>> {
    let mut all: Vec<(String, ArrayRef)> = vec![(CANONICAL.to_string(), canonical.clone())];
    all.extend(candidates.iter().cloned());
    let mut out: Vec<CandidateMeasures> = all
        .iter()
        .map(|(name, _)| CandidateMeasures {
            name: name.clone(),
            ops: Vec::new(),
        })
        .collect();

    for (op_name, op) in &workload.ops {
        let expected = run(canonical, op, ctx).context("operation on the canonical array")?;
        // Check, warm up, size the batch and measure memory, once per candidate.
        let mut timed: Vec<(usize, u32)> = Vec::new();
        for (i, (_, array)) in all.iter().enumerate() {
            let mut entry = OpMeasure {
                op: op_name.clone(),
                correct: false,
                error: None,
                batch: 1,
                samples_ns: Vec::new(),
                converged: false,
                memory: None,
            };
            let region = Region::start();
            let start = Instant::now();
            let result = run(array, op, ctx);
            let once = start.elapsed().as_nanos().max(1) as u64;
            entry.memory = region.finish();
            match result {
                Err(e) => {
                    entry.error = Some(e.to_string().lines().next().unwrap_or_default().into())
                }
                Ok(result) => {
                    entry.correct = match (&result, &expected, op) {
                        (Some(r), Some(e), _) => same(r, e, ctx)?,
                        (None, None, Op::ScalarAt(indices)) => {
                            same_scalars(array, canonical, indices, ctx)?
                        }
                        _ => false,
                    };
                    if entry.correct {
                        entry.batch = u32::try_from(config.min_sample_ns.div_ceil(once))
                            .unwrap_or(u32::MAX)
                            .max(1);
                        timed.push((i, entry.batch));
                    }
                }
            }
            out[i].ops.push(entry);
        }

        // Round-robin samples until every candidate converges or a cap is hit.
        let started = Instant::now();
        let mut done = vec![false; all.len()];
        for round in 0..config.max_samples {
            for &(i, batch) in &timed {
                if done[i] {
                    continue;
                }
                let array = &all[i].1;
                let start = Instant::now();
                for _ in 0..batch {
                    black_box(run(array, op, ctx)?);
                }
                let per_op = start.elapsed().as_nanos() as f64 / f64::from(batch);
                let entry = out[i].ops.last_mut().context("entry")?;
                entry.samples_ns.push(per_op);
                if round + 1 >= config.min_samples
                    && median_ci(&entry.samples_ns) <= config.target_ci
                {
                    entry.converged = true;
                    done[i] = true;
                }
            }
            if timed.iter().all(|&(i, _)| done[i])
                || started.elapsed().as_nanos() as u64 > config.budget_ns
            {
                break;
            }
        }
    }
    Ok(out)
}

/// Times a reference kernel: copying `bytes` bytes, median of 11, ns.
///
/// Recorded beside every observation so that runs whose machine was slower than usual can be
/// flagged, and timings normalised across runs.
pub fn reference_ns(bytes: usize) -> u64 {
    let src = vec![1u8; bytes.max(4096)];
    let mut dst = vec![0u8; src.len()];
    let mut samples: Vec<u64> = (0..11)
        .map(|_| {
            let start = Instant::now();
            dst.copy_from_slice(black_box(&src));
            black_box(&dst);
            start.elapsed().as_nanos() as u64
        })
        .collect();
    samples.sort_unstable();
    samples[5]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_ci_shrinks_with_tight_samples() {
        assert!(median_ci(&[100.0; 11]) < 1e-9);
        assert!(median_ci(&[100.0, 200.0, 50.0, 400.0, 10.0]) > 0.05);
        assert!(median_ci(&[1.0]).is_infinite());
    }
}
