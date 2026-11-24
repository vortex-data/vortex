// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A model-driven compressor.
//!
//! The model, trained by `train.py` and exported as JSON, predicts for each candidate (a root
//! scheme forced over the production compressor, the production compressor itself, or the size
//! model) the compressed size and the decode time. The compressor proposes the candidate with the
//! lowest `bytes / bandwidth + decode time`, compresses it next to production when it is predicted
//! to be at least `gate` cheaper, and keeps whichever has the lower real size-based cost.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Context;
use serde::Deserialize;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::features;

/// One regression tree, as exported from scikit-learn's histogram gradient boosting.
#[derive(Debug, Deserialize)]
struct Tree {
    feature: Vec<usize>,
    threshold: Vec<f64>,
    left: Vec<usize>,
    right: Vec<usize>,
    leaf: Vec<bool>,
    value: Vec<f64>,
}

impl Tree {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut node = 0;
        while !self.leaf[node] {
            node = if x[self.feature[node]] <= self.threshold[node] {
                self.left[node]
            } else {
                self.right[node]
            };
        }
        self.value[node]
    }
}

/// A boosted ensemble: a baseline plus the sum of its trees.
#[derive(Debug, Deserialize)]
struct Ensemble {
    baseline: f64,
    trees: Vec<Tree>,
}

impl Ensemble {
    fn predict(&self, x: &[f64]) -> f64 {
        self.baseline + self.trees.iter().map(|t| t.predict(x)).sum::<f64>()
    }
}

#[derive(Debug, Deserialize)]
struct CandidateModel {
    /// Logit of "this candidate compresses without error"; absent when always feasible.
    feasible: Option<Ensemble>,
    /// `log2(bytes / canonical bytes)`.
    bytes: Ensemble,
    /// `log2(decode ns per value)`.
    decode: Ensemble,
    /// `log2(compress ns per value)`.
    #[serde(default)]
    compress: Option<Ensemble>,
}

/// A trained selector.
#[derive(Debug, Deserialize)]
pub struct Model {
    features: Vec<String>,
    candidates: BTreeMap<String, CandidateModel>,
}

/// A prediction for one candidate.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub name: String,
    pub bytes: f64,
    pub decode_ns: f64,
    pub compress_ns: f64,
}

impl Prediction {
    fn cost(&self, bandwidth: f64) -> f64 {
        self.bytes / bandwidth + self.decode_ns / 1e9
    }
}

impl Model {
    /// Loads a model exported by `train.py`.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))
    }

    /// Predicts every feasible candidate for a chunk.
    pub fn predict(&self, base: &[f64; 20], canonical_bytes: f64) -> Vec<Prediction> {
        // Training fills undefined features with -1 before deriving the estimates.
        let filled = base.map(|v| if v.is_finite() { v } else { -1.0 });
        let named = features::with_derived(&filled);
        let x: Vec<f64> = self
            .features
            .iter()
            .map(|name| {
                named
                    .iter()
                    .find(|(n, _)| *n == name.as_str())
                    .map(|(_, v)| if v.is_finite() { *v } else { -1.0 })
                    .unwrap_or(-1.0)
            })
            .collect();
        let len = filled[2].max(1.0);
        self.candidates
            .iter()
            .filter(|(_, m)| m.feasible.as_ref().is_none_or(|f| f.predict(&x) > 0.0))
            .map(|(name, m)| Prediction {
                name: name.clone(),
                bytes: m.bytes.predict(&x).exp2() * canonical_bytes,
                decode_ns: m.decode.predict(&x).exp2() * len,
                compress_ns: m
                    .compress
                    .as_ref()
                    .map_or(0.0, |c| c.predict(&x).exp2() * len),
            })
            .collect()
    }
}

/// The outcome of one model-driven compression.
pub struct Outcome {
    pub array: ArrayRef,
    pub proposed: String,
    pub tried: bool,
    pub kept: String,
}

/// Compresses with production, plus the model's proposal when it promises enough, and keeps the
/// cheaper by real size and predicted decode time.
/// Cumulative nanoseconds spent in features, canonical sizing, inference and production compression.
pub static PROFILE: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn record(slot: usize, start: std::time::Instant) {
    PROFILE[slot].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// The objective per chunk is `compress time / reads + bytes / bandwidth + decode time`. Production
/// is always compressed; an alternative is only tried when `reads` times its predicted serving
/// saving outweighs its predicted compression time, and its serving cost is at least `gate` lower.
#[allow(clippy::too_many_arguments)]
pub fn compress(
    model: &Model,
    candidates: &BTreeMap<String, &BtrBlocksCompressor>,
    bandwidth: f64,
    reads: f64,
    gate: f64,
    top_k: usize,
    input: &ArrayRef,
    session: &VortexSession,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Outcome> {
    let production = candidates
        .get("production")
        .context("the production compressor is missing")?;

    // Value of information: skip the model when even the best plausible serving saving over all
    // reads cannot repay the model's own cost. Constants are this machine's measured rates.
    const PROD_RATIO: f64 = 5.5;
    const DECODE_NS_PER_VALUE: f64 = 1.0;
    const MODEL_NS_PER_VALUE: f64 = 2.3;
    const MAX_SAVING: f64 = 0.6;
    let len = input.len() as f64;
    let width = input.dtype().as_ptype().byte_width() as f64;
    let serving_est = len * width / PROD_RATIO / bandwidth + len * DECODE_NS_PER_VALUE / 1e9;
    if reads * MAX_SAVING * serving_est < len * MODEL_NS_PER_VALUE / 1e9 {
        return Ok(Outcome {
            array: production.compress(input, ctx)?,
            proposed: "skipped".to_string(),
            tried: false,
            kept: "production".to_string(),
        });
    }

    let t = std::time::Instant::now();
    let primitive = input.clone().execute::<PrimitiveArray>(ctx)?;
    let base = features::from_primitive(&primitive, ctx)?;
    record(0, t);
    let t = std::time::Instant::now();
    let canonical_bytes = crate::blob::serialized_size(input, session)? as f64;
    record(1, t);
    let t = std::time::Instant::now();
    let predictions = model.predict(&base, canonical_bytes);
    record(2, t);

    let t = std::time::Instant::now();
    let prod_array = production.compress(input, ctx)?;
    record(3, t);
    let prod_pred = predictions.iter().find(|p| p.name == "production");

    let Some(prod_pred) = prod_pred else {
        return Ok(Outcome {
            array: prod_array,
            proposed: "production".to_string(),
            tried: false,
            kept: "production".to_string(),
        });
    };
    // Net benefit of also trying a candidate: serving savings over all reads, minus its compression.
    let benefit = |p: &Prediction| {
        reads * (prod_pred.cost(bandwidth) - p.cost(bandwidth)) - p.compress_ns / 1e9
    };
    // Rank alternatives by predicted net benefit. `top_k = usize::MAX` tries every candidate, which
    // finds the best measured encoding at the price of compressing with all of them.
    let mut ranked: Vec<&Prediction> = predictions
        .iter()
        .filter(|p| p.name != "production" && candidates.contains_key(&p.name))
        .collect();
    ranked.sort_by(|a, b| benefit(b).total_cmp(&benefit(a)));
    let proposed = ranked
        .first()
        .map_or_else(|| "production".to_string(), |p| p.name.clone());
    let exhaustive = top_k == usize::MAX;
    // Predicted savings are optimistic, so the saving must cover the extra compression twice over.
    let worth_trying = |p: &Prediction| {
        exhaustive
            || (benefit(p) > p.compress_ns / 1e9
                && p.cost(bandwidth) <= prod_pred.cost(bandwidth) * (1.0 - gate))
    };
    let to_try: Vec<&Prediction> = ranked
        .into_iter()
        .filter(|p| worth_trying(p))
        .take(top_k)
        .collect();

    let tried = !to_try.is_empty();
    let mut best_cost = if tried {
        measured_cost(&prod_array, bandwidth, session, ctx)?
    } else {
        f64::INFINITY
    };
    let mut best = (prod_array, "production".to_string());
    for candidate in to_try {
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            candidates[&candidate.name].compress(input, ctx)
        }));
        let Ok(Ok(array)) = attempt else {
            continue;
        };
        let cost = measured_cost(&array, bandwidth, session, ctx)?;
        if cost < best_cost {
            best_cost = cost;
            best = (array, candidate.name.clone());
        }
    }
    Ok(Outcome {
        array: best.0,
        proposed,
        tried,
        kept: best.1,
    })
}

/// Real serving cost of an encoding: its serialized bytes over the bandwidth plus a measured
/// decode time. Decoding is ~20x cheaper than compressing, so measuring it during verification is
/// affordable; the minimum of three decodes damps timing noise.
pub fn measured_cost(
    array: &ArrayRef,
    bandwidth: f64,
    session: &VortexSession,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<f64> {
    let mut decode_ns = u128::MAX;
    for _ in 0..3 {
        let start = std::time::Instant::now();
        std::hint::black_box(array.clone().execute::<vortex_array::Canonical>(ctx)?);
        decode_ns = decode_ns.min(start.elapsed().as_nanos());
    }
    Ok(crate::blob::serialized_size(array, session)? as f64 / bandwidth + decode_ns as f64 / 1e9)
}
