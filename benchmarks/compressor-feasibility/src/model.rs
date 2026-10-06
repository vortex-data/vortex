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
}

impl Prediction {
    fn cost(&self, bandwidth: f64) -> f64 {
        self.bytes / bandwidth + self.decode_ns / 1e9
    }
}

impl Model {
    /// Loads a model exported by `train.py`.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        serde_json::from_slice(&fs::read(path).with_context(|| format!("reading {}", path.display()))?)
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
pub static PROFILE: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn record(slot: usize, start: std::time::Instant) {
    PROFILE[slot].fetch_add(
        start.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub fn compress(
    model: &Model,
    candidates: &BTreeMap<String, &BtrBlocksCompressor>,
    bandwidth: f64,
    gate: f64,
    input: &ArrayRef,
    session: &VortexSession,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Outcome> {
    let t = std::time::Instant::now();
    let primitive = input.clone().execute::<PrimitiveArray>(ctx)?;
    let base = features::from_primitive(&primitive, ctx)?;
    record(0, t);
    let t = std::time::Instant::now();
    let canonical_bytes = crate::serialized_size(input, session)? as f64;
    record(1, t);
    let t = std::time::Instant::now();
    let predictions = model.predict(&base, canonical_bytes);
    record(2, t);

    let production = candidates
        .get("production")
        .context("the production compressor is missing")?;
    let t = std::time::Instant::now();
    let prod_array = production.compress(input, ctx)?;
    record(3, t);
    let prod_pred = predictions.iter().find(|p| p.name == "production");

    let best = predictions
        .iter()
        .filter(|p| candidates.contains_key(&p.name))
        .min_by(|a, b| a.cost(bandwidth).total_cmp(&b.cost(bandwidth)));
    let (Some(best), Some(prod_pred)) = (best, prod_pred) else {
        return Ok(Outcome {
            array: prod_array,
            proposed: "production".to_string(),
            tried: false,
            kept: "production".to_string(),
        });
    };
    if best.name == "production" || best.cost(bandwidth) > prod_pred.cost(bandwidth) * (1.0 - gate) {
        return Ok(Outcome {
            array: prod_array,
            proposed: best.name.clone(),
            tried: false,
            kept: "production".to_string(),
        });
    }

    let alternative = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        candidates[&best.name].compress(input, ctx)
    }));
    let Ok(Ok(alt_array)) = alternative else {
        return Ok(Outcome {
            array: prod_array,
            proposed: best.name.clone(),
            tried: true,
            kept: "production".to_string(),
        });
    };
    // Decoding is ~20x cheaper than compressing, so verify with a measured decode time.
    let mut real = |array: &ArrayRef| -> anyhow::Result<f64> {
        let mut decode_ns = u128::MAX;
        for _ in 0..2 {
            let start = std::time::Instant::now();
            std::hint::black_box(array.clone().execute::<vortex_array::Canonical>(ctx)?);
            decode_ns = decode_ns.min(start.elapsed().as_nanos());
        }
        Ok(crate::serialized_size(array, session)? as f64 / bandwidth + decode_ns as f64 / 1e9)
    };
    if real(&alt_array)? < real(&prod_array)? {
        Ok(Outcome {
            array: alt_array,
            proposed: best.name.clone(),
            tried: true,
            kept: best.name.clone(),
        })
    } else {
        Ok(Outcome {
            array: prod_array,
            proposed: best.name.clone(),
            tried: true,
            kept: "production".to_string(),
        })
    }
}
