// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Seeded synthetic integer columns, for filling gaps real data leaves near decision boundaries.
//!
//! Every generator takes optional `null_frac`. A generator is a pure function of its parameters
//! and seed, so a synthetic chunk is identified by them alone.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::bail;
use arrow_array::ArrayRef;
use arrow_array::PrimitiveArray;
use arrow_array::types::Int8Type;
use arrow_array::types::Int16Type;
use arrow_array::types::Int32Type;
use arrow_array::types::Int64Type;
use arrow_array::types::UInt8Type;
use arrow_array::types::UInt16Type;
use arrow_array::types::UInt32Type;
use arrow_array::types::UInt64Type;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;

use crate::spec::ParamValue;
use crate::spec::SyntheticPType;

/// The generators and the parameters each reads.
pub const GENERATORS: &[(&str, &str)] = &[
    ("uniform", "bits"),
    ("zipf", "distinct, skew"),
    ("runs", "mean_run, bits"),
    ("sorted", "step_max"),
    ("sequence", "step, noise_frac"),
    ("sparse", "top_frac, bits"),
];

fn int(params: &BTreeMap<String, ParamValue>, name: &str, default: i64) -> i64 {
    match params.get(name) {
        Some(ParamValue::Int(v)) => *v,
        Some(ParamValue::Float(v)) => *v as i64,
        _ => default,
    }
}

fn float(params: &BTreeMap<String, ParamValue>, name: &str, default: f64) -> f64 {
    match params.get(name) {
        Some(ParamValue::Float(v)) => *v,
        Some(ParamValue::Int(v)) => *v as f64,
        _ => default,
    }
}

fn below_bits(rng: &mut StdRng, bits: i64) -> i64 {
    let bits = bits.clamp(1, 62);
    rng.random_range(0..(1i64 << bits))
}

/// Generates one synthetic column.
pub fn generate(
    generator: &str,
    ptype: SyntheticPType,
    params: &BTreeMap<String, ParamValue>,
    seed: u64,
    rows: u64,
) -> anyhow::Result<ArrayRef> {
    let rows = usize::try_from(rows)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let values: Vec<i64> = match generator {
        "uniform" => {
            let bits = int(params, "bits", 16);
            (0..rows).map(|_| below_bits(&mut rng, bits)).collect()
        }
        "zipf" => {
            let distinct = usize::try_from(int(params, "distinct", 1000).max(1))?;
            let skew = float(params, "skew", 1.0);
            let vocab: Vec<i64> = (0..distinct).map(|_| below_bits(&mut rng, 31)).collect();
            let weights: Vec<f64> = (1..=distinct)
                .map(|r| 1.0 / (r as f64).powf(skew))
                .collect();
            let total: f64 = weights.iter().sum();
            let mut cdf = Vec::with_capacity(distinct);
            let mut acc = 0.0;
            for w in &weights {
                acc += w / total;
                cdf.push(acc);
            }
            (0..rows)
                .map(|_| {
                    let u: f64 = rng.random();
                    let idx = cdf.partition_point(|c| *c < u).min(distinct - 1);
                    vocab[idx]
                })
                .collect()
        }
        "runs" => {
            let mean_run = int(params, "mean_run", 8).max(1);
            let bits = int(params, "bits", 16);
            let mut out = Vec::with_capacity(rows);
            while out.len() < rows {
                let run = usize::try_from(rng.random_range(1..=2 * mean_run - 1))?;
                let v = below_bits(&mut rng, bits);
                out.extend(std::iter::repeat_n(v, run.min(rows - out.len())));
            }
            out
        }
        "sorted" => {
            let step_max = int(params, "step_max", 4).max(0);
            let mut acc = 0i64;
            (0..rows)
                .map(|_| {
                    acc += rng.random_range(0..=step_max);
                    acc
                })
                .collect()
        }
        "sequence" => {
            let step = int(params, "step", 1);
            let noise = float(params, "noise_frac", 0.0);
            (0..rows)
                .map(|i| {
                    if rng.random::<f64>() < noise {
                        below_bits(&mut rng, 31)
                    } else {
                        i as i64 * step
                    }
                })
                .collect()
        }
        "sparse" => {
            let top = float(params, "top_frac", 0.9);
            let bits = int(params, "bits", 16);
            (0..rows)
                .map(|_| {
                    if rng.random::<f64>() < top {
                        0
                    } else {
                        below_bits(&mut rng, bits)
                    }
                })
                .collect()
        }
        other => bail!(
            "unknown generator `{other}`; known: {}",
            GENERATORS
                .iter()
                .map(|(n, p)| format!("{n} ({p})"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };

    let null_frac = float(params, "null_frac", 0.0);
    let valid: Vec<bool> = (0..rows)
        .map(|_| rng.random::<f64>() >= null_frac)
        .collect();
    let opt = |v: i64, i: usize| valid[i].then_some(v);

    macro_rules! build {
        ($t:ty, $native:ty) => {
            Arc::new(PrimitiveArray::<$t>::from_iter(
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| opt(*v, i).map(|v| v as $native)),
            )) as ArrayRef
        };
    }
    Ok(match ptype {
        SyntheticPType::U8 => build!(UInt8Type, u8),
        SyntheticPType::U16 => build!(UInt16Type, u16),
        SyntheticPType::U32 => build!(UInt32Type, u32),
        SyntheticPType::U64 => build!(UInt64Type, u64),
        SyntheticPType::I8 => build!(Int8Type, i8),
        SyntheticPType::I16 => build!(Int16Type, i16),
        SyntheticPType::I32 => build!(Int32Type, i32),
        SyntheticPType::I64 => build!(Int64Type, i64),
        SyntheticPType::F32 | SyntheticPType::F64 => {
            bail!("float synthetic columns are not supported yet")
        }
    })
}
