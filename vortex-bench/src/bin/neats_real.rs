// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NeaTS size/timing report.
//!
//! Modes:
//!
//! 1. `cargo run -p vortex-bench --release --bin neats-real -- synthetic`
//!    Runs five synthetic time-series shapes (uniform random, linear ramp, piecewise linear with
//!    noise, sine with drift, stock-like random walk) at three sizes and reports compress + decompress
//!    time and pre/post-cascade size for each (shape, mode).
//!
//! 2. `cargo run -p vortex-bench --release --bin neats-real -- /path/to/some.parquet`
//!    Walks every f32/f64 column of the parquet and runs the same suite. With no argument it tries
//!    the Yellow Taxi parquet that `vortex-bench` already wires up (may 403 in restricted networks).
//!
//! For each input it reports the raw bytes (`N * 8`), the NeaTS array's pre-cascade `nbytes()`, the
//! post-cascade size after running through `BtrBlocksCompressor`, the compress/decompress walltimes,
//! and the round-trip `max_abs_err`.

#![expect(clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use rand::RngExt as _;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressor;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::taxi_data::taxi_data_parquet;
use vortex_neats::NeaTSArraySlotsExt;
use vortex_neats::NeaTSOptions;
use vortex_neats::neats_encode;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = env::args().nth(1).unwrap_or_else(|| "taxi".to_string());

    let inputs: Vec<(String, Vec<f64>)> = match mode.as_str() {
        "synthetic" => synthetic_inputs(),
        other => {
            let parquet_path = if other == "taxi" {
                println!("# no path provided, downloading the taxi parquet (cached on disk)");
                taxi_data_parquet().await?
            } else {
                PathBuf::from(other)
            };
            println!("# loading parquet: {}", parquet_path.display());
            let chunked = parquet_to_vortex_chunks(parquet_path).await?;
            let mut ctx = SESSION.create_execution_ctx();
            let chunks: Vec<ArrayRef> = chunked.chunks().to_vec();
            extract_float_columns(&chunks, &mut ctx)?
        }
    };
    if inputs.is_empty() {
        println!("# no fp inputs found");
        return Ok(());
    }

    println!(
        "{:<36} {:>10} {:>11} {:>12} {:>12} {:>13} {:>13} {:>8} {:>9} {:>11} {:>12} {:>14}",
        "input/mode",
        "rows",
        "raw_bytes",
        "neats_bytes",
        "btr_raw_bytes",
        "btr_neats_bytes",
        "per_slot_bytes",
        "btr_x",
        "neats_x",
        "compress_us",
        "decomp_us",
        "max_abs_err",
    );

    for (name, values) in inputs {
        let array = PrimitiveArray::from_iter(values.iter().copied());
        let raw_bytes = (values.len() * size_of::<f64>()) as f64;

        // BtrBlocks baseline: compress the raw f64 array directly. Same compressor that runs on
        // top of NeaTS in `btr_neats_bytes`, but without NeaTS first. This is "what Vortex's own
        // sampling compressor picks for this column today".
        let mut ctx_raw = SESSION.create_execution_ctx();
        let btr_raw =
            BtrBlocksCompressor::default().compress(&array.clone().into_array(), &mut ctx_raw)?;
        let btr_raw_bytes = btr_raw.nbytes() as f64;

        for (label, epsilon) in [
            ("lossless", None),
            ("eps=1e-6", Some(1e-6)),
            ("eps=1e-3", Some(1e-3)),
        ] {
            let opts = NeaTSOptions {
                epsilon,
                ..NeaTSOptions::default()
            };
            let mut ctx_enc = SESSION.create_execution_ctx();
            let t0 = Instant::now();
            let encoded = neats_encode(array.as_view(), opts, &mut ctx_enc)?;
            let compress_time = t0.elapsed();
            let neats_bytes = encoded.as_ref().nbytes() as f64;

            // Two compressed-size measurements:
            //
            // 1. `btr_bytes`: BtrBlocks operating on the NeaTS array directly. Today this just
            //    canonicalises the array back to f64 and compresses that, so it's the "what
            //    BtrBlocks does without NeaTS-awareness" baseline.
            // 2. `per_slot_bytes`: cascade each NeaTS child slot individually and sum. This is
            //    what a NeaTS-aware writer would emit — residuals bit-pack heavily, model_ids
            //    becomes a constant, etc.
            let mut ctx2 = SESSION.create_execution_ctx();
            let btr = BtrBlocksCompressor::default()
                .compress(&encoded.clone().into_array(), &mut ctx2)?;
            let btr_bytes = btr.nbytes() as f64;

            let mut per_slot_bytes = 0u64;
            let mut slot_sizes = [0u64; 6];
            let slot_names = [
                "piece_starts",
                "model_ids",
                "coeff_a",
                "coeff_b",
                "coeff_c",
                "residuals",
            ];
            for (i, slot) in [
                encoded.piece_starts(),
                encoded.model_ids(),
                encoded.coeff_a(),
                encoded.coeff_b(),
                encoded.coeff_c(),
                encoded.residuals(),
            ]
            .into_iter()
            .enumerate()
            {
                let compressed = BtrBlocksCompressor::default().compress(slot, &mut ctx2)?;
                slot_sizes[i] = compressed.nbytes();
                per_slot_bytes += compressed.nbytes();
            }
            let per_slot_bytes = per_slot_bytes as f64;

            let mut ctx3 = SESSION.create_execution_ctx();
            let t1 = Instant::now();
            let decoded = encoded
                .clone()
                .into_array()
                .execute::<PrimitiveArray>(&mut ctx3)?;
            let decomp_time = t1.elapsed();
            let decoded_slice = decoded.as_slice::<f64>();
            let max_abs_err = values
                .iter()
                .zip(decoded_slice.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            println!(
                "{:<36} {:>10} {:>11.0} {:>12.0} {:>13.0} {:>13.0} {:>13.0} {:>7.2}x {:>8.2}x {:>11} {:>12} {:>14.3e}",
                format!("{name}/{label}"),
                values.len(),
                raw_bytes,
                neats_bytes,
                btr_raw_bytes,
                btr_bytes,
                per_slot_bytes,
                raw_bytes / btr_raw_bytes.max(1.0),
                raw_bytes / per_slot_bytes.max(1.0),
                compress_time.as_micros(),
                decomp_time.as_micros(),
                max_abs_err,
            );
            // Per-slot breakdown so we can see which child is dominating bytes.
            let slot_breakdown = slot_names
                .iter()
                .zip(slot_sizes.iter())
                .map(|(n, b)| format!("{n}={b}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!("    slots: {slot_breakdown}");
        }
    }

    Ok(())
}

fn extract_float_columns(
    chunks: &[ArrayRef],
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    let mut out: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for chunk in chunks {
        let DType::Struct(..) = chunk.dtype() else {
            continue;
        };
        let s = chunk
            .as_opt::<Struct>()
            .expect("dtype said struct but array is not StructArray");
        let names = s.names().clone();
        for (i, name) in names.iter().enumerate() {
            let field = s.unmasked_field(i).clone();
            match field.dtype() {
                DType::Primitive(PType::F32, _) => {
                    let p = field.execute::<PrimitiveArray>(ctx)?;
                    let entry = out.entry(name.to_string()).or_default();
                    entry.extend(p.as_slice::<f32>().iter().map(|v| *v as f64));
                }
                DType::Primitive(PType::F64, _) => {
                    let p = field.execute::<PrimitiveArray>(ctx)?;
                    let entry = out.entry(name.to_string()).or_default();
                    entry.extend(p.as_slice::<f64>().iter().copied());
                }
                _ => {}
            }
        }
    }
    Ok(out.into_iter().collect())
}

fn synthetic_inputs() -> Vec<(String, Vec<f64>)> {
    let sizes = [1_000usize, 10_000, 100_000];
    let mut out = Vec::new();
    for &n in &sizes {
        out.push((format!("uniform_random[{n}]"), uniform_random(n)));
        out.push((format!("linear_ramp[{n}]"), linear_ramp(n)));
        out.push((
            format!("piecewise_linear_noisy[{n}]"),
            piecewise_linear_noisy(n),
        ));
        out.push((format!("sine_drift[{n}]"), sine_drift(n)));
        out.push((format!("stock_walk[{n}]"), stock_walk(n)));
        // Harder shapes: smooth-but-not-repetitive signals where every value is distinct, so
        // RLE / dictionary / constant schemes (BtrBlocks's natural toolkit) can't help. NeaTS
        // should pull ahead here.
        out.push((format!("hf_sensor[{n}]"), hf_sensor(n)));
        out.push((format!("gps_trace[{n}]"), gps_trace(n)));
        out.push((format!("brownian_bridge[{n}]"), brownian_bridge(n)));
        out.push((format!("ecg_like[{n}]"), ecg_like(n)));
        out.push((format!("temperature_diurnal[{n}]"), temperature_diurnal(n)));
    }
    out
}

fn uniform_random(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn linear_ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.001 * i as f64).collect()
}

fn piecewise_linear_noisy(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut out = Vec::with_capacity(n);
    let regime = 1024usize;
    let mut slope = 0.0;
    let mut offset = 0.0;
    for i in 0..n {
        if i % regime == 0 {
            slope = rng.random_range(-0.01..0.01);
            offset = rng.random_range(-1.0..1.0);
        }
        out.push(offset + slope * ((i % regime) as f64) + rng.random_range(-0.001..0.001));
    }
    out
}

fn sine_drift(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.01).sin() + 0.0005 * i as f64)
        .collect()
}

fn stock_walk(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut v = 100.0_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        v *= 1.0 + rng.random_range(-0.005..0.005);
        out.push(v);
    }
    out
}

/// High-frequency mean-reverting sensor: a stationary AR(1) process with small step noise.
/// Every value is distinct; no RLE / dictionary structure to exploit.
fn hf_sensor(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut v = 0.0_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // AR(1) with phi=0.95 and gaussian-ish noise.
        v = 0.95 * v + rng.random_range(-0.1..0.1);
        out.push(v);
    }
    out
}

/// GPS-like monotonic latitude trace: cumulative sum of small steps. Smooth, every value
/// distinct, slowly drifting — a textbook NeaTS target.
fn gps_trace(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut lat = 37.42_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        lat += rng.random_range(-1e-5..1e-5);
        out.push(lat);
    }
    out
}

/// Brownian bridge: standard random walk pinned at start and end. Smooth, no repeats.
fn brownian_bridge(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut steps = Vec::with_capacity(n);
    let mut acc = 0.0_f64;
    for _ in 0..n {
        acc += rng.random_range(-1.0..1.0);
        steps.push(acc);
    }
    // Pin to zero at the end.
    let end = steps[n - 1];
    for (i, v) in steps.iter_mut().enumerate() {
        *v -= end * (i as f64) / (n as f64);
    }
    steps
}

/// ECG-like: low baseline plus periodic Gaussian spikes. Smooth between spikes; sharp transitions
/// at spike crests. Pieces should split tightly around each spike.
fn ecg_like(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut out = Vec::with_capacity(n);
    let period = 200.0_f64;
    for i in 0..n {
        let t = i as f64;
        let phase = (t / period).fract();
        let base = 0.05 * (t * 0.02).sin();
        // Spike: narrow Gaussian centered at phase ~0.5.
        let dx = (phase - 0.5) * 16.0;
        let spike = (-(dx * dx)).exp();
        out.push(base + spike + rng.random_range(-0.005..0.005));
    }
    out
}

/// Temperature with a diurnal cycle plus drifting baseline plus noise — every value distinct.
fn temperature_diurnal(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let diurnal = 5.0 * (t * 2.0 * std::f64::consts::PI / 1440.0).sin();
        let drift = 0.0001 * t;
        let noise = rng.random_range(-0.05..0.05);
        out.push(15.0 + drift + diurnal + noise);
    }
    out
}
