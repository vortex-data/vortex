// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Four-encoder comparison table for time-series-shaped floating-point columns.
//!
//! For each input, runs:
//!
//! - **NeaTS lossless** (residual scale = data-range ULP, exact round-trip)
//! - **NeaTS lossy ε=1e-3**
//! - **BtrBlocks** with the workspace-default sampling compressor
//! - **PCO** (pcodec) at level 8 with default paging
//!
//! Reports compressed bytes, ratio vs raw, compress + decompress walltime, and round-trip
//! max absolute error.
//!
//! Modes:
//!   cargo run -p vortex-bench --release --bin neats-table -- synthetic
//!   cargo run -p vortex-bench --release --bin neats-table -- <path-to-parquet>

#![expect(clippy::expect_used)]

use std::env;
use std::path::PathBuf;
use std::time::Duration;
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
use vortex_neats::NeaTSOptions;
use vortex_neats::neats_encode;
use vortex_pco::Pco;

#[derive(Default, Debug, Clone, Copy)]
#[allow(dead_code, reason = "fields populated for ad-hoc inspection")]
struct Row {
    bytes: f64,
    compress: Duration,
    decompress: Duration,
    max_abs_err: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = env::args()
        .nth(1)
        .unwrap_or_else(|| "synthetic".to_string());
    let inputs: Vec<(String, Vec<f64>)> = match mode.as_str() {
        "synthetic" => synthetic_inputs(),
        "real" => load_real_datasets("/home/user/vortex/benchmarks/real-data")?,
        path if path.ends_with(".csv") => load_csv_columns(path)?,
        path => {
            let parquet_path = PathBuf::from(path);
            println!("# loading parquet: {}", parquet_path.display());
            let chunked = parquet_to_vortex_chunks(parquet_path).await?;
            let mut ctx = SESSION.create_execution_ctx();
            let chunks: Vec<ArrayRef> = chunked.chunks().to_vec();
            extract_float_columns(&chunks, &mut ctx)?
        }
    };

    // Compare across residual encodings: pco-default, PerPieceBitPack (paper), BtrBlocks, PCO.
    println!("## COMPRESSED BYTES (and ratio vs raw)");
    println!(
        "{:<28} {:>8} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6} | {:>10} {:>6}",
        "input",
        "rows",
        "neats_pco",
        "ratio",
        "pco_lossy",
        "ratio",
        "neats_ppb",
        "ratio",
        "ppb_lossy",
        "ratio",
        "btr",
        "ratio",
        "pco",
        "ratio",
    );

    for (name, values) in inputs {
        let raw_bytes = (values.len() * size_of::<f64>()) as f64;
        let neats = run_neats(&values, None)?;
        let neats_lossy = run_neats(&values, Some(1e-3))?;
        let neats_ppb = run_neats_perpiece(&values, None)?;
        let neats_ppb_lossy = run_neats_perpiece(&values, Some(1e-3))?;
        let btr = run_btr(&values)?;
        let pco = run_pco(&values)?;

        println!(
            "{:<28} {:>8} | {:>10.0} {:>5.2}x | {:>10.0} {:>5.2}x | {:>10.0} {:>5.2}x | {:>10.0} {:>5.2}x | {:>10.0} {:>5.2}x | {:>10.0} {:>5.2}x",
            name,
            values.len(),
            neats.bytes,
            raw_bytes / neats.bytes.max(1.0),
            neats_lossy.bytes,
            raw_bytes / neats_lossy.bytes.max(1.0),
            neats_ppb.bytes,
            raw_bytes / neats_ppb.bytes.max(1.0),
            neats_ppb_lossy.bytes,
            raw_bytes / neats_ppb_lossy.bytes.max(1.0),
            btr.bytes,
            raw_bytes / btr.bytes.max(1.0),
            pco.bytes,
            raw_bytes / pco.bytes.max(1.0),
        );
    }

    Ok(())
}

fn run_neats(values: &[f64], epsilon: Option<f64>) -> anyhow::Result<Row> {
    // NeaTS-aware "what the writer would emit": per-slot BtrBlocks cascade summed.
    let array = PrimitiveArray::from_iter(values.iter().copied());
    let opts = NeaTSOptions {
        epsilon,
        ..NeaTSOptions::default()
    };
    let mut enc_ctx = SESSION.create_execution_ctx();
    let t0 = Instant::now();
    let encoded = neats_encode(array.as_view(), opts, &mut enc_ctx)?;
    let compress = t0.elapsed();

    // The encoder now cascades all slots internally — just sum nbytes() of the encoded array.
    let bytes = encoded.as_ref().nbytes();

    let mut ctx2 = SESSION.create_execution_ctx();
    let t1 = Instant::now();
    let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx2)?;
    let decompress = t1.elapsed();
    let max_abs_err = max_abs_err(values, decoded.as_slice::<f64>());

    Ok(Row {
        bytes: bytes as f64,
        compress,
        decompress,
        max_abs_err,
    })
}

/// Force the paper-faithful "PerPieceBitPack" residual encoding.
fn run_neats_perpiece(values: &[f64], epsilon: Option<f64>) -> anyhow::Result<Row> {
    let array = PrimitiveArray::from_iter(values.iter().copied());
    let opts = NeaTSOptions {
        epsilon,
        residual_encoding: vortex_neats::ResidualEncoding::PerPieceBitPack,
        ..NeaTSOptions::default()
    };
    let mut enc_ctx = SESSION.create_execution_ctx();
    let t0 = Instant::now();
    let encoded = neats_encode(array.as_view(), opts, &mut enc_ctx)?;
    let compress = t0.elapsed();

    let bytes = encoded.as_ref().nbytes();

    let mut ctx2 = SESSION.create_execution_ctx();
    let t1 = Instant::now();
    let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx2)?;
    let decompress = t1.elapsed();
    let max_abs_err = max_abs_err(values, decoded.as_slice::<f64>());

    Ok(Row {
        bytes: bytes as f64,
        compress,
        decompress,
        max_abs_err,
    })
}

/// Force the legacy "BitPack on residuals" path (no PCO) — for comparison.
#[allow(dead_code)]
fn run_neats_bitpack(values: &[f64], epsilon: Option<f64>) -> anyhow::Result<Row> {
    let array = PrimitiveArray::from_iter(values.iter().copied());
    let opts = NeaTSOptions {
        epsilon,
        residual_encoding: vortex_neats::ResidualEncoding::BitPack,
        ..NeaTSOptions::default()
    };
    let mut enc_ctx = SESSION.create_execution_ctx();
    let t0 = Instant::now();
    let encoded = neats_encode(array.as_view(), opts, &mut enc_ctx)?;
    let compress = t0.elapsed();

    let bytes = encoded.as_ref().nbytes();

    let mut ctx2 = SESSION.create_execution_ctx();
    let t1 = Instant::now();
    let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx2)?;
    let decompress = t1.elapsed();
    let max_abs_err = max_abs_err(values, decoded.as_slice::<f64>());

    Ok(Row {
        bytes: bytes as f64,
        compress,
        decompress,
        max_abs_err,
    })
}

fn run_btr(values: &[f64]) -> anyhow::Result<Row> {
    let array = PrimitiveArray::from_iter(values.iter().copied()).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let t0 = Instant::now();
    let compressed = BtrBlocksCompressor::default().compress(&array, &mut ctx)?;
    let compress = t0.elapsed();

    let mut ctx2 = SESSION.create_execution_ctx();
    let t1 = Instant::now();
    let decoded = compressed.clone().execute::<PrimitiveArray>(&mut ctx2)?;
    let decompress = t1.elapsed();
    let max_abs_err = max_abs_err(values, decoded.as_slice::<f64>());

    Ok(Row {
        bytes: compressed.nbytes() as f64,
        compress,
        decompress,
        max_abs_err,
    })
}

fn run_pco(values: &[f64]) -> anyhow::Result<Row> {
    let array = PrimitiveArray::from_iter(values.iter().copied());
    let mut ctx = SESSION.create_execution_ctx();
    let t0 = Instant::now();
    // level 8 (default), values_per_page = 0 → use pco's internal default (262_144).
    let compressed = Pco::from_primitive(array.as_view(), 8, 0, &mut ctx)?;
    let compress = t0.elapsed();

    let bytes: u64 = compressed.as_ref().nbytes();

    let mut ctx2 = SESSION.create_execution_ctx();
    let t1 = Instant::now();
    let decoded = compressed
        .into_array()
        .execute::<PrimitiveArray>(&mut ctx2)?;
    let decompress = t1.elapsed();
    let max_abs_err = max_abs_err(values, decoded.as_slice::<f64>());

    Ok(Row {
        bytes: bytes as f64,
        compress,
        decompress,
        max_abs_err,
    })
}

/// Load every numeric column from a CSV file. Columns where every row parses as f64 are kept;
/// columns with any non-numeric value (text, dates, etc.) are skipped.
fn load_csv_columns(path: &str) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut lines = reader.lines();
    let header_line = match lines.next() {
        Some(Ok(line)) => line,
        _ => return Ok(vec![]),
    };
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();
    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); headers.len()];
    for line in lines.map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        for (i, field) in fields.iter().enumerate() {
            if i >= cols.len() {
                break;
            }
            let trimmed = field.trim().trim_matches('"');
            cols[i].push(trimmed.parse::<f64>().ok());
        }
    }
    let basename = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "csv".to_string());
    let mut out = Vec::new();
    for (i, col) in cols.into_iter().enumerate() {
        // Keep columns where the bulk of rows parsed as f64.
        let valid_count = col.iter().filter(|v| v.is_some()).count();
        if valid_count < col.len() / 2 || valid_count < 16 {
            continue;
        }
        let values: Vec<f64> = col.iter().filter_map(|v| *v).collect();
        let name = headers.get(i).cloned().unwrap_or_else(|| format!("col{i}"));
        out.push((format!("{basename}/{name}"), values));
    }
    Ok(out)
}

fn load_real_datasets(dir: &str) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)?;
    let mut paths: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("csv"))
        .collect();
    paths.sort();
    for p in paths {
        if let Some(path_str) = p.to_str() {
            out.extend(load_csv_columns(path_str)?);
        }
    }
    Ok(out)
}

fn max_abs_err(original: &[f64], decoded: &[f64]) -> f64 {
    original
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ---- synthetic shapes ----

fn synthetic_inputs() -> Vec<(String, Vec<f64>)> {
    let n = 100_000usize;
    vec![
        (format!("uniform_random[{n}]"), uniform_random(n)),
        (format!("linear_ramp[{n}]"), linear_ramp(n)),
        (
            format!("piecewise_linear_noisy[{n}]"),
            piecewise_linear_noisy(n),
        ),
        (format!("sine_drift[{n}]"), sine_drift(n)),
        (format!("stock_walk[{n}]"), stock_walk(n)),
        (format!("hf_sensor[{n}]"), hf_sensor(n)),
        (format!("gps_trace[{n}]"), gps_trace(n)),
        (format!("brownian_bridge[{n}]"), brownian_bridge(n)),
        (format!("ecg_like[{n}]"), ecg_like(n)),
        (format!("temperature_diurnal[{n}]"), temperature_diurnal(n)),
    ]
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
fn hf_sensor(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut v = 0.0_f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        v = 0.95 * v + rng.random_range(-0.1..0.1);
        out.push(v);
    }
    out
}
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
fn brownian_bridge(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut steps = Vec::with_capacity(n);
    let mut acc = 0.0_f64;
    for _ in 0..n {
        acc += rng.random_range(-1.0..1.0);
        steps.push(acc);
    }
    let end = steps[n - 1];
    for (i, v) in steps.iter_mut().enumerate() {
        *v -= end * (i as f64) / (n as f64);
    }
    steps
}
fn ecg_like(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut out = Vec::with_capacity(n);
    let period = 200.0_f64;
    for i in 0..n {
        let t = i as f64;
        let phase = (t / period).fract();
        let base = 0.05 * (t * 0.02).sin();
        let dx = (phase - 0.5) * 16.0;
        let spike = (-(dx * dx)).exp();
        out.push(base + spike + rng.random_range(-0.005..0.005));
    }
    out
}
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
