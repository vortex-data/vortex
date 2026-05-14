// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Analysis binary for the layered-pco-bench crate.
//!
//! Compares five compressors on a small set of i64 columns. Two are
//! synthetic (kept from P6 so the comparison stays anchored), and the rest
//! are pulled from TPC-H at scale factor 0.1 via `tpchgen-arrow`. Other
//! real datasets (ClickBench, NYC taxi) are skipped — see RESULTS.md for
//! the rationale.
//!
//! - `pco_default` — full pco, default page size.
//! - `pco_1k` — full pco, 1024 values per page.
//! - `btrblocks_only` — vanilla BtrBlocks cascade, no pco layers.
//! - `hybrid` — pco-style structural top (OrderedLatent or
//!   ConsecutiveDelta) with each leaf compressed by BtrBlocks.
//! - `layered_plain` — the same structural top, but raw `PrimitiveArray`
//!   leaves. No entropy coder. Useful as a "no-bottom" baseline.
//!
//! Outputs a markdown table to stdout and writes a copy to `RESULTS.md`.

use std::fs::File;
use std::io::Write;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use layered_pco_bench::datasets::DatasetColumn;
use layered_pco_bench::datasets::tpch_columns;
use layered_pco_bench::hybrid_compress;
use layered_pco_bench::layered_plain_compress;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_pco::Pco;
use vortex_session::VortexSession;

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------

const SYNTHETIC_N: usize = 1_000_000;
const SCALAR_AT_SAMPLES: usize = 1_000;
const DECODE_RUNS: usize = 5;
const SEED: u64 = 42;
const PCO_LEVEL: usize = 0;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

// ----------------------------------------------------------------------------
// Synthetic datasets (carried over from P6 verbatim)
// ----------------------------------------------------------------------------

fn build_monotone_timestamps() -> Buffer<i64> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let base: i64 = 1_700_000_000_000;
    let mut out = BufferMut::<i64>::with_capacity(SYNTHETIC_N);
    for i in 0..SYNTHETIC_N {
        let noise: i64 = rng.random_range(-50i64..=50);
        out.push(base + (i as i64) * 1000 + noise);
    }
    out.freeze()
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "f64->i64 OK, range is bounded by *1000.0"
)]
fn build_cube_distributed() -> Buffer<i64> {
    let mut rng = SmallRng::seed_from_u64(SEED ^ 0xCAFE_C0DE_F00D_FEED);
    let mut out = BufferMut::<i64>::with_capacity(SYNTHETIC_N);
    for _ in 0..SYNTHETIC_N {
        let u: f64 = rng.random::<f64>();
        out.push((u.powi(3) * 1000.0) as i64);
    }
    out.freeze()
}

fn to_primitive(buf: Buffer<i64>) -> PrimitiveArray {
    PrimitiveArray::new(buf, Validity::NonNullable)
}

fn sample_indices(n: usize) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    (0..SCALAR_AT_SAMPLES)
        .map(|_| rng.random_range(0..n))
        .collect()
}

// ----------------------------------------------------------------------------
// Column input shared across synthetic + real loaders
// ----------------------------------------------------------------------------

struct ColumnInput {
    dataset: &'static str,
    column: &'static str,
    array: PrimitiveArray,
}

impl ColumnInput {
    fn from_dataset_column(c: DatasetColumn) -> Self {
        Self {
            dataset: c.dataset,
            column: c.column,
            array: c.array,
        }
    }

    fn synthetic(dataset: &'static str, column: &'static str, buf: Buffer<i64>) -> Self {
        Self {
            dataset,
            column,
            array: to_primitive(buf),
        }
    }

    fn len(&self) -> usize {
        self.array.len()
    }

    fn raw_bytes(&self) -> u64 {
        (self.array.len() * size_of::<i64>()) as u64
    }
}

// ----------------------------------------------------------------------------
// Variants
// ----------------------------------------------------------------------------

#[derive(Copy, Clone)]
enum Variant {
    PcoDefault,
    Pco1k,
    BtrblocksOnly,
    Hybrid,
    LayeredPlain,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::PcoDefault => "pco_default",
            Variant::Pco1k => "pco_1k",
            Variant::BtrblocksOnly => "btrblocks_only",
            Variant::Hybrid => "hybrid",
            Variant::LayeredPlain => "layered_plain",
        }
    }

    fn compress(self, parray: &PrimitiveArray) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        match self {
            Variant::PcoDefault => {
                Ok(Pco::from_primitive(parray.as_view(), PCO_LEVEL, 0, &mut ctx)?.into_array())
            }
            Variant::Pco1k => {
                Ok(Pco::from_primitive(parray.as_view(), PCO_LEVEL, 1024, &mut ctx)?.into_array())
            }
            Variant::BtrblocksOnly => {
                let compressor = BtrBlocksCompressor::default();
                compressor.compress(&parray.clone().into_array(), &mut ctx)
            }
            Variant::Hybrid => hybrid_compress(parray.as_view(), &mut ctx),
            Variant::LayeredPlain => layered_plain_compress(parray.as_view(), &mut ctx),
        }
    }
}

const ALL_VARIANTS: &[Variant] = &[
    Variant::PcoDefault,
    Variant::Pco1k,
    Variant::BtrblocksOnly,
    Variant::Hybrid,
    Variant::LayeredPlain,
];

// ----------------------------------------------------------------------------
// Measurement
// ----------------------------------------------------------------------------

struct Measurement {
    variant: Variant,
    compressed_bytes: u64,
    ratio: f64,
    decode_mb_s: f64,
    scalar_at_ns: f64,
}

fn measure_decode(compressed: &ArrayRef, n: usize) -> VortexResult<f64> {
    let bytes_per_run = (n * size_of::<i64>()) as f64;
    let mut best = Duration::from_secs(u64::MAX);
    for _ in 0..DECODE_RUNS {
        let mut ctx = SESSION.create_execution_ctx();
        let start = Instant::now();
        let decoded = compressed.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let elapsed = start.elapsed();
        std::hint::black_box(decoded);
        if elapsed < best {
            best = elapsed;
        }
    }
    Ok(bytes_per_run / best.as_secs_f64() / 1_000_000.0)
}

fn measure_scalar_at(compressed: &ArrayRef, indices: &[usize]) -> VortexResult<f64> {
    let mut ctx = SESSION.create_execution_ctx();
    // Warmup pass: prime any one-shot decoders.
    for &i in indices.iter().take(16) {
        std::hint::black_box(compressed.execute_scalar(i, &mut ctx)?);
    }
    let start = Instant::now();
    for &i in indices {
        std::hint::black_box(compressed.execute_scalar(i, &mut ctx)?);
    }
    let elapsed = start.elapsed();
    Ok(elapsed.as_nanos() as f64 / indices.len() as f64)
}

fn measure_one(
    variant: Variant,
    parray: &PrimitiveArray,
    indices: &[usize],
    raw_bytes: u64,
    n: usize,
) -> VortexResult<Measurement> {
    let compressed = variant.compress(parray)?;

    // Sanity check that the variant produces the original column.
    {
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = compressed.clone().execute::<PrimitiveArray>(&mut ctx)?;
        if decoded.as_slice::<i64>() != parray.as_slice::<i64>() {
            vortex_error::vortex_bail!("variant {} failed round trip", variant.name());
        }
    }

    let compressed_bytes = compressed.nbytes();
    let ratio = raw_bytes as f64 / compressed_bytes.max(1) as f64;
    let decode_mb_s = measure_decode(&compressed, n)?;
    let scalar_at_ns = measure_scalar_at(&compressed, indices)?;

    Ok(Measurement {
        variant,
        compressed_bytes,
        ratio,
        decode_mb_s,
        scalar_at_ns,
    })
}

// ----------------------------------------------------------------------------
// Reporting
// ----------------------------------------------------------------------------

struct ColumnReport {
    dataset: &'static str,
    column: &'static str,
    n: usize,
    measurements: Vec<Measurement>,
}

fn format_combined_table(reports: &[ColumnReport]) -> String {
    let mut out = String::new();
    out.push_str(
        "| column | dataset | N | variant | bytes | ratio× | decode MB/s | scalar_at ns/op |\n",
    );
    out.push_str("|---|---|---:|---|---:|---:|---:|---:|\n");
    for r in reports {
        let best_bytes = r
            .measurements
            .iter()
            .map(|m| m.compressed_bytes)
            .min()
            .unwrap_or(u64::MAX);
        let best_ratio = r
            .measurements
            .iter()
            .map(|m| m.ratio)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_decode = r
            .measurements
            .iter()
            .map(|m| m.decode_mb_s)
            .fold(f64::NEG_INFINITY, f64::max);
        let best_scalar = r
            .measurements
            .iter()
            .map(|m| m.scalar_at_ns)
            .fold(f64::INFINITY, f64::min);

        for m in &r.measurements {
            let bytes_str = if m.compressed_bytes == best_bytes {
                format!("**{}**", m.compressed_bytes)
            } else {
                format!("{}", m.compressed_bytes)
            };
            let ratio_str = if (m.ratio - best_ratio).abs() < 1e-6 {
                format!("**{:.2}**", m.ratio)
            } else {
                format!("{:.2}", m.ratio)
            };
            let decode_str = if (m.decode_mb_s - best_decode).abs() < 1e-3 {
                format!("**{:.1}**", m.decode_mb_s)
            } else {
                format!("{:.1}", m.decode_mb_s)
            };
            let scalar_str = if (m.scalar_at_ns - best_scalar).abs() < 1e-3 {
                format!("**{:.0}**", m.scalar_at_ns)
            } else {
                format!("{:.0}", m.scalar_at_ns)
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                r.column,
                r.dataset,
                r.n,
                m.variant.name(),
                bytes_str,
                ratio_str,
                decode_str,
                scalar_str,
            ));
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------

fn collect_inputs() -> Vec<ColumnInput> {
    let mut inputs = Vec::new();

    // Synthetic columns (kept from P6).
    inputs.push(ColumnInput::synthetic(
        "synthetic",
        "monotone_timestamps",
        build_monotone_timestamps(),
    ));
    inputs.push(ColumnInput::synthetic(
        "synthetic",
        "cube_distributed",
        build_cube_distributed(),
    ));

    // Real columns.
    for c in tpch_columns() {
        inputs.push(ColumnInput::from_dataset_column(c));
    }

    inputs
}

fn run() -> VortexResult<String> {
    let inputs = collect_inputs();

    let mut report = String::new();
    report.push_str("# layered-pco-bench results\n\n");
    report.push_str(&format!(
        "Generated by `layered-pco-bench` (P7). decode = best of {} runs, \
         scalar_at = average over {} random indices.\n\n",
        DECODE_RUNS, SCALAR_AT_SAMPLES
    ));
    report.push_str(
        "## Datasets\n\n\
         - **synthetic**: the two columns carried over from P6 verbatim.\n\
         - **tpch_sf0p1_lineitem / tpch_sf0p1_orders**: TPC-H tables at scale\n  \
           factor 0.1, generated in-process by `tpchgen-arrow`. Lineitem has \
           ~600k rows; orders has ~150k rows. Decimal128 columns are cast to \
           i64 cents and date32 columns are widened to i64 so every variant \
           sees the same `Primitive<I64>` input.\n\
         - **ClickBench / NYC taxi**: skipped (no in-tree loader, would need \
           multi-GB downloads and a Parquet read path; budget on dataset \
           acquisition was capped at 20 minutes per task spec).\n\n",
    );
    report.push_str("## Results\n\n");

    let mut column_reports = Vec::new();
    for input in &inputs {
        let n = input.len();
        let raw_bytes = input.raw_bytes();
        let indices = sample_indices(n);
        eprintln!("=== {}/{} (N={}) ===", input.dataset, input.column, n);
        let mut measurements = Vec::new();
        for &variant in ALL_VARIANTS {
            eprintln!("  variant: {}", variant.name());
            let m = measure_one(variant, &input.array, &indices, raw_bytes, n)?;
            eprintln!(
                "    bytes={}, ratio={:.2}x, decode={:.1} MB/s, scalar_at={:.0} ns/op",
                m.compressed_bytes, m.ratio, m.decode_mb_s, m.scalar_at_ns
            );
            measurements.push(m);
        }
        column_reports.push(ColumnReport {
            dataset: input.dataset,
            column: input.column,
            n,
            measurements,
        });
    }

    report.push_str(&format_combined_table(&column_reports));
    report.push('\n');
    Ok(report)
}

fn main() -> VortexResult<()> {
    let report = run()?;
    println!("{}", report);

    // Persist a copy next to the binary's source for easy commit/inspection.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/RESULTS.md");
    let mut f = File::create(path).map_err(vortex_error::VortexError::from)?;
    f.write_all(report.as_bytes())
        .map_err(vortex_error::VortexError::from)?;
    eprintln!("wrote {}", path);
    Ok(())
}
