// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant distortion measurement on real vector datasets.
//!
//! Reports the normalized mean square error (`||x - x'||^2 / ||x||^2`) and the squared
//! cosine-similarity error (`(cos(y_i, x_i) - cos(y_i, x'_i))^2`) against a set of independently
//! sampled unit-norm probe vectors `y_i`, after a full encode and decode roundtrip through the
//! [`vortex_tensor::encodings::turboquant`] scheme.
//!
//! NMSE rather than raw SSE because TurboQuant internally normalizes each input to unit
//! norm before quantizing (storing `||x||` separately), so the paper's Stage-1 bound
//! `E[||unit(x) - unit(x')||^2] <= (sqrt(3) * pi / 2) * 4^(-b)` applies to NMSE directly;
//! raw `||x - x'||^2` sits at `||x||^2` times that bound and isn't comparable across rows.

use std::io::Write;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use tabled::settings::Style;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::error::VortexExpect;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::vector_dataset;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_tensor::encodings::turboquant::TurboQuantConfig;
use vortex_tensor::encodings::turboquant::turboquant_encode;

use crate::SESSION;
use crate::ingest::transform_chunk;

/// Inputs to a distortion run.
#[derive(Debug, Clone)]
pub struct DistortionConfig {
    /// Dataset to load vectors from.
    pub dataset: VectorDataset,
    /// Train-split layout (used to locate the local parquet shards).
    pub layout: TrainLayout,
    /// Bits per quantized coordinate.
    pub bits: u8,
    /// Seed for the SORF rotation.
    pub seed: u64,
    /// Number of sign-diagonal plus Walsh-Hadamard rounds in the SORF transform.
    pub rounds: u8,
    /// Number of base vectors to sample from the first train shard.
    pub samples: usize,
}

/// Mean, median, and max of a sample of distortion measurements.
#[derive(Debug, Clone)]
pub struct DistortionStats {
    /// Arithmetic mean.
    pub mean: f32,
    /// Median (mid element after a partial sort).
    pub median: f32,
    /// Maximum.
    pub max: f32,
}

/// Per-dataset distortion report ready to render as markdown.
#[derive(Debug, Clone)]
pub struct DistortionReport {
    /// Dataset the vectors came from.
    pub dataset: VectorDataset,
    /// Train-split layout used to locate the shard.
    pub layout: TrainLayout,
    /// Vector dimensionality.
    pub dim: u32,
    /// Bits per quantized coordinate.
    pub bits: u8,
    /// Seed for the SORF rotation.
    pub seed: u64,
    /// Number of SORF rounds.
    pub rounds: u8,
    /// Number of base vectors sampled.
    pub samples: usize,
    /// Normalized squared reconstruction error per row, `||x - x'||^2 / ||x||^2`.
    pub reconstruction: DistortionStats,
    /// Squared cosine-similarity error per row against a random unit-norm probe `y_i`,
    /// `(cos(y_i, x_i) - cos(y_i, x'_i))^2`.
    pub decoded_cosine: DistortionStats,
}

/// Compute reconstruction error and cosine-similarity error for a TurboQuant roundtrip.
pub async fn run_distortion(config: &DistortionConfig) -> Result<DistortionReport> {
    let dataset = config.dataset;
    let layout = config.layout;

    let paths = vector_dataset::download(dataset, layout)
        .await
        .with_context(|| format!("download {}", dataset.name()))?;
    let train_path = paths
        .train_files
        .first()
        .with_context(|| format!("dataset {} has no train shards", dataset.name()))?
        .clone();

    let mut ctx = SESSION.create_execution_ctx();

    let chunked = parquet_to_vortex_chunks(train_path).await?;
    let struct_array: StructArray = chunked.into_array().execute(&mut ctx)?;
    let transformed = transform_chunk(struct_array.into_array(), &mut ctx)?;
    let emb_full = transformed
        .as_opt::<Struct>()
        .with_context(|| {
            format!(
                "transform_chunk did not return a Struct, got {}",
                transformed.dtype()
            )
        })?
        .unmasked_field_by_name("emb")
        .context("transformed chunk missing `emb` field")?
        .clone();

    let n = config.samples.min(emb_full.len());
    if n == 0 {
        bail!(
            "distortion: need at least one sampled vector, got 0 (dataset {})",
            dataset.name(),
        );
    }
    let emb = emb_full.slice(0..n)?;

    let original = extract_flat_f32(&emb, &mut ctx)?;
    let dim = pairs_per_row(&original, n)?;

    let tq_config = TurboQuantConfig {
        bit_width: config.bits,
        seed: config.seed,
        num_rounds: config.rounds,
    };
    let encoded = turboquant_encode(emb, &tq_config, &mut ctx)?;
    let decoded_ext: ExtensionArray = encoded.execute(&mut ctx)?;
    let decoded = decoded_ext.into_array();
    let decoded_flat = extract_flat_f32(&decoded, &mut ctx)?;

    let reconstruction = stats(&reconstruction_nmse(&original, &decoded_flat, dim, n));

    // Sample independent unit-norm probe vectors `y_i` (one per row). The TurboQuant Stage-2
    // bound `E[(<y, x> - <y, x'>)^2] <= sqrt(3) * pi^2 / d * 4^(-b)` holds for any fixed `y`,
    // so drawing `y` from the unit sphere is a reasonable empirical sweep.
    let probes = random_unit_vectors(n, dim, config.seed)?;
    let decoded_cosine = stats(&squared_cosine_errors(
        &original,
        &decoded_flat,
        &probes,
        dim,
        n,
    ));

    Ok(DistortionReport {
        dataset,
        layout,
        dim: u32::try_from(dim).context("dim must fit in u32")?,
        bits: config.bits,
        seed: config.seed,
        rounds: config.rounds,
        samples: n,
        reconstruction,
        decoded_cosine,
    })
}

/// Extract a flat `f32` slice from a `Vector<f32, dim>` extension array.
fn extract_flat_f32(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Result<Vec<f32>> {
    let ext: ExtensionArray = array.clone().execute(ctx)?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    Ok(elements.as_slice::<f32>().to_vec())
}

fn pairs_per_row(flat: &[f32], num_rows: usize) -> Result<usize> {
    if num_rows == 0 {
        bail!("distortion: cannot derive dim from zero rows");
    }
    if !flat.len().is_multiple_of(num_rows) {
        bail!(
            "distortion: flat element count {} not divisible by row count {num_rows}",
            flat.len(),
        );
    }
    Ok(flat.len() / num_rows)
}

/// Normalized squared reconstruction error per row, `||x - x'||^2 / ||x||^2`. Zero-norm
/// rows are dropped from the sample because NMSE is undefined when `||x|| = 0`, and our
/// vector datasets are not expected to contain zero vectors.
fn reconstruction_nmse(
    original: &[f32],
    reconstructed: &[f32],
    dim: usize,
    num_rows: usize,
) -> Vec<f32> {
    (0..num_rows)
        .filter_map(|row| {
            let start = row * dim;
            let end = start + dim;
            let orig = &original[start..end];
            let recon = &reconstructed[start..end];
            let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
            if norm_sq == 0.0 {
                return None;
            }
            let err_sq: f32 = orig
                .iter()
                .zip(recon.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            Some(err_sq / norm_sq)
        })
        .collect()
}

/// Sample `num_rows` independent `dim`-D vectors with standard-normal entries and normalize each
/// row to unit L2 norm. Used as probe vectors `y_i` for the squared cosine-similarity error.
fn random_unit_vectors(num_rows: usize, dim: usize, seed: u64) -> Result<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).context("constructing Normal(0, 1)")?;
    let mut buf = vec![0.0_f32; num_rows * dim];
    for row in 0..num_rows {
        let start = row * dim;
        let end = start + dim;
        for v in &mut buf[start..end] {
            *v = normal.sample(&mut rng);
        }
        let norm = buf[start..end].iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut buf[start..end] {
                *v /= norm;
            }
        }
    }
    Ok(buf)
}

/// Cosine similarity of two equal-length vectors, returning `0.0` if either has zero norm.
/// A zero-norm decoded vector represents genuine quantizer failure, so the caller still
/// gets a defined per-row error that reflects the lost direction.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Per-row squared cosine-similarity error against probe `y_i`,
/// `(cos(y_i, x_i) - cos(y_i, x'_i))^2`. Rows whose original `x_i` has zero norm are
/// dropped, matching [`reconstruction_nmse`].
fn squared_cosine_errors(
    original: &[f32],
    reconstructed: &[f32],
    probes: &[f32],
    dim: usize,
    num_rows: usize,
) -> Vec<f32> {
    (0..num_rows)
        .filter_map(|row| {
            let start = row * dim;
            let end = start + dim;
            let xi = &original[start..end];
            let xi_dec = &reconstructed[start..end];
            let yi = &probes[start..end];
            if xi.iter().map(|&v| v * v).sum::<f32>() == 0.0 {
                return None;
            }
            let diff = cosine(yi, xi) - cosine(yi, xi_dec);
            Some(diff * diff)
        })
        .collect()
}

fn stats(samples: &[f32]) -> DistortionStats {
    if samples.is_empty() {
        return DistortionStats {
            mean: f32::NAN,
            median: f32::NAN,
            max: f32::NAN,
        };
    }

    let sum: f64 = samples.iter().map(|&v| f64::from(v)).sum();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "casting an f64 mean back to f32 is intentional and matches the input precision"
    )]
    let mean = (sum / samples.len() as f64) as f32;

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    };

    let max = samples
        .iter()
        .copied()
        .reduce(f32::max)
        .vortex_expect("samples is non-empty per the early return above");

    DistortionStats { mean, median, max }
}

impl DistortionReport {
    /// Render the report as a markdown header line followed by a tabled table.
    pub fn render(&self, writer: &mut dyn Write) -> Result<()> {
        writeln!(
            writer,
            "## {} | dim={} | layout={} | bits={} | samples={} | seed={} | rounds={}",
            self.dataset.name(),
            self.dim,
            self.layout.label(),
            self.bits,
            self.samples,
            self.seed,
            self.rounds,
        )?;

        let rows: &[(&str, f32)] = &[
            ("reconstruction NMSE mean", self.reconstruction.mean),
            ("reconstruction NMSE median", self.reconstruction.median),
            ("reconstruction NMSE max", self.reconstruction.max),
            ("decoded cosine sqerr mean", self.decoded_cosine.mean),
            ("decoded cosine sqerr median", self.decoded_cosine.median),
            ("decoded cosine sqerr max", self.decoded_cosine.max),
        ];

        let mut builder = tabled::builder::Builder::new();
        builder.push_record(["metric", "value"]);
        for &(metric, value) in rows {
            builder.push_record([metric.to_owned(), format_metric(value)]);
        }
        let mut table = builder.build();
        table.with(Style::modern());
        writeln!(writer, "{table}")?;
        Ok(())
    }
}

fn format_metric(value: f32) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value == 0.0 {
        "0".to_owned()
    } else if value.abs() < 1e-3 || value.abs() >= 1e4 {
        format!("{value:.3e}")
    } else {
        format!("{value:.6}")
    }
}
