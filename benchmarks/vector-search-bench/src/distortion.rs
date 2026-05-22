// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant distortion measurement on real vector datasets.
//!
//! Reports per-vector normalized reconstruction error (`||x - x'||^2 / ||x||^2`) and pairwise
//! cosine-similarity error (`|cos(x_i, x_j) - cos(x'_i, x'_j)|`) after a full encode and decode
//! roundtrip through the [`vortex_tensor::encodings::turboquant`] scheme. This is the same
//! TurboQuant implementation the search subcommand stores on disk via
//! [`BtrBlocksCompressorBuilder::with_turboquant`](vortex_btrblocks::BtrBlocksCompressorBuilder).

use std::io::Write;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use tabled::settings::Style;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::ScalarFnArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::vector_dataset;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_tensor::encodings::turboquant::TurboQuantConfig;
use vortex_tensor::encodings::turboquant::turboquant_encode;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;

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
    /// Per-vector normalized squared L2 reconstruction error.
    pub reconstruction: DistortionStats,
    /// Pairwise cosine-similarity error after decoding both sides.
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
    if n < 2 {
        bail!(
            "distortion: need at least 2 sampled vectors for cosine pairs, got {n} (dataset {})",
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
    let encoded = turboquant_encode(emb.clone(), &tq_config, &mut ctx)?;
    let decoded_ext: ExtensionArray = encoded.execute(&mut ctx)?;
    let decoded = decoded_ext.into_array();
    let decoded_flat = extract_flat_f32(&decoded, &mut ctx)?;

    let reconstruction = stats(&reconstruction_errors(&original, &decoded_flat, dim, n));

    let half = n / 2;
    let mut shuffled: Vec<usize> = (0..n).collect();
    shuffled.shuffle(&mut StdRng::seed_from_u64(config.seed));
    let lhs_indices = indices_to_array(&shuffled[..half]);
    let rhs_indices = indices_to_array(&shuffled[half..2 * half]);

    let true_cosines = compute_cosines(
        emb.take(lhs_indices.clone())?,
        emb.take(rhs_indices.clone())?,
        &mut ctx,
    )?;
    let decoded_cosines = compute_cosines(
        decoded.take(lhs_indices)?,
        decoded.take(rhs_indices)?,
        &mut ctx,
    )?;
    let decoded_cosine = stats(&abs_diff(&true_cosines, &decoded_cosines));

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

/// Compute one cosine per row over two equal-length tensor-like arrays.
fn compute_cosines(lhs: ArrayRef, rhs: ArrayRef, ctx: &mut ExecutionCtx) -> Result<Vec<f32>> {
    let len = lhs.len();
    let sfn: ScalarFnArray = CosineSimilarity::try_new_array(lhs, rhs, len)?;
    let prim: PrimitiveArray = sfn.into_array().execute(ctx)?;
    Ok(prim.as_slice::<f32>().to_vec())
}

/// Build a non-nullable `PrimitiveArray<u64>` of row indices for use with [`ArrayRef::take`].
fn indices_to_array(indices: &[usize]) -> ArrayRef {
    let buf: Buffer<u64> = indices.iter().map(|&i| i as u64).collect();
    PrimitiveArray::new::<u64>(buf, Validity::NonNullable).into_array()
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

/// Per-vector normalized reconstruction squared error (NMSE). Rows whose original squared norm is
/// below `1e-10` are dropped because their normalized error is numerically undefined.
fn reconstruction_errors(
    original: &[f32],
    reconstructed: &[f32],
    dim: usize,
    num_rows: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_rows);
    for row in 0..num_rows {
        let start = row * dim;
        let end = start + dim;
        let orig = &original[start..end];
        let recon = &reconstructed[start..end];
        let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
        if norm_sq < 1e-10 {
            continue;
        }
        let err_sq: f32 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        out.push(err_sq / norm_sq);
    }
    out
}

fn abs_diff(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| (a - b).abs())
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

    let max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);

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
            ("decoded cosine err mean", self.decoded_cosine.mean),
            ("decoded cosine err median", self.decoded_cosine.median),
            ("decoded cosine err max", self.decoded_cosine.max),
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
