// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hand-rolled Rust cosine similarity baseline.
//!
//! This module provides the *compute-cost floor* the other Vortex variants are measured
//! against. It is **not** a realistic "parquet in a DBMS" baseline — it's the minimum
//! amount of work a Rust programmer could get away with if they wrote a vector-search
//! scan by hand with no query engine, no scalar-function dispatch, and no Arrow compute
//! kernels.
//!
//! Two distinct phases run per iteration, and the benchmark times them separately so the
//! dashboard can separate storage-read cost from compute cost:
//!
//! 1. **Decompress** ([`read_parquet_embedding_column`]) — reads the canonical parquet
//!    file via `parquet-rs`, downcasts the `emb` column to an Arrow `Float32Array`, and
//!    copies every value into a flat `Vec<f32>`. This phase is the only place Arrow is
//!    actually used — only for the decode. The `memcpy` at the end is incidental: we
//!    could operate directly on `Float32Array::values()` with identical performance,
//!    but taking ownership of a `Vec<f32>` frees the Arrow `RecordBatch` lifetimes.
//! 2. **Compute** ([`cosine_loop`] and [`filter_loop`]) — runs a plain scalar Rust loop
//!    over `&[f32]`. Arrow is no longer involved. There's no SIMD, no unrolling
//!    annotations, no dispatch overhead, no output-array allocation beyond a single
//!    `Vec<f32>`. This is deliberately "the fastest you could possibly make it go
//!    without writing SIMD intrinsics".
//!
//! Calling this "the parquet baseline" would be misleading, because:
//!
//! - The compute layer has nothing to do with parquet — parquet is only the input
//!   encoding, not the execution substrate.
//! - Real parquet-on-DBMS engines (DuckDB's `list_cosine_similarity`, DataFusion with a
//!   vector UDF, etc.) would pay substantial dispatch / planner / row-iterator cost
//!   that this loop skips entirely.
//!
//! Think of it as: "If you didn't have Vortex and didn't feel like reaching for a query
//! engine, what's the minimum scan cost you could get away with on this data?" That's
//! the question this module answers, and it's intentionally a lower bound rather than a
//! fair DBMS comparison. Future work could add DuckDB / DataFusion baselines alongside
//! this one for the DBMS-level comparison.

use std::borrow::Cow;
use std::fs::File;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use arrow_array::Array;
use arrow_array::FixedSizeListArray;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::GenericListArray;
use arrow_array::LargeListArray;
use arrow_array::ListArray;
use arrow_array::OffsetSizeTrait;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_bench::Format;
use vortex_bench::measurements::CompressionTimingMeasurement;
use vortex_bench::measurements::CustomUnitMeasurement;

use crate::DEFAULT_THRESHOLD;
use crate::VariantTimings;
use crate::verify::VerificationKind;
use crate::verify::verify_and_report_scores;

/// Read the entire `emb` column of a parquet file into a single flat `Vec<f32>`, along
/// with the dimension and row count. This is the *decompress* phase of the hand-rolled
/// baseline — it's the only place Arrow is actually used. `parquet-rs` does the file
/// decode, we downcast to `Float32Array`, and then memcpy into a plain `Vec<f32>` so
/// the compute loop can operate over a raw slice without holding any Arrow
/// `RecordBatch` references.
///
/// Kept under its `parquet` name because this function *actually reads parquet*; only
/// the compute-side wrappers take the `handrolled` label.
pub fn read_parquet_embedding_column(parquet_path: &Path) -> Result<HandrolledBaselineData> {
    let file = File::open(parquet_path)
        .with_context(|| format!("open parquet file {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    // Locate the `emb` column and sanity-check its type.
    let (_, emb_field) = builder
        .schema()
        .column_with_name("emb")
        .context("parquet schema missing `emb` column")?;

    // VectorDBBench parquet files use `list<float>`; some others use `fixed_size_list`.
    // Both need to be supported — the canonical parquet emit from arrow-rs is `list<f32>`
    // since parquet has no fixed-size-list logical type.
    let element_dtype = match emb_field.data_type() {
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            field.data_type().clone()
        }
        other => bail!("emb column must be a list of float, got {other:?}"),
    };
    if !matches!(element_dtype, DataType::Float32 | DataType::Float64) {
        bail!(
            "emb column element type must be Float32 or Float64, got {:?}",
            element_dtype
        );
    }

    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    let mut data = Vec::<f32>::new();
    let mut num_rows = 0usize;
    let mut inferred_dim: Option<usize> = None;

    for batch in batches.iter() {
        let column = batch
            .column_by_name("emb")
            .context("emb column missing from record batch")?;
        append_batch(column, &mut data, &mut inferred_dim, &mut num_rows)?;
    }

    let dim = inferred_dim.context("parquet file has zero rows — cannot infer dimension")?;
    Ok(HandrolledBaselineData {
        elements: data,
        dim,
        num_rows,
    })
}

fn append_batch(
    column: &dyn Array,
    data: &mut Vec<f32>,
    inferred_dim: &mut Option<usize>,
    num_rows: &mut usize,
) -> Result<()> {
    if let Some(fsl) = column.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = fsl.value_length() as usize;
        maybe_set_dim(inferred_dim, dim)?;
        extend_float_values(fsl.values(), data)?;
        *num_rows += fsl.len();
        return Ok(());
    }

    // `ListArray` and `LargeListArray` are both `GenericListArray<O>`, differing
    // only in their offset width (`i32` vs `i64`). Real VectorDBBench parquet files
    // canonicalize to `LargeList<f32>` on read, while some smaller test fixtures
    // still arrive as `List<f32>` — handle both through the same generic helper.
    if let Some(list) = column.as_any().downcast_ref::<ListArray>() {
        return append_generic_list(list, data, inferred_dim, num_rows);
    }
    if let Some(list) = column.as_any().downcast_ref::<LargeListArray>() {
        return append_generic_list(list, data, inferred_dim, num_rows);
    }

    bail!(
        "emb column has unsupported arrow type {:?}",
        column.data_type()
    );
}

/// Flatten a `GenericListArray<O>` of `Float32` values into `data`.
///
/// The offsets are used purely to validate that every row has the same length;
/// they're effectively discarded on the output side, since flattening `N` uniform
/// rows of length `dim` into one contiguous `Vec<f32>` just yields the total span
/// `values[first_offset..last_offset]` regardless of how that span is sliced by
/// per-row offsets. That's why this helper is generic over `OffsetSizeTrait` and
/// works verbatim for both `ListArray` (i32 offsets) and `LargeListArray` (i64
/// offsets) — the only difference between the two is how wide an integer we cast
/// to `usize`, which `OffsetSizeTrait::as_usize` handles for us.
fn append_generic_list<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    data: &mut Vec<f32>,
    inferred_dim: &mut Option<usize>,
    num_rows: &mut usize,
) -> Result<()> {
    let offsets = list.value_offsets();
    for i in 0..list.len() {
        let start = offsets[i].as_usize();
        let end = offsets[i + 1].as_usize();
        let row_len = end - start;
        maybe_set_dim(inferred_dim, row_len)?;
        extend_float_values_range(list.values(), data, start, end)?;
        *num_rows += 1;
    }
    Ok(())
}

/// Extend `data` with f32 values from an arrow array. Accepts both Float32 (zero-copy)
/// and Float64 (lossy narrowing cast).
fn extend_float_values(values: &dyn Array, data: &mut Vec<f32>) -> Result<()> {
    if let Some(f32s) = values.as_any().downcast_ref::<Float32Array>() {
        data.extend_from_slice(f32s.values());
    } else if let Some(f64s) = values.as_any().downcast_ref::<Float64Array>() {
        #[expect(clippy::cast_possible_truncation)]
        data.extend(f64s.values().iter().map(|&v| v as f32));
    } else {
        bail!(
            "emb column values must be Float32 or Float64, got {:?}",
            values.data_type()
        );
    }
    Ok(())
}

/// Like [`extend_float_values`] but only appends a sub-range `[start..end)`.
fn extend_float_values_range(
    values: &dyn Array,
    data: &mut Vec<f32>,
    start: usize,
    end: usize,
) -> Result<()> {
    if let Some(f32s) = values.as_any().downcast_ref::<Float32Array>() {
        data.extend_from_slice(&f32s.values()[start..end]);
    } else if let Some(f64s) = values.as_any().downcast_ref::<Float64Array>() {
        #[expect(clippy::cast_possible_truncation)]
        data.extend(f64s.values()[start..end].iter().map(|&v| v as f32));
    } else {
        bail!(
            "emb column values must be Float32 or Float64, got {:?}",
            values.data_type()
        );
    }
    Ok(())
}

fn maybe_set_dim(inferred_dim: &mut Option<usize>, new_dim: usize) -> Result<()> {
    match inferred_dim {
        Some(d) if *d == new_dim => Ok(()),
        Some(d) => bail!("inconsistent emb dimensions: saw {d} then {new_dim}"),
        None if new_dim == 0 => bail!("emb row has zero elements"),
        None => {
            *inferred_dim = Some(new_dim);
            Ok(())
        }
    }
}

/// The flattened representation of an embedding column, suitable for a hand-rolled
/// distance loop. Intentionally decoupled from any format — the compute side doesn't
/// care how the data got into this `Vec<f32>`.
///
/// The benchmark's "size" measurement for the handrolled baseline comes from
/// [`crate::PreparedDataset::parquet_bytes`] (which is populated once in
/// [`crate::prepare_dataset`]), not from this struct. We deliberately don't carry
/// the file size in here — doing so would duplicate state between two places that
/// can go out of sync.
pub struct HandrolledBaselineData {
    /// All rows concatenated: `elements.len() == num_rows * dim`.
    pub elements: Vec<f32>,
    /// Vector dimensionality.
    pub dim: usize,
    /// Number of rows.
    pub num_rows: usize,
}

/// Result of running the hand-rolled baseline timing loop.
///
/// Carries both the best-of-N timing numbers **and** the cosine scores from the final
/// iteration. The scores are exposed so the caller can feed them into
/// [`crate::verify::verify_and_report_scores`] for the correctness check without
/// re-reading the parquet file. Because `cosine_loop` is deterministic, the scores
/// from any iteration equal the scores from every other iteration; using the last
/// one is simply the most convenient snapshot.
pub struct HandrolledBaselineResult {
    /// Best-of-N wall times for decompress / cosine / filter.
    pub timings: VariantTimings,
    /// Cosine-similarity scores from the final iteration. Length equals the dataset
    /// row count.
    pub last_scores: Vec<f32>,
}

/// Run the decompress / cosine / filter microbenchmarks for the hand-rolled baseline
/// and return the best-of-N wall times along with the last iteration's cosine scores.
///
/// The decompress phase re-reads the parquet file from disk on each iteration (matches
/// how the Vortex variants re-execute their tree from scratch each iteration), and the
/// compute phase runs [`cosine_loop`] and [`filter_loop`] over the flat `Vec<f32>` the
/// decompress phase produced. Returning the last iteration's scores lets the caller
/// perform correctness verification against the Vortex baseline without a redundant
/// parquet read.
///
/// # Errors
///
/// Returns an error if `iterations == 0`. The benchmark CLI defaults to 5 and the
/// lowest meaningful value is 1 (single-shot best-of-1).
pub fn run_handrolled_baseline_timings(
    parquet_path: &Path,
    query: &[f32],
    threshold: f32,
    iterations: usize,
) -> Result<HandrolledBaselineResult> {
    anyhow::ensure!(
        iterations > 0,
        "run_handrolled_baseline_timings requires iterations >= 1"
    );

    let mut decompress = Duration::MAX;
    let mut cosine = Duration::MAX;
    let mut filter = Duration::MAX;
    let mut last_scores: Vec<f32> = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        let mut data = read_parquet_embedding_column(parquet_path)?;
        decompress = decompress.min(start.elapsed());

        // Normalize in-place between decompress and cosine timing, matching the Vortex
        // variants which normalize outside their timing window.
        normalize_in_place(&mut data.elements, data.num_rows, data.dim);

        let start = Instant::now();
        let scores = cosine_loop(&data.elements, data.num_rows, data.dim, query);
        cosine = cosine.min(start.elapsed());
        debug_assert_eq!(scores.len(), data.num_rows);

        let start = Instant::now();
        let matches = filter_loop(&scores, threshold);
        filter = filter.min(start.elapsed());
        debug_assert_eq!(matches.len(), data.num_rows);

        last_scores = scores;
    }

    Ok(HandrolledBaselineResult {
        timings: VariantTimings {
            decompress,
            cosine,
            filter,
        },
        last_scores,
    })
}

/// Normalize every row of the flat element buffer to unit L2 norm in-place. Zero-norm
/// rows are left as all zeros.
fn normalize_in_place(elements: &mut [f32], num_rows: usize, dim: usize) {
    for row in elements.chunks_exact_mut(dim) {
        let norm = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
    debug_assert_eq!(elements.len(), num_rows * dim);
}

/// Compute cosine similarity for every row against `query`. The data rows are assumed to
/// be unit-norm (via [`normalize_in_place`]). Returns one f32 score per row; a zero-norm
/// query produces all-zero scores.
pub fn cosine_loop(elements: &[f32], num_rows: usize, dim: usize, query: &[f32]) -> Vec<f32> {
    assert_eq!(query.len(), dim);
    assert_eq!(elements.len(), num_rows * dim);

    let query_norm = query.iter().map(|&q| q * q).sum::<f32>().sqrt();
    let mut out = Vec::with_capacity(num_rows);
    if query_norm == 0.0 {
        out.resize(num_rows, 0.0);
        return out;
    }

    let inv_query_norm = 1.0 / query_norm;
    for slice in elements.chunks_exact(dim) {
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;

        let chunks = slice.chunks_exact(4);
        let q_chunks = query.chunks_exact(4);
        let rem = chunks.remainder();
        let q_rem = q_chunks.remainder();

        for (s, q) in chunks.zip(q_chunks) {
            dot0 += s[0] * q[0];
            dot1 += s[1] * q[1];
            dot2 += s[2] * q[2];
            dot3 += s[3] * q[3];
        }
        for (&s, &q) in rem.iter().zip(q_rem.iter()) {
            dot0 += s * q;
        }

        out.push((dot0 + dot1 + dot2 + dot3) * inv_query_norm);
    }
    out
}

/// Build the `cosine > threshold` boolean mask — **strict greater-than**, matching the
/// Vortex-side path which uses `Operator::Gt` in
/// [`vortex_tensor::vector_search::build_similarity_search_tree`]. Keep these two in
/// sync: if one changes the comparison semantics, the correctness-verification pass will
/// start reporting a mismatch for the lossless variants.
pub fn filter_loop(scores: &[f32], threshold: f32) -> Vec<bool> {
    scores.iter().map(|&s| s > threshold).collect()
}

/// Run the hand-rolled timing + correctness pipeline for one dataset and
/// append the resulting measurements into the caller's collection vecs.
///
/// This is the "push-side" wrapper around [`run_handrolled_baseline_timings`]
/// and [`verify_and_report_scores`]: the bench-loop in `main.rs` used to
/// inline this block of code alongside the vortex-variant loop, which was
/// long enough to obscure the actual dataset iteration. Extracting it keeps
/// `main.rs::main` focused on the outer control flow.
///
/// The emitted measurement grammar is identical to the inlined version
/// (same names, same [`Format::Parquet`] target), which matters because
/// `gh-json` output is what CI consumes.
///
/// # Parameters
///
/// - `parquet_path`: on-disk parquet file for this dataset (the decompress
///   phase re-reads it on every iteration).
/// - `dataset_name`: the [`crate::PreparedDataset::name`] used as the
///   dataset segment of every metric name.
/// - `parquet_bytes`: size of `parquet_path` in bytes, emitted as the
///   `handrolled size/<dataset>` measurement.
/// - `query`: the single-row query vector, forwarded to the cosine loop.
/// - `baseline_scores`: ground-truth cosine scores for the verification
///   pass. A drift from these bails the whole run.
/// - `iterations`: number of timed iterations per phase (best-of-N).
/// - `timings`, `sizes`, `verification`: the caller's collection vecs.
///   This function **appends** to them — it does not replace or sort.
#[allow(clippy::too_many_arguments)]
pub fn run_handrolled_and_collect(
    parquet_path: &Path,
    dataset_name: &str,
    parquet_bytes: u64,
    query: &[f32],
    baseline_scores: &[f32],
    iterations: usize,
    timings: &mut Vec<CompressionTimingMeasurement>,
    sizes: &mut Vec<CustomUnitMeasurement>,
    verification: &mut Vec<CustomUnitMeasurement>,
) -> Result<()> {
    let label = "handrolled";
    let bench_name = format!("{label}/{dataset_name}");

    // Timing runs first and returns the cosine scores from its final
    // iteration; verification then reuses those scores rather than
    // re-reading the parquet file. `cosine_loop` is deterministic, so
    // the last-iteration scores equal what a separate pre-timing
    // verification pass would produce — we just save one parquet read
    // per dataset. If the scores drift from the Vortex baseline,
    // `verify_and_report_scores` bails here (after the timing already
    // ran, which is acceptable because the handrolled loop is cheap and
    // we'd rather run it twice than skip correctness).
    let result =
        run_handrolled_baseline_timings(parquet_path, query, DEFAULT_THRESHOLD, iterations)?;

    let report = verify_and_report_scores(
        &bench_name,
        &result.last_scores,
        baseline_scores,
        VerificationKind::Lossless,
    )?;
    tracing::info!(
        "{} verification (Lossless): max_abs_diff={:.2e}, mean_abs_diff={:.2e}",
        bench_name,
        report.max_abs_diff,
        report.mean_abs_diff,
    );

    verification.push(CustomUnitMeasurement {
        name: format!("correctness-max-diff/{bench_name}"),
        format: Format::Parquet,
        unit: Cow::from("abs-diff"),
        value: report.max_abs_diff,
    });
    sizes.push(CustomUnitMeasurement {
        name: format!("{label} size/{dataset_name}"),
        format: Format::Parquet,
        unit: Cow::from("bytes"),
        value: parquet_bytes as f64,
    });
    timings.push(CompressionTimingMeasurement {
        name: format!("decompress time/{bench_name}"),
        format: Format::Parquet,
        time: result.timings.decompress,
    });
    timings.push(CompressionTimingMeasurement {
        name: format!("cosine-similarity time/{bench_name}"),
        format: Format::Parquet,
        time: result.timings.cosine,
    });
    timings.push(CompressionTimingMeasurement {
        name: format!("cosine-filter time/{bench_name}"),
        format: Format::Parquet,
        time: result.timings.filter,
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_array::builder::FixedSizeListBuilder;
    use arrow_array::builder::Float32Builder;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;
    use tempfile::NamedTempFile;

    use super::*;

    /// Build a minimal parquet file with an `emb: FixedSizeList<f32, dim>` column and
    /// verify the baseline pipeline produces the expected scores.
    fn write_tiny_fsl_parquet(dim: i32, rows: &[&[f32]]) -> Result<NamedTempFile> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        )]));

        let file = NamedTempFile::new()?;
        let mut writer =
            ArrowWriter::try_new(File::create(file.path())?, Arc::clone(&schema), None)?;

        let dim_usize = usize::try_from(dim).unwrap();
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for row in rows {
            assert_eq!(row.len(), dim_usize);
            for &v in row.iter() {
                builder.values().append_value(v);
            }
            builder.append(true);
        }
        let array = builder.finish();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)])?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(file)
    }

    #[test]
    fn handrolled_baseline_reads_fsl_column() {
        let file =
            write_tiny_fsl_parquet(3, &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]])
                .unwrap();

        let data = read_parquet_embedding_column(file.path()).unwrap();
        assert_eq!(data.dim, 3);
        assert_eq!(data.num_rows, 3);
        assert_eq!(data.elements.len(), 9);

        let query = [1.0f32, 0.0, 0.0];
        let scores = cosine_loop(&data.elements, data.num_rows, data.dim, &query);
        assert_eq!(scores, vec![1.0, 0.0, 1.0]);

        let mask = filter_loop(&scores, 0.5);
        assert_eq!(mask, vec![true, false, true]);
    }

    #[test]
    fn run_handrolled_baseline_timings_returns_last_iteration_scores() {
        // Verifies the new `last_scores` contract: the timing loop returns the
        // cosine scores from the final iteration, and those scores match what we'd
        // get from a one-shot `cosine_loop` on the same data. Callers of
        // `run_handrolled_baseline_timings` rely on this for verification (so they
        // don't need a second parquet read to compute ground-truth scores).
        let file =
            write_tiny_fsl_parquet(3, &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]])
                .unwrap();
        let query = [1.0f32, 0.0, 0.0];

        let result = run_handrolled_baseline_timings(file.path(), &query, 0.5, 3).unwrap();

        // Deterministic expected scores: rows 0 and 2 match the query exactly,
        // row 1 is orthogonal.
        assert_eq!(result.last_scores, vec![1.0, 0.0, 1.0]);
        assert!(result.timings.decompress > Duration::ZERO);
        assert!(result.timings.cosine > Duration::ZERO);
        assert!(result.timings.filter > Duration::ZERO);
    }

    #[test]
    fn run_handrolled_baseline_timings_rejects_zero_iterations() {
        let file =
            write_tiny_fsl_parquet(3, &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]])
                .unwrap();
        let query = [1.0f32, 0.0, 0.0];
        let err = match run_handrolled_baseline_timings(file.path(), &query, 0.5, 0) {
            Ok(_) => panic!("expected zero-iteration handrolled timings to fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("iterations >= 1"), "unexpected error: {err}");
    }
}
