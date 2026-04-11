// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parquet-Arrow hand-rolled cosine similarity baseline.
//!
//! This module provides the "what you'd do without Vortex" floor for the vector-search
//! benchmark. It reads the canonical parquet file for a dataset via `parquet::arrow`,
//! decodes the `emb` column to an Arrow `FixedSizeListArray<f32>`, and then runs a
//! straightforward Rust cosine-similarity loop — no scalar functions, no lazy expressions,
//! no index.
//!
//! The four measurements produced mirror those of the Vortex variants so dashboards can
//! put the parquet bar right next to the vortex bars:
//!
//! 1. Compressed size — the on-disk parquet file in bytes.
//! 2. Full-scan decode time — parquet → arrow record batches → concatenated
//!    `FixedSizeListArray<f32>`.
//! 3. Cosine-similarity execute time — hand-rolled loop producing a `Vec<f32>` of scores.
//! 4. Filter execute time — the same loop materializing into a `Vec<bool>` where
//!    `score > threshold`.
//!
//! This module does *not* include the parquet decode time in the cosine/filter wall
//! times. Decoding is treated as its own measurement. This matches how the Vortex variants
//! separate decode from compute.

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
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::VariantTimings;

/// Read the entire `emb` column of a parquet file into a single flat `Vec<f32>`, along
/// with the dimension and row count.
pub fn read_parquet_embedding_column(parquet_path: &Path) -> Result<ParquetBaselineData> {
    let file = File::open(parquet_path)
        .with_context(|| format!("open parquet file {}", parquet_path.display()))?;
    let file_size = file.metadata()?.len();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    // Locate the `emb` column and sanity-check its type.
    let (emb_idx, emb_field) = builder
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
    if !matches!(element_dtype, DataType::Float32) {
        bail!(
            "emb column element type must be Float32, got {:?}",
            element_dtype
        );
    }
    let _ = emb_idx;

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
    Ok(ParquetBaselineData {
        elements: data,
        dim,
        num_rows,
        file_size,
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
        let values = fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("FSL emb column must have Float32 values")?;
        data.extend_from_slice(values.values());
        *num_rows += fsl.len();
        return Ok(());
    }

    if let Some(list) = column.as_any().downcast_ref::<ListArray>() {
        let values: &Float32Array = list
            .values()
            .as_primitive_opt::<arrow_array::types::Float32Type>()
            .context("List emb column must have Float32 values")?;
        let offsets = list.value_offsets();
        for i in 0..list.len() {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let row_len = end - start;
            maybe_set_dim(inferred_dim, row_len)?;
            data.extend_from_slice(&values.values()[start..end]);
            *num_rows += 1;
        }
        return Ok(());
    }

    bail!(
        "emb column has unsupported arrow type {:?}",
        column.data_type()
    );
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

/// The flattened representation of a parquet file's embedding column, suitable for a
/// hand-rolled distance loop.
pub struct ParquetBaselineData {
    /// All rows concatenated: `elements.len() == num_rows * dim`.
    pub elements: Vec<f32>,
    /// Vector dimensionality.
    pub dim: usize,
    /// Number of rows.
    pub num_rows: usize,
    /// On-disk size of the parquet file in bytes.
    pub file_size: u64,
}

/// Run the decode / cosine / filter baseline microbenchmarks and return the best-of-N
/// wall times. Decoding is re-parquet-reading from disk on each iteration (matches how
/// the Vortex variants also re-execute from scratch each iteration).
pub fn run_parquet_baseline_timings(
    parquet_path: &Path,
    query: &[f32],
    threshold: f32,
    iterations: usize,
) -> Result<VariantTimings> {
    let mut decompress = Duration::MAX;
    let mut cosine = Duration::MAX;
    let mut filter = Duration::MAX;

    for _ in 0..iterations {
        let start = Instant::now();
        let data = read_parquet_embedding_column(parquet_path)?;
        decompress = decompress.min(start.elapsed());

        let start = Instant::now();
        let scores = cosine_loop(&data.elements, data.num_rows, data.dim, query);
        cosine = cosine.min(start.elapsed());
        debug_assert_eq!(scores.len(), data.num_rows);

        let start = Instant::now();
        let matches = filter_loop(&scores, threshold);
        filter = filter.min(start.elapsed());
        debug_assert_eq!(matches.len(), data.num_rows);
    }

    Ok(VariantTimings {
        decompress,
        cosine,
        filter,
    })
}

/// Compute cosine similarity for every row against `query`. The query is assumed to match
/// the database vectors' dimension. Returns one f32 score per row; scores for zero-norm
/// rows or a zero-norm query are 0.0 by convention.
pub fn cosine_loop(elements: &[f32], num_rows: usize, dim: usize, query: &[f32]) -> Vec<f32> {
    assert_eq!(query.len(), dim);
    assert_eq!(elements.len(), num_rows * dim);

    let query_norm = query.iter().map(|&q| q * q).sum::<f32>().sqrt();
    let mut out = Vec::with_capacity(num_rows);
    if query_norm == 0.0 {
        out.resize(num_rows, 0.0);
        return out;
    }

    for row in 0..num_rows {
        let base = row * dim;
        let slice = &elements[base..base + dim];
        let mut dot = 0.0f32;
        let mut sq = 0.0f32;
        for i in 0..dim {
            dot += slice[i] * query[i];
            sq += slice[i] * slice[i];
        }
        let norm = sq.sqrt();
        if norm == 0.0 {
            out.push(0.0);
        } else {
            out.push(dot / (norm * query_norm));
        }
    }
    out
}

/// Build the `cosine > threshold` boolean mask.
pub fn filter_loop(scores: &[f32], threshold: f32) -> Vec<bool> {
    scores.iter().map(|&s| s > threshold).collect()
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
    fn parquet_baseline_reads_fsl_column() {
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
}
