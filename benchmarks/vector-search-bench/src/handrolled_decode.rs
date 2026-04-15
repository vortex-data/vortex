// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parquet → flat `Vec<f32>` decoder used by the handrolled scan.
//!
//! Kept in its own module so the compute loop in [`crate::handrolled`] is reading-friendly:
//! one file = one decode pass, returns a [`HandrolledShard`] the cosine loop can iterate
//! over without holding any Arrow references.

use std::fs::File;
use std::path::Path;

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

/// Flattened embedding column from one parquet shard. `elements.len() == num_rows * dim`.
pub struct HandrolledShard {
    pub elements: Vec<f32>,
    pub dim: usize,
    pub num_rows: usize,
}

/// Decode the `emb` column of a parquet file into a single flat `Vec<f32>`.
///
/// Accepts `List<f32>`, `LargeList<f32>`, or `FixedSizeList<f32, dim>`, with `Float64`
/// element values automatically narrowed to `f32` (the bench operates entirely in f32).
pub fn decode_parquet_emb(parquet_path: &Path) -> Result<HandrolledShard> {
    let file =
        File::open(parquet_path).with_context(|| format!("open {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    let (_, emb_field) = builder
        .schema()
        .column_with_name("emb")
        .context("parquet schema missing `emb` column")?;
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
    for batch in &batches {
        let column = batch
            .column_by_name("emb")
            .context("emb column missing from record batch")?;
        append_batch(column, &mut data, &mut inferred_dim, &mut num_rows)?;
    }

    let dim = inferred_dim.context("parquet file has zero rows — cannot infer dimension")?;
    Ok(HandrolledShard {
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
