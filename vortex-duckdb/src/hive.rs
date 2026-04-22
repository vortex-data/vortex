// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hive-style partition column support for `read_vortex` table functions.
//!
//! This module provides the core data structures and algorithms for:
//!
//! * Extracting `key=value` pairs from file path segments.
//! * Building per-column partition data from a list of file paths.
//! * Evaluating DuckDB table filters against pre-known string partition values
//!   (used for early file pruning in `init_global`).
//! * Interleaving partition constant arrays into a `StructArray` at the exact
//!   column positions DuckDB's projection map requires.

use std::sync::Arc;

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructDataParts;
use vortex::array::validity::Validity;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::scalar::Scalar;

use crate::duckdb::ExtractedValue;
use crate::duckdb::TableFilterClass;
use crate::duckdb::TableFilterRef;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A hive-style partition column produced during bind.
///
/// Each column has a name (the partition key) and one string value per file,
/// in the same order as the files resolved from the glob pattern.
#[derive(Debug, Clone)]
pub(crate) struct HivePartitionColumn {
    pub name: String,
    /// One value per file, in file-discovery order.
    pub values: Vec<String>,
}

// ---------------------------------------------------------------------------
// Path parsing
// ---------------------------------------------------------------------------

/// Extracts hive-style `key=value` pairs from path segments.
///
/// For example, given `/data/year=2023/month=01/file.vortex`,
/// returns `[("year", "2023"), ("month", "01")]`.
pub(crate) fn extract_hive_partitions(path: &str) -> Vec<(String, String)> {
    let mut partitions = Vec::new();
    for segment in path.split('/') {
        if let Some((key, value)) = segment.split_once('=') {
            if !key.is_empty() && !value.is_empty() {
                partitions.push((key.to_string(), value.to_string()));
            }
        }
    }
    partitions
}

/// Builds per-column hive partition data from a list of file paths.
///
/// All files must have the same partition keys in the same order.
pub(crate) fn extract_hive_partition_columns(
    file_paths: &[String],
) -> VortexResult<Vec<HivePartitionColumn>> {
    if file_paths.is_empty() {
        return Ok(vec![]);
    }

    let all_partitions: Vec<Vec<(String, String)>> = file_paths
        .iter()
        .map(|p| extract_hive_partitions(p))
        .collect();

    let first = &all_partitions[0];
    for (idx, partitions) in all_partitions.iter().enumerate() {
        if partitions.len() != first.len() {
            vortex_bail!(
                "Hive partition mismatch: file {} has {} partition keys but expected {}",
                file_paths[idx],
                partitions.len(),
                first.len()
            );
        }
        for (i, (key, _)) in partitions.iter().enumerate() {
            if key != &first[i].0 {
                vortex_bail!(
                    "Hive partition key mismatch: file {} has key '{}' but expected '{}'",
                    file_paths[idx],
                    key,
                    first[i].0
                );
            }
        }
    }

    let mut columns: Vec<HivePartitionColumn> = first
        .iter()
        .map(|(key, _)| HivePartitionColumn {
            name: key.clone(),
            values: Vec::with_capacity(file_paths.len()),
        })
        .collect();

    for partitions in &all_partitions {
        for (i, (_, value)) in partitions.iter().enumerate() {
            columns[i].values.push(value.clone());
        }
    }

    Ok(columns)
}

// ---------------------------------------------------------------------------
// Partition filter evaluation (for early file pruning)
// ---------------------------------------------------------------------------

/// Returns `true` when `value` satisfies the DuckDB table filter.
///
/// Partition column values are always non-null UTF-8 strings. Unknown filter types
/// conservatively return `true` (do not prune the file).
pub(crate) fn partition_value_matches_filter(value: &str, filter: &TableFilterRef) -> bool {
    use crate::cpp::DUCKDB_VX_EXPR_TYPE;

    match filter.as_class() {
        TableFilterClass::ConstantComparison(const_) => {
            let filter_str = match const_.value.extract() {
                ExtractedValue::Varchar(v) => v.to_string(),
                _ => return true, // non-varchar comparison: don't prune
            };
            let f = filter_str.as_str();
            match const_.operator {
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => value == f,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL => value != f,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => value > f,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => value < f,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => {
                    value >= f
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => {
                    value <= f
                }
                _ => true,
            }
        }
        TableFilterClass::ConjunctionAnd(conj) => conj
            .children()
            .all(|child| partition_value_matches_filter(value, child)),
        TableFilterClass::ConjunctionOr(disj) => disj
            .children()
            .any(|child| partition_value_matches_filter(value, child)),
        TableFilterClass::InFilter(values) => values.iter().any(|v| match v.extract() {
            ExtractedValue::Varchar(fv) => value == fv.to_string().as_str(),
            _ => false,
        }),
        TableFilterClass::IsNull => false, // partition values are never null
        TableFilterClass::IsNotNull => true,
        _ => true, // unknown filter type: conservatively pass
    }
}

// ---------------------------------------------------------------------------
// Output column interleaving
// ---------------------------------------------------------------------------

/// Rebuilds `array_result` with hive partition columns interleaved at the positions DuckDB
/// expects, as recorded in `output_schema_cols`.
///
/// DuckDB's projection may reorder columns (e.g. filter column first), so we cannot simply
/// append partition columns at the end of the `StructArray`. Instead we iterate
/// `output_schema_cols` — the schema column index per output `DataChunk` vector — and place
/// each file column (taken in order from `array_result`) or partition constant at the right
/// slot.
///
/// # Parameters
///
/// * `array_result` — `StructArray` produced by the file reader (file columns only).
/// * `file_column_count` — number of columns that come from the file.
/// * `file_column_names` — name of each file column, indexed by schema column index.
/// * `hive_partition_columns` — partition metadata (name + per-file values).
/// * `file_index` — index of the current file in `hive_partition_columns.values`.
/// * `output_schema_cols` — schema column index for each `DataChunk` output vector position,
///   as computed from `projection_ids` / `column_ids` during `init_global`.
pub(crate) fn interleave_partition_columns(
    array_result: StructArray,
    file_column_count: usize,
    file_column_names: &[String],
    hive_partition_columns: &[HivePartitionColumn],
    file_index: usize,
    output_schema_cols: &[usize],
) -> VortexResult<StructArray> {
    let row_count = array_result.len();

    if output_schema_cols.is_empty() {
        // Zero-column projection: just append partition cols at the end (won't affect output).
        let mut result = array_result;
        for partition_col in hive_partition_columns {
            let value = &partition_col.values[file_index];
            let constant_array = ConstantArray::new(Scalar::from(value.as_str()), row_count);
            result =
                result.with_column(partition_col.name.as_str(), constant_array.into_array())?;
        }
        return Ok(result);
    }

    // Extract file field arrays in their current order (matching the file-column positions
    // in output_schema_cols, which is also the order extract_projection_expr requested them).
    let StructDataParts { fields: file_arrays, .. } = array_result.into_data_parts();

    let mut file_col_cursor = 0usize;
    let mut out_names: Vec<Arc<str>> = Vec::with_capacity(output_schema_cols.len());
    let mut out_arrays: Vec<ArrayRef> = Vec::with_capacity(output_schema_cols.len());

    for &schema_col in output_schema_cols {
        if schema_col < file_column_count {
            let name = Arc::from(file_column_names[schema_col].as_str());
            let arr = file_arrays[file_col_cursor].clone();
            out_names.push(name);
            out_arrays.push(arr);
            file_col_cursor += 1;
        } else {
            let part_idx = schema_col - file_column_count;
            let part_col = &hive_partition_columns[part_idx];
            let value = &part_col.values[file_index];
            let constant_array = ConstantArray::new(Scalar::from(value.as_str()), row_count);
            out_names.push(Arc::from(part_col.name.as_str()));
            out_arrays.push(constant_array.into_array());
        }
    }

    let names: FieldNames = out_names.into_iter().collect();
    StructArray::try_new(names, out_arrays, row_count, Validity::NonNullable.into())
}
