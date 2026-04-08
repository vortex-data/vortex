// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic data generation for the PolarSignals benchmark.
//!
//! Data is generated in sorted order by labels then `time_nanos`. Label values
//! come from a pre-enumerated set of `NUM_LABEL_SETS` distinct label
//! combinations that are sorted lexicographically (null < any value). Rows are
//! distributed evenly across these sorted sets so that each label achieves its
//! target cardinality while the overall row order remains sorted.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_array::builder::ListBuilder;
use arrow_array::builder::StructBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::schema::Int64DictBuilder;
use super::schema::LABELS;
use super::schema::STACKTRACES_SCHEMA;
use super::schema::StringDictBuilder;
use super::schema::UInt64DictBuilder;
use super::schema::label_fields;
use super::schema::lines_fields;
use super::schema::location_fields;

pub const BASE_TIMESTAMP_NS: i64 = 0;

/// Fixed step between consecutive rows (0.5ms = 500µs).
const TIMESTAMP_STEP_NS: i64 = 500_000;

const FUNCTION_NAME_POOL_SIZE: usize = 500;

const FRAME_TYPES: &[&str] = &["regular", "kernel", "inline", "native"];
const MAPPING_FILES: &[&str] = &[
    "/usr/bin/app",
    "/lib/x86_64-linux-gnu/libc.so.6",
    "/lib/x86_64-linux-gnu/libpthread.so.0",
    "[vdso]",
    "[kernel]",
];

/// Number of distinct label-set combinations to enumerate.
///
/// Must be >= the largest `num_distinct` across all labels so that every label
/// achieves its target cardinality.
const NUM_LABEL_SETS: usize = 1000;

/// Pre-computed label sets sorted by ascending cardinality.
///
/// `field_indices[pos]` gives the LABELS field index for position `pos` in each
/// set vec. Values within each set are stored in ascending-cardinality order
/// (ties broken alphabetically) so that plain lexicographic Vec sort produces
/// the correct row ordering (low-cardinality labels as outermost sort keys).
struct LabelSets {
    field_indices: Vec<usize>,
    sets: Vec<Vec<Option<usize>>>,
}

fn generate_sorted_label_sets() -> LabelSets {
    // Sort LABELS indices by ascending cardinality, alphabetical tiebreak.
    let mut field_indices: Vec<usize> = (0..LABELS.len()).collect();
    field_indices.sort_by(|&a, &b| {
        LABELS[a]
            .2
            .cmp(&LABELS[b].2)
            .then_with(|| LABELS[a].0.cmp(LABELS[b].0))
    });

    // Generate sets with values in cardinality order.
    let mut sets: Vec<Vec<Option<usize>>> = (0..NUM_LABEL_SETS)
        .map(|s| {
            field_indices
                .iter()
                .map(|&idx| {
                    let (_, fill, distinct) = LABELS[idx];
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let null_count = ((1.0 - fill) * NUM_LABEL_SETS as f64).round() as usize;
                    if s < null_count || distinct == 0 {
                        None
                    } else {
                        Some((s - null_count) % distinct)
                    }
                })
                .collect()
        })
        .collect();

    // Plain lexicographic sort: None < Some, low-cardinality fields outermost.
    sets.sort();
    LabelSets {
        field_indices,
        sets,
    }
}

/// Map a global row index to the label-set index it belongs to.
///
/// Rows are distributed evenly: each of the `NUM_LABEL_SETS` sets gets
/// `n_rows / S` rows, with the first `n_rows % S` sets receiving one extra.
fn set_for_row(row_idx: usize, n_rows: usize) -> usize {
    let base = n_rows / NUM_LABEL_SETS;
    let extra = n_rows % NUM_LABEL_SETS;
    let big = extra * (base + 1);
    if row_idx < big {
        row_idx / (base + 1)
    } else {
        extra + (row_idx - big) / base
    }
}

/// Format a label value string for a given field and value index.
fn format_label_value(field_index: usize, value_index: usize) -> String {
    format!("{}_{value_index}", LABELS[field_index].0)
}

/// Generate synthetic PolarSignals profiling data and write to Parquet.
pub fn generate_polarsignals_parquet(n_rows: usize, output_path: &Path) -> Result<()> {
    let schema = Arc::new(STACKTRACES_SCHEMA.clone());
    let label_sets = generate_sorted_label_sets();

    let function_names: Arc<[String]> = (0..FUNCTION_NAME_POOL_SIZE)
        .map(|i| format!("func_{i}"))
        .collect();
    let function_filenames: Arc<[String]> = (0..100)
        .map(|i| format!("/src/pkg{}/{}.go", i / 10, i % 10))
        .collect();
    let build_ids: Arc<[String]> = (0..20).map(|i| format!("build_{i:040x}")).collect();
    let label_sets = Arc::new(label_sets);

    let file = std::fs::File::create(output_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;

    let batch_size = 10_000;
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let batch_ranges: Vec<(usize, usize)> = (0..n_rows)
        .step_by(batch_size)
        .map(|start| (start, batch_size.min(n_rows - start)))
        .collect();

    batch_ranges.chunks(num_threads).try_for_each(|chunk| {
        chunk
            .iter()
            .map(|&(start, len)| {
                let schema = Arc::clone(&schema);
                let label_sets = Arc::clone(&label_sets);
                let function_names = Arc::clone(&function_names);
                let function_filenames = Arc::clone(&function_filenames);
                let build_ids = Arc::clone(&build_ids);
                std::thread::spawn(move || {
                    let mut rng = StdRng::seed_from_u64(42 + start as u64);
                    build_batch(
                        &schema,
                        len,
                        &mut rng,
                        start,
                        n_rows,
                        &label_sets,
                        &function_names,
                        &function_filenames,
                        &build_ids,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .try_for_each(|h| -> Result<()> {
                writer.write(&h.join().unwrap()?)?;
                Ok(())
            })
    })?;

    writer.close()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_batch(
    schema: &Arc<Schema>,
    n: usize,
    rng: &mut StdRng,
    batch_start_row: usize,
    n_rows: usize,
    label_sets: &LabelSets,
    function_names: &[String],
    function_filenames: &[String],
    build_ids: &[String],
) -> Result<RecordBatch> {
    let labels_array = build_labels(n, batch_start_row, n_rows, label_sets);
    let locations_array = build_locations(n, rng, function_names, function_filenames, build_ids);

    let mut value_builder = arrow_array::builder::Int64Builder::with_capacity(n);
    let mut producer_builder = StringDictBuilder::new();
    let mut sample_type_builder = StringDictBuilder::new();
    let mut sample_unit_builder = StringDictBuilder::new();
    let mut period_type_builder = StringDictBuilder::new();
    let mut period_unit_builder = StringDictBuilder::new();
    let mut temporality_builder = StringDictBuilder::new();
    let mut period_builder = arrow_array::builder::Int64Builder::with_capacity(n);
    let mut duration_builder = arrow_array::builder::Int64Builder::with_capacity(n);
    let mut time_nanos_builder =
        arrow_array::builder::TimestampNanosecondBuilder::with_capacity(n).with_timezone("UTC");

    for i in 0..n {
        value_builder.append_value(1);
        producer_builder.append_value("parca_agent");
        sample_type_builder.append_value("samples");
        sample_unit_builder.append_value("count");
        period_type_builder.append_value("cpu");
        period_unit_builder.append_value("nanoseconds");
        temporality_builder.append_value("delta");
        period_builder.append_value(52_631_578);
        duration_builder.append_value(1_000_000_000);

        let row_idx = batch_start_row + i;
        let ts = BASE_TIMESTAMP_NS + (row_idx as i64) * TIMESTAMP_STEP_NS;
        time_nanos_builder.append_value(ts);
    }

    let batch = RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(labels_array),
            Arc::new(locations_array),
            Arc::new(value_builder.finish()),
            Arc::new(producer_builder.finish()),
            Arc::new(sample_type_builder.finish()),
            Arc::new(sample_unit_builder.finish()),
            Arc::new(period_type_builder.finish()),
            Arc::new(period_unit_builder.finish()),
            Arc::new(temporality_builder.finish()),
            Arc::new(period_builder.finish()),
            Arc::new(duration_builder.finish()),
            Arc::new(time_nanos_builder.finish()),
        ],
    )?;

    Ok(batch)
}

fn build_labels(
    n: usize,
    batch_start_row: usize,
    n_rows: usize,
    label_sets: &LabelSets,
) -> arrow_array::StructArray {
    let fields = label_fields();
    let num_fields = LABELS.len();

    let field_builders: Vec<Box<dyn arrow_array::builder::ArrayBuilder>> = (0..num_fields)
        .map(|_| Box::new(StringDictBuilder::new()) as Box<dyn arrow_array::builder::ArrayBuilder>)
        .collect();

    let mut struct_builder = StructBuilder::new(fields.to_vec(), field_builders);

    for i in 0..n {
        let row_idx = batch_start_row + i;
        let set_idx = set_for_row(row_idx, n_rows);
        let set = &label_sets.sets[set_idx];

        // set stores values in cardinality order; map back to LABELS field order.
        for (pos, &val) in set.iter().enumerate() {
            let field_idx = label_sets.field_indices[pos];
            let fb = struct_builder
                .field_builder::<StringDictBuilder>(field_idx)
                .unwrap();
            match val {
                Some(val_idx) => {
                    fb.append_value(format_label_value(field_idx, val_idx));
                }
                None => {
                    fb.append_null();
                }
            }
        }
        struct_builder.append(true);
    }

    struct_builder.finish()
}

fn make_lines_struct_builder() -> StructBuilder {
    let fields = lines_fields().to_vec();
    let builders: Vec<Box<dyn arrow_array::builder::ArrayBuilder>> = vec![
        Box::new(Int64DictBuilder::new()),  // line
        Box::new(StringDictBuilder::new()), // function_name
        Box::new(StringDictBuilder::new()), // function_system_name
        Box::new(StringDictBuilder::new()), // function_filename
        Box::new(Int64DictBuilder::new()),  // function_start_line
    ];
    StructBuilder::new(fields, builders)
}

fn make_location_struct_builder() -> StructBuilder {
    let lines_list_builder = ListBuilder::new(make_lines_struct_builder()).with_field(Field::new(
        "item",
        DataType::Struct(lines_fields()),
        true,
    ));

    let fields = location_fields().to_vec();
    let builders: Vec<Box<dyn arrow_array::builder::ArrayBuilder>> = vec![
        Box::new(UInt64DictBuilder::new()), // address
        Box::new(StringDictBuilder::new()), // frame_type
        Box::new(StringDictBuilder::new()), // mapping_file
        Box::new(StringDictBuilder::new()), // mapping_build_id
        Box::new(lines_list_builder),       // lines
    ];
    StructBuilder::new(fields, builders)
}

fn build_locations(
    n: usize,
    rng: &mut StdRng,
    function_names: &[String],
    function_filenames: &[String],
    build_ids: &[String],
) -> arrow_array::ListArray {
    let location_builder = make_location_struct_builder();
    let mut list_builder = ListBuilder::new(location_builder).with_field(Field::new(
        "item",
        DataType::Struct(location_fields()),
        true,
    ));

    for _ in 0..n {
        let num_locations = rng.random_range(5i32..50);
        let loc_struct = list_builder.values();

        for _ in 0..num_locations {
            // address
            loc_struct
                .field_builder::<UInt64DictBuilder>(0)
                .unwrap()
                .append_value(rng.random::<u64>());
            // frame_type
            loc_struct
                .field_builder::<StringDictBuilder>(1)
                .unwrap()
                .append_value(FRAME_TYPES[rng.random_range(0..FRAME_TYPES.len())]);
            // mapping_file
            loc_struct
                .field_builder::<StringDictBuilder>(2)
                .unwrap()
                .append_value(MAPPING_FILES[rng.random_range(0..MAPPING_FILES.len())]);
            // mapping_build_id
            loc_struct
                .field_builder::<StringDictBuilder>(3)
                .unwrap()
                .append_value(&build_ids[rng.random_range(0..build_ids.len())]);

            // lines
            let lines_list = loc_struct
                .field_builder::<ListBuilder<StructBuilder>>(4)
                .unwrap();
            let num_lines = rng.random_range(1i32..5);
            let lines_struct = lines_list.values();
            for _ in 0..num_lines {
                // line
                lines_struct
                    .field_builder::<Int64DictBuilder>(0)
                    .unwrap()
                    .append_value(rng.random_range(1..2000));
                // function_name
                lines_struct
                    .field_builder::<StringDictBuilder>(1)
                    .unwrap()
                    .append_value(&function_names[rng.random_range(0..function_names.len())]);
                // function_system_name
                lines_struct
                    .field_builder::<StringDictBuilder>(2)
                    .unwrap()
                    .append_value(&function_names[rng.random_range(0..function_names.len())]);
                // function_filename
                lines_struct
                    .field_builder::<StringDictBuilder>(3)
                    .unwrap()
                    .append_value(
                        &function_filenames[rng.random_range(0..function_filenames.len())],
                    );
                // function_start_line
                lines_struct
                    .field_builder::<Int64DictBuilder>(4)
                    .unwrap()
                    .append_value(rng.random_range(1..2000));
                lines_struct.append(true);
            }
            lines_list.append(true);

            loc_struct.append(true);
        }

        list_builder.append(true);
    }

    list_builder.finish()
}

#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_label_cardinality() {
        let ls = generate_sorted_label_sets();

        // field_indices maps cardinality-order position → LABELS field index.
        for (pos, &field_idx) in ls.field_indices.iter().enumerate() {
            let (name, fill, distinct) = LABELS[field_idx];
            let unique_values: HashSet<Option<usize>> = ls.sets.iter().map(|s| s[pos]).collect();

            if distinct == 0 {
                assert_eq!(
                    unique_values.len(),
                    1,
                    "label {name} should be all null but has {} distinct values",
                    unique_values.len()
                );
                assert!(unique_values.contains(&None));
            } else {
                let non_null_count = unique_values.iter().filter(|v| v.is_some()).count();
                assert_eq!(
                    non_null_count, distinct,
                    "label {name} (fill={fill}) should have {distinct} distinct non-null values, got {non_null_count}"
                );
            }
        }
    }

    #[test]
    fn test_set_for_row_covers_all_sets() {
        let n_rows: usize = 5000;
        let mut seen = HashSet::new();
        for i in 0..n_rows {
            seen.insert(set_for_row(i, n_rows));
        }
        assert_eq!(
            seen.len(),
            NUM_LABEL_SETS,
            "not all label sets are used: got {} of {NUM_LABEL_SETS}",
            seen.len()
        );
    }
}
