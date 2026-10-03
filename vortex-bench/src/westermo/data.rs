// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of the Westermo CSVs into Prometheus layout.
//!
//! Each CSV holds one node_exporter target: a `timestamp` column in seconds since the start of
//! collection, then one column per metric. Each (file, metric column) pair becomes one series
//! with labels `__name__`, `instance` and `job`. Metric names have `-` replaced by `_` so they
//! are valid Prometheus names, and `instance` is the file stem, such as `system-7`.
//!
//! Rows are written series-major: series sorted by label set, samples within a series in file
//! order, which is ascending time. Missing scrapes in the source stay missing, so the timestamp
//! column keeps its real gaps.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use arrow_array::ArrayRef;
use arrow_array::DictionaryArray;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StructArray;
use arrow_array::UInt32Array;
use arrow_array::types::UInt32Type;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;

use super::schema::JOB;
use super::schema::SAMPLES_SCHEMA;
use super::schema::label_fields;

/// Commit of the upstream repository the CSVs are pinned to.
pub(super) const DATASET_COMMIT: &str = "47e0ccdc10c30e86327d6eb828ed48d164469fd4";

/// The data set has one CSV per test system, `system-1.csv` to `system-19.csv`.
pub(super) const NUM_SYSTEMS: usize = 19;

pub(super) fn csv_url(system: usize) -> String {
    format!(
        "https://raw.githubusercontent.com/westermo/test-system-performance-dataset/{DATASET_COMMIT}/data/system-{system}.csv"
    )
}

/// One parsed CSV: every metric of one node_exporter target.
struct SystemTable {
    instance: String,
    timestamps_ms: Vec<i64>,
    /// Prometheus metric name and its values, in CSV column order.
    metrics: Vec<(String, Vec<f64>)>,
}

fn metric_name(column: &str) -> String {
    column.trim().replace('-', "_")
}

fn parse_system_csv(instance: String, contents: &str) -> Result<SystemTable> {
    let mut lines = contents.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow!("{instance}: empty CSV"))?;
    let mut columns = header.split(',');
    let first = columns.next().map(str::trim);
    ensure!(
        first == Some("timestamp"),
        "{instance}: first column must be timestamp, found {first:?}"
    );
    let mut metrics: Vec<(String, Vec<f64>)> =
        columns.map(|c| (metric_name(c), Vec::new())).collect();
    let mut timestamps_ms = Vec::new();

    for (idx, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // Line 1 is the header.
        let line_no = idx + 2;
        let mut fields = line.split(',');
        let seconds: i64 = fields
            .next()
            .unwrap_or_default()
            .trim()
            .parse()
            .with_context(|| format!("{instance} line {line_no}: bad timestamp"))?;
        timestamps_ms.push(seconds * 1000);

        let mut parsed = 0;
        for (field, (name, values)) in fields.by_ref().zip(metrics.iter_mut()) {
            values.push(
                field
                    .trim()
                    .parse()
                    .with_context(|| format!("{instance} line {line_no}: bad value for {name}"))?,
            );
            parsed += 1;
        }
        ensure!(
            parsed == metrics.len() && fields.next().is_none(),
            "{instance} line {line_no}: expected {} values",
            metrics.len()
        );
    }

    Ok(SystemTable {
        instance,
        timestamps_ms,
        metrics,
    })
}

/// Build the record batch for one series.
fn series_batch(schema: &Arc<Schema>, system: &SystemTable, metric: usize) -> Result<RecordBatch> {
    let (name, values) = &system.metrics[metric];
    let n = values.len();

    // Every row of a series has the same labels, so each label is a one-entry dictionary.
    let label_arrays = [name.as_str(), system.instance.as_str(), JOB]
        .into_iter()
        .map(|value| -> Result<ArrayRef> {
            let keys = UInt32Array::from(vec![0u32; n]);
            let dictionary = Arc::new(StringArray::from(vec![value]));
            Ok(Arc::new(DictionaryArray::<UInt32Type>::try_new(
                keys, dictionary,
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    let labels = StructArray::try_new(label_fields(), label_arrays, None)?;

    let ts = Int64Array::from(system.timestamps_ms.clone());

    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(labels),
            Arc::new(ts),
            Arc::new(Float64Array::from(values.clone())),
        ],
    )?)
}

/// Parse every CSV and write all series to one Parquet file in Prometheus layout.
///
/// `inputs` pairs each `instance` label value with the CSV it comes from.
pub(super) fn convert_to_parquet(inputs: &[(String, PathBuf)], output_path: &Path) -> Result<()> {
    let systems: Vec<SystemTable> = std::thread::scope(|scope| {
        let handles: Vec<_> = inputs
            .iter()
            .map(|(instance, path)| {
                scope.spawn(move || -> Result<SystemTable> {
                    let contents = std::fs::read_to_string(path)
                        .with_context(|| format!("reading {}", path.display()))?;
                    parse_system_csv(instance.clone(), &contents)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(table) => table,
                Err(_) => Err(anyhow!("CSV parser thread panicked")),
            })
            .collect::<Result<Vec<_>>>()
    })?;

    // Sort series the way Prometheus orders label sets: by `__name__`, then `instance`.
    let mut series: Vec<(usize, usize)> = systems
        .iter()
        .enumerate()
        .flat_map(|(system, table)| (0..table.metrics.len()).map(move |metric| (system, metric)))
        .collect();
    series.sort_by(|&(sa, ma), &(sb, mb)| {
        let a = (&systems[sa].metrics[ma].0, &systems[sa].instance);
        let b = (&systems[sb].metrics[mb].0, &systems[sb].instance);
        a.cmp(&b)
    });

    let schema = Arc::new(SAMPLES_SCHEMA.clone());
    let file = std::fs::File::create(output_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;
    for (system, metric) in series {
        writer.write(&series_batch(&schema, &systems[system], metric)?)?;
    }
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;

    use super::*;

    const CSV: &str = "timestamp,load-1m,sys-mem-total\n0,0.22,16662700032\n30,0.26,16662700032\n90,0.5,16662700032\n";

    #[test]
    fn parses_header_and_values() -> Result<()> {
        let table = parse_system_csv("system-1".to_string(), CSV)?;
        assert_eq!(table.timestamps_ms, vec![0, 30_000, 90_000]);
        let names: Vec<&str> = table.metrics.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["load_1m", "sys_mem_total"]);
        assert_eq!(table.metrics[0].1, vec![0.22, 0.26, 0.5]);
        assert_eq!(table.metrics[1].1, vec![16_662_700_032.0; 3]);
        Ok(())
    }

    #[test]
    fn rejects_short_rows() {
        let csv = "timestamp,load-1m,load-5m\n0,0.1\n";
        assert!(parse_system_csv("system-1".to_string(), csv).is_err());
    }

    #[test]
    fn rejects_missing_timestamp_column() {
        let csv = "load-1m,load-5m\n0.1,0.2\n";
        assert!(parse_system_csv("system-1".to_string(), csv).is_err());
    }

    #[test]
    fn series_batch_matches_schema() -> Result<()> {
        let schema = Arc::new(SAMPLES_SCHEMA.clone());
        let table = parse_system_csv("system-7".to_string(), CSV)?;
        let batch = series_batch(&schema, &table, 1)?;

        assert_eq!(batch.schema(), schema);
        assert_eq!(batch.num_rows(), 3);

        let labels = batch.column(0).as_struct();
        for (column, expected) in [(0, "sys_mem_total"), (1, "system-7"), (2, JOB)] {
            let dict = labels.column(column).as_dictionary::<UInt32Type>();
            let values = dict.values().as_string::<i32>();
            assert_eq!(values.len(), 1);
            assert_eq!(values.value(0), expected);
        }

        let ts = batch.column(1).as_primitive::<Int64Type>();
        assert_eq!(ts.values().as_ref(), &[0, 30_000, 90_000]);
        Ok(())
    }
}
