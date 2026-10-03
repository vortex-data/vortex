// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of the LO2 sample run into Prometheus layout.
//!
//! The archive holds one run of the light-oauth2 microservice under Locust load: 54 test cases,
//! back to back, over about 90 minutes. Each test case directory has one JSON file per metric,
//! which is the result array of a Prometheus `query_range` call at a one second step:
//! `[{"metric": {label: value}, "values": [[unix_seconds, "value"], ...]}, ...]`.
//!
//! The same series appears in every test case, so samples are merged per label set. Rows are
//! written series-major, with series in Prometheus label-set order and samples in time order.
//! Every label key seen in the data becomes a nullable field of the `labels` struct, so a series
//! has nulls for the labels it does not carry.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_array::DictionaryArray;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StructArray;
use arrow_array::UInt32Array;
use arrow_array::types::UInt32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use serde::Deserialize;

/// The sample run from LO2v2, Zenodo record 18937117.
pub(super) const ARCHIVE_URL: &str =
    "https://zenodo.org/api/records/18937117/files/light-oauth2-metrics.zip/content";

/// A series' labels as sorted `(name, value)` pairs. Comparing these vectors gives Prometheus'
/// label-set order.
type Labels = Vec<(String, String)>;

/// Samples per series: millisecond timestamp and value.
type SeriesMap = BTreeMap<Labels, Vec<(i64, f64)>>;

/// One element of a Prometheus `query_range` result.
#[derive(Deserialize)]
struct RawSeries {
    metric: BTreeMap<String, String>,
    values: Vec<(f64, String)>,
}

/// Merge the series of one metric file into `series`.
fn merge_metric_file(json: &str, series: &mut SeriesMap) -> Result<()> {
    let raw: Vec<RawSeries> = serde_json::from_str(json)?;
    for RawSeries { metric, values } in raw {
        let samples = series.entry(metric.into_iter().collect()).or_default();
        for (seconds, value) in values {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Prometheus timestamps are seconds with millisecond precision"
            )]
            let ts = (seconds * 1000.0).round() as i64;
            let value: f64 = value
                .parse()
                .with_context(|| format!("bad sample value {value:?}"))?;
            samples.push((ts, value));
        }
    }
    Ok(())
}

/// Read every metric file in the archive into one map of series.
fn read_archive(archive_path: &Path) -> Result<SeriesMap> {
    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("opening {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut series = SeriesMap::new();
    let mut json = String::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_string();
        if !(name.contains("/metrics/metric_") && name.ends_with(".json")) {
            continue;
        }
        json.clear();
        entry.read_to_string(&mut json)?;
        merge_metric_file(&json, &mut series).with_context(|| format!("parsing {name}"))?;
    }
    for samples in series.values_mut() {
        samples.sort_by_key(|&(ts, _)| ts);
        samples.dedup_by_key(|&mut (ts, _)| ts);
    }
    Ok(series)
}

fn label_fields(label_names: &BTreeSet<&str>) -> Fields {
    label_names
        .iter()
        .map(|name| {
            Field::new(
                *name,
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            )
        })
        .collect()
}

fn samples_schema(label_names: &BTreeSet<&str>) -> Schema {
    Schema::new(vec![
        Field::new("labels", DataType::Struct(label_fields(label_names)), false),
        Field::new("ts", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// Build the record batch for one series.
fn series_batch(
    schema: &Arc<Schema>,
    label_names: &BTreeSet<&str>,
    labels: &Labels,
    samples: &[(i64, f64)],
) -> Result<RecordBatch> {
    let n = samples.len();
    let present: BTreeMap<&str, &str> = labels
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();

    // Every row of a series has the same labels, so each label is a one-entry dictionary, or
    // all nulls when the series does not carry that label.
    let label_arrays = label_names
        .iter()
        .map(|name| -> Result<ArrayRef> {
            let (keys, dictionary) = match present.get(name) {
                Some(value) => (
                    UInt32Array::from(vec![0u32; n]),
                    StringArray::from(vec![*value]),
                ),
                None => (
                    UInt32Array::new_null(n),
                    StringArray::from(Vec::<&str>::new()),
                ),
            };
            Ok(Arc::new(DictionaryArray::<UInt32Type>::try_new(
                keys,
                Arc::new(dictionary),
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    let labels = StructArray::try_new(label_fields(label_names), label_arrays, None)?;

    let ts = Int64Array::from_iter_values(samples.iter().map(|&(ts, _)| ts));
    let values = Float64Array::from_iter_values(samples.iter().map(|&(_, value)| value));

    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![Arc::new(labels), Arc::new(ts), Arc::new(values)],
    )?)
}

/// Convert the LO2 archive to one Parquet file in Prometheus layout.
pub(super) fn convert_to_parquet(archive_path: &Path, output_path: &Path) -> Result<()> {
    let series = read_archive(archive_path)?;
    let label_names: BTreeSet<&str> = series
        .keys()
        .flatten()
        .map(|(name, _)| name.as_str())
        .collect();
    let schema = Arc::new(samples_schema(&label_names));

    let file = std::fs::File::create(output_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;
    for (labels, samples) in &series {
        writer.write(&series_batch(&schema, &label_names, labels, samples)?)?;
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

    const CPU: &str = r#"[
        {"metric": {"__name__": "node_cpu_seconds_total", "cpu": "1", "mode": "idle"},
         "values": [[1739746339, "10.5"], [1739746340, "11.5"]]},
        {"metric": {"__name__": "node_cpu_seconds_total", "cpu": "0", "mode": "idle"},
         "values": [[1739746339, "20"]]}
    ]"#;
    const LATER: &str = r#"[
        {"metric": {"__name__": "node_cpu_seconds_total", "cpu": "0", "mode": "idle"},
         "values": [[1739746400, "+Inf"]]}
    ]"#;

    #[test]
    fn merges_series_across_files() -> Result<()> {
        let mut series = SeriesMap::new();
        merge_metric_file(CPU, &mut series)?;
        merge_metric_file(LATER, &mut series)?;

        let keys: Vec<&str> = series.keys().map(|labels| labels[1].1.as_str()).collect();
        assert_eq!(keys, vec!["0", "1"], "series sorted by label set");

        let cpu0 = series.values().next().expect("two series");
        assert_eq!(cpu0.len(), 2);
        assert_eq!(cpu0[0], (1_739_746_339_000, 20.0));
        assert_eq!(cpu0[1].0, 1_739_746_400_000);
        assert!(cpu0[1].1.is_infinite());
        Ok(())
    }

    #[test]
    fn rejects_bad_values() {
        let json = r#"[{"metric": {"__name__": "x"}, "values": [[1, "abc"]]}]"#;
        assert!(merge_metric_file(json, &mut SeriesMap::new()).is_err());
    }

    #[test]
    fn series_batch_fills_missing_labels_with_nulls() -> Result<()> {
        let label_names: BTreeSet<&str> = ["__name__", "cpu", "device"].into_iter().collect();
        let schema = Arc::new(samples_schema(&label_names));
        let labels: Labels = vec![
            ("__name__".to_string(), "node_load1".to_string()),
            ("cpu".to_string(), "3".to_string()),
        ];
        let batch = series_batch(&schema, &label_names, &labels, &[(1000, 0.5), (2000, 0.75)])?;

        assert_eq!(batch.schema(), schema);
        let labels = batch.column(0).as_struct();
        let name = labels.column(0).as_dictionary::<UInt32Type>();
        assert_eq!(name.values().as_string::<i32>().value(0), "node_load1");
        assert_eq!(labels.column(2).null_count(), 2, "device is absent");

        let ts = batch.column(1).as_primitive::<Int64Type>();
        assert_eq!(ts.values().as_ref(), &[1000, 2000]);
        Ok(())
    }
}
