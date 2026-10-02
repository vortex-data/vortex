// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prometheus-layout schema for the Westermo benchmark.

use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;

/// Label names in Prometheus order: sorted by name, so `__name__` comes first.
pub(super) const LABEL_NAMES: &[&str] = &["__name__", "instance", "job"];

/// The `job` label value. Every series in the data set was scraped from node_exporter.
pub(super) const JOB: &str = "node";

pub(super) fn label_fields() -> Fields {
    LABEL_NAMES
        .iter()
        .map(|name| {
            Field::new(
                *name,
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                false,
            )
        })
        .collect()
}

/// One row per sample: the series labels, a timestamp and a float64 value.
///
/// `ts` is int64 milliseconds, the type Prometheus itself uses for sample timestamps. It also
/// keeps the queries portable across DataFusion and DuckDB, which spell timestamp literals and
/// time bucketing differently.
pub static SAMPLES_SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    Schema::new(vec![
        Field::new("labels", DataType::Struct(label_fields()), false),
        Field::new("ts", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ])
});
