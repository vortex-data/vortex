// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JSONBench queries over Parquet files whose `data` column holds either shredded Parquet Variant
//! values or JSON strings.
//!
//! Every row group is read and aggregated on its own task, with up to one task per core.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringViewArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet_variant::VariantPath;
use parquet_variant::VariantPathElement;
use parquet_variant_compute::GetOptions;
use serde::Deserialize;
use serde_json::Value;
use tokio::fs::File;
use vortex::utils::parallelism::get_available_parallelism;

use crate::agg::Partial;
use crate::agg::Query;
use crate::agg::Row;
use crate::agg::i64_at;
use crate::agg::str_at;

/// Rows per decoded record batch.
const BATCH_SIZE: usize = 8192;

/// How the `data` column of a Parquet file stores each event.
#[derive(Clone, Copy, Debug)]
pub enum Storage {
    /// Shredded Parquet Variant values.
    Variant,
    /// JSON strings.
    Json,
}

pub async fn run(query: Query, storage: Storage, files: Vec<PathBuf>) -> anyhow::Result<Partial> {
    // One task per row group.
    let mut row_groups = Vec::new();
    for path in files {
        let builder = ParquetRecordBatchStreamBuilder::new(File::open(&path).await?).await?;
        for row_group in 0..builder.metadata().num_row_groups() {
            row_groups.push((path.clone(), row_group));
        }
    }

    let parallelism = get_available_parallelism().unwrap_or(1);
    futures::stream::iter(row_groups)
        .map(|(path, row_group)| {
            tokio::spawn(async move { scan_row_group(query, storage, path, row_group).await })
        })
        .buffer_unordered(parallelism)
        .map(|joined| joined?)
        .try_fold(Partial::new(query), |mut total, partial| async move {
            total.merge(partial);
            Ok(total)
        })
        .await
}

async fn scan_row_group(
    query: Query,
    storage: Storage,
    path: PathBuf,
    row_group: usize,
) -> anyhow::Result<Partial> {
    let builder = ParquetRecordBatchStreamBuilder::new(File::open(&path).await?).await?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [0]);
    let mut batches = builder
        .with_row_groups(vec![row_group])
        .with_projection(mask)
        .with_batch_size(BATCH_SIZE)
        .build()?;

    let mut partial = Partial::new(query);
    while let Some(batch) = batches.try_next().await? {
        match storage {
            Storage::Variant => aggregate_variant(query, &batch, &mut partial)?,
            Storage::Json => aggregate_json(query, &batch, &mut partial)?,
        }
    }
    Ok(partial)
}

/// Extract `path` of each Variant row as `data_type` with arrow-rs, which reads shredded
/// `typed_value` columns directly and falls back to the binary `value` columns.
fn extract(data: &ArrayRef, path: &str, data_type: DataType) -> anyhow::Result<ArrayRef> {
    let path = VariantPath::from_iter(path.split('.').map(VariantPathElement::from));
    let options = GetOptions::new_with_path(path)
        .with_as_type(Some(Arc::new(Field::new("value", data_type, true))));
    Ok(parquet_variant_compute::variant_get(data, options)?)
}

fn aggregate_variant(
    query: Query,
    batch: &RecordBatch,
    partial: &mut Partial,
) -> anyhow::Result<()> {
    let data = batch.column(0);
    let str_path = |path: &str| -> anyhow::Result<StringViewArray> {
        Ok(extract(data, path, DataType::Utf8View)?
            .as_string_view()
            .clone())
    };
    let outputs = query.outputs();

    let kind = query
        .filters_commit_creates()
        .then(|| str_path("kind"))
        .transpose()?;
    let operation = query
        .filters_commit_creates()
        .then(|| str_path("commit.operation"))
        .transpose()?;
    let collection = (outputs.collection || query.collection_filter().is_some())
        .then(|| str_path("commit.collection"))
        .transpose()?;
    let did = outputs.did.then(|| str_path("did")).transpose()?;
    let time_us = outputs
        .time_us
        .then(|| -> anyhow::Result<Int64Array> {
            Ok(extract(data, "time_us", DataType::Int64)?
                .as_primitive::<arrow_array::types::Int64Type>()
                .clone())
        })
        .transpose()?;

    for idx in 0..batch.num_rows() {
        let collection_value = str_at(collection.as_ref(), idx);
        if !query.keeps(
            str_at(kind.as_ref(), idx),
            str_at(operation.as_ref(), idx),
            collection_value,
        ) {
            continue;
        }
        partial.add(Row {
            collection: collection_value,
            did: str_at(did.as_ref(), idx),
            time_us: i64_at(time_us.as_ref(), idx),
        });
    }
    Ok(())
}

/// The event fields JSONBench reads. Every other field is skipped while parsing.
#[derive(Deserialize)]
struct Event<'a> {
    #[serde(borrow, default)]
    did: Option<Cow<'a, str>>,
    #[serde(default)]
    time_us: Option<i64>,
    #[serde(borrow, default)]
    kind: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    commit: Option<Commit<'a>>,
}

#[derive(Deserialize)]
struct Commit<'a> {
    #[serde(borrow, default)]
    operation: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    collection: Option<Cow<'a, str>>,
}

/// An event whose fields do not all have the types [`Event`] expects, read leniently: a path whose
/// value is not of the expected type reads as null.
fn lenient_event(json: &str) -> anyhow::Result<Event<'static>> {
    let value: Value = serde_json::from_str(json)?;
    let str_at = |pointer: &str| {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .map(|s| Cow::Owned(s.to_string()))
    };
    Ok(Event {
        did: str_at("/did"),
        time_us: value.pointer("/time_us").and_then(Value::as_i64),
        kind: str_at("/kind"),
        commit: Some(Commit {
            operation: str_at("/commit/operation"),
            collection: str_at("/commit/collection"),
        }),
    })
}

fn aggregate_json(query: Query, batch: &RecordBatch, partial: &mut Partial) -> anyhow::Result<()> {
    let data = batch.column(0).as_string::<i32>();
    for idx in 0..data.len() {
        if data.is_null(idx) {
            continue;
        }
        let json = data.value(idx);
        let event = match serde_json::from_str::<Event<'_>>(json) {
            Ok(event) => event,
            Err(_) => lenient_event(json)?,
        };
        let operation = event.commit.as_ref().and_then(|c| c.operation.as_deref());
        let collection = event.commit.as_ref().and_then(|c| c.collection.as_deref());
        if !query.keeps(event.kind.as_deref(), operation, collection) {
            continue;
        }
        partial.add(Row {
            collection,
            did: event.did.as_deref(),
            time_us: event.time_us,
        });
    }
    Ok(())
}
