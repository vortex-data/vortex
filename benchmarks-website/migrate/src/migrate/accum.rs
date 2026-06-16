// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-fact-table row accumulators + their `RecordBatch` builders.
//!
//! Each `*Accum` collects classified records during the streaming JSONL
//! pass and then materialises one Arrow `RecordBatch` per fact table at
//! flush time. Three of the four use parallel column vectors with a
//! `seen` map keyed by `measurement_id`; `CompressionSizeAccum` is a
//! `HashMap<i64, CompressionSize>` because it has two collision semantics
//! (replace from `data.json.gz`, sum from `file-sizes-*.json.gz`).

use std::sync::Arc;

use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use vortex_bench_server::records::CompressionSize;
use vortex_bench_server::records::CompressionTime;
use vortex_bench_server::records::QueryMeasurement;
use vortex_bench_server::records::RandomAccessTime;
use vortex_utils::aliases::hash_map::HashMap;

use super::MigrationSummary;

/// `query_measurements` accumulator. Parallel column vectors plus a
/// `measurement_id`-keyed seen map; first-write wins on collision.
#[derive(Default)]
pub(super) struct QueryAccum {
    pub(super) measurement_id: Vec<i64>,
    pub(super) commit_sha: Vec<String>,
    pub(super) dataset: Vec<String>,
    pub(super) dataset_variant: Vec<Option<String>>,
    pub(super) scale_factor: Vec<Option<String>>,
    pub(super) query_idx: Vec<i32>,
    pub(super) storage: Vec<String>,
    pub(super) engine: Vec<String>,
    pub(super) format: Vec<String>,
    pub(super) value_ns: Vec<i64>,
    pub(super) all_runtimes_ns: Vec<Vec<i64>>,
    pub(super) peak_physical: Vec<Option<i64>>,
    pub(super) peak_virtual: Vec<Option<i64>>,
    pub(super) physical_delta: Vec<Option<i64>>,
    pub(super) virtual_delta: Vec<Option<i64>>,
    pub(super) env_triple: Vec<Option<String>>,
    /// `mid` -> index in the parallel column vecs. Lets us look up the
    /// kept row's `value_ns` on collision so we can flag conflicts.
    pub(super) seen: HashMap<i64, usize>,
}

impl QueryAccum {
    pub(super) fn push(&mut self, mid: i64, r: QueryMeasurement, summary: &mut MigrationSummary) {
        if let Some(&idx) = self.seen.get(&mid) {
            summary.deduped += 1;
            if self.value_ns[idx] != r.value_ns {
                summary.deduped_with_conflict += 1;
            }
            return;
        }
        let idx = self.measurement_id.len();
        self.seen.insert(mid, idx);
        self.measurement_id.push(mid);
        self.commit_sha.push(r.commit_sha);
        self.dataset.push(r.dataset);
        self.dataset_variant.push(r.dataset_variant);
        self.scale_factor.push(r.scale_factor);
        self.query_idx.push(r.query_idx);
        self.storage.push(r.storage);
        self.engine.push(r.engine);
        self.format.push(r.format);
        self.value_ns.push(r.value_ns);
        self.all_runtimes_ns.push(r.all_runtimes_ns);
        self.peak_physical.push(r.peak_physical);
        self.peak_virtual.push(r.peak_virtual);
        self.physical_delta.push(r.physical_delta);
        self.virtual_delta.push(r.virtual_delta);
        self.env_triple.push(r.env_triple);
    }
}

/// `compression_times` accumulator. Same shape as [`QueryAccum`] minus the
/// query-only columns.
#[derive(Default)]
pub(super) struct CompressionTimeAccum {
    pub(super) measurement_id: Vec<i64>,
    pub(super) commit_sha: Vec<String>,
    pub(super) dataset: Vec<String>,
    pub(super) dataset_variant: Vec<Option<String>>,
    pub(super) format: Vec<String>,
    pub(super) op: Vec<String>,
    pub(super) value_ns: Vec<i64>,
    pub(super) all_runtimes_ns: Vec<Vec<i64>>,
    pub(super) env_triple: Vec<Option<String>>,
    pub(super) seen: HashMap<i64, usize>,
}

impl CompressionTimeAccum {
    pub(super) fn push(&mut self, mid: i64, r: CompressionTime, summary: &mut MigrationSummary) {
        if let Some(&idx) = self.seen.get(&mid) {
            summary.deduped += 1;
            if self.value_ns[idx] != r.value_ns {
                summary.deduped_with_conflict += 1;
            }
            return;
        }
        let idx = self.measurement_id.len();
        self.seen.insert(mid, idx);
        self.measurement_id.push(mid);
        self.commit_sha.push(r.commit_sha);
        self.dataset.push(r.dataset);
        self.dataset_variant.push(r.dataset_variant);
        self.format.push(r.format);
        self.op.push(r.op);
        self.value_ns.push(r.value_ns);
        self.all_runtimes_ns.push(r.all_runtimes_ns);
        self.env_triple.push(r.env_triple);
    }
}

/// `random_access_times` accumulator. Smallest of the three parallel-vec
/// accumulators.
#[derive(Default)]
pub(super) struct RandomAccessAccum {
    pub(super) measurement_id: Vec<i64>,
    pub(super) commit_sha: Vec<String>,
    pub(super) dataset: Vec<String>,
    pub(super) format: Vec<String>,
    pub(super) value_ns: Vec<i64>,
    pub(super) all_runtimes_ns: Vec<Vec<i64>>,
    pub(super) env_triple: Vec<Option<String>>,
    pub(super) seen: HashMap<i64, usize>,
}

impl RandomAccessAccum {
    pub(super) fn push(&mut self, mid: i64, r: RandomAccessTime, summary: &mut MigrationSummary) {
        if let Some(&idx) = self.seen.get(&mid) {
            summary.deduped += 1;
            if self.value_ns[idx] != r.value_ns {
                summary.deduped_with_conflict += 1;
            }
            return;
        }
        let idx = self.measurement_id.len();
        self.seen.insert(mid, idx);
        self.measurement_id.push(mid);
        self.commit_sha.push(r.commit_sha);
        self.dataset.push(r.dataset);
        self.format.push(r.format);
        self.value_ns.push(r.value_ns);
        self.all_runtimes_ns.push(r.all_runtimes_ns);
        self.env_triple.push(r.env_triple);
    }
}

/// `compression_sizes` is fed by both `data.json.gz` (replace-on-collision)
/// and `file-sizes-*.json.gz` (sum-on-collision). Stored as a map; converted
/// to a `RecordBatch` at flush time.
#[derive(Default)]
pub(super) struct CompressionSizeAccum {
    pub(super) rows: HashMap<i64, CompressionSize>,
}

impl CompressionSizeAccum {
    /// data.json.gz path: latest write wins, mirroring the prior
    /// `ON CONFLICT DO UPDATE SET value_bytes = excluded.value_bytes`.
    /// Bumps `deduped_with_conflict` when an existing row's
    /// `value_bytes` differs from the incoming row's, so silent
    /// value-corruption is observable.
    pub(super) fn push_replace(
        &mut self,
        mid: i64,
        r: CompressionSize,
        summary: &mut MigrationSummary,
    ) {
        if let Some(existing) = self.rows.get(&mid)
            && existing.value_bytes != r.value_bytes
        {
            summary.deduped_with_conflict += 1;
        }
        self.rows.insert(mid, r);
    }

    /// file-sizes-*.json.gz path: per-file rows aggregate into one
    /// `(commit, dataset, dataset_variant, format)` row by summing,
    /// mirroring the prior `value_bytes = compression_sizes.value_bytes
    /// + excluded.value_bytes`.
    pub(super) fn push_sum(&mut self, mid: i64, r: CompressionSize) {
        let add = r.value_bytes;
        self.rows
            .entry(mid)
            .and_modify(|x| x.value_bytes += add)
            .or_insert(r);
    }
}

pub(super) fn build_query_batch(a: QueryAccum) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("dataset_variant", DataType::Utf8, true),
        Field::new("scale_factor", DataType::Utf8, true),
        Field::new("query_idx", DataType::Int32, false),
        Field::new("storage", DataType::Utf8, false),
        Field::new("engine", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("value_ns", DataType::Int64, false),
        Field::new(
            "all_runtimes_ns",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        ),
        Field::new("peak_physical", DataType::Int64, true),
        Field::new("peak_virtual", DataType::Int64, true),
        Field::new("physical_delta", DataType::Int64, true),
        Field::new("virtual_delta", DataType::Int64, true),
        Field::new("env_triple", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(a.measurement_id)),
        Arc::new(StringArray::from(a.commit_sha)),
        Arc::new(StringArray::from(a.dataset)),
        Arc::new(StringArray::from(a.dataset_variant)),
        Arc::new(StringArray::from(a.scale_factor)),
        Arc::new(Int32Array::from(a.query_idx)),
        Arc::new(StringArray::from(a.storage)),
        Arc::new(StringArray::from(a.engine)),
        Arc::new(StringArray::from(a.format)),
        Arc::new(Int64Array::from(a.value_ns)),
        Arc::new(build_list_int64(a.all_runtimes_ns)),
        Arc::new(Int64Array::from(a.peak_physical)),
        Arc::new(Int64Array::from(a.peak_virtual)),
        Arc::new(Int64Array::from(a.physical_delta)),
        Arc::new(Int64Array::from(a.virtual_delta)),
        Arc::new(StringArray::from(a.env_triple)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

pub(super) fn build_compression_time_batch(a: CompressionTimeAccum) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("dataset_variant", DataType::Utf8, true),
        Field::new("format", DataType::Utf8, false),
        Field::new("op", DataType::Utf8, false),
        Field::new("value_ns", DataType::Int64, false),
        Field::new(
            "all_runtimes_ns",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        ),
        Field::new("env_triple", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(a.measurement_id)),
        Arc::new(StringArray::from(a.commit_sha)),
        Arc::new(StringArray::from(a.dataset)),
        Arc::new(StringArray::from(a.dataset_variant)),
        Arc::new(StringArray::from(a.format)),
        Arc::new(StringArray::from(a.op)),
        Arc::new(Int64Array::from(a.value_ns)),
        Arc::new(build_list_int64(a.all_runtimes_ns)),
        Arc::new(StringArray::from(a.env_triple)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

pub(super) fn build_random_access_batch(a: RandomAccessAccum) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("value_ns", DataType::Int64, false),
        Field::new(
            "all_runtimes_ns",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            false,
        ),
        Field::new("env_triple", DataType::Utf8, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(a.measurement_id)),
        Arc::new(StringArray::from(a.commit_sha)),
        Arc::new(StringArray::from(a.dataset)),
        Arc::new(StringArray::from(a.format)),
        Arc::new(Int64Array::from(a.value_ns)),
        Arc::new(build_list_int64(a.all_runtimes_ns)),
        Arc::new(StringArray::from(a.env_triple)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

pub(super) fn build_compression_size_batch(a: CompressionSizeAccum) -> Result<RecordBatch> {
    let n = a.rows.len();
    let mut measurement_id = Vec::with_capacity(n);
    let mut commit_sha = Vec::with_capacity(n);
    let mut dataset = Vec::with_capacity(n);
    let mut dataset_variant = Vec::with_capacity(n);
    let mut format = Vec::with_capacity(n);
    let mut value_bytes = Vec::with_capacity(n);
    for (mid, cs) in a.rows {
        measurement_id.push(mid);
        commit_sha.push(cs.commit_sha);
        dataset.push(cs.dataset);
        dataset_variant.push(cs.dataset_variant);
        format.push(cs.format);
        value_bytes.push(cs.value_bytes);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("measurement_id", DataType::Int64, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("dataset_variant", DataType::Utf8, true),
        Field::new("format", DataType::Utf8, false),
        Field::new("value_bytes", DataType::Int64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(measurement_id)),
        Arc::new(StringArray::from(commit_sha)),
        Arc::new(StringArray::from(dataset)),
        Arc::new(StringArray::from(dataset_variant)),
        Arc::new(StringArray::from(format)),
        Arc::new(Int64Array::from(value_bytes)),
    ];
    Ok(RecordBatch::try_new(schema, cols)?)
}

/// Build a non-nullable `List<Int64>` Arrow array from one inner Vec
/// per row. The outer list is non-null; inner i64 values are non-null.
fn build_list_int64(values: Vec<Vec<i64>>) -> ListArray {
    let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
    offsets.push(0);
    let mut flat: Vec<i64> = Vec::new();
    for inner in values {
        flat.extend_from_slice(&inner);
        offsets.push(flat.len() as i32);
    }
    let values_arr = Int64Array::from(flat);
    let field = Arc::new(Field::new("item", DataType::Int64, false));
    ListArray::new(
        field,
        OffsetBuffer::new(offsets.into()),
        Arc::new(values_arr),
        None,
    )
}
