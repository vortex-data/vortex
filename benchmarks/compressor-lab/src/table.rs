// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The store's tables: one Rust struct per row, written to Parquet through `arrow-json`.
//!
//! Every row written also gets `segment` (the writer's segment id) and `written_at` (ms since the
//! epoch) columns, so any row traces back to the process and flush that made it.

use std::sync::Arc;

use anyhow::Context;
use arrow_array::ArrayRef;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_json::ReaderBuilder;
use arrow_json::writer::JsonArray;
use arrow_json::writer::WriterBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// A row type stored as Parquet segments.
pub trait Table: Serialize + DeserializeOwned + Clone {
    /// The table's directory name under `facts/`, or the full directory for non-fact tables.
    const NAME: &'static str;

    /// The Arrow schema of a row, without the `segment` and `written_at` columns.
    fn fields() -> Vec<Field>;

    /// The key readers keep one row per.
    fn dedupe_key(&self) -> String;

    /// The directory a version of the table lives in.
    fn dir(version: u32) -> String {
        format!("facts/{}/v={version}", Self::NAME)
    }
}

fn schema<T: Table>() -> SchemaRef {
    Arc::new(Schema::new(T::fields()))
}

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

fn utf8_opt(name: &str) -> Field {
    Field::new(name, DataType::Utf8, true)
}

fn u64f(name: &str) -> Field {
    Field::new(name, DataType::UInt64, false)
}

fn u64_opt(name: &str) -> Field {
    Field::new(name, DataType::UInt64, true)
}

fn boolf(name: &str) -> Field {
    Field::new(name, DataType::Boolean, false)
}

fn strings(name: &str) -> Field {
    Field::new_list(name, Field::new_list_field(DataType::Utf8, false), false)
}

/// Converts rows to a record batch, adding the `segment` and `written_at` columns.
pub fn to_batch<T: Table>(rows: &[T], segment: &str) -> anyhow::Result<RecordBatch> {
    let schema = schema::<T>();
    let mut decoder = ReaderBuilder::new(Arc::clone(&schema)).build_decoder()?;
    decoder.serialize(rows)?;
    let batch = decoder
        .flush()?
        .unwrap_or_else(|| RecordBatch::new_empty(Arc::clone(&schema)));
    let now = crate::store::now_ms();
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(utf8("segment"));
    fields.push(Field::new("written_at", DataType::Int64, false));
    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(StringArray::from(vec![segment; batch.num_rows()])));
    columns.push(Arc::new(Int64Array::from(vec![now; batch.num_rows()])));
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Converts a record batch back to rows. Columns the row type does not name are ignored.
pub fn from_batch<T: Table>(batch: &RecordBatch) -> anyhow::Result<Vec<T>> {
    let mut writer = WriterBuilder::new()
        .with_explicit_nulls(true)
        .build::<_, JsonArray>(Vec::new());
    writer.write(batch)?;
    writer.finish()?;
    let bytes = writer.into_inner();
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice(&bytes).with_context(|| format!("decoding {} rows", T::NAME))
}

/// One chunk: where its rows come from and how to get them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRow {
    /// The chunk task's key.
    pub task_key: String,
    /// The source name.
    pub source: String,
    /// The column name.
    pub column: String,
    /// The chunk's input as JSON: enough to read or regenerate its rows.
    pub input: String,
    /// The primitive type, e.g. `i64`.
    pub ptype: String,
    /// Whether the array is nullable.
    pub nullable: bool,
    /// The number of rows.
    pub len: u64,
    /// The size of the canonical serialization.
    pub canonical_bytes: u64,
    /// The SHA-256 of the canonical serialization; every reader checks it.
    pub canonical_hash: String,
    /// The encoding ids of the canonical serialization.
    pub canonical_ids: Vec<String>,
    /// How the source's raw data is kept: `generate`, `reference`, `copy` or `local`.
    pub mode: String,
}

impl Table for ChunkRow {
    const NAME: &'static str = "chunks";

    fn fields() -> Vec<Field> {
        vec![
            utf8("task_key"),
            utf8("source"),
            utf8("column"),
            utf8("input"),
            utf8("ptype"),
            boolf("nullable"),
            u64f("len"),
            u64f("canonical_bytes"),
            utf8("canonical_hash"),
            strings("canonical_ids"),
            utf8("mode"),
        ]
    }

    fn dedupe_key(&self) -> String {
        self.task_key.clone()
    }
}

/// One feature value of one site of one chunk, in long form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureRow {
    /// The features task's key.
    pub task_key: String,
    /// The chunk task's key.
    pub chunk_key: String,
    /// The compression site, e.g. `root`.
    pub site: String,
    /// The site's depth in the cascade.
    pub site_depth: u8,
    /// The feature name.
    pub feature: String,
    /// The feature's cost tier.
    pub tier: Option<u8>,
    /// The window index, or `None` for the whole chunk.
    pub window: Option<u8>,
    /// The value, or `None` where undefined.
    pub value: Option<f64>,
}

impl Table for FeatureRow {
    const NAME: &'static str = "features";

    fn fields() -> Vec<Field> {
        vec![
            utf8("task_key"),
            utf8("chunk_key"),
            utf8("site"),
            Field::new("site_depth", DataType::UInt8, false),
            utf8("feature"),
            Field::new("tier", DataType::UInt8, true),
            Field::new("window", DataType::UInt8, true),
            Field::new("value", DataType::Float64, true),
        ]
    }

    fn dedupe_key(&self) -> String {
        format!(
            "{}/{}/{}/{:?}",
            self.task_key, self.site, self.feature, self.window
        )
    }
}

/// One candidate's encoding of one chunk. The encoded bytes are not stored: the plan rebuilds them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingRow {
    /// The candidates task's key.
    pub task_key: String,
    /// The chunk task's key.
    pub chunk_key: String,
    /// The candidate name.
    pub candidate: String,
    /// Whether the candidate encoded the chunk.
    pub ok: bool,
    /// The error, if it did not.
    pub error: Option<String>,
    /// The recorded plan, as printed by `Plan`'s `Display`.
    pub plan: Option<String>,
    /// The SHA-256 of the plan text.
    pub plan_hash: Option<String>,
    /// The SHA-256 of the encoded bytes.
    pub encoded_hash: Option<String>,
    /// The encoding ids the encoded bytes refer to.
    pub encoded_ids: Vec<String>,
    /// Serialized size, buffers plus metadata.
    pub bytes: Option<u64>,
    /// In-memory buffer bytes.
    pub nbytes: Option<u64>,
    /// The root encoding id.
    pub root: Option<String>,
    /// The encoding tree.
    pub tree: Option<String>,
    /// Whether the encoding decoded back to the chunk.
    pub verified: bool,
    /// For oracle front plans: the search's predicted buffer bytes.
    pub predicted_nbytes: Option<u64>,
    /// For oracle front plans: the search's predicted decode time, ns, from the shard machine.
    pub predicted_decode_ns: Option<u64>,
}

impl Table for EncodingRow {
    const NAME: &'static str = "encodings";

    fn fields() -> Vec<Field> {
        vec![
            utf8("task_key"),
            utf8("chunk_key"),
            utf8("candidate"),
            boolf("ok"),
            utf8_opt("error"),
            utf8_opt("plan"),
            utf8_opt("plan_hash"),
            utf8_opt("encoded_hash"),
            strings("encoded_ids"),
            u64_opt("bytes"),
            u64_opt("nbytes"),
            utf8_opt("root"),
            utf8_opt("tree"),
            boolf("verified"),
            u64_opt("predicted_nbytes"),
            u64_opt("predicted_decode_ns"),
        ]
    }

    fn dedupe_key(&self) -> String {
        format!("{}/{}", self.task_key, self.candidate)
    }
}

/// One (site, scheme) the oracle tried on a chunk. Decode times come from the shard machine and
/// are only for pruning and rough analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiteFactRow {
    /// The candidates task's key.
    pub task_key: String,
    /// The chunk task's key.
    pub chunk_key: String,
    /// The site's path in the cascade.
    pub path: String,
    /// The scheme tried.
    pub scheme: String,
    /// Whether it won the site.
    pub won: bool,
    /// Buffer bytes.
    pub nbytes: u64,
    /// Decode time, ns.
    pub decode_ns: u64,
    /// The site's best buffer bytes.
    pub best_nbytes: u64,
    /// The site's best decode time, ns.
    pub best_decode_ns: u64,
}

impl Table for SiteFactRow {
    const NAME: &'static str = "sites";

    fn fields() -> Vec<Field> {
        vec![
            utf8("task_key"),
            utf8("chunk_key"),
            utf8("path"),
            utf8("scheme"),
            boolf("won"),
            u64f("nbytes"),
            u64f("decode_ns"),
            u64f("best_nbytes"),
            u64f("best_decode_ns"),
        ]
    }

    fn dedupe_key(&self) -> String {
        format!("{}/{}/{}", self.task_key, self.path, self.scheme)
    }
}

/// The oracle's search effort on one chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchRow {
    /// The candidates task's key.
    pub task_key: String,
    /// The chunk task's key.
    pub chunk_key: String,
    /// Sites visited.
    pub sites: u64,
    /// Compressions run.
    pub compressions: u64,
    /// Memo hits.
    pub memo_hits: u64,
    /// Sites that hit the compression cap.
    pub capped_sites: u64,
    /// Wall time, ms.
    pub search_ms: u64,
}

impl Table for SearchRow {
    const NAME: &'static str = "search";

    fn fields() -> Vec<Field> {
        vec![
            utf8("task_key"),
            utf8("chunk_key"),
            u64f("sites"),
            u64f("compressions"),
            u64f("memo_hits"),
            u64f("capped_sites"),
            u64f("search_ms"),
        ]
    }

    fn dedupe_key(&self) -> String {
        self.task_key.clone()
    }
}

/// A finished fact task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerRow {
    /// The task key.
    pub task_key: String,
    /// The task kind.
    pub kind: String,
    /// The hash of the task's rows, used to detect nondeterminism.
    pub output_hash: String,
    /// How many rows the task wrote.
    pub rows: u64,
    /// Wall-clock milliseconds the task took.
    pub millis: u64,
    /// The code the task ran with.
    pub git_commit: String,
}

impl Table for LedgerRow {
    const NAME: &'static str = "ledger";

    fn fields() -> Vec<Field> {
        vec![
            utf8("task_key"),
            utf8("kind"),
            utf8("output_hash"),
            u64f("rows"),
            u64f("millis"),
            utf8("git_commit"),
        ]
    }

    fn dedupe_key(&self) -> String {
        self.task_key.clone()
    }
}

/// One timing sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingRow {
    /// The hardware the sample was taken on.
    pub machine_id: String,
    /// The build the sample was taken with: commit, compiler, target features and profile.
    pub build_id: String,
    /// The timing run.
    pub run_id: String,
    /// The code the sample was taken with.
    pub git_commit: String,
    /// The candidates task's key.
    pub task_key: String,
    /// The chunk task's key.
    pub chunk_key: String,
    /// The candidate, for `compress`; `None` for ops on the encoding alone.
    pub candidate: Option<String>,
    /// The SHA-256 of the encoded bytes measured.
    pub encoded_hash: String,
    /// Whether replaying the plan reproduced `encoded_hash`.
    pub replay_matched: bool,
    /// The operation: `compress` (replaying the plan), `adaptive_compress` (the sampling
    /// compressor, for `production` and `sizemodel`), `decode`, …
    pub op: String,
    /// The repetition index.
    pub rep: u32,
    /// Nanoseconds.
    pub ns: u64,
}

impl Table for TimingRow {
    const NAME: &'static str = "timings";

    fn fields() -> Vec<Field> {
        vec![
            utf8("machine_id"),
            utf8("build_id"),
            utf8("run_id"),
            utf8("git_commit"),
            utf8("task_key"),
            utf8("chunk_key"),
            utf8_opt("candidate"),
            utf8("encoded_hash"),
            boolf("replay_matched"),
            utf8("op"),
            Field::new("rep", DataType::UInt32, false),
            u64f("ns"),
        ]
    }

    fn dedupe_key(&self) -> String {
        format!(
            "{}/{}/{}/{:?}/{}/{}/{}",
            self.machine_id,
            self.run_id,
            self.task_key,
            self.candidate,
            self.encoded_hash,
            self.op,
            self.rep
        )
    }
}
