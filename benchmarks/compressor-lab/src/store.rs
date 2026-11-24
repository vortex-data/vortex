// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared store: chunks, Parquet fact segments, the ledger and timings, on local disk or any
//! object store.
//!
//! ```text
//! <store>/
//!   plans/<plan_id>/plan.json                 the plan header of every plan that ran here
//!   plans/<plan_id>/RETIRED                   optional: GC no longer keeps this plan's data
//!   chunks/<hh>/<sha256>                      chunks of `copy` sources, content-addressed
//!   facts/<table>/v=<version>/<segment>.parquet
//!   ledger/kind=<kind>/<segment>.parquet      one row per finished task, written after its facts
//!   timings/machine=<id>/<segment>.parquet    timing samples, append-only
//!   machines/<machine_id>.json                what each machine id is
//!   workers/<worker_id>.json                  heartbeats
//! ```
//!
//! Every key is content-addressed or contains a segment id unique to its writer, so any number of
//! processes and machines write at once with no coordination. Tasks are deterministic, so a task
//! written twice is harmless: readers keep one row per key.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use arrow_array::RecordBatch;
use futures::TryStreamExt;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutMode;
use object_store::PutOptions;
use object_store::PutPayload;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::registry::ObjectStoreRegistry;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::key::TaskKey;
use crate::plan::TaskKind;
use crate::table::LedgerRow;
use crate::table::Table;

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// The runtime the store's IO runs on, started on first use.
fn runtime() -> anyhow::Result<&'static tokio::runtime::Runtime> {
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .context("starting the IO runtime")?;
    Ok(RUNTIME.get_or_init(|| runtime))
}

/// A store at a URL (`s3://bucket/prefix`, `gs://…`, `file:///…`) or a local directory.
#[derive(Debug, Clone)]
pub struct Store {
    url: String,
    inner: Arc<dyn ObjectStore>,
    prefix: ObjectPath,
    local: bool,
}

/// A store object's key and size.
#[derive(Debug, Clone)]
pub struct Listed {
    /// The key, relative to the store root.
    pub key: String,
    /// The size in bytes.
    pub size: u64,
    /// When it was last written, in milliseconds since the epoch.
    pub modified_ms: i64,
}

fn fan_out(name: &str) -> &str {
    name.get(..2).unwrap_or("00")
}

/// Milliseconds since the epoch.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

impl Store {
    /// Opens a store. A string with `://` is resolved as a URL through `vortex-cloud`, reading
    /// credentials from the environment; anything else is a local directory, created if missing.
    pub fn open(location: impl AsRef<Path>) -> anyhow::Result<Self> {
        let location = location.as_ref();
        let text = location.to_string_lossy().to_string();
        if text.contains("://") && !text.starts_with("file://") {
            let url =
                url::Url::parse(&text).with_context(|| format!("parsing store URL {text}"))?;
            let (inner, prefix) = vortex_cloud::Registry::new()
                .resolve(&url)
                .with_context(|| format!("opening store {text}"))?;
            return Ok(Self {
                url: text,
                inner,
                prefix,
                local: false,
            });
        }
        let dir = Path::new(text.strip_prefix("file://").unwrap_or(&text));
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating store {}", dir.display()))?;
        let dir = dir.canonicalize()?;
        Ok(Self {
            url: dir.display().to_string(),
            inner: Arc::new(LocalFileSystem::new_with_prefix(&dir)?),
            prefix: ObjectPath::default(),
            local: true,
        })
    }

    /// Where the store is.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Whether the store is on local disk.
    pub fn is_local(&self) -> bool {
        self.local
    }

    fn path(&self, key: &str) -> ObjectPath {
        let mut path = self.prefix.clone();
        for part in key.split('/').filter(|p| !p.is_empty()) {
            path = path.join(part);
        }
        path
    }

    fn relative(&self, path: &ObjectPath) -> String {
        let full = path.as_ref();
        let prefix = self.prefix.as_ref();
        full.strip_prefix(prefix)
            .unwrap_or(full)
            .trim_start_matches('/')
            .to_string()
    }

    /// Writes `bytes` at `key` unless something is already there. Returns whether it wrote.
    pub fn put_create(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<bool> {
        let path = self.path(key);
        let options = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        match runtime()?.block_on(self.inner.put_opts(&path, PutPayload::from(bytes), options)) {
            Ok(_) => Ok(true),
            Err(object_store::Error::AlreadyExists { .. }) => Ok(false),
            Err(e) => Err(e).with_context(|| format!("writing {key}")),
        }
    }

    /// Writes `bytes` at `key`, replacing anything there.
    pub fn put(&self, key: &str, bytes: Vec<u8>) -> anyhow::Result<()> {
        runtime()?
            .block_on(self.inner.put(&self.path(key), PutPayload::from(bytes)))
            .with_context(|| format!("writing {key}"))?;
        Ok(())
    }

    /// Reads `key`, or `None` if it does not exist.
    pub fn get(&self, key: &str) -> anyhow::Result<Option<Vec<u8>>> {
        let path = self.path(key);
        runtime()?.block_on(async {
            match self.inner.get(&path).await {
                Ok(result) => Ok(Some(result.bytes().await?.to_vec())) as anyhow::Result<_>,
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(anyhow::Error::from(e).context(format!("reading {key}"))),
            }
        })
    }

    /// Every object under `prefix`, sorted by key.
    pub fn list(&self, prefix: &str) -> anyhow::Result<Vec<Listed>> {
        let path = self.path(prefix);
        let metas: Vec<ObjectMeta> = runtime()?.block_on(async {
            match self.inner.list(Some(&path)).try_collect().await {
                Ok(metas) => Ok(metas),
                Err(object_store::Error::NotFound { .. }) => Ok(Vec::new()),
                Err(e) => Err(anyhow::Error::from(e).context(format!("listing {prefix}"))),
            }
        })?;
        let mut listed: Vec<Listed> = metas
            .iter()
            .map(|m| Listed {
                key: self.relative(&m.location),
                size: m.size,
                modified_ms: m.last_modified.timestamp_millis(),
            })
            .filter(|l| !l.key.contains(".tmp"))
            .collect();
        listed.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(listed)
    }

    /// Deletes `key`. Deleting a missing key is not an error.
    pub fn delete(&self, key: &str) -> anyhow::Result<()> {
        match runtime()?.block_on(self.inner.delete(&self.path(key))) {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(e).with_context(|| format!("deleting {key}")),
        }
    }

    /// Writes a JSON document, replacing anything there.
    pub fn put_json<T: Serialize>(&self, key: &str, value: &T) -> anyhow::Result<()> {
        self.put(key, serde_json::to_vec_pretty(value)?)
    }

    /// Reads a JSON document, if present.
    pub fn get_json<T: DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        self.get(key)?
            .map(|bytes| serde_json::from_slice(&bytes).with_context(|| format!("parsing {key}")))
            .transpose()
    }

    /// The key a stored chunk lives at.
    pub fn chunk_key(hash: &str) -> String {
        format!("chunks/{}/{hash}", fan_out(hash))
    }

    /// Writes a parquet segment of rows. Returns its key.
    pub fn put_segment<T: Table>(
        &self,
        dir: &str,
        segment: &str,
        rows: &[T],
    ) -> anyhow::Result<String> {
        let key = format!("{dir}/{segment}.parquet");
        let batch = crate::table::to_batch(rows, segment)?;
        self.put(&key, parquet_bytes(&batch)?)?;
        Ok(key)
    }

    /// Reads every row of every segment under `dir`, in segment order.
    pub fn read_rows<T: Table>(&self, dir: &str) -> anyhow::Result<Vec<T>> {
        let mut rows = Vec::new();
        for listed in self.list(dir)? {
            if !listed.key.ends_with(".parquet") {
                continue;
            }
            rows.extend(self.read_segment::<T>(&listed.key)?);
        }
        Ok(rows)
    }

    /// Reads the rows of one segment.
    pub fn read_segment<T: Table>(&self, key: &str) -> anyhow::Result<Vec<T>> {
        let bytes = self.get(key)?.with_context(|| format!("{key} vanished"))?;
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))?.build()?;
        let mut rows = Vec::new();
        for batch in reader {
            rows.extend(crate::table::from_batch::<T>(&batch?)?);
        }
        Ok(rows)
    }

    /// Reads a fact table, one row per [`Table::dedupe_key`], the first written winning.
    pub fn read_table<T: Table>(&self, version: u32) -> anyhow::Result<Vec<T>> {
        let mut seen = BTreeSet::new();
        let mut rows = self.read_rows::<T>(&T::dir(version))?;
        rows.retain(|row| seen.insert(row.dedupe_key()));
        Ok(rows)
    }

    /// The finished tasks of a kind, by key.
    pub fn ledger(&self, kind: TaskKind) -> anyhow::Result<BTreeMap<TaskKey, LedgerRow>> {
        let mut done = BTreeMap::new();
        for row in self.read_rows::<LedgerRow>(&ledger_dir(kind))? {
            done.entry(TaskKey::from_hex(&row.task_key)).or_insert(row);
        }
        Ok(done)
    }

    /// Records what a machine id describes.
    pub fn put_machine(&self, info: &crate::machine::MachineInfo) -> anyhow::Result<()> {
        self.put_json(&format!("machines/{}.json", info.machine_id), info)
    }

    /// The machine ids that have timings.
    pub fn timing_machines(&self) -> anyhow::Result<Vec<String>> {
        let mut ids: BTreeSet<String> = BTreeSet::new();
        for listed in self.list("timings")? {
            if let Some(id) = listed
                .key
                .split('/')
                .find_map(|part| part.strip_prefix("machine="))
            {
                ids.insert(id.to_string());
            }
        }
        Ok(ids.into_iter().collect())
    }
}

/// The ledger directory of a task kind.
pub fn ledger_dir(kind: TaskKind) -> String {
    format!("ledger/kind={}", kind.as_str())
}

/// The timings directory of a machine.
pub fn timings_dir(machine_id: &str) -> String {
    format!("timings/machine={machine_id}")
}

/// Encodes a batch as zstd-compressed Parquet.
pub fn parquet_bytes(batch: &RecordBatch) -> anyhow::Result<Vec<u8>> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(out)
}
