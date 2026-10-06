// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The on-disk store: facts, observations, blobs and the ledger.
//!
//! ```text
//! <store>/
//!   ledger/<kk>/<task key>.json            done marker for a fact task (written last)
//!   facts/<kind>/<kk>/<task key>.json      a fact task's output; rewriting it is an upsert
//!   blobs/<hh>/<sha256>                    content-addressed array bytes, written once
//!   obs/<kind>/machine=<id>/<task key>/<run id>.json
//!                                          observations: every run appends a new file
//!   machines/<id>.json                     what each machine id was
//! ```
//!
//! Every path is derived from a task key or a content hash, so parallel writers never collide and
//! re-running a task rewrites exactly the files it wrote before.

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::key::TaskKey;
use crate::plan::TaskKind;

/// A store rooted at a directory.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

/// What a done marker records.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct LedgerEntry {
    /// The task key.
    pub key: TaskKey,
    /// The task kind.
    pub kind: TaskKind,
    /// The hash of the fact's JSON, used to detect nondeterminism on re-runs.
    pub output_hash: String,
    /// Wall-clock milliseconds the task took.
    pub millis: u128,
}

fn fan_out(name: &str) -> &str {
    name.get(..2).unwrap_or("00")
}

impl Store {
    /// Opens (and creates) a store.
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).with_context(|| format!("creating store {}", root.display()))?;
        Ok(Self { root })
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn ledger_path(&self, key: &TaskKey) -> PathBuf {
        self.root
            .join("ledger")
            .join(fan_out(key.as_str()))
            .join(format!("{key}.json"))
    }

    fn fact_path(&self, kind: TaskKind, key: &TaskKey) -> PathBuf {
        self.root
            .join("facts")
            .join(kind.as_str())
            .join(fan_out(key.as_str()))
            .join(format!("{key}.json"))
    }

    fn blob_path(&self, hash: &str) -> PathBuf {
        self.root.join("blobs").join(fan_out(hash)).join(hash)
    }

    /// Whether a fact task is done.
    pub fn is_done(&self, key: &TaskKey) -> bool {
        self.ledger_path(key).exists()
    }

    /// The done marker of a fact task, if any.
    pub fn ledger(&self, key: &TaskKey) -> anyhow::Result<Option<LedgerEntry>> {
        read_json_opt(&self.ledger_path(key))
    }

    /// Writes a fact, then its done marker. Returns the fact's hash.
    pub fn put_fact<T: Serialize>(
        &self,
        kind: TaskKind,
        key: &TaskKey,
        fact: &T,
        millis: u128,
    ) -> anyhow::Result<String> {
        let bytes = serde_json::to_vec(fact)?;
        let output_hash = crate::key::bytes_digest(&bytes);
        write_atomically(&self.fact_path(kind, key), &bytes)?;
        let entry = LedgerEntry {
            key: key.clone(),
            kind,
            output_hash: output_hash.clone(),
            millis,
        };
        write_atomically(&self.ledger_path(key), &serde_json::to_vec(&entry)?)?;
        Ok(output_hash)
    }

    /// Reads a fact.
    pub fn fact<T: DeserializeOwned>(&self, kind: TaskKind, key: &TaskKey) -> anyhow::Result<T> {
        let path = self.fact_path(kind, key);
        serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))
    }

    /// Stores blob bytes under their hash. Existing blobs are left alone.
    pub fn put_blob(&self, hash: &str, bytes: &[u8]) -> anyhow::Result<()> {
        let path = self.blob_path(hash);
        if !path.exists() {
            write_atomically(&path, bytes)?;
        }
        Ok(())
    }

    /// Reads blob bytes.
    pub fn blob(&self, hash: &str) -> anyhow::Result<Vec<u8>> {
        let path = self.blob_path(hash);
        fs::read(&path).with_context(|| format!("reading blob {}", path.display()))
    }

    /// Appends an observation for a task, as one new file per run.
    pub fn put_observation<T: Serialize>(
        &self,
        kind: &str,
        machine_id: &str,
        key: &TaskKey,
        run_id: &str,
        observation: &T,
    ) -> anyhow::Result<()> {
        let path = self
            .root
            .join("obs")
            .join(kind)
            .join(format!("machine={machine_id}"))
            .join(key.as_str())
            .join(format!("{run_id}.json"));
        write_atomically(&path, &serde_json::to_vec(observation)?)
    }

    /// Every observation for a task on a machine, one per run.
    pub fn observations<T: DeserializeOwned>(
        &self,
        kind: &str,
        machine_id: &str,
        key: &TaskKey,
    ) -> anyhow::Result<Vec<T>> {
        let dir = self
            .root
            .join("obs")
            .join(kind)
            .join(format!("machine={machine_id}"))
            .join(key.as_str());
        let Ok(entries) = fs::read_dir(&dir) else {
            return Ok(Vec::new());
        };
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        paths.iter().map(|p| read_json(p)).collect()
    }

    /// The machine ids that have observations of `kind`.
    pub fn machines(&self, kind: &str) -> Vec<String> {
        let Ok(entries) = fs::read_dir(self.root.join("obs").join(kind)) else {
            return Vec::new();
        };
        let mut ids: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.file_name()
                    .to_str()
                    .and_then(|n| n.strip_prefix("machine="))
                    .map(str::to_string)
            })
            .collect();
        ids.sort();
        ids
    }

    /// Records what a machine id describes.
    pub fn put_machine(&self, info: &crate::machine::MachineInfo) -> anyhow::Result<()> {
        let path = self
            .root
            .join("machines")
            .join(format!("{}.json", info.machine_id));
        write_atomically(&path, &serde_json::to_vec_pretty(info)?)
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn read_json_opt<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Option<T>> {
    if path.exists() {
        read_json(path).map(Some)
    } else {
        Ok(None)
    }
}

/// Writes to a temporary file and renames it into place, so readers never see partial files.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(".tmp{}", std::process::id()));
    let temp = PathBuf::from(temp);
    let mut file =
        fs::File::create(&temp).with_context(|| format!("creating {}", temp.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing {}", temp.display()))?;
    file.sync_all()?;
    fs::rename(&temp, path).with_context(|| format!("renaming into {}", path.display()))
}
