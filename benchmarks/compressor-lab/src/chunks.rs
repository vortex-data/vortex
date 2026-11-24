// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Making chunk rows and loading chunks back, whichever way a source's rows are held.
//!
//! A chunk's rows arrive one of three ways (see [`ChunkMode`]): regenerated, re-read from a
//! pinned source, or read from the store. Whichever way, the canonical array is hashed and checked
//! against the chunk row, so a source that changed fails loudly instead of measuring other data.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::bail;
use sha2::Digest;
use sha2::Sha256;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::blob;
use crate::blob::BlobRef;
use crate::key::TaskKey;
use crate::source::ChunkInput;
use crate::spec::ChunkMode;
use crate::store::Store;
use crate::table::ChunkRow;

/// Reads a chunk's rows from its source and makes its row, storing a copy if `mode` asks.
pub fn make(
    store: &Store,
    key: &TaskKey,
    input: &ChunkInput,
    mode: ChunkMode,
    data_root: &Path,
    session: &VortexSession,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<ChunkRow> {
    let array = from_source(input, data_root, ctx)?;
    let (canonical, _) = blob::serialize(&array, session)?;
    let canonical_hash = logical_hash(&array, ctx)?;
    let mut row = ChunkRow {
        task_key: key.to_string(),
        source: input.source().to_string(),
        column: input.column().to_string(),
        input: serde_json::to_string(input)?,
        ptype: canonical.ptype.clone(),
        nullable: canonical.nullable,
        len: canonical.len,
        canonical_bytes: canonical.bytes,
        canonical_hash,
        canonical_ids: canonical.ids,
        mode: mode.as_str().to_string(),
        stored_hash: None,
        stored_bytes: None,
        stored_ids: Vec::new(),
    };
    if mode == ChunkMode::Copy {
        let compressed = BtrBlocksCompressor::from_session(session).compress(&array, ctx)?;
        let (stored, bytes) = blob::serialize(&compressed, session)?;
        store.put_create(&Store::chunk_key(&stored.hash), bytes)?;
        row.stored_hash = Some(stored.hash);
        row.stored_bytes = Some(stored.bytes);
        row.stored_ids = stored.ids;
    }
    Ok(row)
}

fn from_source(
    input: &ChunkInput,
    data_root: &Path,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<ArrayRef> {
    let arrow = crate::load::arrow(input, data_root)?;
    crate::load::to_vortex(&arrow, ctx)
}

/// Loads chunks by key, from wherever their rows are held.
pub struct ChunkLoader<'a> {
    store: &'a Store,
    session: VortexSession,
    data_root: PathBuf,
    cache: Option<PathBuf>,
    rows: HashMap<String, ChunkRow>,
}

impl<'a> ChunkLoader<'a> {
    /// A loader over the store's chunks table. Stored chunks of a remote store are cached in
    /// `cache` (by default `~/.cache/vx-lab/chunks`); they never change, so the cache is never
    /// invalidated.
    pub fn new(
        store: &'a Store,
        session: &VortexSession,
        data_root: &Path,
        cache: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let rows = store
            .read_table::<ChunkRow>(CHUNKS_VERSION)?
            .into_iter()
            .map(|row| (row.task_key.clone(), row))
            .collect();
        let cache = if store.is_local() {
            None
        } else {
            cache.or_else(|| {
                std::env::var_os("HOME")
                    .map(|home| PathBuf::from(home).join(".cache/vx-lab/chunks"))
            })
        };
        Ok(Self {
            store,
            session: session.clone(),
            data_root: data_root.to_path_buf(),
            cache,
            rows,
        })
    }

    /// The chunk row of a chunk task, if it is done.
    pub fn row(&self, key: &TaskKey) -> Option<&ChunkRow> {
        self.rows.get(key.as_str())
    }

    /// Adds rows written since the loader was made.
    pub fn insert(&mut self, row: ChunkRow) {
        self.rows.insert(row.task_key.clone(), row);
    }

    /// Fetches a stored chunk into the local cache without decoding it. Returns whether it was
    /// fetched (as opposed to already cached or not a stored chunk).
    pub fn prefetch(&self, key: &TaskKey) -> anyhow::Result<bool> {
        let Some(row) = self.row(key) else {
            return Ok(false);
        };
        let (Some(hash), Some(cache)) = (&row.stored_hash, &self.cache) else {
            return Ok(false);
        };
        if cache.join(hash).exists() {
            return Ok(false);
        }
        self.stored_bytes(hash)?;
        Ok(true)
    }

    fn stored_bytes(&self, hash: &str) -> anyhow::Result<Vec<u8>> {
        if let Some(cache) = &self.cache {
            let path = cache.join(hash);
            if let Ok(bytes) = std::fs::read(&path) {
                return Ok(bytes);
            }
            let bytes = self.fetch(hash)?;
            std::fs::create_dir_all(cache)?;
            let temp = cache.join(format!("{hash}.tmp{}", std::process::id()));
            std::fs::write(&temp, &bytes)?;
            std::fs::rename(&temp, &path)?;
            return Ok(bytes);
        }
        self.fetch(hash)
    }

    fn fetch(&self, hash: &str) -> anyhow::Result<Vec<u8>> {
        self.store
            .get(&Store::chunk_key(hash))?
            .with_context(|| format!("stored chunk {hash} is missing from {}", self.store.url()))
    }

    /// The chunk's canonical array, checked against its row's hash.
    pub fn load(
        &self,
        key: &TaskKey,
        ctx: &mut ExecutionCtx,
    ) -> anyhow::Result<(ChunkRow, ArrayRef)> {
        let row = self
            .row(key)
            .with_context(|| format!("chunk {key} is not done"))?
            .clone();
        let array = match &row.stored_hash {
            Some(hash) => {
                let blob = BlobRef {
                    hash: hash.clone(),
                    ids: row.stored_ids.clone(),
                    ptype: row.ptype.clone(),
                    nullable: row.nullable,
                    len: row.len,
                    bytes: row.stored_bytes.unwrap_or_default(),
                };
                blob::deserialize(&blob, self.stored_bytes(hash)?, &self.session)?
            }
            None => {
                let input: ChunkInput = serde_json::from_str(&row.input)
                    .with_context(|| format!("parsing the input of chunk {key}"))?;
                from_source(&input, &self.data_root, ctx)?
            }
        };
        let array = array.execute::<Canonical>(ctx)?.into_array();
        if logical_hash(&array, ctx)? != row.canonical_hash {
            bail!(
                "chunk {key} ({} {}, {}) no longer matches its hash: the source changed or moved",
                row.source,
                row.column,
                row.mode
            );
        }
        Ok((row, array))
    }
}

/// A hash of an array's logical contents: its type, length, and each row's validity and value.
///
/// Unlike the hash of a serialization, it does not depend on how the array is represented (an
/// all-valid mask or none, buffer alignment), so a decoded encoding hashes like its input.
pub fn logical_hash(array: &ArrayRef, ctx: &mut ExecutionCtx) -> anyhow::Result<String> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    let len = primitive.len();
    let width = primitive.ptype().byte_width();
    let mask = primitive.validity()?.execute_mask(len, ctx)?;
    let values = primitive.buffer_handle().to_host_sync();
    let values = values.as_slice();
    if values.len() < len * width {
        bail!(
            "{} values need {} bytes, the buffer has {}",
            len,
            len * width,
            values.len()
        );
    }
    let mut hasher = Sha256::new();
    hasher.update(primitive.ptype().to_string().as_bytes());
    hasher.update([u8::from(array.dtype().is_nullable())]);
    hasher.update((len as u64).to_le_bytes());
    if mask.all_true() {
        hasher.update([1]);
        hasher.update(&values[..len * width]);
    } else {
        hasher.update([0]);
        for row in 0..len {
            if mask.value(row) {
                hasher.update([1]);
                hasher.update(&values[row * width..(row + 1) * width]);
            } else {
                hasher.update([0]);
            }
        }
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The version of the chunks table.
pub const CHUNKS_VERSION: u32 = 1;
