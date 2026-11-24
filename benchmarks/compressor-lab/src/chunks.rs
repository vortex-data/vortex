// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Making chunk rows and loading chunks back from their sources.
//!
//! The store keeps no copy of a chunk. Its rows are regenerated (TPC-H, synthetic), re-read from
//! the source's raw file (kept in the store under `raw/` for `copy` sources, fetched into a cache
//! for `reference` ones), or read from a local file. Whichever way, the canonical array is hashed
//! and checked against the chunk row, so a source that changed fails loudly instead of measuring
//! other data.

use std::collections::BTreeMap;

use anyhow::Context;
use anyhow::bail;
use sha2::Digest;
use sha2::Sha256;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;

use crate::blob;
use crate::key::TaskKey;
use crate::load::DataLocation;
use crate::source::ChunkInput;
use crate::spec::Storage;
use crate::store::Store;
use crate::table::ChunkRow;

/// The version of the chunks table.
pub const CHUNKS_VERSION: u32 = 1;

/// Reads a chunk's rows from its source and makes its row.
pub fn make(
    key: &TaskKey,
    input: &ChunkInput,
    storage: Storage,
    data: &DataLocation,
    session: &VortexSession,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<ChunkRow> {
    let array = from_source(input, data, ctx)?;
    let (canonical, _) = blob::serialize(&array, session)?;
    Ok(ChunkRow {
        task_key: key.to_string(),
        source: input.source().to_string(),
        column: input.column().to_string(),
        input: serde_json::to_string(input)?,
        ptype: canonical.ptype,
        nullable: canonical.nullable,
        len: canonical.len,
        canonical_bytes: canonical.bytes,
        canonical_hash: logical_hash(&array, ctx)?,
        canonical_ids: canonical.ids,
        mode: storage.as_str().to_string(),
    })
}

fn from_source(
    input: &ChunkInput,
    data: &DataLocation,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<ArrayRef> {
    let arrow = crate::load::arrow(input, data)?;
    crate::load::to_vortex(&arrow, ctx)
}

/// Loads chunks by key from their sources.
pub struct ChunkLoader {
    data: DataLocation,
    rows: BTreeMap<String, ChunkRow>,
}

impl ChunkLoader {
    /// A loader over the store's chunks table.
    pub fn new(store: &Store, data: &DataLocation) -> anyhow::Result<Self> {
        let rows = store
            .read_table::<ChunkRow>(CHUNKS_VERSION)?
            .into_iter()
            .map(|row| (row.task_key.clone(), row))
            .collect();
        Ok(Self {
            data: data.clone(),
            rows,
        })
    }

    /// The chunk row of a chunk task, if it is done.
    pub fn row(&self, key: &TaskKey) -> Option<&ChunkRow> {
        self.rows.get(key.as_str())
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
        let input: ChunkInput = serde_json::from_str(&row.input)
            .with_context(|| format!("parsing the input of chunk {key}"))?;
        let array = from_source(&input, &self.data, ctx)?
            .execute::<Canonical>(ctx)?
            .into_array();
        if logical_hash(&array, ctx)? != row.canonical_hash {
            bail!(
                "chunk {key} ({} {}, {}) no longer matches its hash: the source changed",
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
            "{len} values need {} bytes, the buffer has {}",
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
