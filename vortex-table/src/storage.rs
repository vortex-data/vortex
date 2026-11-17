// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Where a table's files live.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
use vortex_io::std_file::FileReadAt;
use vortex_utils::aliases::hash_map::HashMap;

/// Resolves the paths a manifest names to byte sources.
///
/// Paths are relative to the table, so a table moves with its directory or prefix.
pub trait TableStorage: Send + Sync {
    /// Opens the file at `path`, relative to the table.
    fn open(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>>;
}

/// A table in a local directory.
pub struct LocalStorage {
    root: PathBuf,
    handle: Handle,
}

impl LocalStorage {
    /// A table under `root`, reading on `handle`.
    pub fn new(root: impl Into<PathBuf>, handle: Handle) -> Self {
        Self {
            root: root.into(),
            handle,
        }
    }

    /// The local path of `path`.
    pub fn path(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }
}

impl TableStorage for LocalStorage {
    fn open(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(FileReadAt::open(
            self.path(path),
            self.handle.clone(),
        )?))
    }
}

/// A table held in memory, for tests and examples.
#[derive(Default)]
pub struct MemoryStorage {
    files: RwLock<HashMap<String, ByteBuffer>>,
}

impl MemoryStorage {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `bytes` at `path`, replacing what was there.
    pub fn put(&self, path: impl Into<String>, bytes: ByteBuffer) {
        self.files.write().insert(path.into(), bytes);
    }
}

impl TableStorage for MemoryStorage {
    fn open(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let bytes = self
            .files
            .read()
            .get(path)
            .cloned()
            .ok_or_else(|| vortex_err!("no file at {path}"))?;
        Ok(Arc::new(bytes))
    }
}
