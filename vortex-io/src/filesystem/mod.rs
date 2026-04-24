// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A filesystem abstraction for discovering and opening Vortex files.
//!
//! [`FileSystem`] provides a storage-agnostic interface for listing files under a prefix
//! and opening them for reading. Implementations can target local filesystems, object stores,
//! or any other storage backend.

mod glob;
mod prefix;

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::VortexReadAt;

/// A file discovered during listing, with its path and optional size in bytes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileListing {
    /// The file path (relative to the filesystem root).
    pub path: String,
    /// The file size in bytes, if known from the listing metadata.
    pub size: Option<u64>,
}

/// A reference-counted handle to a file system.
pub type FileSystemRef = Arc<dyn FileSystem>;

/// A storage-agnostic filesystem interface for discovering and reading Vortex files.
///
/// Implementations handle the details of a particular storage backend (local disk, S3, GCS, etc.)
/// while consumers work through this uniform interface.
///
/// # Future Work
///
/// An `open_write` method will be added once [`VortexWrite`](crate::VortexWrite) is
/// object-safe (it currently uses `impl Future` return types which prevent trait-object usage).
#[async_trait]
pub trait FileSystem: Debug + Send + Sync {
    /// Recursively list files whose paths start with `prefix`.
    ///
    /// When `prefix` is empty, all files are listed. Implementations must recurse into
    /// subdirectories so that the returned stream contains every file reachable under the prefix.
    ///
    /// Returns a stream of [`FileListing`] entries. The stream may yield entries in any order;
    /// callers should sort if deterministic ordering is required.
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>>;

    /// Open a file for reading at the given path.
    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>>;

    async fn delete(&self, path: &str) -> VortexResult<()>;
}
