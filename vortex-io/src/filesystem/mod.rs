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
/// # Paths
///
/// Path strings are *literal* object keys / file paths: the characters are used verbatim, with no
/// shell-style `~` expansion (`~` is a literal tilde, not the home directory) and no
/// percent-encoding or -decoding applied by this layer (`%20` is the three characters `%`, `2`,
/// `0`, not a space). A path produced by [`list`](FileSystem::list) or [`head`](FileSystem::head)
/// is the object's actual key, so it can be passed straight back to
/// [`open_read`](FileSystem::open_read) — including when it contains characters such as `~`, `%`,
/// `[`, `]`, or `#`.
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

    /// Fetch metadata for the file at the exact `path`, if it exists.
    ///
    /// Unlike [`list`](FileSystem::list), which enumerates files *under* a prefix on a
    /// path-segment basis and never yields the prefix itself, `head` looks up the object at
    /// exactly `path`. It is the correct primitive for confirming that a single known file
    /// exists and reading its size.
    ///
    /// Returns `Ok(Some(_))` with the file's [`FileListing`] when it exists, `Ok(None)` when no
    /// file exists at `path`, and `Err(_)` for any other failure (I/O or permission errors, etc.).
    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>>;

    /// Open a file for reading at the given path.
    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>>;

    async fn delete(&self, path: &str) -> VortexResult<()>;
}
