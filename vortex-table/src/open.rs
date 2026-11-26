// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opening a table's files.
//!
//! A scan asks for an [`Openable`] for each file it plans and awaits the opens together. Opening
//! is async because the stores that need it most, object stores and completion-based local IO
//! such as io_uring, are async. A source that is open already is ready at once, through
//! [`ReadAtOpenable`].

use std::fmt;
use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use futures::future::BoxFuture;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::filesystem::FileSystemRef;

/// A byte source that can be opened for positional reads, such as one file of a table.
pub trait Openable: Send + Sync + fmt::Debug + 'static {
    /// What identifies the source, such as its URI. Opens with the same key open the same
    /// source.
    fn key(&self) -> &Arc<str>;

    /// Opens the source for positional reads.
    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>>;
}

/// A source that is open already, such as an in-memory buffer.
#[derive(Clone)]
pub struct ReadAtOpenable {
    key: Arc<str>,
    read: Arc<dyn VortexReadAt>,
}

impl ReadAtOpenable {
    /// Opens to `read`, identified by `key`.
    pub fn new(key: impl Into<Arc<str>>, read: Arc<dyn VortexReadAt>) -> Self {
        Self {
            key: key.into(),
            read,
        }
    }
}

impl fmt::Debug for ReadAtOpenable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ReadAtOpenable").field(&self.key).finish()
    }
}

impl Openable for ReadAtOpenable {
    fn key(&self) -> &Arc<str> {
        &self.key
    }

    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>> {
        future::ready(Ok(Arc::clone(&self.read))).boxed()
    }
}

/// A path on a [`FileSystem`](vortex_io::filesystem::FileSystem), local or an object store.
#[derive(Clone)]
pub struct FileSystemOpenable {
    fs: FileSystemRef,
    path: Arc<str>,
}

impl FileSystemOpenable {
    /// Opens `path` on `fs`. The path is the key.
    pub fn new(fs: FileSystemRef, path: impl Into<Arc<str>>) -> Self {
        Self {
            fs,
            path: path.into(),
        }
    }
}

impl fmt::Debug for FileSystemOpenable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSystemOpenable")
            .field("fs", &self.fs)
            .field("path", &self.path)
            .finish()
    }
}

impl Openable for FileSystemOpenable {
    fn key(&self) -> &Arc<str> {
        &self.path
    }

    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>> {
        let fs = Arc::clone(&self.fs);
        let path = Arc::clone(&self.path);
        async move { fs.open_read(&path).await }.boxed()
    }
}
