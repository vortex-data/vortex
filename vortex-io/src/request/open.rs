// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte sources a planner opens mid-plan.
//!
//! Opening is async because the stores that need it most, object stores and completion-based
//! local IO such as io_uring, are async. A source that is open already is ready at once, through
//! [`ReadAtOpenable`].

use std::fmt;
use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use futures::future::BoxFuture;
use vortex_error::VortexResult;

use super::IoService;
use super::ReadAtIoService;
use crate::VortexReadAt;
use crate::filesystem::FileSystemRef;

/// Something a planner can turn into a byte source, carried by an
/// [`IoTarget::Open`](super::IoTarget::Open).
///
/// The planner builds it, so a plan can open sources of any backend without the plan's
/// [`SourcesIoService`](super::SourcesIoService) knowing about it: a path on a
/// [`FileSystem`](crate::filesystem::FileSystem), a reader that is already open, or a backend
/// with its own submission queue such as io_uring.
pub trait Openable: Send + Sync + fmt::Debug + 'static {
    /// What identifies the source, such as its URI. Opens with the same key within one service
    /// share one source and its id.
    fn key(&self) -> &Arc<str>;

    /// Opens the source for positional reads.
    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>>;

    /// The service the opened source's requests are served by. Reads straight from
    /// [`open`](Self::open)'s reader unless overridden, for example by a backend that batches the
    /// reads of every source it opened into one submission.
    fn open_service(&self) -> BoxFuture<'static, VortexResult<Arc<dyn IoService>>> {
        self.open()
            .map(|read| read.map(|read| Arc::new(ReadAtIoService::new(read)) as Arc<dyn IoService>))
            .boxed()
    }
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

/// A path on a [`FileSystem`](crate::filesystem::FileSystem), local or an object store.
#[derive(Clone)]
pub struct FileSystemOpenable {
    fs: FileSystemRef,
    path: Arc<str>,
}

impl FileSystemOpenable {
    /// Opens `path` on `fs`. The path is the key, so it must be unique within the plan.
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
