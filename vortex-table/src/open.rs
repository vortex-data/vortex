// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opening a table's files.
//!
//! A scan asks an [`Openable`] for each file it plans, by the path the manifest records, and
//! awaits the opens together. Opening is async because the stores that need it most, object
//! stores and completion-based local IO such as io_uring, are async. A synchronous opener, such
//! as a closure over files already open, needs no future of its own: every
//! `Fn(&str) -> VortexResult<Arc<dyn VortexReadAt>>` is an [`Openable`] that is ready at once.

use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use futures::future::BoxFuture;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::filesystem::FileSystem;

/// Opens a table's files for positional reads.
pub trait Openable: Send + Sync {
    /// Opens the file at `path`, as the manifest records it.
    fn open<'a>(&'a self, path: &'a str) -> BoxFuture<'a, VortexResult<Arc<dyn VortexReadAt>>>;
}

impl<F> Openable for F
where
    F: Fn(&str) -> VortexResult<Arc<dyn VortexReadAt>> + Send + Sync,
{
    fn open<'a>(&'a self, path: &'a str) -> BoxFuture<'a, VortexResult<Arc<dyn VortexReadAt>>> {
        future::ready(self(path)).boxed()
    }
}

/// Opens paths on a [`FileSystem`], local or an object store.
impl Openable for dyn FileSystem {
    fn open<'a>(&'a self, path: &'a str) -> BoxFuture<'a, VortexResult<Arc<dyn VortexReadAt>>> {
        self.open_read(path)
    }
}
