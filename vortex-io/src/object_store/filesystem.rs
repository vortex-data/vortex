// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FileSystem`] implementation backed by an [`ObjectStore`].

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use object_store::ObjectStore;
use object_store::path::Path;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::object_store::ObjectStoreReadAt;
use crate::runtime::Handle;

/// A [`FileSystem`] backed by an [`ObjectStore`].
///
// TODO(ngates): we could consider spawning a driver task inside this file system such that we can
//  apply concurrency limits to the overall object store, rather than on a per-file basis.
pub struct ObjectStoreFileSystem {
    store: Arc<dyn ObjectStore>,
    handle: Handle,
}

impl Debug for ObjectStoreFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectStoreFileSystem")
            .field("store", &self.store)
            .finish()
    }
}

impl ObjectStoreFileSystem {
    /// Create a new filesystem backed by the given object store and runtime handle.
    pub fn new(store: Arc<dyn ObjectStore>, handle: Handle) -> Self {
        Self { store, handle }
    }

    /// Create a new filesystem backed by a local file system object store and the given runtime
    /// handle.
    pub fn local(handle: Handle) -> Self {
        Self::new(
            Arc::new(object_store::local::LocalFileSystem::new()),
            handle,
        )
    }
}

#[async_trait]
impl FileSystem for ObjectStoreFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let path = if prefix.is_empty() {
            None
        } else {
            Some(Path::from(prefix))
        };
        self.store
            .list(path.as_ref())
            .map(|result| {
                result
                    .map(|meta| FileListing {
                        path: meta.location.to_string(),
                        size: Some(meta.size),
                    })
                    .map_err(Into::into)
            })
            .boxed()
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(ObjectStoreReadAt::new(
            Arc::clone(&self.store),
            path.into(),
            self.handle.clone(),
        )))
    }
}
