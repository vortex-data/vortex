// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::filesystem::FileListing;
use crate::filesystem::FileSystem;
use crate::runtime::Handle;
use crate::std_file::FileReadAt;

/// A FileSystem over local filesystem.
pub struct StdFileSystem {
    handle: Handle,
}

impl Debug for StdFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdFileSystem").finish()
    }
}

impl StdFileSystem {
    pub fn new(handle: Handle) -> Self {
        Self { handle }
    }
}

fn walk(dir: &Path, out: &mut Vec<FileListing>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            walk(&entry.path(), out)?;
        } else {
            let metadata = entry.metadata()?;
            out.push(FileListing {
                path: entry.path().to_string_lossy().into_owned(),
                size: Some(metadata.len()),
            });
        }
    }
    Ok(())
}

#[async_trait]
impl FileSystem for StdFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let dir = PathBuf::from(prefix);
        let listing = self.handle.spawn_blocking(move || {
            let mut out = Vec::new();
            walk(&dir, &mut out)?;
            Ok::<_, io::Error>(out)
        });
        stream::once(listing)
            .flat_map(|result| match result {
                Ok(listings) => stream::iter(listings.into_iter().map(Ok)).boxed(),
                Err(e) => stream::once(async move { Err(e.into()) }).boxed(),
            })
            .boxed()
    }

    async fn head(&self, path: &str) -> VortexResult<Option<FileListing>> {
        let path = path.to_owned();
        self.handle
            .spawn_blocking(move || match fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() => Ok(Some(FileListing {
                    path,
                    size: Some(metadata.len()),
                })),
                Ok(_) => Ok(None),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e.into()),
            })
            .await
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let handle = self.handle.clone();
        let path = path.to_owned();
        let reader = self
            .handle
            .spawn_blocking(move || FileReadAt::open(path, handle))
            .await?;
        Ok(Arc::new(reader))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        let path = path.to_owned();
        self.handle
            .spawn_blocking(move || fs::remove_file(path))
            .await?;
        Ok(())
    }
}
