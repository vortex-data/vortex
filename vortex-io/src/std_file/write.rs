// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::future::Future;
use std::io;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::IoBuf;
use crate::VortexWrite;
use crate::runtime::Handle;

/// A local file writer that runs blocking filesystem operations on a Vortex runtime.
pub struct FileWrite {
    file: Arc<Mutex<File>>,
    handle: Handle,
}

impl FileWrite {
    /// Create or truncate a file using `handle` for the blocking open operation.
    pub async fn create(path: impl AsRef<Path>, handle: Handle) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = handle.spawn_blocking(move || File::create(path)).await?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            handle,
        })
    }
}

impl VortexWrite for FileWrite {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        async move {
            handle
                .spawn_blocking(move || {
                    file.lock().write_all(buffer.as_slice())?;
                    Ok(buffer)
                })
                .await
        }
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        async move { handle.spawn_blocking(move || file.lock().flush()).await }
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use vortex_error::VortexResult;

    use super::*;
    use crate::runtime::BlockingRuntime;
    use crate::runtime::current::CurrentThreadRuntime;

    #[test]
    fn test_write_file() -> VortexResult<()> {
        let directory = tempdir()?;
        let path = directory.path().join("write.vortex");
        let runtime = CurrentThreadRuntime::new();
        runtime.block_on(async {
            let mut writer = FileWrite::create(&path, runtime.handle()).await?;
            writer.write_all(vec![1, 2, 3]).await?;
            writer.shutdown().await
        })?;

        assert_eq!(std::fs::read(path)?, vec![1, 2, 3]);
        Ok(())
    }
}
