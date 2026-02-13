// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
#[cfg(all(not(unix), not(windows)))]
use std::io::Read;
#[cfg(all(not(unix), not(windows)))]
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::ReadInto;
use crate::WriteTarget;
use crate::read_at::AllocatingReader;
use crate::runtime::Handle;

/// Read exactly `buffer.len()` bytes from `file` starting at `offset`.
/// This is a platform-specific helper that uses the most efficient method available.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut bytes_read = 0;
        while bytes_read < buffer.len() {
            let read = file.seek_read(&mut buffer[bytes_read..], offset + bytes_read as u64)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            bytes_read += read;
        }
        Ok(())
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        use std::io::SeekFrom;
        let mut file_ref = file;
        file_ref.seek(SeekFrom::Start(offset))?;
        file_ref.read_exact(buffer)
    }
}

const COALESCING_CONFIG: CoalesceConfig = CoalesceConfig {
    distance: 1024 * 1024,     // 1MB
    max_size: 4 * 1024 * 1024, // 4MB
};
/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// Low-level file reader that implements [`ReadInto`].
pub struct FileReader {
    file: Arc<File>,
    handle: Handle,
}

impl FileReader {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        let file = Arc::new(File::open(path.as_ref())?);
        Ok(Self { file, handle })
    }
}

impl ReadInto for FileReader {
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn read_into(
        &self,
        mut target: Box<dyn WriteTarget>,
        offset: u64,
    ) -> BoxFuture<'static, VortexResult<Box<dyn WriteTarget>>> {
        let file = self.file.clone();
        let handle = self.handle.clone();
        async move {
            handle
                .spawn_blocking(move || {
                    read_exact_at(&file, target.as_mut_slice(), offset)?;
                    Ok(target)
                })
                .await
        }
        .boxed()
    }
}

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`](crate::VortexReadAt).
///
/// This is a convenience alias for [`AllocatingReader<FileReader>`] using the default allocator.
pub type FileReadAt = AllocatingReader<FileReader>;

impl FileReadAt {
    /// Open a file for reading with the default allocator.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri: Arc<str> = path.to_string_lossy().to_string().into();
        let reader = FileReader::open(path, handle)?;
        Ok(
            AllocatingReader::with_default_allocator(reader, DEFAULT_CONCURRENCY)
                .with_uri(uri)
                .with_coalesce_config(COALESCING_CONFIG),
        )
    }
}
