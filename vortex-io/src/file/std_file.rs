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
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::WriteTarget;
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

const DEFAULT_COALESCING_CONFIG: CoalesceConfig = CoalesceConfig {
    distance: 8 * 1024, // 8KB
    max_size: 8 * 1024, // 8KB
};
const DEFAULT_CONCURRENCY: usize = 32;

const COALESCE_DISTANCE_ENV: &str = "VORTEX_FILE_COALESCE_DISTANCE";
const COALESCE_MAX_SIZE_ENV: &str = "VORTEX_FILE_COALESCE_MAX_SIZE";
const COALESCE_DISABLE_ENV: &str = "VORTEX_FILE_COALESCE_DISABLE";
const READ_CONCURRENCY_ENV: &str = "VORTEX_FILE_READ_CONCURRENCY";

fn read_env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse::<u64>().ok()
}

fn read_env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse::<usize>().ok()
}

fn read_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .map(|value| value != 0)
        .unwrap_or(default)
}

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAdapter {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl FileReadAdapter {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(path)?);

        let mut coalesce_config = Some(DEFAULT_COALESCING_CONFIG);
        if read_env_bool(COALESCE_DISABLE_ENV, false) {
            coalesce_config = None;
        } else if let Some(defaults) = coalesce_config.as_mut() {
            if let Some(distance) = read_env_u64(COALESCE_DISTANCE_ENV) {
                defaults.distance = distance;
            }
            if let Some(max_size) = read_env_u64(COALESCE_MAX_SIZE_ENV) {
                defaults.max_size = max_size;
            }
        }

        let concurrency = read_env_usize(READ_CONCURRENCY_ENV).unwrap_or(DEFAULT_CONCURRENCY);

        Ok(Self {
            uri,
            file,
            handle,
            concurrency,
            coalesce_config,
        })
    }
}

impl VortexReadAt for FileReadAdapter {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = self.file.clone();
        async move {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
        unsafe { buffer.set_len(length) };
        let target: Box<dyn WriteTarget> = Box::new(buffer);
        self.read_at_into(offset, target)
    }

    fn read_at_into(
        &self,
        offset: u64,
        mut target: Box<dyn WriteTarget>,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = self.file.clone();
        let handle = self.handle.clone();
        async move {
            handle
                .spawn_blocking(move || {
                    read_exact_at(&file, target.as_mut_slice(), offset)?;
                    target.into_handle()
                })
                .await
        }
        .boxed()
    }
}
