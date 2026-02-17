// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use url::Url;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::filesystem::FileListing;
use vortex::file::filesystem::FileSystem;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Handle;

use crate::RUNTIME;
use crate::cpp;
use crate::duckdb::ClientContext;
use crate::duckdb::FsFileHandle;
use crate::duckdb::duckdb_fs_glob;

pub struct DuckDbFileSystem {
    base_url: Url,
    ctx: ClientContext,
    handle: Handle,
}

impl DuckDbFileSystem {
    pub fn new(base_url: Url, ctx: ClientContext) -> Self {
        Self {
            base_url,
            ctx,
            handle: RUNTIME.handle(),
        }
    }
}

#[async_trait]
impl FileSystem for DuckDbFileSystem {
    fn list(&self, prefix: Option<&str>) -> BoxStream<'_, VortexResult<FileListing>> {
        let pattern = if let Some(prefix) = prefix {
            // Ensure the prefix is properly joined to the base URL
            let mut joined_url = self.base_url.clone();
            joined_url.set_path(&format!("{}/{}", joined_url.path(), prefix));
        } else {
            self.base_url.to_string();
        };
        stream::iter(
            duckdb_fs_glob(&self.ctx, prefix.unwrap_or(""))?
                .into_iter()
                .map(|url| {
                    // Create path relative to the base URL.
                    let relative_path = url
                        .path()
                        .strip_prefix(self.base_url.path())
                        .unwrap_or(url.path());
                    Ok(FileListing {
                        path: relative_path.to_string(),
                        size: None,
                    })
                }),
        )
        .boxed()
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        todo!()
    }
}

/// A VortexReadAt implementation backed by DuckDB's filesystem (e.g., httpfs/s3).
pub(crate) struct DuckDbFsReader {
    handle: Arc<FsFileHandle>,
    uri: Arc<str>,
    size: Arc<OnceLock<u64>>,
}

impl DuckDbFsReader {
    pub(crate) unsafe fn open_url(
        ctx: cpp::duckdb_vx_client_context,
        url: &url::Url,
    ) -> VortexResult<Self> {
        let c_path = CString::new(url.as_str()).map_err(|e| vortex_err!("Invalid URL: {e}"))?;
        let mut err: cpp::duckdb_vx_error = ptr::null_mut();
        let handle = unsafe { cpp::duckdb_vx_fs_open(ctx, c_path.as_ptr(), &raw mut err) };
        if handle.is_null() {
            return Err(crate::duckdb::file_system::fs_error(err));
        }

        Ok(Self {
            handle: Arc::new(unsafe { FsFileHandle::own(handle) }),
            uri: Arc::from(url.as_str()),
            size: Arc::new(OnceLock::new()),
        })
    }
}

impl VortexReadAt for DuckDbFsReader {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(crate::duckdb::file_system::DEFAULT_COALESCE)
    }

    fn concurrency(&self) -> usize {
        crate::duckdb::file_system::DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let handle = self.handle.clone();
        let size_cell = self.size.clone();

        async move {
            if let Some(size) = size_cell.get() {
                return Ok(*size);
            }

            let runtime = RUNTIME.handle();
            let size = runtime
                .spawn_blocking(move || {
                    let mut err: cpp::duckdb_vx_error = ptr::null_mut();
                    let mut size_out: cpp::idx_t = 0;
                    let status = unsafe {
                        cpp::duckdb_vx_fs_get_size(handle.as_ptr(), &raw mut size_out, &raw mut err)
                    };
                    if status != cpp::duckdb_state::DuckDBSuccess {
                        return Err(crate::duckdb::file_system::fs_error(err));
                    }
                    Ok::<_, VortexError>(size_out as u64)
                })
                .await?;

            let _ = size_cell.set(size);
            Ok(size)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let handle = self.handle.clone();

        async move {
            let runtime = RUNTIME.handle();
            let result: VortexResult<BufferHandle> = runtime
                .spawn_blocking(move || -> VortexResult<BufferHandle> {
                    let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
                    unsafe { buffer.set_len(length) };

                    let mut err: cpp::duckdb_vx_error = ptr::null_mut();
                    let mut out_len: cpp::idx_t = 0;
                    let status = unsafe {
                        cpp::duckdb_vx_fs_read(
                            handle.as_ptr(),
                            offset as cpp::idx_t,
                            length as cpp::idx_t,
                            buffer.as_mut_slice().as_mut_ptr(),
                            &raw mut out_len,
                            &raw mut err,
                        )
                    };

                    if status != cpp::duckdb_state::DuckDBSuccess {
                        return Err(crate::duckdb::file_system::fs_error(err));
                    }

                    let used = usize::try_from(out_len)
                        .map_err(|e| vortex_err!("Invalid read len: {e}"))?;
                    unsafe { buffer.set_len(used) };

                    let frozen = buffer.freeze();
                    Ok::<_, VortexError>(BufferHandle::new_host(frozen))
                })
                .await;
            result
        }
        .boxed()
    }
}

// SAFETY: DuckDB file handles can be used across threads when operations are position-based. The
// C++ bridge opens handles with FILE_FLAGS_PARALLEL_ACCESS, and writes use explicit offsets, so
// there is no shared cursor state.
unsafe impl Send for DuckDbFsReader {}
unsafe impl Sync for DuckDbFsReader {}
