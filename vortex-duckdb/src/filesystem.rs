// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use url::Url;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::filesystem::FileListing;
use vortex::file::filesystem::FileSystem;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::file::object_store;
use vortex::io::file::std_file;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::cpp;
use crate::duckdb::ClientContext;
use crate::duckdb::FsFileHandle;
use crate::duckdb::duckdb_fs_list_dir;
use crate::duckdb::fs_error;

pub struct DuckDbFileSystem {
    base_url: Url,
    ctx: ClientContext,
}

impl Debug for DuckDbFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDbFileSystem")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl DuckDbFileSystem {
    pub fn new(base_url: Url, ctx: ClientContext) -> Self {
        Self { base_url, ctx }
    }
}

#[async_trait]
impl FileSystem for DuckDbFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let mut directory_url = self.base_url.clone();
        if !prefix.is_empty() {
            directory_url.set_path(&format!("{}/{}", directory_url.path(), prefix));
        }
        let directory = directory_url.to_string();

        tracing::debug!(
            "Listing files from {} with prefix {:?} using directory {}",
            self.base_url,
            prefix,
            directory
        );

        let ctx = self.ctx.clone();
        let base_path = self.base_url.path().to_string();

        stream::once(async move {
            RUNTIME
                .handle()
                .spawn_blocking(move || list_recursive(&ctx, &directory, &base_path))
                .await
        })
        .flat_map(|result| match result {
            Ok(listings) => stream::iter(listings.into_iter().map(Ok)).boxed(),
            Err(e) => stream::once(async move { Err(e) }).boxed(),
        })
        .boxed()
    }

    async fn open_read(&self, path: &str) -> VortexResult<Arc<dyn VortexReadAt>> {
        let mut url = self.base_url.clone();
        url.set_path(&format!("{}/{}", url.path(), path));
        let reader = unsafe { DuckDbFsReader::open_url(self.ctx.as_ptr(), &url)? };
        Ok(Arc::new(reader))
    }
}

/// Recursively list all files under `directory`, stripping `base_path` from each
/// returned URL to produce relative paths.
fn list_recursive(
    ctx: &ClientContext,
    directory: &str,
    base_path: &str,
) -> VortexResult<Vec<FileListing>> {
    let mut results = Vec::new();
    let mut stack = vec![directory.to_string()];

    while let Some(dir) = stack.pop() {
        for entry in duckdb_fs_list_dir(ctx, &dir)? {
            let full_path = format!("{}/{}", dir.trim_end_matches('/'), entry.name);
            if entry.is_dir {
                stack.push(full_path);
            } else {
                let url = match Url::parse(&full_path) {
                    Ok(url) => url,
                    Err(_) => {
                        let path = std::path::Path::new(&full_path);
                        let canonical = path.canonicalize().map_err(|e| {
                            vortex_err!("Cannot canonicalize file path {path:?}: {e}")
                        })?;
                        Url::from_file_path(&canonical)
                            .map_err(|_| vortex_err!("Cannot convert path to URL: {full_path}"))?
                    }
                };
                let relative_path = url
                    .path()
                    .strip_prefix(base_path)
                    .unwrap_or_else(|| url.path())
                    .to_string();
                results.push(FileListing {
                    path: relative_path,
                    size: None,
                });
            }
        }
    }

    Ok(results)
}

/// A VortexReadAt implementation backed by DuckDB's filesystem (e.g., httpfs/s3).
pub(crate) struct DuckDbFsReader {
    handle: Arc<FsFileHandle>,
    uri: Arc<str>,
    is_local: bool,
    size: Arc<OnceLock<u64>>,
}

impl DuckDbFsReader {
    pub(crate) unsafe fn open_url(
        ctx: cpp::duckdb_vx_client_context,
        url: &Url,
    ) -> VortexResult<Self> {
        let c_path = CString::new(url.as_str()).map_err(|e| vortex_err!("Invalid URL: {e}"))?;
        let mut err: cpp::duckdb_vx_error = ptr::null_mut();
        let handle = unsafe { cpp::duckdb_vx_fs_open(ctx, c_path.as_ptr(), &raw mut err) };
        if handle.is_null() {
            return Err(fs_error(err));
        }

        let is_local = url.scheme() == "file";

        Ok(Self {
            handle: Arc::new(unsafe { FsFileHandle::own(handle) }),
            uri: Arc::from(url.as_str()),
            is_local,
            size: Arc::new(OnceLock::new()),
        })
    }
}

impl VortexReadAt for DuckDbFsReader {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(if self.is_local {
            CoalesceConfig::local()
        } else {
            CoalesceConfig::object_storage()
        })
    }

    fn concurrency(&self) -> usize {
        if self.is_local {
            std_file::DEFAULT_CONCURRENCY
        } else {
            object_store::DEFAULT_CONCURRENCY
        }
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
                        return Err(fs_error(err));
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
                        return Err(fs_error(err));
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
