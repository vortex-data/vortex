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
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use url::Url;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileListing;
use vortex::io::filesystem::FileSystem;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::cpp;
use crate::duckdb::ClientContextRef;
use crate::duckdb::FsFileHandle;
use crate::duckdb::duckdb_fs_list_dir;
use crate::duckdb::fs_error;

pub(super) fn resolve_filesystem(
    base_url: &Url,
    ctx: &ClientContextRef,
) -> VortexResult<FileSystemRef> {
    let fs_config = ctx
        .try_get_current_setting(c"vortex_filesystem")
        .ok_or_else(|| {
            vortex_err!("Failed to read 'vortex_filesystem' setting from DuckDB config")
        })?;
    let fs_config = fs_config.as_string();

    Ok(if fs_config.as_str() == "duckdb" {
        tracing::debug!(
            "Using DuckDB's built-in filesystem for URL scheme '{}'",
            base_url.scheme()
        );
        // SAFETY: The ClientContext is owned by the Connection and lives for the duration of
        // query execution. DuckDB keeps the connection alive while the filesystem is in use.
        Arc::new(DuckDbFileSystem::new(base_url.clone(), unsafe {
            ctx.erase_lifetime()
        }))
    } else if fs_config.as_str() == "vortex" {
        tracing::debug!(
            "Using Vortex's object store filesystem for URL scheme '{}'",
            base_url.scheme()
        );
        object_store_fs(base_url)?
    } else {
        vortex_bail!(
            "Unsupported filesystem '{}', vortex_filesystem setting must be set to either 'duckdb' or 'vortex'",
            fs_config.as_str()
        );
    })
}

fn object_store_fs(base_url: &Url) -> VortexResult<FileSystemRef> {
    let object_store: Arc<dyn ObjectStore> = if base_url.scheme() == "file" {
        Arc::new(LocalFileSystem::new())
    } else if base_url.scheme() == "s3" {
        Arc::new(
            AmazonS3Builder::from_env()
                .with_bucket_name(base_url.host_str().ok_or_else(|| {
                    vortex_err!("Failed to extract bucket name from URL: {base_url}")
                })?)
                .build()?,
        )
    } else {
        vortex_bail!(
            "Unsupported URL scheme '{}', only 'file' and 's3' are supported with vortex_filesystem='vortex'",
            base_url.scheme()
        );
    };

    let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;

    Ok(Arc::new(ObjectStoreFileSystem::new(
        object_store,
        RUNTIME.handle(),
    )))
}

struct DuckDbFileSystem {
    base_url: Url,
    ctx: &'static ClientContextRef,
}

impl Debug for DuckDbFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDbFileSystem")
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl DuckDbFileSystem {
    pub fn new(base_url: Url, ctx: &'static ClientContextRef) -> Self {
        Self { base_url, ctx }
    }
}

#[async_trait]
impl FileSystem for DuckDbFileSystem {
    fn list(&self, prefix: &str) -> BoxStream<'_, VortexResult<FileListing>> {
        let mut directory_url = self.base_url.clone();
        if !prefix.is_empty() {
            directory_url.set_path(prefix);
        }

        let ctx = self.ctx;

        let base_url = self.base_url.clone();
        stream::once(async move {
            RUNTIME
                .handle()
                .spawn_blocking(move || list_recursive(ctx, &directory_url, &base_url))
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
        url.set_path(path);
        let reader = unsafe { DuckDbFsReader::open_url(self.ctx.as_ptr(), &url)? };
        Ok(Arc::new(reader))
    }

    async fn delete(&self, path: &str) -> VortexResult<()> {
        let mut url = self.base_url.clone();
        url.set_path(path);
        let c_path = CString::new(url.as_str()).map_err(|e| vortex_err!("Invalid URL: {e}"))?;
        let ctx = self.ctx;

        RUNTIME
            .handle()
            .spawn_blocking(move || {
                let mut err: cpp::duckdb_vx_error = ptr::null_mut();
                let status = unsafe {
                    cpp::duckdb_vx_fs_remove(ctx.as_ptr(), c_path.as_ptr(), &raw mut err)
                };
                if status != cpp::duckdb_state::DuckDBSuccess {
                    return Err(fs_error(err));
                }
                Ok::<_, VortexError>(())
            })
            .await
    }
}

/// Recursively list all files under `directory`, stripping `base_path` from each
/// returned URL to produce relative paths.
fn list_recursive(
    ctx: &ClientContextRef,
    directory_url: &Url,
    base_url: &Url,
) -> VortexResult<Vec<FileListing>> {
    // DuckDB's ListFiles expects bare paths for local files, but full URLs
    // for remote schemes (s3://, etc.).
    let directory = if directory_url.scheme() == "file" {
        directory_url.path().to_string()
    } else {
        directory_url.to_string()
    };

    let (base_path, is_remote_path) = if base_url.scheme() == "file" {
        (base_url.path().to_string(), false)
    } else {
        // This is really ugly. As we operate on Strings and not on urls, we
        // must produce a base path with / so as relative url would not have
        // the / and thus match the glob
        (format!("{base_url}/"), true)
    };

    let mut results = Vec::new();
    let mut stack = vec![directory];

    while let Some(dir) = stack.pop() {
        // TODO(myrrc) this doesn't work with curl backend in v1.4, producing
        // "URL using bad/illegal format or missing URL error", see
        // https://github.com/duckdb/duckdb-httpfs/pull/265
        for entry in duckdb_fs_list_dir(ctx, &dir)? {
            // duckdb_fs_list_dir returns relative paths for local files but full
            // paths for s3 files.
            let full_path = if is_remote_path {
                entry.name
            } else {
                format!("{}/{}", dir.trim_end_matches('/'), entry.name)
            };
            if entry.is_dir {
                stack.push(full_path);
            } else {
                let relative_path = full_path
                    .strip_prefix(&base_path)
                    .unwrap_or_else(|| &full_path)
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
        ctx: cpp::duckdb_client_context,
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
            CoalesceConfig::file()
        } else {
            CoalesceConfig::object_storage()
        })
    }

    fn concurrency(&self) -> usize {
        if self.is_local {
            vortex::io::std_file::DEFAULT_CONCURRENCY
        } else {
            vortex::io::object_store::DEFAULT_CONCURRENCY
        }
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let handle = Arc::clone(&self.handle);
        let size_cell = Arc::clone(&self.size);

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
        let handle = Arc::clone(&self.handle);

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
