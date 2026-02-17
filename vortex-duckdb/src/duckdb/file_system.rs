// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::IoBuf;
use vortex::io::VortexReadAt;
use vortex::io::VortexWrite;
use vortex::io::file::object_store;
use vortex::io::file::std_file;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::cpp;
use crate::duckdb::ClientContext;
use crate::lifetime_wrapper;

lifetime_wrapper!(FsFileHandle, cpp::duckdb_vx_file_handle, cpp::duckdb_vx_fs_close, [owned, ref]);
unsafe impl Send for FsFileHandle {}
unsafe impl Sync for FsFileHandle {}
fn fs_error(err: cpp::duckdb_vx_error) -> VortexError {
    if err.is_null() {
        return vortex_err!("DuckDB filesystem error (unknown)");
    }
    let message = unsafe { CStr::from_ptr(cpp::duckdb_vx_error_value(err)) }
        .to_string_lossy()
        .to_string();
    unsafe { cpp::duckdb_vx_error_free(err) };
    vortex_err!("{message}")
}

pub(crate) fn duckdb_fs_glob(ctx: &ClientContext, pattern: &str) -> VortexResult<Vec<url::Url>> {
    let c_pattern = CString::new(pattern).map_err(|e| vortex_err!("Invalid glob pattern: {e}"))?;
    let mut err: cpp::duckdb_vx_error = ptr::null_mut();
    let mut list =
        unsafe { cpp::duckdb_vx_fs_glob(ctx.as_ptr(), c_pattern.as_ptr(), &raw mut err) };
    if !err.is_null() {
        return Err(fs_error(err));
    }

    let mut urls = Vec::with_capacity(list.count);
    for idx in 0..list.count {
        let entry = unsafe { CStr::from_ptr(*list.entries.add(idx)) };
        let entry_str = entry.to_string_lossy();
        let url = match url::Url::parse(entry_str.as_ref()) {
            Ok(url) => url,
            Err(parse_err) => {
                let path = std::path::Path::new(entry_str.as_ref());
                let canonical = path
                    .canonicalize()
                    .map_err(|e| vortex_err!("Cannot canonicalize file path {path:?}: {e}"))?;
                url::Url::from_file_path(&canonical).map_err(|_| {
                    vortex_err!("Invalid URL returned by DuckDB glob {entry_str}: {parse_err}")
                })?
            }
        };
        urls.push(url);
    }

    unsafe { cpp::duckdb_vx_uri_list_free(&raw mut list) };

    Ok(urls)
}

pub(crate) unsafe fn duckdb_fs_create_writer(
    ctx: cpp::duckdb_vx_client_context,
    path: &str,
) -> VortexResult<DuckDbFsWriter> {
    unsafe { DuckDbFsWriter::create(ctx, path) }
}

/// A VortexReadAt implementation backed by DuckDB's filesystem (e.g., httpfs/s3).
pub(crate) struct DuckDbFsReader {
    handle: Arc<FsFileHandle>,
    uri: Arc<str>,
    size: Arc<OnceLock<u64>>,
    is_local: bool,
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
            return Err(fs_error(err));
        }

        let is_local = url.scheme() == "file";

        Ok(Self {
            handle: Arc::new(unsafe { FsFileHandle::own(handle) }),
            uri: Arc::from(url.as_str()),
            size: Arc::new(OnceLock::new()),
            is_local,
        })
    }
}

impl VortexReadAt for DuckDbFsReader {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        if self.is_local {
            Some(CoalesceConfig::local())
        } else {
            Some(CoalesceConfig::object_storage())
        }
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

pub(crate) struct DuckDbFsWriter {
    handle: Arc<FsFileHandle>,
    pos: u64,
}

impl DuckDbFsWriter {
    pub(crate) unsafe fn create(
        ctx: cpp::duckdb_vx_client_context,
        path: &str,
    ) -> VortexResult<Self> {
        let c_path = CString::new(path).map_err(|e| vortex_err!("Invalid path: {e}"))?;
        let mut err: cpp::duckdb_vx_error = ptr::null_mut();
        let handle = unsafe { cpp::duckdb_vx_fs_create(ctx, c_path.as_ptr(), &raw mut err) };
        if handle.is_null() {
            return Err(fs_error(err));
        }

        Ok(Self {
            handle: Arc::new(unsafe { FsFileHandle::own(handle) }),
            pos: 0,
        })
    }
}

impl VortexWrite for DuckDbFsWriter {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> std::io::Result<B> {
        let len = buffer.bytes_init();
        let offset = self.pos;
        let handle = self.handle.clone();
        // IoBuf is not bounded by Send, so it cannot be moved into a
        // spawn_blocking closure. Pass the pointer as usize (which is Send)
        // and keep `buffer` alive in this async fn until the blocking call
        // completes.
        let buf_ptr = buffer.read_ptr() as usize;

        let runtime = RUNTIME.handle();
        runtime
            .spawn_blocking(move || {
                let mut err: cpp::duckdb_vx_error = ptr::null_mut();
                let mut out_len: cpp::idx_t = 0;
                let status = unsafe {
                    cpp::duckdb_vx_fs_write(
                        handle.as_ptr(),
                        offset as cpp::idx_t,
                        len as cpp::idx_t,
                        buf_ptr as *mut u8,
                        &raw mut out_len,
                        &raw mut err,
                    )
                };

                if status != cpp::duckdb_state::DuckDBSuccess {
                    return Err(std::io::Error::other(fs_error(err).to_string()));
                }

                Ok(())
            })
            .await?;

        self.pos = offset + len as u64;
        Ok(buffer)
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        let handle = self.handle.clone();

        let runtime = RUNTIME.handle();
        runtime
            .spawn_blocking(move || {
                let mut err: cpp::duckdb_vx_error = ptr::null_mut();
                let status = unsafe { cpp::duckdb_vx_fs_sync(handle.as_ptr(), &raw mut err) };
                if status != cpp::duckdb_state::DuckDBSuccess {
                    return Err(std::io::Error::other(fs_error(err).to_string()));
                }
                Ok(())
            })
            .await
    }

    async fn shutdown(&mut self) -> std::io::Result<()> {
        self.flush().await
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::duckdb::Database;

    #[test]
    fn test_writer_roundtrip_local() {
        let db = Database::open_in_memory().unwrap();
        let conn = db.connect().unwrap();
        let ctx = conn.client_context().unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path: PathBuf = dir.path().join("writer_local.vortex");
        let path_str = path.to_string_lossy();

        let mut writer = unsafe { duckdb_fs_create_writer(ctx.as_ptr(), &path_str) }.unwrap();

        futures::executor::block_on(async {
            VortexWrite::write_all(&mut writer, vec![1_u8, 2, 3])
                .await
                .unwrap();
            VortexWrite::flush(&mut writer).await.unwrap();
        });

        let data = fs::read(path).unwrap();
        assert_eq!(data, vec![1, 2, 3]);
    }
}
