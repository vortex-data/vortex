// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::IoBuf;
use vortex::io::VortexWrite;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::cpp;
use crate::duckdb::ClientContext;
use crate::lifetime_wrapper;

lifetime_wrapper!(FsFileHandle, cpp::duckdb_vx_file_handle, cpp::duckdb_vx_fs_close, [owned, ref]);
unsafe impl Send for FsFileHandle {}
unsafe impl Sync for FsFileHandle {}

pub(crate) fn fs_error(err: cpp::duckdb_vx_error) -> VortexError {
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
