// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::IoBuf;
use vortex::io::VortexWrite;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::cpp;
use crate::duckdb::ClientContextRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    FsFileHandle,
    cpp::duckdb_vx_file_handle,
    cpp::duckdb_vx_fs_close
);
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

/// An entry returned by [`duckdb_fs_list_dir`].
pub(crate) struct DirEntry {
    /// Full path for S3 files, relative path for local files
    pub name: String,
    pub is_dir: bool,
}

/// Non-recursively list entries in `directory` via DuckDB's filesystem.
///
/// Returns full paths. The caller is responsible for joining paths and
/// recursing into subdirectories.
pub(crate) fn duckdb_fs_list_dir(
    ctx: &ClientContextRef,
    directory: &str,
) -> VortexResult<Vec<DirEntry>> {
    let c_directory =
        CString::new(directory).map_err(|e| vortex_err!("Invalid directory path: {e}"))?;

    let mut entries: Vec<DirEntry> = Vec::new();
    let mut err: cpp::duckdb_vx_error = ptr::null_mut();

    let status = unsafe {
        cpp::duckdb_vx_fs_list_files(
            ctx.as_ptr(),
            c_directory.as_ptr(),
            Some(list_files_callback),
            (&raw mut entries).cast(),
            &raw mut err,
        )
    };

    if status != cpp::duckdb_state::DuckDBSuccess {
        return Err(fs_error(err));
    }

    Ok(entries)
}

/// FFI callback invoked by `duckdb_vx_fs_list_files` for each directory entry.
unsafe extern "C-unwind" fn list_files_callback(
    name: *const std::ffi::c_char,
    is_dir: bool,
    user_data: *mut std::ffi::c_void,
) {
    let entries = unsafe { &mut *user_data.cast::<Vec<DirEntry>>() };
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    entries.push(DirEntry { name, is_dir });
}

pub(crate) struct DuckDbFsWriter {
    handle: Arc<FsFileHandle>,
    pos: u64,
}

impl DuckDbFsWriter {
    pub(crate) fn new(ctx: &ClientContextRef, path: &str) -> VortexResult<Self> {
        let c_path = CString::new(path).map_err(|e| vortex_err!("Invalid path: {e}"))?;
        let mut err: cpp::duckdb_vx_error = ptr::null_mut();
        let file_handle =
            unsafe { cpp::duckdb_vx_fs_create(ctx.as_ptr(), c_path.as_ptr(), &raw mut err) };
        if file_handle.is_null() {
            return Err(fs_error(err));
        }

        Ok(Self {
            handle: Arc::new(unsafe { FsFileHandle::own(file_handle) }),
            pos: 0,
        })
    }
}

impl VortexWrite for DuckDbFsWriter {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> std::io::Result<B> {
        let len = buffer.bytes_init();
        let offset = self.pos;
        let handle = Arc::clone(&self.handle);

        let runtime = RUNTIME.handle();
        let buffer = runtime
            .spawn_blocking(move || {
                let mut err: cpp::duckdb_vx_error = ptr::null_mut();
                let mut out_len: cpp::idx_t = 0;
                let status = unsafe {
                    cpp::duckdb_vx_fs_write(
                        handle.as_ptr(),
                        offset as cpp::idx_t,
                        len as cpp::idx_t,
                        buffer.read_ptr() as *mut u8,
                        &raw mut out_len,
                        &raw mut err,
                    )
                };

                if status != cpp::duckdb_state::DuckDBSuccess {
                    return Err(std::io::Error::other(fs_error(err).to_string()));
                }

                Ok(buffer)
            })
            .await?;

        self.pos = offset + len as u64;
        Ok(buffer)
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        let handle = Arc::clone(&self.handle);

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

        let mut writer = DuckDbFsWriter::new(ctx, &path_str).unwrap();

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
