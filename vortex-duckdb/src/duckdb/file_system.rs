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

/// Expand a glob pattern against a filesystem, returning matching file URLs.
///
/// If the pattern contains no glob characters, it is treated as an exact path.
/// Otherwise, extracts the directory prefix before the first glob character,
/// recursively lists all files under that directory via [`duckdb_fs_list_dir`],
/// and filters the results using [`glob::Pattern`].
pub(crate) fn duckdb_fs_glob(ctx: &ClientContext, pattern: &str) -> VortexResult<Vec<url::Url>> {
    let has_glob = pattern.contains(['*', '?', '[']);

    // No glob characters: treat as an exact file path.
    if !has_glob {
        let url = path_to_url(pattern)?;
        return Ok(vec![url]);
    }

    let glob_pattern = glob::Pattern::new(pattern)
        .map_err(|e| vortex_err!("Invalid glob pattern '{pattern}': {e}"))?;

    // Find the directory prefix before the first glob character.
    let glob_pos = pattern.find(['*', '?', '[']).unwrap_or(pattern.len());
    let prefix = match pattern[..glob_pos].rfind('/') {
        Some(slash_pos) => &pattern[..=slash_pos],
        None => "",
    };

    let directory = if prefix.is_empty() {
        ".".to_string()
    } else {
        prefix.trim_end_matches('/').to_string()
    };

    let all_files = list_files_recursive(ctx, &directory)?;

    let mut urls = Vec::new();
    for full_path in all_files {
        if glob_pattern.matches(&full_path) {
            urls.push(path_to_url(&full_path)?);
        }
    }

    Ok(urls)
}

/// Convert a path string to a [`url::Url`], canonicalizing local paths.
fn path_to_url(path: &str) -> VortexResult<url::Url> {
    match url::Url::parse(path) {
        Ok(url) => Ok(url),
        Err(_) => {
            let p = std::path::Path::new(path);
            let canonical = p
                .canonicalize()
                .map_err(|e| vortex_err!("Cannot canonicalize file path {p:?}: {e}"))?;
            url::Url::from_file_path(&canonical)
                .map_err(|_| vortex_err!("Cannot convert path to URL: {path}"))
        }
    }
}

/// Recursively list all file paths under `directory`.
fn list_files_recursive(ctx: &ClientContext, directory: &str) -> VortexResult<Vec<String>> {
    let mut results = Vec::new();
    let mut stack = vec![directory.to_string()];

    while let Some(dir) = stack.pop() {
        for entry in duckdb_fs_list_dir(ctx, &dir)? {
            let full_path = format!("{}/{}", dir.trim_end_matches('/'), entry.name);
            if entry.is_dir {
                stack.push(full_path);
            } else {
                results.push(full_path);
            }
        }
    }

    Ok(results)
}

/// An entry returned by [`duckdb_fs_list_dir`].
pub(crate) struct DirEntry {
    pub name: String,
    pub is_dir: bool,
}

/// Non-recursively list entries in `directory` via DuckDB's filesystem.
///
/// Returns file and subdirectory names (not full paths). The caller is
/// responsible for joining paths and recursing into subdirectories.
pub(crate) fn duckdb_fs_list_dir(
    ctx: &ClientContext,
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
