// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for DuckDB scans and pushdown.
//!
//! These tests exercise the registered extension through scans, spatial pushdown, and S3 reads.

#[cfg(test)]
use std::ffi::CStr;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::fs::OpenOptions;
#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
use parking_lot::Mutex;

#[cfg(test)]
mod s3_test;
#[cfg(all(test, duckdb_release))]
mod spatial_pushdown_test;
#[cfg(test)]
mod vortex_scan_test;

/// Stage DuckDB's `spatial` extension once into a synchronized cache directory and load it into
/// `conn`.
///
/// DuckDB's `INSTALL spatial` writes directly into the extension directory without cross-process
/// locking, so concurrent `nextest` processes (or threads under `cargo test`) race when they all
/// install into `~/.duckdb/extensions`. First try `LOAD spatial` against DuckDB's default
/// extension directory so pre-installed environments avoid redundant downloads; otherwise stage
/// into a user-scoped versioned directory guarded by an in-process mutex and an OS file lock.
#[cfg(test)]
pub(crate) fn load_spatial(conn: &crate::duckdb::Connection) {
    static PROCESS_LOCK: Mutex<()> = Mutex::new(());

    if conn.query("LOAD spatial;").is_ok() {
        return;
    }

    // SAFETY: `duckdb_library_version` returns a static NUL-terminated version string.
    let version: String = unsafe { CStr::from_ptr(crate::duckdb_library_version()) }
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();

    let cache_base = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| {
            PathBuf::from(home)
                .join(".duckdb")
                .join("vortex-test-extensions")
        })
        .unwrap_or_else(std::env::temp_dir);
    let ext_dir = cache_base.join(format!("vortex-duckdb-extensions-{version}"));
    fs::create_dir_all(&ext_dir).expect("create DuckDB extension cache directory");

    let sql_dir = ext_dir
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    conn.query(&format!("SET extension_directory = '{sql_dir}'"))
        .expect("set DuckDB extension_directory");

    let marker = ext_dir.join(".spatial.installed");
    if marker.exists() && conn.query("LOAD spatial;").is_ok() {
        return;
    }

    let _thread_guard = PROCESS_LOCK.lock();
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(ext_dir.join(".spatial.lock"))
        .expect("open DuckDB spatial install lock file");
    lock_file
        .lock()
        .expect("acquire DuckDB spatial install lock");

    if marker.exists() && conn.query("LOAD spatial;").is_ok() {
        return;
    }

    conn.query("FORCE INSTALL spatial;")
        .expect("install DuckDB spatial extension");
    fs::write(&marker, b"installed\n").expect("write DuckDB spatial install marker");
    conn.query("LOAD spatial;")
        .expect("load DuckDB spatial extension");
}
