// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use cpp::duckdb_database;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::Config;
use crate::duckdb::connection::Connection;
use crate::duckdb_try;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// A DuckDB database instance.
    Database,
    duckdb_database,
    cpp::duckdb_close
);

impl Database {
    /// Creates a new DuckDB database instance in memory.
    pub fn open_in_memory() -> VortexResult<Self> {
        let mut ptr: duckdb_database = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_open(ptr::null(), &raw mut ptr) },
            "Failed to open in-memory DuckDB database"
        );
        Ok(unsafe { Self::own(ptr) })
    }

    /// Opens a DuckDB database from a file path.
    ///
    /// Creates a new file in case the path does not exist.
    pub fn open<P: AsRef<Path>>(path: P) -> VortexResult<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| vortex_err!("Invalid path: path contains non-UTF8 characters"))?;
        let path_cstr = CString::new(path_str)
            .map_err(|_| vortex_err!("Invalid path: path contains null bytes"))?;

        let mut ptr: duckdb_database = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_open(path_cstr.as_ptr(), &raw mut ptr) },
            "Failed to open DuckDB database at path: {}",
            path_str
        );
        Ok(unsafe { Self::own(ptr) })
    }

    /// Opens a DuckDB database from a file path with custom configuration.
    ///
    /// Creates a new file in case the path does not exist.
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: Config) -> VortexResult<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| vortex_err!("Invalid path: path contains non-UTF8 characters"))?;
        let path_cstr = CString::new(path_str)
            .map_err(|_| vortex_err!("Invalid path: path contains null bytes"))?;

        let mut ptr: duckdb_database = ptr::null_mut();
        let mut error: *mut c_char = ptr::null_mut();

        // duckdb_open_ext borrows the config (copies it internally), so we pass as_ptr()
        // and let the Config drop naturally at the end of this function.
        let result = unsafe {
            cpp::duckdb_open_ext(
                path_cstr.as_ptr(),
                &raw mut ptr,
                config.as_ptr(),
                &raw mut error,
            )
        };

        if result != cpp::duckdb_state::DuckDBSuccess {
            if !error.is_null() {
                let error_msg = unsafe {
                    std::ffi::CStr::from_ptr(error)
                        .to_string_lossy()
                        .to_string()
                };
                unsafe { cpp::duckdb_free(error.cast()) };
                vortex_bail!(
                    "Failed to open DuckDB database at path '{}': {}",
                    path_str,
                    error_msg
                );
            } else {
                vortex_bail!("Failed to open DuckDB database at path: {}", path_str);
            }
        }

        Ok(unsafe { Self::own(ptr) })
    }

    /// Opens an in-memory DuckDB database with custom configuration.
    pub fn open_in_memory_with_config(config: Config) -> VortexResult<Self> {
        let mut ptr: duckdb_database = ptr::null_mut();
        let mut error: *mut c_char = ptr::null_mut();

        // duckdb_open_ext borrows the config (copies it internally), so we pass as_ptr()
        // and let the Config drop naturally at the end of this function.
        let result = unsafe {
            cpp::duckdb_open_ext(ptr::null(), &raw mut ptr, config.as_ptr(), &raw mut error)
        };

        if result != cpp::duckdb_state::DuckDBSuccess {
            if !error.is_null() {
                let error_msg = unsafe {
                    std::ffi::CStr::from_ptr(error)
                        .to_string_lossy()
                        .to_string()
                };
                unsafe { cpp::duckdb_free(error.cast()) };
                vortex_bail!("Failed to open in-memory DuckDB database: {}", error_msg);
            } else {
                vortex_bail!("Failed to open in-memory DuckDB database");
            }
        }

        Ok(unsafe { Self::own(ptr) })
    }
}

impl DatabaseRef {
    /// Connects to the DuckDB database.
    pub fn connect(&self) -> VortexResult<Connection> {
        Connection::connect(self)
    }

    pub fn register_vortex_scan_replacement(&self) -> VortexResult<()> {
        duckdb_try!(
            unsafe { cpp::duckdb_vx_register_scan_replacement(self.as_ptr()) },
            "Failed to register vortex scan replacement"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_with_config() {
        let mut config = Config::new().unwrap();
        config.set("memory_limit", "512MB").unwrap();
        config.set("threads", "1").unwrap();

        let db = Database::open_in_memory_with_config(config);
        assert!(db.is_ok());

        let conn = db.unwrap().connect();
        assert!(conn.is_ok());
    }

    #[test]
    fn test_file_database_with_config() {
        let mut config = Config::new().unwrap();
        config.set("memory_limit", "256MB").unwrap();

        let db = Database::open_with_config("test_config_unit.db", config);
        assert!(db.is_ok());

        let conn = db.unwrap().connect();
        assert!(conn.is_ok());

        std::fs::remove_file("test_config_unit.db").ok();
    }
}
