// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use vortex::error::{VortexResult, vortex_err};

use crate::duckdb::connection::Connection;
use crate::{cpp, duckdb_try, wrapper};

wrapper!(
    /// A DuckDB database instance.
    Database,
    cpp::duckdb_database,
    cpp::duckdb_close
);

impl Database {
    /// Creates a new DuckDB database instance in memory.
    pub fn open_in_memory() -> VortexResult<Self> {
        let mut ptr: cpp::duckdb_database = ptr::null_mut();
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

        let mut ptr: cpp::duckdb_database = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_open(path_cstr.as_ptr(), &raw mut ptr) },
            "Failed to open DuckDB database at path: {}",
            path_str
        );
        Ok(unsafe { Self::own(ptr) })
    }

    /// Connects to the DuckDB database.
    pub fn connect(&self) -> VortexResult<Connection> {
        Connection::connect(self)
    }
}
