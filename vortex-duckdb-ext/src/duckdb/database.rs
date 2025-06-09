use std::ptr;

use vortex::error::VortexResult;

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
            unsafe { cpp::duckdb_open(ptr::null(), &mut ptr) },
            "Failed to open in-memory DuckDB database"
        );
        Ok(unsafe { Self::own(ptr) })
    }

    /// Connects to the DuckDB database.
    pub fn connect(&self) -> VortexResult<Connection> {
        Connection::connect(self)
    }
}
