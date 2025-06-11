use std::ptr;

use vortex::error::VortexResult;

use crate::duckdb::Database;
use crate::{cpp, duckdb_try, wrapper};

wrapper!(
    /// A DuckDB connection.
    Connection,
    cpp::duckdb_connection,
    cpp::duckdb_disconnect
);

impl Connection {
    pub fn connect(db: &Database) -> VortexResult<Self> {
        let mut ptr: cpp::duckdb_connection = ptr::null_mut();
        duckdb_try!(
            unsafe { cpp::duckdb_connect(db.as_ptr(), &raw mut ptr) },
            "Failed to connect to DuckDB database"
        );
        Ok(unsafe { Self::own(ptr) })
    }
}
