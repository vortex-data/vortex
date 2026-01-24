// Allow std::sync::Mutex for this standalone prototype (no parking_lot dependency)
#![allow(clippy::disallowed_types)]

//! DuckDB connection management.
//!
//! While DuckDB supports concurrent operations at the database level, the Rust
//! `Connection` type uses `RefCell` internally (for statement caching) which
//! is not `Sync`. To share the connection across Leptos async contexts, we
//! wrap it in a `Mutex`.
//!
//! For this prototype with light load, this is sufficient. For production,
//! consider using a connection pool or multiple connections.

use std::sync::Arc;
use std::sync::Mutex;

use duckdb::Connection;
use duckdb::Result as DuckResult;

/// Shared DuckDB connection for Leptos server functions.
///
/// Uses `Arc<Mutex<Connection>>` because:
/// - `Arc` allows cheap cloning for sharing across contexts
/// - `Mutex` is required because `Connection` contains `RefCell` (not `Sync`)
///
/// # Example
///
/// ```ignore
/// let db = DbPool::new_with_mock_data()?;
/// let data = db.query(|conn| {
///     // Use conn to execute queries
/// })?;
/// ```
#[derive(Clone)]
pub struct DbPool {
    conn: Arc<Mutex<Connection>>,
}

impl DbPool {
    /// Creates a new in-memory DuckDB database with the benchmark schema.
    ///
    /// The schema includes:
    /// - `commits` table for git commit metadata
    /// - `random_access` table for benchmark results
    pub fn new_in_memory() -> DuckResult<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Creates a new database populated with mock benchmark data.
    ///
    /// Useful for development and testing. Generates 100 commits over 100 days
    /// with realistic benchmark patterns including gradual improvements and
    /// occasional regressions.
    pub fn new_with_mock_data() -> DuckResult<Self> {
        let pool = Self::new_in_memory()?;
        crate::mock_data::populate_mock_data(&pool)?;
        Ok(pool)
    }

    /// Initializes the database schema.
    fn init_schema(conn: &Connection) -> DuckResult<()> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS commits (
                commit_hash VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                message VARCHAR,
                author VARCHAR
            );

            CREATE INDEX IF NOT EXISTS idx_commits_timestamp
                ON commits(timestamp DESC);

            CREATE TABLE IF NOT EXISTS random_access (
                commit_hash VARCHAR NOT NULL REFERENCES commits(commit_hash),
                vortex_ns UBIGINT,
                parquet_ns UBIGINT,
                lance_ns UBIGINT,
                PRIMARY KEY (commit_hash)
            );
        "#,
        )?;
        Ok(())
    }

    /// Executes a read query against the database.
    ///
    /// Acquires the mutex lock for the duration of the closure.
    pub fn query<T, F>(&self, f: F) -> DuckResult<T>
    where
        F: FnOnce(&Connection) -> DuckResult<T>,
    {
        let conn = self.conn.lock().expect("connection lock poisoned");
        f(&conn)
    }

    /// Executes a write operation against the database.
    ///
    /// Acquires the mutex lock for the duration of the closure.
    pub fn execute<F>(&self, f: F) -> DuckResult<()>
    where
        F: FnOnce(&Connection) -> DuckResult<()>,
    {
        let conn = self.conn.lock().expect("connection lock poisoned");
        f(&conn)
    }
}
