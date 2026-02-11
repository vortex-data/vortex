// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion_sqllogictest::DFColumnType;
use indicatif::ProgressBar;
use sqllogictest::DBOutput;
use sqllogictest::runner::AsyncDB;
use vortex::error::VortexError;
use vortex_duckdb::LogicalType;
use vortex_duckdb::Value;
use vortex_duckdb::duckdb::Connection;
use vortex_duckdb::duckdb::Database;
use vortex_duckdb::register_table_functions;

#[derive(Debug, thiserror::Error)]
pub enum DuckDBTestError {
    Other(String),
    Vortex(#[from] VortexError),
}

impl std::fmt::Display for DuckDBTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuckDBTestError::Other(msg) => write!(f, "Other: {msg}"),
            DuckDBTestError::Vortex(inner) => write!(f, "Vortex: {inner}"),
        }
    }
}

struct Inner {
    conn: Connection,
    _db: Database,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct DuckDB {
    inner: Arc<Inner>,
    pb: ProgressBar,
}

impl DuckDB {
    pub fn try_new(pb: ProgressBar) -> Result<Self, DuckDBTestError> {
        let db = Database::open_in_memory()?;
        db.register_vortex_scan_replacement()?;
        let conn = db.connect()?;

        register_table_functions(&conn)?;
        Ok(Self {
            pb,
            inner: Arc::new(Inner { conn, _db: db }),
        })
    }

    /// Turn the DuckDB logical type into a `DFColumnType`, which
    /// tells the runner what types they are. We use the one from DataFusion
    /// as its richer than the default one.
    fn normalize_column_type(dtype: LogicalType) -> DFColumnType {
        let type_id = dtype.as_type_id();
        if type_id == LogicalType::int32().as_type_id()
            || type_id == LogicalType::int64().as_type_id()
            || type_id == LogicalType::uint64().as_type_id()
        {
            DFColumnType::Integer
        } else if type_id == LogicalType::varchar().as_type_id() {
            DFColumnType::Text
        } else if type_id == LogicalType::bool().as_type_id() {
            DFColumnType::Boolean
        } else if type_id == LogicalType::float32().as_type_id()
            || type_id == LogicalType::float64().as_type_id()
        {
            DFColumnType::Float
        } else {
            DFColumnType::Another
        }
    }
}

#[async_trait]
impl AsyncDB for DuckDB {
    type Error = DuckDBTestError;
    type ColumnType = DFColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        let result = {
            let r = self.inner.conn.query(sql)?;

            if r.column_count() == 0 && r.row_count() == 0 {
                Ok(DBOutput::StatementComplete(0))
            } else {
                let mut types = Vec::default();
                let mut rows = Vec::default();

                for col_idx in 0..r.column_count() {
                    let col_idx = usize::try_from(col_idx).map_err(VortexError::from)?;
                    let dtype = r.column_type(col_idx);
                    types.push(Self::normalize_column_type(dtype));
                }

                for chunk in r.into_iter() {
                    for row_idx in 0..chunk.len() {
                        let mut current_row = Vec::new();
                        for col_idx in 0..chunk.column_count() {
                            let vector = chunk.get_vector(col_idx);
                            match vector.get_value(row_idx, chunk.len()) {
                                Some(value) => current_row.push(value.to_string()),
                                None => current_row
                                    .push(Value::null(&vector.logical_type()).to_string()),
                            }
                        }

                        rows.push(current_row);
                    }
                }

                Ok(DBOutput::Rows { types, rows })
            }
        };

        self.pb.inc(1);

        result
    }

    async fn shutdown(&mut self) {}

    fn engine_name(&self) -> &str {
        "DuckDB"
    }

    async fn sleep(dur: Duration) {
        tokio::time::sleep(dur).await
    }

    async fn run_command(command: Command) -> std::io::Result<std::process::Output> {
        tokio::process::Command::from(command).output().await
    }
}
