// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use log::trace;
use url::Url;
use vortex::error::VortexExpect;
use vortex_duckdb::duckdb::{Config, Connection, Database};

use crate::statpopgen::StatPopGenBenchmark;
use crate::{BenchmarkDataset, Format, IdempotentPath};

#[derive(Debug, Clone)]
enum DuckDBObject {
    Table,
    View,
}

impl DuckDBObject {
    fn to_str(&self) -> &str {
        match self {
            DuckDBObject::Table => "TABLE",
            DuckDBObject::View => "VIEW",
        }
    }
}

/// DuckDB context for benchmarks.
pub struct DuckDBCtx {
    pub db: Database,
    pub connection: Connection,
    pub db_path: Option<PathBuf>,
}

impl DuckDBCtx {
    pub fn new(dataset: BenchmarkDataset, format: Format, delete_database: bool) -> Result<Self> {
        let dir = match dataset {
            BenchmarkDataset::ClickBench { flavor, .. } => {
                format!("clickbench_{}/{}", flavor, format.name()).to_data_path()
            }
            BenchmarkDataset::TpcH { scale_factor } => {
                format!("tpch/{scale_factor}/{}", format.name()).to_data_path()
            }
            BenchmarkDataset::TpcDS { scale_factor } => {
                format!("tpcds/{scale_factor}/{}", format.name()).to_data_path()
            }
            BenchmarkDataset::PublicBi { .. } => todo!(),
            BenchmarkDataset::StatPopGen { n_rows } => {
                format!("statpopgen/{n_rows}/{}", format.name()).to_data_path()
            }
            BenchmarkDataset::Fineweb => format!("fineweb/{}", format.name()).to_data_path(),
            BenchmarkDataset::GhArchive => format!("gharchive/{}", format.name()).to_data_path(),
        };
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("duckdb.db");
        if delete_database {
            std::fs::remove_file(&db_path)?;
        }

        let (db, connection) = Self::open_and_setup_database(Some(db_path.clone()))?;

        Ok(Self {
            db,
            connection,
            db_path: Some(db_path),
        })
    }

    pub fn open_and_setup_database(path: Option<PathBuf>) -> Result<(Database, Connection)> {
        let config = Config::new().vortex_expect("failed to create duckdb config");

        let db = match path {
            Some(path) => Database::open_with_config(path, config),
            None => Database::open_in_memory_with_config(config),
        }?;
        let connection = db.connect()?;
        vortex_duckdb::register_table_functions(&connection)?;

        // Enable Parquet metadata cache for all benchmark runs.
        //
        // `parquet_metadata_cache` is an extension-specific option that's
        // only available after the Parquet extension is loaded. The Parquet
        // extension is loaded after the connection is established.
        //
        // Passing the option to `open_with_config` before leads to
        // "Invalid Input Error: The following options were not recognized:
        // parquet_metadata_cache" when running DuckDB in debug mode.
        connection.query("SET parquet_metadata_cache = true")?;

        Ok((db, connection))
    }

    pub fn reopen(&mut self) -> Result<()> {
        // take ownership of the connection & database
        let mut connection = unsafe { Connection::borrow(self.connection.as_ptr()) };
        std::mem::swap(&mut self.connection, &mut connection);
        let mut db = unsafe { Database::borrow(self.db.as_ptr()) };
        std::mem::swap(&mut self.db, &mut db);

        // drop the connection, then the database (order might be important?)
        // NB: self.db and self.connection will be dangling pointers, which we'll fix below
        drop(connection);
        drop(db);

        let (mut db, mut connection) = Self::open_and_setup_database(self.db_path.clone())?;

        std::mem::swap(&mut self.connection, &mut connection);
        std::mem::swap(&mut self.db, &mut db);

        Ok(())
    }

    pub fn new_in_memory() -> Result<Self> {
        let db = Database::open_in_memory()?;
        let connection = db.connect()?;
        vortex_duckdb::register_table_functions(&connection)?;
        Ok(Self {
            db,
            connection,
            db_path: None,
        })
    }

    /// Execute DuckDB queries for benchmarks using the internal connection
    pub fn execute_query(&self, query: &str) -> Result<(Duration, usize)> {
        trace!("execute duckdb query: {query}");
        let time_instant = Instant::now();
        let result = self.connection.query(query)?;
        let query_time = time_instant.elapsed();
        trace!("query completed in {:.3}s", query_time.as_secs_f64());

        Ok((
            query_time,
            usize::try_from(result.row_count()).vortex_expect("row count overflow"),
        ))
    }

    /// Register tables for benchmarks using the internal connection
    pub fn register_tables(
        &self,
        base_url: &Url,
        file_format: Format,
        dataset: &BenchmarkDataset,
    ) -> Result<()> {
        let object = match file_format {
            Format::Parquet | Format::OnDiskVortex | Format::VortexCompact => DuckDBObject::View,
            Format::OnDiskDuckDB => DuckDBObject::Table,
            #[cfg(feature = "lance")]
            Format::Lance => {
                anyhow::bail!(
                    "Lance format is not supported for DuckDB engine. \
                    Please use DataFusion engine instead (e.g., --targets datafusion:lance)"
                );
            }
            format => anyhow::bail!("Format {format} isn't supported for DuckDB"),
        };

        let load_format = match file_format {
            // Duckdb loads values from parquet to duckdb
            Format::Parquet | Format::OnDiskDuckDB => Format::Parquet,
            f => f,
        };

        let effective_url = self.resolve_storage_url(base_url, load_format, dataset)?;
        let extension = match load_format {
            Format::Parquet | Format::OnDiskVortex | Format::VortexCompact => load_format.ext(),
            other => anyhow::bail!("Format {other} isn't supported for DuckDB"),
        };

        // Generate and execute table registration commands
        let commands = self.generate_table_commands(&effective_url, extension, dataset, object);
        trace!("Executing table registration commands: {commands}");
        self.execute_query(&commands)?;

        Ok(())
    }

    /// Resolves the storage URL based on dataset and format requirements
    fn resolve_storage_url(
        &self,
        base_url: &Url,
        file_format: Format,
        dataset: &BenchmarkDataset,
    ) -> Result<Url> {
        if file_format == Format::OnDiskVortex
            || file_format == Format::Parquet
            || file_format == Format::VortexCompact
        {
            match dataset.format_path(file_format, base_url) {
                Ok(vortex_url) => Ok(vortex_url),
                Err(_) => Ok(base_url.clone()),
            }
        } else {
            Ok(base_url.clone())
        }
    }

    /// Generate SQL commands for table registration.
    fn generate_table_commands(
        &self,
        base_url: &Url,
        extension: &str,
        dataset: &BenchmarkDataset,
        duckdb_object: DuckDBObject,
    ) -> String {
        // Base path contains trailing /.
        let base_dir = base_url.as_str();
        let base_dir = base_dir.strip_prefix("file://").unwrap_or(base_dir);
        match dataset {
            BenchmarkDataset::TpcH { .. } => {
                let mut commands = String::new();
                let tables = [
                    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region",
                    "supplier",
                ];

                for table_name in &tables {
                    let table_path = format!("{base_dir}{table_name}_*.{extension}");
                    commands.push_str(&format!(
                                "CREATE {} IF NOT EXISTS {table_name} AS SELECT * FROM read_{extension}('{table_path}');\n",
                                duckdb_object.to_str(),
                            ));
                }
                commands
            }
            BenchmarkDataset::ClickBench { .. } => {
                format!(
                    "CREATE {} IF NOT EXISTS hits AS SELECT * FROM read_{extension}('{base_dir}*.{extension}');",
                    duckdb_object.to_str()
                )
            }
            dataset @ BenchmarkDataset::TpcDS { .. } => {
                let mut commands = String::new();
                let tables = dataset.tables();

                for table_name in tables {
                    let table_path = format!("{base_dir}{table_name}.{extension}");
                    commands.push_str(&format!(
                                "CREATE {} IF NOT EXISTS {table_name} AS SELECT * FROM read_{extension}('{table_path}');\n",
                                duckdb_object.to_str(),
                            ));
                }
                commands
            }
            BenchmarkDataset::PublicBi { .. } => todo!(),
            BenchmarkDataset::StatPopGen { .. } => {
                let path = format!("{base_dir}{}.{extension}", StatPopGenBenchmark::FILE_NAME);
                format!(
                    "CREATE {} IF NOT EXISTS statpopgen AS SELECT * FROM read_{extension}('{path}');",
                    duckdb_object.to_str()
                )
            }
            BenchmarkDataset::Fineweb => {
                let path = format!("{base_dir}*.{extension}");
                format!(
                    "CREATE {} IF NOT EXISTS fineweb AS SELECT * FROM read_{extension}('{path}');",
                    duckdb_object.to_str(),
                )
            }
            BenchmarkDataset::GhArchive => {
                let path = format!("{base_dir}*.{extension}");
                format!(
                    "CREATE {} IF NOT EXISTS events AS SELECT * FROM read_{extension}('{path}');",
                    duckdb_object.to_str(),
                )
            }
        }
    }
}
