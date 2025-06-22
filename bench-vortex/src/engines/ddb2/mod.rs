use std::time::{Duration, Instant};

use anyhow::Result;
use log::trace;
use url::Url;
use vortex_duckdb_ext::duckdb::{Connection, Database};

use crate::{BenchmarkDataset, Format};

// TODO: handle S3

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
}

impl DuckDBCtx {
    pub fn new() -> Result<Self> {
        let db = Database::open_in_memory()?;
        let connection = db.connect()?;
        vortex_duckdb_ext::init(&connection)?;
        Ok(Self { db, connection })
    }

    /// Execute DuckDB queries for benchmarks using the internal connection
    pub fn execute_query(&self, query: &str) -> Result<Duration> {
        // TODO: handle multiple queries
        trace!("execute duckdb query: {}", query);
        let time_instant = Instant::now();
        self.connection.execute(query)?;
        let query_time = time_instant.elapsed();
        trace!("query completed in {:.3}s", query_time.as_secs_f64());

        Ok(query_time)
    }

    /// Register tables for benchmarks using the internal connection
    pub fn register_tables(
        &self,
        base_url: &Url,
        file_format: Format,
        dataset: BenchmarkDataset,
    ) -> Result<()> {
        let object = match file_format {
            Format::Parquet | Format::OnDiskVortex => DuckDBObject::View,
            Format::OnDiskDuckDB => DuckDBObject::Table,
            format => anyhow::bail!("Format {format} isn't supported for DuckDB"),
        };

        let load_format = match file_format {
            // Duckdb loads values from parquet to duckdb
            Format::Parquet | Format::OnDiskDuckDB => Format::Parquet,
            f => f,
        };

        let effective_url = self.resolve_storage_url(base_url, load_format, dataset)?;
        let extension = match load_format {
            Format::Parquet => "parquet",
            Format::OnDiskVortex => "vortex",
            other => anyhow::bail!("Format {other} isn't supported for DuckDB"),
        };

        // Generate and execute table registration commands
        let commands = self.generate_table_commands(&effective_url, extension, dataset, object);
        self.execute_query(&commands)?;
        trace!("Executing table registration commands: {}", commands);

        Ok(())
    }

    /// Resolves the storage URL based on dataset and format requirements
    fn resolve_storage_url(
        &self,
        base_url: &Url,
        file_format: Format,
        dataset: BenchmarkDataset,
    ) -> Result<Url> {
        if file_format == Format::OnDiskVortex {
            match dataset.vortex_path(base_url) {
                Ok(vortex_url) => {
                    // Check if the directory exists (for file:// URLs)
                    if vortex_url.scheme() == "file" {
                        let path = std::path::Path::new(vortex_url.path());
                        if !path.exists() {
                            log::warn!(
                                "Vortex directory doesn't exist at: {}. Run with DataFusion engine first to generate Vortex files.",
                                path.display()
                            );
                        }
                    }
                    Ok(vortex_url)
                }
                Err(_) => Ok(base_url.clone()),
            }
        } else if file_format == Format::Parquet {
            match dataset.parquet_path(base_url) {
                Ok(parquet_url) => Ok(parquet_url),
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
        dataset: BenchmarkDataset,
        duckdb_object: DuckDBObject,
    ) -> String {
        // Base path contains trailing /.
        let base_dir = base_url.as_str();
        let base_dir = base_dir.strip_prefix("file://").unwrap_or(base_dir);

        match dataset {
            BenchmarkDataset::TpcH => {
                let mut commands = String::new();
                let tables = [
                    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region",
                    "supplier",
                ];

                for table_name in &tables {
                    let table_path = format!("{base_dir}{table_name}.{extension}");
                    commands.push_str(&format!(
                        "CREATE {} IF NOT EXISTS {table_name} AS SELECT * FROM read_{extension}('{table_path}');\n",
                        duckdb_object.to_str(),
                    ));
                }
                commands
            }
            BenchmarkDataset::ClickBench { single_file } => {
                let file_glob = if single_file {
                    format!("{base_dir}hits.{extension}")
                } else {
                    format!("{base_dir}*.{extension}")
                };

                format!(
                    "CREATE {} IF NOT EXISTS hits AS SELECT * FROM read_{extension}('{file_glob}');",
                    duckdb_object.to_str()
                )
            }
            BenchmarkDataset::TpcDS => {
                let mut commands = String::new();
                let tables = BenchmarkDataset::TpcDS.tables();

                for table_name in tables {
                    let table_path = format!("{base_dir}{table_name}.{extension}");
                    commands.push_str(&format!(
                        "CREATE {} IF NOT EXISTS {table_name} AS SELECT * FROM read_{extension}('{table_path}');\n",
                        duckdb_object.to_str(),
                    ));
                }
                commands
            }
        }
    }
}
