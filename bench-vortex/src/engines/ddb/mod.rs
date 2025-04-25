use std::path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::time::{Duration, Instant};

use log::{info, trace};
use path::Path;
use url::Url;
use vortex::error::vortex_panic;
use {anyhow, log};

use crate::Format;
use crate::datasets::BenchmarkDataset;

#[derive(Debug, Clone)]
pub struct DuckDBExecutor {
    duckdb_path: PathBuf,
    duckdb_file: PathBuf,
}

impl DuckDBExecutor {
    fn command(&self) -> Command {
        let mut command = Command::new(&self.duckdb_path);
        command.arg(&self.duckdb_file);
        command
    }

    pub fn new(duckdb_path: PathBuf, duckdb_file: PathBuf) -> Self {
        Self {
            duckdb_path,
            duckdb_file,
        }
    }
}

/// Finds the path to the DuckDB executable
pub fn build_and_get_executable_path(user_supplied_path_flag: &Option<PathBuf>) -> PathBuf {
    let validate_path = |duckdb_path: &PathBuf| {
        if !duckdb_path.as_path().exists() {
            panic!(
                "failed to find duckdb executable at: {}",
                duckdb_path.display()
            );
        }
    };

    // User supplied path takes priority.
    if let Some(duckdb_path) = user_supplied_path_flag {
        validate_path(duckdb_path);
        return duckdb_path.to_owned();
    }

    // Try to find the 'vortex' top-level directory. This is preferred over logic along
    // the lines of `git rev-parse --show-toplevel`, as the repository uses submodules.
    let mut repo_root = None;
    let mut current_dir = std::env::current_dir().expect("failed to get current dir");

    while current_dir.file_name().is_some() {
        if current_dir.file_name().and_then(|name| name.to_str()) == Some("vortex") {
            repo_root = Some(current_dir.to_string_lossy().into_owned());
            break;
        }

        if !current_dir.pop() {
            break;
        }
    }

    let duckdb_vortex_path = PathBuf::from_str(&repo_root.unwrap_or_else(|| ".".to_string()))
        .expect("failed to find the vortex repo")
        .join("duckdb-vortex");

    let mut command = Command::new("make");
    command
        .current_dir(&duckdb_vortex_path)
        .env("GEN", "ninja")
        .arg("release");

    info!(
        "Building duckdb vortex extension at {}, with command {:?}",
        duckdb_vortex_path.display(),
        command
    );

    let output = command
        .output()
        .expect("Trying to build duckdb vortex extension");

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("duckdb failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }

    info!(
        "Built duckdb vortex extension at {}",
        duckdb_vortex_path.display()
    );

    let duckdb_path = duckdb_vortex_path.join("build/release/duckdb");

    validate_path(&duckdb_path);

    duckdb_path
}

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

fn create_table_registration(
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
                    "CREATE {} {table_name} AS SELECT * FROM read_{extension}('{table_path}');\n",
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
                "CREATE {} hits AS SELECT * FROM read_{extension}('{file_glob}');",
                duckdb_object.to_str()
            )
        }
    }
}

/// Resolves the storage URL based on dataset and format requirements
fn resolve_storage_url(base_url: &Url, file_format: Format, dataset: BenchmarkDataset) -> Url {
    if file_format == Format::OnDiskVortex {
        match dataset.vortex_path(base_url) {
            Ok(vortex_url) => {
                // Check if the directory exists (for file:// URLs)
                if vortex_url.scheme() == "file" {
                    let path = Path::new(vortex_url.path());
                    if !path.exists() {
                        log::warn!(
                            "Vortex directory doesn't exist at: {}. Run with DataFusion engine first to generate Vortex files.",
                            path.display()
                        );
                    }
                }
                vortex_url
            }
            Err(_) => base_url.clone(),
        }
    } else if file_format == Format::Parquet {
        match dataset.parquet_path(base_url) {
            Ok(parquet_url) => parquet_url,
            Err(_) => base_url.clone(),
        }
    } else {
        base_url.clone()
    }
}

pub fn register_tables(
    duckdb_executor: &DuckDBExecutor,
    base_url: &Url,
    file_format: Format,
    dataset: BenchmarkDataset,
) -> anyhow::Result<()> {
    let object = match file_format {
        Format::Parquet | Format::OnDiskVortex => DuckDBObject::View,
        Format::OnDiskDuckDB => DuckDBObject::Table,
        format => todo!("cannot run {format}"),
    };

    let load_format = match file_format {
        // Duckdb loads values from parquet to duckdb
        Format::Parquet | Format::OnDiskDuckDB => Format::Parquet,
        f => f,
    };

    let effective_url = resolve_storage_url(base_url, load_format, dataset);
    let extension = match load_format {
        Format::Parquet => "parquet",
        Format::OnDiskVortex => "vortex",
        other => vortex_panic!("Format {other} isn't supported for DuckDB"),
    };

    let mut command = duckdb_executor.command();

    command.arg("-c").arg(create_table_registration(
        &effective_url,
        extension,
        dataset,
        object,
    ));

    trace!("register duckdb tables with command: {:?}", command);

    let output = command.output()?;

    // DuckDB does not return non-zero exit codes in case of failures.
    // Therefore, we need to additionally check whether stderr is set.
    if !output.status.success() || !output.stderr.is_empty() {
        anyhow::bail!(
            "DuckDB query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    };

    Ok(())
}

/// Execute DuckDB queries for benchmarks
pub fn execute_query(
    queries: &[String],
    duckdb_executor: &DuckDBExecutor,
) -> anyhow::Result<Duration> {
    let mut command = duckdb_executor.command();

    for query in queries {
        command.arg("-c").arg(query);
    }

    trace!("execute duckdb query with command: {:?}", command);

    let time_instant = Instant::now();
    let output = command.output()?;
    let time = time_instant.elapsed();

    // DuckDB does not return non-zero exit codes in case of failures.
    // Therefore, we need to additionally check whether stderr is set.
    if !output.status.success() || !output.stderr.is_empty() {
        anyhow::bail!(
            "DuckDB query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(time)
}

/// Convenience wrapper for TPC-H benchmarks
pub fn execute_tpch_query(
    queries: &[String],
    duckdb_executor: &DuckDBExecutor,
) -> anyhow::Result<Duration> {
    execute_query(queries, duckdb_executor)
}

/// Convenience wrapper for ClickBench benchmarks
pub fn execute_clickbench_query(
    query_string: &str,
    duckdb_executor: &DuckDBExecutor,
) -> anyhow::Result<Duration> {
    execute_query(&[query_string.to_string()], duckdb_executor)
}
