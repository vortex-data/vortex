use std::path;
use std::process::Command;
use std::str::FromStr;
use std::time::{Duration, Instant};

use path::Path;
use url::Url;
use vortex::error::vortex_panic;
use {anyhow, log};

use crate::Format;
use crate::datasets::BenchmarkDataset;

/// Finds the path to the DuckDB executable
pub fn executable_path(user_supplied_path_flag: &Option<path::PathBuf>) -> path::PathBuf {
    let validate_path = |duckdb_path: &path::PathBuf| {
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

    let duckdb_path = path::PathBuf::from_str(&format!(
        "{}/duckdb-vortex/build/release/duckdb",
        repo_root.unwrap_or_default()
    ))
    .expect("failed to create DuckDB executable path");

    validate_path(&duckdb_path);

    duckdb_path
}

fn create_table_registration(base_url: &Url, extension: &str, dataset: BenchmarkDataset) -> String {
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

            for table in &tables {
                let table_path = format!("{base_dir}{table}.{extension}");
                commands.push_str(&format!(
                    "CREATE VIEW {table} AS SELECT * FROM read_{extension}('{table_path}');\n"
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

            format!("CREATE VIEW hits AS SELECT * FROM read_{extension}('{file_glob}');")
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

/// Execute DuckDB queries for benchmarks
pub fn execute_query(
    queries: &[String],
    base_url: &Url,
    file_format: Format,
    dataset: BenchmarkDataset,
    duckdb_path: &Path,
) -> anyhow::Result<Duration> {
    let extension = match file_format {
        Format::Parquet => "parquet",
        Format::OnDiskVortex => "vortex",
        other => vortex_panic!("Format {other} isn't supported for DuckDB"),
    };

    let effective_url = resolve_storage_url(base_url, file_format, dataset);
    let mut command = Command::new(duckdb_path);
    let register_tables = create_table_registration(&effective_url, extension, dataset);
    command.arg("-c").arg(register_tables);
    for query in queries {
        command.arg("-c").arg(query);
    }

    let time_instant = Instant::now();
    let output = command.output()?;

    // DuckDB does not return non-zero exit codes in case of failures.
    // Therefore, we need to additionally check whether stderr is set.
    if !output.status.success() || !output.stderr.is_empty() {
        anyhow::bail!(
            "DuckDB query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(time_instant.elapsed())
}

/// Convenience wrapper for TPC-H benchmarks
pub fn execute_tpch_query(
    queries: &[String],
    base_url: &Url,
    file_format: Format,
    duckdb_path: &Path,
) -> anyhow::Result<Duration> {
    execute_query(
        queries,
        base_url,
        file_format,
        BenchmarkDataset::TpcH,
        duckdb_path,
    )
}

/// Convenience wrapper for ClickBench benchmarks
pub fn execute_clickbench_query(
    query_string: &str,
    base_url: &Url,
    file_format: Format,
    single_file: bool,
    duckdb_path: &Path,
) -> anyhow::Result<Duration> {
    let dataset = BenchmarkDataset::ClickBench { single_file };

    execute_query(
        &[query_string.to_string()],
        base_url,
        file_format,
        dataset,
        duckdb_path,
    )
}
