mod timing;

use std::path;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::bail;
use log::{info, trace};
use path::Path;
use url::Url;
use vortex::error::vortex_panic;
use {anyhow, log};

use crate::Format;
use crate::datasets::BenchmarkDataset;
use crate::ddb::timing::parse_query_output;

#[derive(Debug, Clone)]
pub struct DuckDBExecutor {
    duckdb_path: PathBuf,
    duckdb_file: PathBuf,
}

impl DuckDBExecutor {
    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.duckdb_path);
        command.arg("-unsigned").arg(&self.duckdb_file);
        command
    }

    pub fn new(duckdb_path: PathBuf, duckdb_file: PathBuf) -> Self {
        Self {
            duckdb_path,
            duckdb_file,
        }
    }
}

fn validate_path(duckdb_path: &Path) {
    assert!(
        duckdb_path.exists(),
        "failed to find duckdb executable at: {}",
        duckdb_path.display()
    );
}

pub fn vortex_duckdb_folder() -> PathBuf {
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

    PathBuf::from_str(&repo_root.unwrap_or_else(|| ".".to_string()))
        .expect("failed to find the vortex repo")
        .join("duckdb-vortex")
}

pub fn vortex_duckdb_extension_path() -> PathBuf {
    vortex_duckdb_folder().join("build/release/extension/vortex/vortex.duckdb_extension")
}

pub fn duckdb_executable_path(user_supplied_path_flag: &Option<PathBuf>) -> PathBuf {
    // User supplied path takes priority.
    if let Some(duckdb_path) = user_supplied_path_flag {
        validate_path(duckdb_path);
        return duckdb_path.to_owned();
    };
    // Use the binary
    PathBuf::from("duckdb")
}

/// Finds the path to the DuckDB executable
pub fn build_vortex_duckdb() {
    let duckdb_vortex_path = vortex_duckdb_folder();

    let mut command = Command::new("make");
    command
        .current_dir(&duckdb_vortex_path)
        // The version of DuckDB and its Vortex extension is either implicitly set by Git tag, e.g.
        // v1.2.2, or commit SHA if the current commit does not have a tag. The implicitly set
        // version can be overridden by defining the `OVERRIDE_GIT_DESCRIBE` environment variable.
        .env("OVERRIDE_GIT_DESCRIBE", "v1.2.2")
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
        vortex_panic!("duckdb failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }

    info!(
        "Built duckdb vortex extension at {}",
        duckdb_vortex_path.display()
    );
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

/// Resolves the storage URL based on dataset and format requirements
#[allow(dead_code)]
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

    let vortex_path = vortex_duckdb_extension_path();
    command
        .arg("-c")
        .arg(format!("load \"{}\";", vortex_path.to_string_lossy()));

    command
        .arg("-c")
        .arg("SET autoinstall_known_extensions=1;")
        .arg("-c")
        .arg("SET autoload_known_extensions=1;");

    command.arg("-c").arg(
        "CREATE OR REPLACE SECRET secret (
            TYPE s3,
            PROVIDER credential_chain,
            CHAIN config,
            REGION 'eu-west-1'
        );",
    );

    command.arg("-c").arg(create_table_registration(
        &effective_url,
        extension,
        dataset,
        object,
    ));

    trace!("register duckdb tables with command: {:?}", command);

    // Pass along OS env vars (for aws creds)
    // Don't trace env vars.
    command.envs(std::env::vars_os());
    let output = command.output()?;

    // DuckDB does not return non-zero exit codes in case of failures.
    // Therefore, we need to additionally check whether stderr is set.
    if !output.status.success() || !output.stderr.is_empty() {
        anyhow::bail!(
            "DuckDB query failed: stdout=({})\n, stderr=({})",
            String::from_utf8_lossy(&output.stdout),
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

    let vortex_path = vortex_duckdb_extension_path();
    command
        .arg("-c")
        .arg(format!("load \"{}\";", vortex_path.to_string_lossy()));

    command
        .arg("-c")
        .arg("SET autoinstall_known_extensions=1;")
        .arg("-c")
        .arg("SET autoload_known_extensions=1;");

    let query = queries.join(";") + ";";
    command
        .arg("-c")
        .arg(".timer on")
        .arg("-c")
        .arg(".once /dev/null")
        .arg("-c")
        .arg(query);

    trace!("execute duckdb query with command: {:?}", command);

    let time_instant = Instant::now();
    let output = command.output()?;
    let binary_runtime = time_instant.elapsed();

    // DuckDB does not return non-zero exit codes in case of failures.
    // Therefore, we need to additionally check whether stderr is set.
    if !output.status.success() || !output.stderr.is_empty() {
        bail!(
            "DuckDB query failed, stdout: {} stderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let output = String::from_utf8_lossy(&output.stdout);

    trace!("query output {output}");

    let query_time = parse_query_output(&output)?;
    trace!(
        "query ran with time real {}, user {}, sys {}",
        query_time.real.as_secs_f64(),
        query_time.user.as_secs_f64(),
        query_time.sys.as_secs_f64()
    );

    // We know that the report runtime must be less than the total binary runtime.
    assert!(binary_runtime >= query_time.real);

    Ok(query_time.real)
}

/// Convenience wrapper for TPC-H benchmarks
pub fn execute_tpch_query(
    queries: &[String],
    duckdb_executor: &DuckDBExecutor,
) -> anyhow::Result<Duration> {
    execute_query(queries, duckdb_executor)
}

/// Convenience wrapper for TPC-DS benchmarks
pub fn execute_tpcds_query(
    query_string: &str,
    duckdb_executor: &DuckDBExecutor,
) -> anyhow::Result<Duration> {
    execute_query(&[query_string.to_string()], duckdb_executor)
}

/// Convenience wrapper for ClickBench benchmarks
pub fn execute_clickbench_query(
    query_string: &str,
    duckdb_executor: &DuckDBExecutor,
) -> anyhow::Result<Duration> {
    execute_query(&[query_string.to_string()], duckdb_executor)
}
