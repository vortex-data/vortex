// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "datafusion")]
mod datafusion_engine;

use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::LazyLock;

#[cfg(feature = "datafusion")]
use datafusion_engine::drive_datafusion;
use sqllogictest::Runner;
use sqllogictest::harness::Arguments;
use sqllogictest::harness::Failed;
use sqllogictest::harness::Trial;
use sqllogictest::strict_column_validator;
use vortex_sqllogictest::duckdb::DuckDB;
use vortex_sqllogictest::duckdb::duckdb_validator;
use vortex_sqllogictest::normalize::PathNormalizing;
use vortex_sqllogictest::normalize::WORK_DIR_VAR;
use vortex_sqllogictest::normalize::value_normalizer;
use vortex_sqllogictest::scratch::WorkDirGuard;
use vortex_sqllogictest::scratch::reset_dir;
use vortex_sqllogictest::scratch::work_dir_for;
use vortex_sqllogictest::utils::list_files;

static SLT_ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_path.join("slt")
});

/// Whether to verify a file against its expected output or rewrite it.
#[derive(Clone, Copy)]
enum Mode {
    Run,
    Complete,
}

#[cfg(not(feature = "datafusion"))]
fn drive_datafusion(_path: &Path, _work_dir: &Path, _mode: Mode) -> anyhow::Result<()> {
    anyhow::bail!("vortex-sqllogictest is built without the datafusion feature")
}

/// Runs or completes a single `.slt` file against DuckDB.
fn drive_duckdb(path: &Path, work_dir: &Path, mode: Mode) -> anyhow::Result<()> {
    reset_dir(work_dir)?;
    let _guard = WorkDirGuard::new(work_dir.to_path_buf());
    // Windows get us backslashes slashes but Duckdb produces forward slashes
    let work_dir = work_dir.to_string_lossy().replace('\\', "/");

    // Deliberately not a Tokio runtime. DuckDB scans drive Vortex's own runtime, and a Vortex
    // runtime driven from a thread inside `tokio::runtime::Runtime::block_on` loses the wakeups
    // that complete its I/O and stalls forever (#9817). `AsyncDB::run` for DuckDB is synchronous
    // and no DuckDB `.slt` uses the `sleep` or `system` directives, so nothing here needs Tokio.
    futures::executor::block_on(async {
        let mut runner = Runner::new(|| async {
            DuckDB::try_new().map(|db| PathNormalizing::new(db, work_dir.clone()))
        });
        runner.set_var(WORK_DIR_VAR.to_string(), work_dir.clone());
        runner.add_label("duckdb");
        runner.with_column_validator(strict_column_validator);
        runner.with_normalizer(value_normalizer);
        runner.with_validator(duckdb_validator);

        run_or_complete(&mut runner, path, mode, duckdb_validator).await
    })
}

/// Either validates `path` or rewrites its expected output, depending on `mode`.
async fn run_or_complete<D, M>(
    runner: &mut Runner<D, M>,
    path: &Path,
    mode: Mode,
    validator: sqllogictest::Validator,
) -> anyhow::Result<()>
where
    D: sqllogictest::runner::AsyncDB,
    M: sqllogictest::connection::MakeConnection<Conn = D>,
{
    match mode {
        Mode::Run => runner
            .run_file_async(path)
            .await
            .map_err(|e| anyhow::anyhow!("{e}")),
        // Completion rewrites the file's expected output in place.
        Mode::Complete => runner
            .update_test_file(
                path,
                " ",
                validator,
                value_normalizer,
                strict_column_validator,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}")),
    }
}

/// Determines which engines a file runs on from its path: a `duckdb/` directory
/// is DuckDB-only, a `datafusion/` directory is DataFusion-only, else both.
fn engines_for(path: &Path) -> (bool, bool) {
    let in_dir = |dir: &str| path.components().any(|c| c.as_os_str() == dir);
    let datafusion = cfg!(feature = "datafusion") && !in_dir("duckdb");
    let duckdb = !in_dir("datafusion");
    (datafusion, duckdb)
}

/// Suites whose tables come from a `generate_data.sh` script rather than the
/// test itself: the `slt/` subdirectory holding the suite, and a file whose
/// Vortex and Parquet versions both have to exist for the suite to run.
const GENERATED_DATASETS: &[(&str, &str)] = &[
    ("tpch", "tpch/data/lineitem"),
    ("tpcds", "tpcds/data/store_sales"),
    ("clickbench", "clickbench/data/hits"),
];

/// Whether `path` belongs to a generated-data suite whose data is absent.
fn missing_generated_data(path: &Path) -> bool {
    GENERATED_DATASETS.iter().any(|(dir, fixture)| {
        path.components().any(|c| c.as_os_str() == *dir)
            && !["vortex", "parquet"]
                .into_iter()
                .all(|format| SLT_ROOT.join(format!("{fixture}.{format}")).exists())
    })
}

/// Rewrites the expected output of each file in place, completing from a single
/// reference engine per file (DuckDB for `duckdb/` files, DataFusion otherwise).
fn complete_files(args: &Arguments, files: &[PathBuf], slt_root: &Path) -> anyhow::Result<()> {
    for path in files {
        if missing_generated_data(path) {
            continue;
        }
        let name = path
            .strip_prefix(slt_root)
            .unwrap_or(path)
            .display()
            .to_string();
        if let Some(filter) = &args.filter
            && !name.contains(filter)
        {
            continue;
        }

        // A `duckdb/` file completes from DuckDB; DataFusion is the reference for
        // everything else (including files that also run on DuckDB).
        if path.components().any(|c| c.as_os_str() == "duckdb") {
            let test_name = format!("slt::duckdb::{name}");
            drive_duckdb(path, &work_dir_for(&test_name), Mode::Complete)?;
        } else {
            let test_name = format!("slt::datafusion::{name}");
            drive_datafusion(path, &work_dir_for(&test_name), Mode::Complete)?;
        }
        eprintln!("completed {name}");
    }
    Ok(())
}

fn main() -> anyhow::Result<ExitCode> {
    drop(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init(),
    );
    let mut raw_args: Vec<String> = std::env::args().collect();
    // We remove the `--complete` flag that isn't standard before we pass the rest.
    let complete = {
        let flag = "--complete";
        let present = raw_args.iter().any(|arg| arg == flag);
        raw_args.retain(|arg| arg != flag);
        present
    };
    let args = Arguments::from_iter(raw_args);

    let mut files = list_files(SLT_ROOT.as_path())?;
    files.sort();

    if complete {
        complete_files(&args, &files, SLT_ROOT.as_path())?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut trials = Vec::new();
    for path in files {
        let (run_datafusion, run_duckdb) = engines_for(&path);
        // Generated-data trials (TPC-H, TPC-DS, ClickBench) are ignored (rather
        // than removed) when the data is absent, so `--list` and the run summary
        // still account for them.
        let ignored = missing_generated_data(&path);
        let name = path
            .strip_prefix(SLT_ROOT.as_path())
            .unwrap_or(&path)
            .display()
            .to_string();

        if run_datafusion {
            let path = path.clone();
            let test_name = format!("slt::datafusion::{name}");
            let work_dir = work_dir_for(&test_name);
            trials.push(
                Trial::test(test_name, move || {
                    drive_datafusion(&path, &work_dir, Mode::Run)
                        .map_err(|e| Failed::from(e.to_string()))
                })
                .with_ignored_flag(ignored),
            );
        }

        if run_duckdb {
            let path = path.clone();
            let test_name = format!("slt::duckdb::{name}");
            let work_dir = work_dir_for(&test_name);
            trials.push(
                Trial::test(test_name, move || {
                    drive_duckdb(&path, &work_dir, Mode::Run)
                        .map_err(|e| Failed::from(e.to_string()))
                })
                .with_ignored_flag(ignored),
            );
        }
    }

    Ok(sqllogictest::harness::run(&args, trials).exit_code())
}
