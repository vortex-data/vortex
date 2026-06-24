// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::LazyLock;

use datafusion::common::GetExt;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionConfig;
use datafusion::prelude::SessionContext;
use datafusion_sqllogictest::DataFusion;
use datafusion_sqllogictest::df_value_validator;
use datafusion_sqllogictest::value_normalizer;
use indicatif::ProgressBar;
use sqllogictest::Runner;
use sqllogictest::harness::Arguments;
use sqllogictest::harness::Failed;
use sqllogictest::harness::Trial;
use sqllogictest::strict_column_validator;
use vortex_datafusion::VortexFormatFactory;
use vortex_datafusion::VortexTableOptions;
use vortex_sqllogictest::duckdb::DuckDB;
use vortex_sqllogictest::duckdb::duckdb_validator;
use vortex_sqllogictest::normalize::PathNormalizing;
use vortex_sqllogictest::normalize::WORK_DIR_VAR;
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

/// Builds a single-threaded Tokio runtime for one test file.
///
/// `libtest-mimic` runs each trial on its own thread, so a current-thread
/// runtime keeps blocking DuckDB calls and async DataFusion work isolated per
/// file instead of contending for shared multi-threaded runtime workers.
fn build_runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

/// Runs or completes a single `.slt` file against DataFusion reading Vortex files.
fn drive_datafusion(path: &Path, work_dir: &Path, mode: Mode) -> anyhow::Result<()> {
    reset_dir(work_dir)?;
    let _guard = WorkDirGuard::new(work_dir.to_path_buf());
    let work_dir = work_dir.to_string_lossy().into_owned();

    let rt = build_runtime()?;
    rt.block_on(async {
        let config = SessionConfig::default().with_option_extension(VortexTableOptions::default());
        let factory = Arc::new(VortexFormatFactory::new());
        let session_state_builder = SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_table_factory(
                factory.get_ext().to_uppercase(),
                Arc::new(DefaultTableFactory::new()),
            )
            .with_file_formats(vec![factory]);
        let session =
            SessionContext::new_with_state(session_state_builder.build()).enable_url_table();

        let mut runner = Runner::new(|| async {
            Ok(PathNormalizing::new(
                DataFusion::new(session.clone(), path.to_path_buf(), ProgressBar::hidden()),
                work_dir.clone(),
            ))
        });
        runner.set_var(WORK_DIR_VAR.to_string(), work_dir.clone());
        runner.add_label("datafusion");
        runner.with_column_validator(strict_column_validator);
        runner.with_normalizer(value_normalizer);
        runner.with_validator(df_value_validator);

        run_or_complete(&mut runner, path, mode, df_value_validator).await
    })
}

/// Runs or completes a single `.slt` file against DuckDB reading Vortex files.
fn drive_duckdb(path: &Path, work_dir: &Path, mode: Mode) -> anyhow::Result<()> {
    reset_dir(work_dir)?;
    let _guard = WorkDirGuard::new(work_dir.to_path_buf());
    let work_dir = work_dir.to_string_lossy().into_owned();

    let rt = build_runtime()?;
    rt.block_on(async {
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
    let datafusion = !in_dir("duckdb");
    let duckdb = !in_dir("datafusion");
    (datafusion, duckdb)
}

fn is_tpch(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == "tpch")
}

/// Rewrites the expected output of each file in place, completing from a single
/// reference engine per file (DuckDB for `duckdb/` files, DataFusion otherwise).
fn complete_files(
    args: &Arguments,
    files: &[PathBuf],
    slt_root: &Path,
    has_tpch_data: bool,
) -> anyhow::Result<()> {
    for path in files {
        if is_tpch(path) && !has_tpch_data {
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
    let mut raw_args: Vec<String> = std::env::args().collect();
    // We remove the `--complete` flag that isn't standard before we pass the rest.
    let complete = {
        let flag = "--complete";
        let present = raw_args.iter().any(|arg| arg == flag);
        raw_args.retain(|arg| arg != flag);
        present
    };
    let args = Arguments::from_iter(raw_args);

    let has_tpch_data = SLT_ROOT.join("tpch/data/lineitem.vortex").exists();

    let mut files = list_files(SLT_ROOT.as_path())?;
    files.sort();

    if complete {
        complete_files(&args, &files, SLT_ROOT.as_path(), has_tpch_data)?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut trials = Vec::new();
    for path in files {
        let (run_datafusion, run_duckdb) = engines_for(&path);
        // TPC-H trials are ignored (rather than removed) when the generated data
        // is absent, so `--list` and the run summary still account for them.
        let ignored = is_tpch(&path) && !has_tpch_data;
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
