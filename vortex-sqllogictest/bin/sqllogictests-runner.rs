// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use clap::Parser;
use datafusion::common::GetExt;
use datafusion::datasource::provider::DefaultTableFactory;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_sqllogictest::DataFusion;
use datafusion_sqllogictest::df_value_validator;
use datafusion_sqllogictest::value_normalizer;
use futures::StreamExt;
use futures::TryStreamExt;
use indicatif::MultiProgress;
use indicatif::ProgressBar;
use indicatif::ProgressDrawTarget;
use sqllogictest::Record;
use sqllogictest::Runner;
use sqllogictest::parse_file;
use sqllogictest::strict_column_validator;
use vortex::error::VortexExpect;
use vortex_datafusion::VortexFormatFactory;
use vortex_sqllogictest::args::Args;
use vortex_sqllogictest::duckdb::DuckDB;
use vortex_sqllogictest::duckdb::DuckDBTestError;
use vortex_sqllogictest::duckdb::duckdb_validator;
use vortex_sqllogictest::utils::list_files;
use vortex_sqllogictest::utils::pb_style;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.list {
        eprintln!("Ignoring `--list` which is unsupported by `sqlogictests-runner`");

        return Ok(());
    }

    let mpb = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));

    let crate_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = crate_path.join("slt/");
    let has_tpch_data = crate_path.join("slt/tpch/data/lineitem.vortex").exists();

    let all_errors = futures::stream::iter(
        list_files(path)?
            .into_iter()
            .filter(|path| {
                has_tpch_data || !path.components().any(|comp| comp.as_os_str() == "tpch")
            })
            .collect::<Vec<_>>(),
    )
    .map(|path| {
        let mpb = mpb.clone();
        let filter = args.filter.clone();

        async move {
            let path = path.canonicalize()?;

            let mut errors = vec![];
            let factory = Arc::new(VortexFormatFactory::new());
            let session_state_builder = SessionStateBuilder::new()
                .with_default_features()
                .with_table_factory(
                    factory.get_ext().to_uppercase(),
                    Arc::new(DefaultTableFactory::new()),
                )
                .with_file_formats(vec![factory]);

            let session =
                SessionContext::new_with_state(session_state_builder.build()).enable_url_table();

            let filename = path
                .file_name()
                .vortex_expect("must be file")
                .to_string_lossy();

            if filter.is_some_and(|f| !filename.contains(f.as_str())) {
                return anyhow::Ok(vec![]);
            }

            let records = parse_file(path.as_path())?;

            let exec_statements = records
                .iter()
                .filter(|r| {
                    matches!(
                        r,
                        Record::Query { .. } | Record::Statement { .. } | Record::Let { .. }
                    )
                })
                .count() as u64;

            if !path.components().any(|comp| comp.as_os_str() == "duckdb") {
                let df_pb = mpb.add(ProgressBar::new(exec_statements));
                df_pb.set_message(format!("DataFusion {filename}"));
                df_pb.set_style(pb_style());

                let mut df_runner = Runner::new(|| async {
                    Ok(DataFusion::new(
                        session.clone(),
                        path.clone(),
                        df_pb.clone(),
                    ))
                });

                df_runner.add_label("datafusion");
                df_runner.with_column_validator(strict_column_validator);
                df_runner.with_normalizer(value_normalizer);
                df_runner.with_validator(df_value_validator);

                for record in records.iter() {
                    if let Record::Halt { .. } = record {
                        break;
                    }

                    if let Err(e) = df_runner.run_async(record.clone()).await {
                        errors.push(format!("DF Failure: {e}"));
                    }
                }

                df_pb.finish();
            }

            if !path
                .components()
                .any(|comp| comp.as_os_str() == "datafusion")
            {
                let duckdb_pb = mpb.add(ProgressBar::new(exec_statements));
                duckdb_pb.set_message(format!("DuckDB {filename}"));
                duckdb_pb.set_style(pb_style());

                let mut duckdb_runner = Runner::new(|| async {
                    DuckDB::try_new(duckdb_pb.clone())
                        .map_err(|e| DuckDBTestError::Other(e.to_string()))
                });

                duckdb_runner.add_label("duckdb");
                duckdb_runner.with_column_validator(strict_column_validator);
                duckdb_runner.with_normalizer(value_normalizer);
                duckdb_runner.with_validator(duckdb_validator);

                for record in records.iter() {
                    if let Record::Halt { .. } = record {
                        break;
                    }

                    if let Err(e) = duckdb_runner.run_async(record.clone()).await {
                        errors.push(format!("DuckDB Failure: {e}"));
                    }
                }

                duckdb_pb.finish();
            }

            anyhow::Ok(errors)
        }
    })
    .buffer_unordered(args.test_threads)
    .try_collect::<Vec<_>>()
    .await?;

    let errors = all_errors.into_iter().flatten().collect::<Vec<_>>();
    for err in &errors {
        eprintln!("Failure: {err}");
    }

    if !has_tpch_data {
        eprintln!("Skipping TPC-H sqllogictests because slt/tpch/data is not present.");
    }

    if !errors.is_empty() {
        anyhow::bail!("{} sqllogictest failure(s)", errors.len());
    }

    Ok(())
}
