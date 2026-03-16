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
use indicatif::ProgressStyle;
use sqllogictest::Record;
use sqllogictest::Runner;
use sqllogictest::parse_file;
use sqllogictest::strict_column_validator;
use vortex::error::VortexExpect;
use vortex_datafusion::VortexFormatFactory;
use vortex_sqllogictest::args::Args;
use vortex_sqllogictest::duckdb::DuckDB;
use vortex_sqllogictest::duckdb::DuckDBTestError;
use vortex_sqllogictest::utils::list_files;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.list {
        eprintln!("Ignoring `--list` which is unsupported by `sqlogictests-runner`");

        return Ok(());
    }

    if args.filter.is_some() {
        eprintln!("Ignoring test filter for sqllogictest");
    }

    let mpb = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(1));

    let crate_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = crate_path.join("slt/");

    let all_errors = futures::stream::iter(list_files(path)?)
        .map(|path| {
            let mpb = mpb.clone();

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

                let session = SessionContext::new_with_state(session_state_builder.build())
                    .enable_url_table();

                let filename = path
                    .file_name()
                    .vortex_expect("must be file")
                    .to_string_lossy();
                let records = parse_file(path.as_path())?;

                if !path.components().any(|comp| comp.as_os_str() == "duckdb") {
                    let df_pb = mpb.add(ProgressBar::new(records.len() as u64));
                    df_pb.set_message(format!("DF {filename}"));
                    df_pb.set_style(ProgressStyle::default_spinner());

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

                    df_pb.finish_and_clear();
                }

                if !path
                    .components()
                    .any(|comp| comp.as_os_str() == "datafusion")
                {
                    let duckdb_pb = mpb.add(ProgressBar::new(records.len() as u64));
                    duckdb_pb.set_message(format!("DuckDB {filename}"));

                    let mut duckdb_runner = Runner::new(|| async {
                        DuckDB::try_new(duckdb_pb.clone())
                            .map_err(|e| DuckDBTestError::Other(e.to_string()))
                    });

                    duckdb_runner.add_label("duckdb");
                    duckdb_runner.with_column_validator(strict_column_validator);
                    duckdb_runner.with_normalizer(value_normalizer);

                    for record in records.iter() {
                        if let Record::Halt { .. } = record {
                            break;
                        }

                        if let Err(e) = duckdb_runner.run_async(record.clone()).await {
                            errors.push(format!("DuckDB Failure: {e}"));
                        }
                    }

                    duckdb_pb.finish_and_clear();
                }

                anyhow::Ok(errors)
            }
        })
        .buffer_unordered(args.test_threads)
        .try_collect::<Vec<_>>()
        .await?;

    for err in all_errors.into_iter().flatten() {
        eprintln!("Failure: {err}");
    }

    Ok(())
}
