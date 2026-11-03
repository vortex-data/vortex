// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::sync::Arc;

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
use sqllogictest::strict_column_validator;
use vortex::VortexSessionDefault;
use vortex::editions::CORE_2026_08_3;
use vortex::editions::EditionSessionExt;
use vortex::session::VortexSession;
use vortex_datafusion::VortexFormatFactory;
use vortex_datafusion::VortexTableOptions;
use vortex_sqllogictest::normalize::PathNormalizing;
use vortex_sqllogictest::normalize::WORK_DIR_VAR;
use vortex_sqllogictest::scratch::WorkDirGuard;
use vortex_sqllogictest::scratch::reset_dir;

use crate::Mode;
use crate::run_or_complete;

/// Builds a single-threaded Tokio runtime for one DataFusion test file.
///
/// `libtest-mimic` runs each trial on its own thread, so a current-thread
/// runtime keeps async DataFusion work isolated per file instead of contending
/// for shared multi-threaded runtime workers.
fn build_runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

/// Runs or completes a single `.slt` file against DataFusion.
pub fn drive_datafusion(path: &Path, work_dir: &Path, mode: Mode) -> anyhow::Result<()> {
    reset_dir(work_dir)?;
    let _guard = WorkDirGuard::new(work_dir.to_path_buf());
    // Windows get us backslashes slashes but Duckdb produces forward slashes
    let work_dir = work_dir.to_string_lossy().replace('\\', "/");

    let rt = build_runtime()?;
    rt.block_on(async {
        // Keep EXPLAIN plans independent of the host's CPU count.
        let config = SessionConfig::default()
            .with_target_partitions(4)
            .with_option_extension(VortexTableOptions::default());
        let vortex_session = VortexSession::default();
        vortex_session.enable_edition(CORE_2026_08_3)?;
        let factory = Arc::new(VortexFormatFactory::new_with_session(vortex_session));
        let session_state_builder = SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_table_factory(
                factory.get_ext().to_uppercase(),
                Arc::new(DefaultTableFactory::new()),
            );
        let mut session_state = session_state_builder.build();
        session_state.register_file_format(factory, false)?;
        // The workspace builds `datafusion` without the `nested_expressions` feature, so array
        // functions (e.g. `make_array`, `array_length`) are not registered by default. Register
        // them explicitly so SLT files can construct and query list columns.
        datafusion_functions_nested::register_all(&mut session_state)?;
        let session = SessionContext::new_with_state(session_state).enable_url_table();

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
