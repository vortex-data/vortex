// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::IsTerminal;

use tracing::level_filters::LevelFilter;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

/// Initialize logging/tracing for a benchmark
pub fn setup_logging_and_tracing(verbose: bool, tracing: bool) -> anyhow::Result<()> {
    let filter = default_env_filter(verbose);

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(std::io::stderr().is_terminal());

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing
                .then(|| {
                    Ok::<_, anyhow::Error>(
                        PerfettoLayer::new(File::create("trace.json")?)
                            .with_debug_annotations(true),
                    )
                })
                .transpose()?,
        )
        .with(fmt_layer)
        .init();

    Ok(())
}

pub fn default_env_filter(is_verbose: bool) -> EnvFilter {
    match EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_e) => {
            let default_level = if is_verbose {
                LevelFilter::TRACE
            } else {
                LevelFilter::INFO
            };

            EnvFilter::builder()
                .with_default_directive(default_level.into())
                .from_env_lossy()
        }
    }
}
