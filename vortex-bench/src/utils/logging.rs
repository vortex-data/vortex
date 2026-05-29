// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::IsTerminal;

use clap::ValueEnum;
use tracing::level_filters::LevelFilter;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::prelude::*;

/// Format for the primary stderr log sink.
///
/// `Text` is the default human-readable formatter matching the historical behavior of this crate.
/// `Json` emits one newline-delimited JSON object per event, suitable for piping into `jq` or a log
/// aggregator.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    #[default]
    Text,
    Json,
}

/// Initialize logging/tracing for a benchmark, hardcoding [`LogFormat::Text`].
///
/// See [`setup_logging_and_tracing_with_format`] if you want to select JSON
/// output from a CLI flag.
pub fn setup_logging_and_tracing(verbose: bool, perfetto: bool) -> anyhow::Result<()> {
    setup_logging_and_tracing_with_format(verbose, perfetto, LogFormat::Text)
}

/// Initialize logging/tracing for a benchmark with an explicit stderr format.
///
/// - `verbose`: when `RUST_LOG` is unset, raises the default filter from `INFO` to `TRACE`. Has no
///   effect when `RUST_LOG` is set (the env var wins).
/// - `perfetto`: when `true`, additionally attaches a [`tracing_perfetto::PerfettoLayer`] that
///   writes span begin/end events to `trace.json` in the current directory. Intended to be loaded
///   into the Perfetto UI for flamegraph visualization.
/// - `format`: controls the primary stderr sink's formatting. See [`LogFormat`].
pub fn setup_logging_and_tracing_with_format(
    verbose: bool,
    perfetto: bool,
    format: LogFormat,
) -> anyhow::Result<()> {
    let filter = default_env_filter(verbose);

    let perfetto_layer = perfetto
        .then(|| {
            Ok::<_, anyhow::Error>(
                PerfettoLayer::new(File::create("trace.json")?).with_debug_annotations(true),
            )
        })
        .transpose()?;

    // `fmt::layer()` and `fmt::layer().json()` produce different concrete types,
    // so erase each to a `dyn Layer` via `.boxed()` and keep the registry uniform.
    let fmt_layer: Box<dyn Layer<_> + Send + Sync> = match format {
        LogFormat::Text => tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .with_ansi(std::io::stderr().is_terminal())
            .boxed(),
        LogFormat::Json => tracing_subscriber::fmt::layer()
            .json()
            .with_writer(std::io::stderr)
            .with_current_span(true)
            .with_span_list(true)
            .boxed(),
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(perfetto_layer)
        .with(fmt_layer)
        .init();

    Ok(())
}

/// Lance crates emit chatty `INFO`-level logs (dataset open/commit details, fragment reads, etc.)
/// that drown out benchmark output. We quiet the whole `lance` crate family to `WARN` by default.
/// Targets are module paths, so hyphenated crate names appear with underscores.
const QUIET_LANCE_TARGETS: &[&str] = &[
    "lance",
    "lance_arrow",
    "lance_core",
    "lance_datafusion",
    "lance_datagen",
    "lance_encoding",
    "lance_file",
    "lance_index",
    "lance_io",
    "lance_linalg",
    "lance_table",
];

pub fn default_env_filter(is_verbose: bool) -> EnvFilter {
    match EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_e) => {
            let default_level = if is_verbose {
                LevelFilter::TRACE
            } else {
                LevelFilter::INFO
            };

            let mut filter = EnvFilter::builder()
                .with_default_directive(default_level.into())
                .from_env_lossy();

            // Only silence Lance when the user has not explicitly opted into verbose output;
            // `--verbose` (or `RUST_LOG`) should still surface everything.
            if !is_verbose {
                for target in QUIET_LANCE_TARGETS {
                    filter = filter.add_directive(
                        format!("{target}=warn")
                            .parse()
                            .expect("hardcoded lance log directive is valid"),
                    );
                }
            }

            filter
        }
    }
}
