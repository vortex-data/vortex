// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "perfetto")]
use std::fs::File;
use std::io::IsTerminal;

use clap::ValueEnum;
use tracing::level_filters::LevelFilter;
#[cfg(feature = "perfetto")]
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

    #[cfg(feature = "perfetto")]
    let perfetto_layer = perfetto
        .then(|| {
            Ok::<_, anyhow::Error>(
                PerfettoLayer::new(File::create("trace.json")?).with_debug_annotations(true),
            )
        })
        .transpose()?;

    #[cfg(not(feature = "perfetto"))]
    if perfetto {
        eprintln!(
            "warning: tracing/Perfetto export was requested but vortex-bench was built \
             without the `perfetto` feature; no trace will be written"
        );
    }

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

    let registry = tracing_subscriber::registry().with(filter);
    #[cfg(feature = "perfetto")]
    let registry = registry.with(perfetto_layer);
    registry.with(fmt_layer).init();

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
