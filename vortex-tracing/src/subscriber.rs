// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper for installing a [`tracing`] subscriber that writes a Perfetto JSON
//! trace file, compatible with <https://ui.perfetto.dev>.
//!
//! This module is gated behind the `perfetto` feature because it pulls in
//! `tracing-perfetto` and `tracing-subscriber`. Binaries that already
//! configure their own subscriber can ignore this module and simply construct
//! the wrappers ([`crate::TracingReadAt`], [`crate::TracingSegmentSource`],
//! [`crate::TracingLayoutReader`]); the spans will flow into whichever
//! subscriber is installed.

use std::fs::File;
use std::path::Path;

use tracing::level_filters::LevelFilter;
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

/// Install a [`tracing`] subscriber that writes spans to a Perfetto JSON file.
///
/// The subscriber honours `RUST_LOG` when set (via [`EnvFilter`]); when unset,
/// it defaults to [`LevelFilter::INFO`] so that the [`crate::TARGET_IO`],
/// [`crate::TARGET_SEGMENT`], and [`crate::TARGET_LAYOUT`] spans emitted by
/// the Vortex wrappers are captured.
///
/// The file can be opened in <https://ui.perfetto.dev> directly. Span fields
/// (offsets, lengths, expressions, row ranges, durations) appear in the
/// per-span details panel when a span is selected.
///
/// Call this once from your binary or test entry point, typically before
/// constructing any wrapped Vortex reader.
pub fn install_perfetto(path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path.as_ref())?;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy()
    });
    let layer = PerfettoLayer::new(file).with_debug_annotations(true);
    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init()?;
    Ok(())
}

/// Build a Perfetto [`tracing_subscriber::Layer`] that can be composed into a
/// caller-owned subscriber registry.
///
/// Use this when you already configure a subscriber (for example, to attach a
/// `fmt` layer for stderr logs) and want to add Perfetto output alongside it.
///
/// ```ignore
/// use tracing_subscriber::prelude::*;
/// use tracing_subscriber::EnvFilter;
///
/// let perfetto = vortex_tracing::subscriber::perfetto_layer("trace.json")?;
/// tracing_subscriber::registry()
///     .with(EnvFilter::from_default_env())
///     .with(perfetto)
///     .with(tracing_subscriber::fmt::layer())
///     .init();
/// ```
pub fn perfetto_layer(path: impl AsRef<Path>) -> std::io::Result<PerfettoLayer<File>> {
    Ok(PerfettoLayer::new(File::create(path.as_ref())?).with_debug_annotations(true))
}
