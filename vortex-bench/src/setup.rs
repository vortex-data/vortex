// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dataset materialization: the [`SetupCtx`] handed to [`Benchmark::setup`] and the driver
//! that turns natively-produced formats into every requested one.
//!
//! # Model
//!
//! A benchmark declares the formats it can produce without help via
//! [`Benchmark::native_formats`], and materializes one of them in
//! [`Benchmark::setup`]. Everything else is derived by [`prepare_data`].
//!
//! Parquet and Vortex convert in both directions, so either can act as the pivot: a
//! Parquet-native suite derives Vortex, and a Vortex-native suite derives Parquet. A suite
//! that produces neither can only serve the formats it lists.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use parking_lot::Mutex;
use reqwest::Client;
use tracing::info;

use crate::Benchmark;
use crate::CompactionStrategy;
use crate::Format;
use crate::conversions::convert_parquet_directory_to_vortex;
use crate::conversions::convert_vortex_directory_to_parquet;
use crate::datasets::data_downloads::download_many;
use crate::datasets::data_downloads::http_client;

/// A parquet (or other native-format) file produced by [`Benchmark::setup`], tagged with the
/// table it belongs to.
///
/// Tables are many-to-one with files: ClickBench emits 100 files for the single `hits` table,
/// while Appian emits nine files for nine tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emitted {
    pub table: String,
    pub path: PathBuf,
}

/// Context handed to [`Benchmark::setup`].
///
/// Owns the staging directory the benchmark writes into, the shared download pool, and the
/// list of files the benchmark has produced so far.
pub struct SetupCtx {
    staging: PathBuf,
    emitted: Mutex<Vec<Emitted>>,
}

impl SetupCtx {
    /// Create a context staging into `staging`, creating the directory if needed.
    ///
    /// `staging` is deliberately stable across runs rather than a fresh temp dir, so a failure
    /// partway through a multi-file dataset leaves the already-fetched files in place for the
    /// next attempt.
    pub fn new(staging: impl Into<PathBuf>) -> Result<Self> {
        let staging = staging.into();
        std::fs::create_dir_all(&staging)?;
        Ok(Self {
            staging,
            emitted: Mutex::new(Vec::new()),
        })
    }

    /// Directory this benchmark should write its output into.
    pub fn staging(&self) -> &Path {
        &self.staging
    }

    /// Idempotently fetch `(url, path)` pairs through the shared download pool.
    ///
    /// Pass every file the benchmark needs in one call: the pool ramps concurrency across the
    /// whole batch and renders a single progress block. Files already on disk are skipped.
    pub async fn download<I, S, P>(&self, files: I) -> Result<Vec<PathBuf>>
    where
        I: IntoIterator<Item = (S, P)>,
        S: Into<String>,
        P: Into<PathBuf>,
    {
        // `download_many` takes `(path, url)`; flip so call sites read `(url, path)`.
        let downloads: Vec<(PathBuf, String)> = files
            .into_iter()
            .map(|(url, path)| (path.into(), url.into()))
            .collect();
        download_many(downloads).await
    }

    /// The shared HTTP client, for datasets that must consume a response stream rather than
    /// land a file — see [`crate::statpopgen`], which decodes a multi-hundred-gigabyte VCF on
    /// the fly and stops after a fixed row count.
    pub fn http(&self) -> &'static Client {
        http_client()
    }

    /// Register a file this benchmark produced as (part of) `table`.
    pub fn emit(&self, table: impl Into<String>, path: impl Into<PathBuf>) {
        self.emitted.lock().push(Emitted {
            table: table.into(),
            path: path.into(),
        });
    }

    /// Every file registered via [`SetupCtx::emit`], in emission order.
    pub fn emitted(&self) -> Vec<Emitted> {
        self.emitted.lock().clone()
    }
}

/// Materialize `formats` for `benchmark`.
///
/// Formats the benchmark lists in [`Benchmark::native_formats`] are produced directly by
/// [`Benchmark::setup`]. The Vortex formats are derived from Parquet when the benchmark does
/// not produce them natively. Benchmarks whose data already lives remotely (a non-`file://`
/// [`Benchmark::data_url`]) are skipped entirely.
pub async fn prepare_data(benchmark: &dyn Benchmark, formats: &[Format]) -> Result<()> {
    if benchmark.data_url().scheme() != "file" {
        info!(
            "{}: data is remote ({}), nothing to materialize",
            benchmark.dataset_name(),
            benchmark.data_url(),
        );
        return Ok(());
    }

    let base_path = benchmark
        .data_url()
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

    let native = benchmark.native_formats();

    let plans = formats
        .iter()
        // Lance and DuckDB are built by their own bench binaries from the Parquet below.
        .filter(|f| !matches!(f, Format::Lance | Format::OnDiskDuckDB | Format::Csv))
        .map(|&f| plan(native, f))
        .collect::<Result<Vec<_>>>()
        .with_context(|| format!("benchmark {}", benchmark.dataset_name()))?;

    // Run `setup` once per source format any plan depends on, before any derivation reads it.
    let mut sources: Vec<Format> = plans.iter().map(Plan::source).collect();
    sources.sort_unstable_by_key(|f| f.name());
    sources.dedup();
    for source in sources {
        setup_format(benchmark, &base_path, source).await?;
    }

    for plan in plans {
        match plan {
            // Produced by the source loop above.
            Plan::Native(_) => {}
            Plan::DeriveFromParquet(format, compaction) => {
                convert_parquet_directory_to_vortex(&base_path, compaction).await?;
                benchmark.prepare_format(format, &base_path).await?;
            }
            Plan::DeriveFromVortex(format) => {
                convert_vortex_directory_to_parquet(&base_path).await?;
                benchmark.prepare_format(format, &base_path).await?;
            }
        }
    }

    Ok(())
}

/// Run `benchmark`'s setup for one natively-produced format, staging into `<base>/<format>/`.
async fn setup_format(benchmark: &dyn Benchmark, base_path: &Path, format: Format) -> Result<()> {
    let staging = base_path.join(format.name());
    let ctx = SetupCtx::new(&staging)?;

    benchmark.setup(&ctx, format).await?;

    let emitted = ctx.emitted();
    info!(
        "{}: setup produced {} {format} file(s) in {}",
        benchmark.dataset_name(),
        emitted.len(),
        staging.display(),
    );

    benchmark.prepare_format(format, base_path).await?;
    Ok(())
}

/// How one requested format gets produced.
///
/// Split out so the routing is testable without running a download or a generator.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// `setup` is called for this format directly.
    Native(Format),
    /// Parquet is produced, then converted with this strategy.
    DeriveFromParquet(Format, CompactionStrategy),
    /// Vortex is produced, then converted back to Parquet.
    DeriveFromVortex(Format),
}

impl Plan {
    /// The format `setup` must produce before this plan can run.
    fn source(&self) -> Format {
        match self {
            Plan::Native(format) => *format,
            Plan::DeriveFromParquet(..) => Format::Parquet,
            Plan::DeriveFromVortex(_) => Format::OnDiskVortex,
        }
    }
}

/// Pick how to produce `requested` given what the benchmark makes natively.
///
/// Parquet and Vortex convert in both directions, so either can serve as the pivot: a
/// Parquet-native suite derives Vortex, and a Vortex-native suite derives Parquet.
pub(crate) fn plan(native: &[Format], requested: Format) -> Result<Plan> {
    if native.contains(&requested) {
        return Ok(Plan::Native(requested));
    }

    match requested {
        Format::OnDiskVortex if native.contains(&Format::Parquet) => Ok(Plan::DeriveFromParquet(
            requested,
            CompactionStrategy::Default,
        )),
        Format::VortexCompact if native.contains(&Format::Parquet) => Ok(Plan::DeriveFromParquet(
            requested,
            CompactionStrategy::Compact,
        )),
        // Reverse direction: only plain on-disk Vortex is a valid source. Reading back a
        // compacted file would produce identical Parquet, so it is not worth a second path.
        Format::Parquet if native.contains(&Format::OnDiskVortex) => {
            Ok(Plan::DeriveFromVortex(requested))
        }
        other => bail!("cannot produce {other}: native formats are {native:?}"),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    /// StatPopGen streams HTTP straight to Parquet, then Vortex comes off that Parquet — the
    /// case that a URL-to-path download descriptor could not express.
    #[test]
    fn streaming_only_suite_derives_vortex_from_its_parquet() -> Result<()> {
        let native = [Format::Parquet];
        assert_eq!(
            plan(&native, Format::Parquet)?,
            Plan::Native(Format::Parquet)
        );
        assert_eq!(
            plan(&native, Format::OnDiskVortex)?,
            Plan::DeriveFromParquet(Format::OnDiskVortex, CompactionStrategy::Default),
        );
        Ok(())
    }

    /// TPC-H writes Vortex from its row generator, so it must not round-trip through Parquet.
    #[test]
    fn generator_that_writes_vortex_natively_skips_the_parquet_round_trip() -> Result<()> {
        let native = [Format::Parquet, Format::OnDiskVortex];
        assert_eq!(
            plan(&native, Format::OnDiskVortex)?,
            Plan::Native(Format::OnDiskVortex),
        );
        Ok(())
    }

    /// Conversion runs both ways, so a Vortex-only suite can serve a Parquet run.
    #[test]
    fn vortex_only_suite_derives_parquet() -> Result<()> {
        let native = [Format::OnDiskVortex];
        assert_eq!(
            plan(&native, Format::Parquet)?,
            Plan::DeriveFromVortex(Format::Parquet),
        );
        Ok(())
    }

    /// A format with no conversion path from anything native is still a hard error.
    #[test]
    fn unreachable_format_is_an_error() {
        assert!(plan(&[Format::OnDiskVortex], Format::Lance).is_err());
        // VortexCompact is only derivable from Parquet, which this suite does not produce.
        assert!(plan(&[Format::OnDiskVortex], Format::VortexCompact).is_err());
    }

    #[test]
    fn source_is_what_setup_must_produce() {
        assert_eq!(Plan::Native(Format::Parquet).source(), Format::Parquet);
        assert_eq!(
            Plan::DeriveFromParquet(Format::OnDiskVortex, CompactionStrategy::Default).source(),
            Format::Parquet,
        );
        assert_eq!(
            Plan::DeriveFromVortex(Format::Parquet).source(),
            Format::OnDiskVortex,
        );
    }

    #[test]
    fn emit_records_table_and_path_in_order() -> Result<()> {
        let dir = tempdir()?;
        let ctx = SetupCtx::new(dir.path())?;

        ctx.emit("hits", dir.path().join("hits_0.parquet"));
        ctx.emit("hits", dir.path().join("hits_1.parquet"));

        let emitted = ctx.emitted();
        assert_eq!(emitted.len(), 2);
        assert!(emitted.iter().all(|e| e.table == "hits"));
        assert_eq!(emitted[0].path, dir.path().join("hits_0.parquet"));
        assert_eq!(emitted[1].path, dir.path().join("hits_1.parquet"));
        Ok(())
    }

    #[test]
    fn new_creates_the_staging_directory() -> Result<()> {
        let dir = tempdir()?;
        let staging = dir.path().join("nested").join("parquet");
        let ctx = SetupCtx::new(&staging)?;
        assert!(staging.is_dir());
        assert_eq!(ctx.staging(), staging);
        Ok(())
    }
}
