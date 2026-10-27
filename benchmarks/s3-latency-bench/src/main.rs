// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures object store (S3) request latency for Vortex-like files and scan request patterns.
//!
//! - `prepare` writes a dataset of synthetic Vortex-like files and a manifest.
//! - `run` replays the workloads and writes one CSV row per request, for fitting a latency model.
//! - `sweep` deletes objects older than a maximum age, giving the dataset a retention period.

#![allow(
    clippy::cast_possible_truncation,
    reason = "sizes, counts and durations here are far below the truncation limits"
)]

mod layout;
mod recorder;
mod workloads;

use std::collections::BTreeMap;
use std::fs;
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::bail;
use bytes::Bytes;
use clap::Parser;
use clap::Subcommand;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use serde::Deserialize;
use serde::Serialize;
use url::Url;

use crate::layout::FileLayout;
use crate::recorder::CSV_HEADER;
use crate::recorder::Recorder;
use crate::recorder::csv_row;
use crate::workloads::Ctx;
use crate::workloads::Dataset;

#[derive(Parser)]
#[command(about = "Object store latency benchmark for Vortex-like scans")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate and upload the dataset.
    Prepare {
        /// `s3://bucket/prefix` or `file:///absolute/dir`.
        #[arg(long)]
        url: Url,
        #[arg(long, default_value_t = 100)]
        files: usize,
        #[arg(long, default_value_t = 20)]
        file_size_mib: u64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 16)]
        upload_concurrency: usize,
    },
    /// Run the workloads against a prepared dataset.
    Run {
        #[arg(long)]
        url: Url,
        /// Label stored in every CSV row, e.g. the CI attempt number.
        #[arg(long, default_value = "local")]
        run_label: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "cold,size_sweep,suffix,concurrency,scan"
        )]
        phases: Vec<String>,
        /// Multiplies repetitions and file counts.
        #[arg(long, default_value_t = 1.0)]
        scale: f64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Process-wide cap on requests in flight.
        #[arg(long, default_value_t = 1024)]
        max_inflight: usize,
        /// Extra `key=value` metadata recorded in `meta.json`.
        #[arg(long)]
        meta: Vec<String>,
    },
    /// Delete objects under the prefix older than `max_age_hours`.
    Sweep {
        #[arg(long)]
        url: Url,
        #[arg(long, default_value_t = 3.0)]
        max_age_hours: f64,
    },
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    file_size: u64,
    seed: u64,
    created_unix: u64,
    names: Vec<String>,
    sizes: Vec<u64>,
}

fn layouts(count: usize, file_size: u64, seed: u64) -> anyhow::Result<Vec<FileLayout>> {
    (0..count)
        .map(|i| {
            FileLayout::generate(
                format!("file-{i:04}.vortex"),
                file_size,
                seed.wrapping_add(i as u64),
            )
        })
        .collect::<anyhow::Result<_>>()
}

fn open_store(url: &Url) -> anyhow::Result<(Arc<dyn ObjectStore>, Path)> {
    match url.scheme() {
        "s3" => {
            let bucket = url.host_str().context("s3 url has no bucket")?;
            let store = AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?;
            Ok((Arc::new(store), Path::from(url.path())))
        }
        "file" => {
            let dir = url
                .to_file_path()
                .map_err(|()| anyhow::anyhow!("bad file url {url}"))?;
            fs::create_dir_all(&dir)?;
            Ok((
                Arc::new(LocalFileSystem::new_with_prefix(dir)?),
                Path::default(),
            ))
        }
        other => bail!("unsupported scheme {other}"),
    }
}

async fn prepare(
    url: &Url,
    files: usize,
    file_size_mib: u64,
    seed: u64,
    concurrency: usize,
) -> anyhow::Result<()> {
    let (store, prefix) = open_store(url)?;
    let layouts = layouts(files, file_size_mib << 20, seed)?;
    let started = std::time::Instant::now();
    stream::iter(layouts.iter().enumerate())
        .map(|(i, layout)| {
            let store = Arc::clone(&store);
            let path = prefix.clone().join(layout.name.as_str());
            let layout = layout.clone();
            async move {
                let bytes =
                    tokio::task::spawn_blocking(move || layout.bytes(seed + i as u64)).await?;
                store
                    .put(&path, PutPayload::from(Bytes::from(bytes)))
                    .await?;
                anyhow::Ok(())
            }
        })
        .buffer_unordered(concurrency)
        .try_collect::<Vec<_>>()
        .await?;

    let manifest = Manifest {
        file_size: file_size_mib << 20,
        seed,
        created_unix: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        names: layouts.iter().map(|l| l.name.clone()).collect(),
        sizes: layouts.iter().map(|l| l.size).collect(),
    };
    store
        .put(
            &prefix.clone().join("manifest.json"),
            PutPayload::from(serde_json::to_vec_pretty(&manifest)?),
        )
        .await?;
    let total: u64 = manifest.sizes.iter().sum();
    let (segments, footers_spilled) = layouts.iter().fold((0, 0), |(s, f), l| {
        (
            s + l.segments.len(),
            f + usize::from(l.open_reads().1.is_some()),
        )
    });
    println!(
        "uploaded {files} files, {:.1} MiB total, {segments} segments, {footers_spilled} files \
         need a second footer read, in {:.1}s",
        total as f64 / f64::from(1 << 20),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

#[expect(clippy::too_many_arguments)]
async fn run(
    url: &Url,
    run_label: &str,
    output: &std::path::Path,
    phases: &[String],
    scale: f64,
    seed: u64,
    max_inflight: usize,
    meta: &[String],
) -> anyhow::Result<()> {
    let (store, prefix) = open_store(url)?;
    let manifest: Manifest = serde_json::from_slice(
        &store
            .get(&prefix.clone().join("manifest.json"))
            .await?
            .bytes()
            .await?,
    )?;
    let files = layouts(manifest.names.len(), manifest.file_size, manifest.seed)?;
    for (layout, size) in files.iter().zip(&manifest.sizes) {
        if layout.size != *size {
            bail!("layout of {} does not match the manifest", layout.name);
        }
    }

    let rec = Arc::new(Recorder::new(Arc::clone(&store), max_inflight));
    let ctx = Ctx {
        rec: Arc::clone(&rec),
        ds: Arc::new(Dataset { prefix, files }),
        scale,
        seed,
    };

    fs::create_dir_all(output)?;
    let mut csv = BufWriter::new(fs::File::create(output.join("requests.csv"))?);
    writeln!(csv, "{CSV_HEADER}")?;
    let mut scans = Vec::new();
    let mut phase_times = BTreeMap::new();
    for phase in phases {
        let start = rec.elapsed_us();
        match phase.as_str() {
            "cold" => workloads::cold(&ctx).await,
            "size_sweep" => workloads::size_sweep(&ctx).await,
            "suffix" => workloads::suffix(&ctx).await,
            "concurrency" => workloads::concurrency(&ctx).await,
            "scan" => scans.extend(workloads::scans(&ctx).await),
            other => bail!("unknown phase {other}"),
        }
        let end = rec.elapsed_us();
        phase_times.insert(phase.clone(), (start, end));

        let records = rec.take_records();
        let mut by_kind: BTreeMap<(String, &str), Vec<u64>> = BTreeMap::new();
        let mut errors = 0;
        for r in &records {
            writeln!(csv, "{}", csv_row(run_label, r))?;
            errors += usize::from(r.error.is_some());
            by_kind
                .entry((r.tag.scenario.clone(), r.tag.kind))
                .or_default()
                .push(r.total_us());
        }
        println!(
            "== {phase}: {} requests, {errors} errors, {:.1}s",
            records.len(),
            (end - start) as f64 / 1e6
        );
        if phase != "size_sweep" && phase != "concurrency" {
            for ((scenario, kind), mut lat) in by_kind {
                lat.sort_unstable();
                println!(
                    "  {scenario:<24} {kind:<12} n={:<6} p50={:>7.1}ms p90={:>7.1}ms p99={:>7.1}ms",
                    lat.len(),
                    percentile(&lat, 0.5) as f64 / 1e3,
                    percentile(&lat, 0.9) as f64 / 1e3,
                    percentile(&lat, 0.99) as f64 / 1e3,
                );
            }
        }
    }
    csv.flush()?;

    for s in &scans {
        let secs = (s.end_us - s.start_us) as f64 / 1e6;
        println!(
            "  scan {:<24} #{:<2} {:>4} files {:>8.1} MiB in {:>6.2}s = {:>7.1} MiB/s",
            s.scenario,
            s.scan_id,
            s.files,
            s.bytes as f64 / f64::from(1 << 20),
            secs,
            s.bytes as f64 / f64::from(1 << 20) / secs.max(1e-9),
        );
    }
    fs::write(
        output.join("scans.json"),
        serde_json::to_vec_pretty(&scans)?,
    )?;

    let meta: BTreeMap<&str, &str> = meta.iter().filter_map(|kv| kv.split_once('=')).collect();
    let meta_json = serde_json::json!({
        "run_label": run_label,
        "url": url.as_str(),
        "scale": scale,
        "seed": seed,
        "max_inflight": max_inflight,
        "files": ctx.ds.files.len(),
        "file_size": manifest.file_size,
        "phases": phase_times,
        "meta": meta,
    });
    fs::write(
        output.join("meta.json"),
        serde_json::to_vec_pretty(&meta_json)?,
    )?;
    Ok(())
}

async fn sweep(url: &Url, max_age_hours: f64) -> anyhow::Result<()> {
    let (store, prefix) = open_store(url)?;
    let cutoff = chrono_cutoff(max_age_hours)?;
    let expired: Vec<Path> = store
        .list(Some(&prefix))
        .try_filter(|meta| {
            let old = meta.last_modified.timestamp() < cutoff;
            async move { old }
        })
        .map_ok(|meta| meta.location)
        .try_collect()
        .await?;
    println!(
        "deleting {} objects under {url} older than {max_age_hours}h",
        expired.len()
    );
    let deleted: Vec<Path> = store
        .delete_stream(stream::iter(expired.into_iter().map(Ok)).boxed())
        .try_collect()
        .await?;
    println!("deleted {}", deleted.len());
    Ok(())
}

fn chrono_cutoff(max_age_hours: f64) -> anyhow::Result<i64> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let age = Duration::from_secs_f64(max_age_hours * 3600.0);
    Ok(i64::try_from(now.saturating_sub(age).as_secs())?)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Prepare {
            url,
            files,
            file_size_mib,
            seed,
            upload_concurrency,
        } => prepare(&url, files, file_size_mib, seed, upload_concurrency).await,
        Command::Run {
            url,
            run_label,
            output,
            phases,
            scale,
            seed,
            max_inflight,
            meta,
        } => {
            run(
                &url,
                &run_label,
                &output,
                &phases,
                scale,
                seed,
                max_inflight,
                &meta,
            )
            .await
        }
        Command::Sweep { url, max_age_hours } => sweep(&url, max_age_hours).await,
    }
}
