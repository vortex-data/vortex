// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rewrite benchmark data through the default Vortex writer while auditing its size estimates.
//!
//! For every table of a benchmark, this writes the Parquet source to Vortex (the `data-gen` path),
//! then scans that Vortex file and writes it again (Vortex to Vortex). Run it with
//! `VORTEX_NBYTES_AUDIT=<file>` to record `nbytes` against `exact_nbytes` at each coalescing site,
//! and with `VORTEX_NBYTES_POLICY=exact` to make repartitioning use exact sizes.
//!
//! Tables are converted one at a time so audit records can be labelled per table. One JSON line
//! per written file, summarising its segments, is appended to `--summary`.

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use clap::value_parser;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::array::IntoArray;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::layout::nbytes_audit;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::SESSION;
use vortex_bench::conversions::convert_parquet_file_to_vortex;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::create_benchmark;
use vortex_bench::datasets::Dataset;
use vortex_bench::datasets::feature_vectors::feature_vectors_parquet;
use vortex_bench::datasets::nested_lists::nested_lists_parquet;
use vortex_bench::datasets::nested_structs::nested_structs_parquet;
use vortex_bench::datasets::struct_list_of_ints::StructListOfInts;

#[derive(Parser)]
struct Args {
    /// A query benchmark whose tables are generated locally (e.g. tpch, tpcds, polarsignals).
    #[arg(long, value_enum, conflicts_with = "dataset")]
    benchmark: Option<BenchmarkArg>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,

    /// A synthetic compress/random-access dataset: `struct-list-of-ints:<cols>:<rows>:<chunks>`,
    /// `feature-vectors`, `nested-lists` or `nested-structs`.
    #[arg(long)]
    dataset: Option<String>,

    /// Only convert tables whose file stem matches this regex.
    #[arg(long)]
    tables: Option<regex::Regex>,

    /// Directory for the Vortex files written; each is deleted once summarised.
    #[arg(long)]
    out_dir: PathBuf,

    /// JSONL file to append one summary line per written file to.
    #[arg(long)]
    summary: PathBuf,

    /// Stream a synthetic dataset from Parquet instead of loading it whole, for datasets too
    /// large to hold in memory.
    #[arg(long)]
    streaming: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.out_dir)?;

    let (name, tables) = if let Some(benchmark) = args.benchmark {
        let benchmark = create_benchmark(benchmark, &Opts::from(args.options))?;
        benchmark.generate_base_data().await?;
        let base = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("benchmark data is not local"))?;
        let mut tables = Vec::new();
        collect_parquet(&base.join("parquet"), &mut tables)?;
        tables.sort();
        (benchmark.dataset_name().to_string(), tables)
    } else if let Some(dataset) = args.dataset.as_deref() {
        (dataset.to_string(), vec![synthetic_parquet(dataset).await?])
    } else {
        anyhow::bail!("pass --benchmark or --dataset");
    };

    let in_memory = args.dataset.is_some() && !args.streaming;
    for parquet in tables {
        let stem = parquet
            .file_stem()
            .and_then(|stem| stem.to_str())
            .context("parquet file name")?
            .to_string();
        if args
            .tables
            .as_ref()
            .is_some_and(|tables| !tables.is_match(&stem))
        {
            continue;
        }

        let from_parquet = args.out_dir.join(format!("{stem}.from-parquet.vortex"));
        nbytes_audit::set_label(format!("{name}/{stem}/from-parquet"));
        if in_memory {
            // compress-bench and random-access-bench load the whole file as chunks first.
            let chunks = parquet_to_vortex_chunks(parquet.clone()).await?;
            write_stream(&from_parquet, chunks.into_array().to_array_stream()).await?;
        } else {
            convert_parquet_file_to_vortex(&parquet, &from_parquet, CompactionStrategy::Default)
                .await?;
        }
        summarise(&args.summary, &name, &stem, "from-parquet", &from_parquet).await?;

        let from_vortex = args.out_dir.join(format!("{stem}.from-vortex.vortex"));
        nbytes_audit::set_label(format!("{name}/{stem}/from-vortex"));
        let source = SESSION.open_options().open_path(&from_parquet).await?;
        write_stream(&from_vortex, source.scan()?.into_array_stream()?).await?;
        summarise(&args.summary, &name, &stem, "from-vortex", &from_vortex).await?;

        fs::remove_file(&from_parquet)?;
        fs::remove_file(&from_vortex)?;
    }
    Ok(())
}

async fn write_stream(
    path: &Path,
    stream: impl vortex::array::stream::ArrayStream + Send + Unpin + 'static,
) -> anyhow::Result<()> {
    let mut file = File::create(path).await?;
    SESSION.write_options().write(&mut file, stream).await?;
    file.flush().await?;
    Ok(())
}

async fn synthetic_parquet(dataset: &str) -> anyhow::Result<PathBuf> {
    let mut parts = dataset.split(':');
    match parts.next() {
        Some("struct-list-of-ints") => {
            let mut next = || -> anyhow::Result<usize> {
                Ok(parts
                    .next()
                    .context("struct-list-of-ints:<cols>:<rows>:<chunks>")?
                    .parse()?)
            };
            let (cols, rows, chunks) = (next()?, next()?, next()?);
            StructListOfInts::new(cols, rows, chunks)
                .to_parquet_path()
                .await
        }
        Some("feature-vectors") => feature_vectors_parquet().await,
        Some("nested-lists") => nested_lists_parquet().await,
        Some("nested-structs") => nested_structs_parquet().await,
        _ => anyhow::bail!("unknown dataset {dataset}"),
    }
}

fn collect_parquet(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_parquet(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "parquet") {
            out.push(path);
        }
    }
    Ok(())
}

/// Append one line describing the file's size and segment-size distribution.
async fn summarise(
    summary: &Path,
    name: &str,
    table: &str,
    path_kind: &str,
    file: &Path,
) -> anyhow::Result<()> {
    let file_bytes = fs::metadata(file)?.len();
    let vortex_file = SESSION.open_options().open_path(file).await?;
    let mut segments: Vec<u64> = vortex_file
        .footer()
        .segment_map()
        .iter()
        .map(|segment| u64::from(segment.length))
        .collect();
    segments.sort_unstable();
    let quantile = |q: f64| -> u64 {
        if segments.is_empty() {
            return 0;
        }
        let idx = ((segments.len() - 1) as f64 * q).round() as usize;
        segments[idx]
    };

    let line = serde_json::json!({
        "dataset": name,
        "table": table,
        "path": path_kind,
        "policy": if nbytes_audit::use_exact_policy() { "exact" } else { "nbytes" },
        "rows": vortex_file.row_count(),
        "file_bytes": file_bytes,
        "segments": segments.len(),
        "segment_p10": quantile(0.1),
        "segment_p50": quantile(0.5),
        "segment_p90": quantile(0.9),
        "segment_max": segments.last().copied().unwrap_or(0),
    });
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(summary)?;
    writeln!(out, "{line}")?;
    println!("{line}");
    Ok(())
}
