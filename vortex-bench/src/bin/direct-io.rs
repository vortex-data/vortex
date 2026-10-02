// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measure direct I/O reads and block-aligned segment writes.
//!
//! `convert` writes Vortex files under each [`SegmentPadding`] policy, `analyze` replays every
//! policy against an already-written file's segment map to price it without rewriting, and `scan`
//! times full scans through the layout reader's `ScanBuilder` with buffered and direct reads.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use futures::StreamExt;
use futures::TryStreamExt;
use humansize::DECIMAL;
use humansize::format_size;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use vortex::array::ArrayRef;
use vortex::array::memory::BufferAllocatorRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Alignment;
use vortex::dtype::FieldName;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::SegmentSpec;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::segments::SegmentPadding;
use vortex::io::runtime::Handle;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::DEFAULT_CONCURRENCY;
use vortex::io::std_file::FileReadAt;
use vortex::io::std_file::FileReadAtOptions;
use vortex::layout::segments::SegmentId;
use vortex_arrow::ArrowSessionExt;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_stream;
use vortex_bench::setup_logging_and_tracing;

/// Block size every policy in this tool aligns to.
const BLOCK: u64 = 4096;

/// The same block size as an [`Alignment`], for building padding policies.
const BLOCK_ALIGNMENT: Alignment = Alignment::new(4096);

#[derive(Parser)]
#[command(
    name = "direct-io",
    about = "Direct I/O read and segment alignment measurements"
)]
struct Args {
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert Parquet files to Vortex under each padding policy.
    Convert {
        /// Parquet files, or directories of them.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Directory to write Vortex files into.
        #[arg(short, long)]
        out: PathBuf,
        /// Padding policies to write.
        #[arg(long, value_delimiter = ',', default_values = ["none", "block"])]
        padding: Vec<Padding>,
    },
    /// Price every padding policy against already-written Vortex files.
    Analyze {
        /// Vortex files, or directories of them.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Overhead ratios to sweep for the proportional policy.
        #[arg(long, value_delimiter = ',', default_values_t = [4u32, 8, 16, 32, 64, 128, 256])]
        ratios: Vec<u32>,
    },
    /// Time full scans with buffered and direct reads.
    Scan {
        /// Vortex files, or directories of them.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(short, long, default_value_t = 3)]
        iterations: usize,
        /// Restrict the scan to these top-level columns.
        #[arg(long, value_delimiter = ',')]
        columns: Vec<String>,
        /// Drop the page cache before every iteration, measuring cold reads.
        #[arg(long)]
        cold: bool,
        /// Resolve every segment without decoding, isolating the read pipeline from decompression.
        #[arg(long)]
        io_only: bool,
        /// Number of files to scan concurrently.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Padding {
    /// Pack segments contiguously.
    None,
    /// Start every segment on a 4KiB boundary.
    Block,
    /// Pack consecutive segments into shared 4KiB blocks.
    Grouped,
    /// Block-align a segment when the padding is within 1/64 of its length.
    Proportional,
}

impl Padding {
    fn policy(self) -> SegmentPadding {
        match self {
            Self::None => SegmentPadding::None,
            Self::Block => SegmentPadding::block_aligned(),
            Self::Grouped => SegmentPadding::grouped(),
            Self::Proportional => SegmentPadding::proportional(),
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Block => "block",
            Self::Grouped => "grouped",
            Self::Proportional => "proportional",
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, false)?;

    match args.command {
        Command::Convert {
            inputs,
            out,
            padding,
        } => convert(&collect(&inputs, "parquet")?, &out, &padding).await,
        Command::Analyze { inputs, ratios } => analyze(&collect(&inputs, "vortex")?, &ratios).await,
        Command::Scan {
            inputs,
            iterations,
            columns,
            cold,
            io_only,
            concurrency,
        } => {
            scan(
                &collect(&inputs, "vortex")?,
                iterations,
                ScanConfig {
                    columns,
                    cold,
                    io_only,
                    concurrency,
                },
            )
            .await
        }
    }
}

/// Expand directories into the files inside them with the given extension.
fn collect(inputs: &[PathBuf], extension: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            for entry in fs::read_dir(input)? {
                let path = entry?.path();
                if path.extension().is_some_and(|ext| ext == extension) {
                    files.push(path);
                }
            }
        } else {
            files.push(input.clone());
        }
    }
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no .{extension} files found");
    Ok(files)
}

async fn convert(inputs: &[PathBuf], out: &Path, padding: &[Padding]) -> anyhow::Result<()> {
    fs::create_dir_all(out)?;

    let mut totals: BTreeMap<&str, u64> = BTreeMap::new();
    let mut parquet_total = 0;
    for input in inputs {
        let stem = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("data");
        parquet_total += fs::metadata(input)?.len();

        for padding in padding {
            let output = out.join(format!("{stem}.{}.vortex", padding.suffix()));
            let elapsed = if output.exists() {
                Duration::ZERO
            } else {
                let start = Instant::now();
                write_vortex(input, &output, padding.policy()).await?;
                start.elapsed()
            };
            let size = fs::metadata(&output)?.len();
            *totals.entry(padding.suffix()).or_default() += size;
            println!(
                "{stem:<24} {:<13} {:>12}  ({:.1}s)",
                padding.suffix(),
                format_size(size, DECIMAL),
                elapsed.as_secs_f64(),
            );
        }
    }

    println!("\nparquet total: {}", format_size(parquet_total, DECIMAL));
    let baseline = totals.get("none").copied();
    for (policy, size) in &totals {
        let growth = baseline
            .map(|baseline| format!("  {:+.3}%", percent_change(baseline, *size)))
            .unwrap_or_default();
        println!("{policy:<13} {:>12}{growth}", format_size(*size, DECIMAL));
    }
    Ok(())
}

async fn write_vortex(input: &Path, output: &Path, padding: SegmentPadding) -> anyhow::Result<()> {
    let file = tokio::fs::File::open(input).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let dtype = SESSION
        .arrow()
        .from_arrow_schema(builder.schema().as_ref())?;
    let stream = parquet_to_vortex_stream(builder.build()?);

    let mut out = tokio::fs::File::create(output).await?;
    SESSION
        .write_options()
        .with_segment_padding(padding)
        .write(
            &mut out,
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream)),
        )
        .await?;
    Ok(())
}

/// Coalescing window the file reader applies to local files, mirroring `CoalesceConfig::file`.
const COALESCE_DISTANCE: u64 = 1 << 20;
const COALESCE_MAX_SIZE: u64 = 4 << 20;

/// What a padding policy would cost, replayed over an existing file's segment map.
#[derive(Default)]
struct Replay {
    file_size: u64,
    padding: u64,
    aligned_segments: usize,
    /// Segment bytes living in a block-aligned segment.
    aligned_bytes: u64,
    /// Bytes a direct-I/O reader transfers when each segment is read on its own.
    uncoalesced_bytes: u64,
    /// Bytes a direct-I/O reader transfers once neighbouring segments are coalesced.
    coalesced_bytes: u64,
}

fn replay(specs: &[SegmentSpec], start: u64, policy: SegmentPadding) -> Replay {
    let mut offset = start;
    let mut replay = Replay::default();
    // Placed segments, in offset order, so the coalescing pass can walk them.
    let mut placed: Vec<(u64, u64)> = Vec::with_capacity(specs.len());

    for spec in specs {
        let length = u64::from(spec.length);
        let pad = policy.padding(offset, length, spec.alignment);
        offset += pad;
        replay.padding += pad;
        if offset.is_multiple_of(BLOCK) {
            replay.aligned_segments += 1;
            replay.aligned_bytes += length;
        }
        // A direct read is widened to the blocks it overlaps, so a segment straddling a boundary
        // transfers one more block than its length alone implies.
        replay.uncoalesced_bytes += widen(offset, length);
        placed.push((offset, length));
        offset += length;
    }
    replay.file_size = offset;

    // Coalesce neighbours the way the read driver does, then widen each physical read once.
    let mut run: Option<(u64, u64)> = None;
    for (offset, length) in placed {
        run = Some(match run {
            Some((start, end))
                if offset.saturating_sub(end) <= COALESCE_DISTANCE
                    && offset + length - start <= COALESCE_MAX_SIZE =>
            {
                (start, end.max(offset + length))
            }
            Some((start, end)) => {
                replay.coalesced_bytes += widen(start, end - start);
                (offset, offset + length)
            }
            None => (offset, offset + length),
        });
    }
    if let Some((start, end)) = run {
        replay.coalesced_bytes += widen(start, end - start);
    }

    replay
}

/// Bytes transferred by a direct read of `offset..offset + length`, widened to whole blocks.
fn widen(offset: u64, length: u64) -> u64 {
    if length == 0 {
        return 0;
    }
    (offset % BLOCK + length).div_ceil(BLOCK) * BLOCK
}

async fn analyze(inputs: &[PathBuf], ratios: &[u32]) -> anyhow::Result<()> {
    let mut all_specs = Vec::new();
    let mut baseline_size = 0;
    let mut trailer = 0;

    for input in inputs {
        let file = SESSION.open_options().open_path(input).await?;
        let specs = file.footer().segment_map().to_vec();
        let size = fs::metadata(input)?.len();
        let segment_end = specs
            .iter()
            .map(|spec| spec.offset + u64::from(spec.length))
            .max()
            .unwrap_or(0);
        trailer += size - segment_end;
        baseline_size += size;
        all_specs.push(specs);
    }

    let segments: Vec<SegmentSpec> = all_specs.iter().flatten().copied().collect();
    let data: u64 = segments.iter().map(|s| u64::from(s.length)).sum();
    println!("files:          {}", inputs.len());
    println!("segments:       {}", segments.len());
    println!("segment bytes:  {}", format_size(data, DECIMAL));
    println!("file bytes:     {}", format_size(baseline_size, DECIMAL));
    print_size_distribution(&segments);

    println!(
        "\n{:<28} {:>13} {:>8} {:>9} {:>9} {:>13} {:>13}",
        "policy", "file size", "growth", "aligned", "of bytes", "1 seg/io", "coalesced"
    );
    // The read columns are deltas against the packed baseline, so it must be replayed first.
    let mut policies = vec![
        ("none".to_string(), SegmentPadding::None),
        ("always (4KiB)".to_string(), SegmentPadding::block_aligned()),
        ("grouped (4KiB)".to_string(), SegmentPadding::grouped()),
    ];
    for ratio in ratios {
        policies.push((
            format!("proportional 1/{ratio}"),
            SegmentPadding::Proportional {
                block: BLOCK_ALIGNMENT,
                max_overhead_ratio: *ratio,
            },
        ));
    }

    // Every file starts its segments after the magic bytes, so replay each separately and sum.
    let totals: Vec<(String, Replay)> = policies
        .into_iter()
        .map(|(name, policy)| {
            let mut total = Replay::default();
            for specs in &all_specs {
                let start = specs.first().map(|spec| spec.offset).unwrap_or(0);
                let replayed = replay(specs, start, policy);
                total.file_size += replayed.file_size;
                total.padding += replayed.padding;
                total.aligned_segments += replayed.aligned_segments;
                total.aligned_bytes += replayed.aligned_bytes;
                total.uncoalesced_bytes += replayed.uncoalesced_bytes;
                total.coalesced_bytes += replayed.coalesced_bytes;
            }
            (name, total)
        })
        .collect();

    // Price every policy against the contiguously packed layout it would replace.
    let base = &totals[0].1;
    for (name, total) in &totals {
        println!(
            "{name:<28} {:>13} {:>7.3}% {:>8.1}% {:>8.1}% {:>7.2}% {:>7.2}%",
            format_size(total.file_size + trailer, DECIMAL),
            percent_change(baseline_size, total.file_size + trailer),
            100.0 * total.aligned_segments as f64 / segments.len() as f64,
            100.0 * total.aligned_bytes as f64 / data as f64,
            percent_change(base.uncoalesced_bytes, total.uncoalesced_bytes),
            percent_change(base.coalesced_bytes, total.coalesced_bytes),
        );
    }
    println!(
        "\nbytes read, packed baseline: {} (1 seg/io), {} (coalesced), for {} of segments",
        format_size(base.uncoalesced_bytes, DECIMAL),
        format_size(base.coalesced_bytes, DECIMAL),
        format_size(data, DECIMAL),
    );
    println!(
        "\n\"read\" columns are the bytes a direct-I/O reader transfers to read every segment,\n\
         widened to {BLOCK}-byte blocks. The coalesced column applies the reader's own\n\
         {}/{} coalescing window first.",
        format_size(COALESCE_DISTANCE, DECIMAL),
        format_size(COALESCE_MAX_SIZE, DECIMAL),
    );
    Ok(())
}

fn print_size_distribution(segments: &[SegmentSpec]) {
    let mut lengths: Vec<u64> = segments.iter().map(|s| u64::from(s.length)).collect();
    lengths.sort_unstable();
    let quantile = |numerator: usize, denominator: usize| {
        lengths[(lengths.len() * numerator / denominator).min(lengths.len() - 1)]
    };
    println!(
        "segment size:   p50={} p90={} p99={} max={}",
        format_size(quantile(1, 2), DECIMAL),
        format_size(quantile(9, 10), DECIMAL),
        format_size(quantile(99, 100), DECIMAL),
        format_size(*lengths.last().unwrap_or(&0), DECIMAL),
    );
    for threshold in [BLOCK, 4 * BLOCK, 16 * BLOCK] {
        let count = lengths.partition_point(|len| *len < threshold);
        let bytes: u64 = lengths.iter().take_while(|len| **len < threshold).sum();
        println!(
            "  < {:>7}: {:>7} segments ({:>4.1}%), {:>10} ({:.1}% of bytes)",
            format_size(threshold, DECIMAL),
            count,
            100.0 * count as f64 / lengths.len() as f64,
            format_size(bytes, DECIMAL),
            100.0 * bytes as f64 / lengths.iter().sum::<u64>().max(1) as f64,
        );
    }
}

fn percent_change(baseline: u64, value: u64) -> f64 {
    100.0 * (value as f64 - baseline as f64) / baseline as f64
}

struct ScanConfig {
    columns: Vec<String>,
    cold: bool,
    io_only: bool,
    concurrency: usize,
}

async fn scan(inputs: &[PathBuf], iterations: usize, config: ScanConfig) -> anyhow::Result<()> {
    let bytes: u64 = inputs
        .iter()
        .map(|input| fs::metadata(input).map(|meta| meta.len()))
        .sum::<Result<_, _>>()?;

    println!(
        "{} file(s), {}, {iterations} iteration(s), {} cache, {}",
        inputs.len(),
        format_size(bytes, DECIMAL),
        if config.cold { "cold" } else { "warm" },
        if config.io_only {
            "segments only"
        } else {
            "full scan"
        },
    );
    println!(
        "\n{:<10} {:>10} {:>10} {:>10} {:>12}",
        "mode", "min", "median", "max", "throughput"
    );

    let mut baseline = None;
    for direct in [false, true] {
        let options = read_options(direct);
        let mut timings = Vec::with_capacity(iterations);
        let mut rows = 0;
        for _ in 0..iterations {
            if config.cold {
                drop_caches()?;
            }
            let start = Instant::now();
            rows = scan_once(inputs, &config, options).await?;
            timings.push(start.elapsed());
        }
        timings.sort();
        let median = timings[timings.len() / 2];
        let throughput = bytes as f64 / median.as_secs_f64() / 1e6;
        let label = if direct { "direct" } else { "buffered" };
        let delta = baseline
            .map(|base: Duration| {
                format!(
                    "  {:+.1}%",
                    100.0 * (median.as_secs_f64() - base.as_secs_f64()) / base.as_secs_f64()
                )
            })
            .unwrap_or_default();
        println!(
            "{label:<10} {:>9.3}s {:>9.3}s {:>9.3}s {throughput:>9.0} MB/s{delta}",
            timings[0].as_secs_f64(),
            median.as_secs_f64(),
            timings[timings.len() - 1].as_secs_f64(),
        );
        baseline.get_or_insert(median);
        anyhow::ensure!(rows > 0, "scan produced no rows");
    }
    Ok(())
}

fn read_options(direct: bool) -> FileReadAtOptions {
    #[cfg(target_os = "linux")]
    if direct {
        return FileReadAtOptions::default().with_direct_io();
    }
    let _ = direct;
    FileReadAtOptions::default()
}

async fn scan_once(
    inputs: &[PathBuf],
    config: &ScanConfig,
    options: FileReadAtOptions,
) -> anyhow::Result<usize> {
    let handle = SESSION.handle();
    let scans = inputs.iter().map(|input| {
        let handle = handle.clone();
        async move { scan_file(input, config, options, handle).await }
    });
    let rows: Vec<usize> = futures::stream::iter(scans)
        .buffer_unordered(config.concurrency.max(1))
        .try_collect()
        .await?;
    Ok(rows.into_iter().sum())
}

async fn scan_file(
    input: &Path,
    config: &ScanConfig,
    options: FileReadAtOptions,
    handle: Handle,
) -> anyhow::Result<usize> {
    let reader = FileReadAt::open_with_options(
        input,
        handle,
        BufferAllocatorRef::statically_allocated(),
        options,
    )?;
    let file: VortexFile = SESSION.open_options().open(Arc::new(reader)).await?;

    if config.io_only {
        return read_all_segments(&file).await;
    }

    let mut scan = file.scan()?;
    if !config.columns.is_empty() {
        let names: Vec<FieldName> = config
            .columns
            .iter()
            .map(|name| FieldName::from(name.as_str()))
            .collect();
        let projection = select(names, root())
            .optimize_recursive(file.dtype())?
            .bind(file.dtype())?;
        scan = scan.with_projection(projection);
    }

    let mut stream = Box::pin(scan.into_array_stream()?);
    let mut rows = 0;
    while let Some(array) = stream.next().await {
        let array: ArrayRef = array?;
        rows += array.len();
    }
    Ok(rows)
}

/// Pull every segment through the file's read pipeline without decoding any of them.
///
/// This exercises the same request registration, coalescing, and concurrency limiting that a scan
/// drives, so it isolates the cost of the reads themselves from decompression.
async fn read_all_segments(file: &VortexFile) -> anyhow::Result<usize> {
    let source = file.segment_source();
    let segment_count = file.footer().segment_map().len();
    let requests = (0..u32::try_from(segment_count)?).map(|id| {
        let request = source.request(SegmentId::from(id));
        async move { request.await.map(|buffer| buffer.len()) }
    });
    let lengths: Vec<usize> = futures::stream::iter(requests)
        .buffered(DEFAULT_CONCURRENCY)
        .try_collect()
        .await?;
    Ok(lengths.into_iter().sum())
}

/// Evict the page cache so the next scan reads from the device.
fn drop_caches() -> anyhow::Result<()> {
    use std::io::Write;

    fs::File::create("/proc/sys/vm/drop_caches")
        .and_then(|mut file| file.write_all(b"3"))
        .map_err(|e| anyhow::anyhow!("cannot drop the page cache (needs root): {e}"))
}
