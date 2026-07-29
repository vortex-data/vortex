// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use clap::Parser;
use clap::ValueEnum;
use indicatif::ProgressBar;
use regex::Regex;
use string_bench::ColumnResult;
use string_bench::DirectCandidate;
use string_bench::SESSION;
use string_bench::SerializedResult;
use string_bench::StringColumn;
use string_bench::StringEncoder;
use string_bench::bench_column;
use string_bench::bench_serialized_with_session;
use string_bench::load_clickbench_url;
use string_bench::load_tpch_l_comment;
use tabled::builder::Builder;
use tabled::settings::Style;
use vortex::VortexSessionDefault;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_bench::LogFormat;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::display::print_measurements_json;
use vortex_bench::measurements::CustomUnitMeasurement;
use vortex_bench::setup_logging_and_tracing_with_format;

/// The benchmark ID used for the output path.
const BENCHMARK_ID: &str = "string";

/// Repo-relative path of the suite explainer linked from CI benchmark PR comments.
const DOC_PATH: &str = "benchmarks/string-bench/README.md";

const CODEC_ONPAIR_DICT_BITS: [u8; 2] = [12, 16];

#[async_trait]
trait StringColumnSource {
    fn name(&self) -> &str;

    async fn load(&self, ctx: &mut ExecutionCtx) -> Result<StringColumn>;
}

struct ClickBenchUrl {
    shard: u32,
}

#[async_trait]
impl StringColumnSource for ClickBenchUrl {
    fn name(&self) -> &str {
        "URL"
    }

    async fn load(&self, ctx: &mut ExecutionCtx) -> Result<StringColumn> {
        load_clickbench_url(self.shard, ctx).await
    }
}

struct TpchLComment;

#[async_trait]
impl StringColumnSource for TpchLComment {
    fn name(&self) -> &str {
        "l_comment"
    }

    async fn load(&self, ctx: &mut ExecutionCtx) -> Result<StringColumn> {
        load_tpch_l_comment(ctx).await
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchmarkSuite {
    /// Run only the direct whole-column codec microbenchmark.
    #[value(name = "codec")]
    Codec,
    /// Run only the Vortex serialized write/read benchmark.
    #[value(name = "vortex")]
    Vortex,
    /// Run both benchmark paths.
    #[value(name = "both")]
    Both,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Benchmark path to run: direct codec, Vortex serialized, or both.
    #[arg(long, value_enum, default_value_t = BenchmarkSuite::Codec)]
    suite: BenchmarkSuite,
    /// Timed runs per (column, encoder, config); the median is reported.
    #[arg(short, long, default_value = "10")]
    iterations: NonZeroUsize,
    /// Untimed warm-up runs before timing, per (column, encoder, config). At
    /// least one warm-up always runs; correctness validation is also untimed.
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Regex filter matched against configured input-column identifiers.
    #[arg(long)]
    columns: Option<String>,
    /// ClickBench shard index to read the `URL` column from.
    #[arg(long, default_value_t = 0)]
    clickbench_shard: u32,
    /// Skip the one-time canonicalized-output correctness check before timing.
    #[arg(long)]
    no_verify: bool,
    /// Encoders to benchmark (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        StringEncoder::OnPair,
        StringEncoder::Fsst,
    ])]
    encoders: Vec<StringEncoder>,
    /// Output format: `table` for humans, `gh-json` for machine-readable JSONL.
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    /// Write output to this file instead of stdout.
    #[arg(short, long)]
    output_path: Option<PathBuf>,
    /// Enable verbose (debug-level) logging.
    #[arg(short, long)]
    verbose: bool,
    /// Enable span tracing output.
    #[arg(long)]
    tracing: bool,
    /// Format for the primary stderr log sink.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
}

fn contains_duplicates<T: PartialEq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
}

fn main() -> Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing_with_format(args.verbose, args.tracing, args.log_format)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run(args))
}

async fn run(args: Args) -> Result<()> {
    let iterations = args.iterations.get();
    let run_codec = matches!(args.suite, BenchmarkSuite::Codec | BenchmarkSuite::Both);
    let run_vortex = matches!(args.suite, BenchmarkSuite::Vortex | BenchmarkSuite::Both);

    if args.encoders.is_empty() {
        bail!("--encoders must select at least one encoder");
    }
    if contains_duplicates(&args.encoders) {
        bail!("--encoders must not contain duplicates");
    }
    let run_onpair = args.encoders.contains(&StringEncoder::OnPair);
    let run_fsst = args.encoders.contains(&StringEncoder::Fsst);

    let mut candidates: Vec<DirectCandidate> = Vec::new();
    if run_codec && run_onpair {
        for bits in CODEC_ONPAIR_DICT_BITS {
            candidates.push(DirectCandidate::on_pair(bits)?);
        }
    }
    if run_codec && run_fsst {
        candidates.push(DirectCandidate::Fsst);
    }

    let filter = args.columns.as_deref().map(Regex::new).transpose()?;
    let clickbench_url = ClickBenchUrl {
        shard: args.clickbench_shard,
    };
    let column_sources: Vec<&dyn StringColumnSource> = [
        &clickbench_url as &dyn StringColumnSource,
        &TpchLComment as &dyn StringColumnSource,
    ]
    .into_iter()
    .filter(|source| {
        filter
            .as_ref()
            .is_none_or(|filter| filter.is_match(source.name()))
    })
    .collect();
    if column_sources.is_empty() {
        bail!("no input columns matched the --columns filter");
    }

    let mut ctx = SESSION.create_execution_ctx();
    let mut columns = Vec::with_capacity(column_sources.len());
    for source in column_sources {
        columns.push(source.load(&mut ctx).await?);
    }

    let verify = !args.no_verify;

    let total_steps = if run_codec {
        columns.len() * candidates.len()
    } else {
        0
    };
    let total_steps = total_steps
        + if run_vortex {
            columns.len() * args.encoders.len()
        } else {
            0
        };
    let progress = ProgressBar::new(total_steps as u64);

    let mut column_results: Vec<ColumnResult> = Vec::new();
    let mut serialized_results: Vec<SerializedResult> = Vec::new();

    for column in &columns {
        if run_codec {
            column_results.extend(run_codec_column(
                column,
                iterations,
                args.warmup,
                verify,
                &candidates,
                &progress,
            )?);
        }

        if run_vortex {
            serialized_results.extend(run_vortex_column(
                column,
                iterations,
                args.warmup,
                verify,
                &args.encoders,
                &progress,
            )?);
        }
    }
    progress.finish();

    let mut writer = create_output_writer(&args.display_format, args.output_path, BENCHMARK_ID)?;
    match args.display_format {
        DisplayFormat::Table => {
            if !column_results.is_empty() {
                render_inmemory_table(&mut writer, &column_results)?;
            }
            if !serialized_results.is_empty() {
                render_serialized_table(&mut writer, &serialized_results)?;
            }
        }
        DisplayFormat::GhJson => {
            let mut measurements: Vec<CustomUnitMeasurement> = Vec::new();
            for result in &column_results {
                measurements.extend(result.measurements());
            }
            for rt in &serialized_results {
                measurements.extend(rt.measurements());
            }
            print_measurements_json(&mut writer, measurements, DOC_PATH)?;
        }
    }

    Ok(())
}

fn run_codec_column(
    column: &StringColumn,
    iterations: usize,
    warmup: usize,
    verify: bool,
    candidates: &[DirectCandidate],
    progress: &ProgressBar,
) -> Result<Vec<ColumnResult>> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let result = bench_column(column, iterations, warmup, candidate, verify)?;
        tracing::info!(
            "{} [{}]: {} rows, {:.3} GB canonical, codec compress {:.2} MB/s, \
             encoded size {:.2}%",
            result.name,
            result.encoder,
            result.rows,
            result.uncompressed_bytes as f64 / 1e9,
            result.compression_mbps(),
            result.encoded_size_pct(),
        );
        results.push(result);
        progress.inc(1);
    }
    Ok(results)
}

fn run_vortex_column(
    column: &StringColumn,
    iterations: usize,
    warmup: usize,
    verify: bool,
    encoders: &[StringEncoder],
    progress: &ProgressBar,
) -> Result<Vec<SerializedResult>> {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    let mut results = Vec::with_capacity(encoders.len());
    for &encoder in encoders {
        let result = runtime.block_on(bench_serialized_with_session(
            &session, column, iterations, warmup, verify, encoder,
        ))?;
        record_vortex_result(&result, progress);
        results.push(result);
    }
    Ok(results)
}

fn record_vortex_result(result: &SerializedResult, progress: &ProgressBar) {
    tracing::info!(
        "{} [{}]: file size {:.2}% | write {:.2} MB/s | \
         canonicalize {:.2} MB/s | staged read {:.2} MB/s",
        result.name,
        result.encoder,
        result.file_size_pct(),
        result.write_mbps(),
        result.canonicalize_mbps(),
        result.staged_read_mbps(),
    );
    progress.inc(1);
}

/// Render the direct codec microbenchmark metrics.
fn render_inmemory_table(writer: &mut dyn Write, results: &[ColumnResult]) -> Result<()> {
    let mut builder = Builder::default();
    builder.push_record([
        "Column",
        "Encoder",
        "Encoded buffers (% canonical)",
        "Codec compress MB/s",
    ]);
    for r in results {
        builder.push_record([
            r.name.clone(),
            r.encoder.clone(),
            format!("{:.2}", r.encoded_size_pct()),
            format!("{:.2}", r.compression_mbps()),
        ]);
    }
    let mut table = builder.build();
    table.with(Style::modern());
    writeln!(writer, "Direct codec microbenchmark")?;
    writeln!(
        writer,
        "  One whole-column codec state (dictionary or symbol table); no Vortex layout, child compression, or file I/O; \
         MB/s = canonical uncompressed bytes per second"
    )?;
    writeln!(writer, "{table}")?;
    Ok(())
}

/// Render the Vortex serialized write/read metrics as one row per (column, encoder).
fn render_serialized_table(writer: &mut dyn Write, results: &[SerializedResult]) -> Result<()> {
    let mut builder = Builder::default();
    builder.push_record([
        "Column",
        "Encoder",
        "File size (% canonical)",
        "Write MB/s",
        "Canonicalize MB/s",
        "Staged Read (serial, in-memory) MB/s",
    ]);
    for r in results {
        builder.push_record([
            r.name.clone(),
            r.encoder.clone(),
            format!("{:.2}", r.file_size_pct()),
            format!("{:.2}", r.write_mbps()),
            format!("{:.2}", r.canonicalize_mbps()),
            format!("{:.2}", r.staged_read_mbps()),
        ]);
    }
    let mut table = builder.build();
    table.with(Style::modern());
    writeln!(writer, "\nVortex serialized write and read")?;
    writeln!(
        writer,
        "  Single-threaded Vortex write = encode + layout + child compression + serialization; \
         Canonicalize = encoded arrays to canonical arrays; \
         Staged Read = open + scan-all + canonicalize-all; \
         MB/s = canonical uncompressed bytes per second"
    )?;
    writeln!(writer, "{table}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_iterations() {
        assert!(Args::try_parse_from(["string-bench", "--iterations", "0"]).is_err());
    }

    #[test]
    fn suite_defaults_to_codec() -> Result<()> {
        let args = Args::try_parse_from(["string-bench"])?;

        assert_eq!(args.suite, BenchmarkSuite::Codec);
        Ok(())
    }
}
