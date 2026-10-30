// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use indicatif::ProgressBar;
use mimalloc::MiMalloc;
use regex::Regex;
use string_bench::Candidate;
use string_bench::ColumnResult;
use string_bench::HF_COLUMNS;
use string_bench::HfColumn;
use string_bench::SESSION;
use string_bench::SerializedResult;
use string_bench::StringColumn;
use string_bench::StringEncoder;
use string_bench::bench_column;
use string_bench::bench_serialized_with_session;
use string_bench::load_clickbench_url;
use string_bench::load_hf_column;
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

/// OnPair dictionary budgets (in bits) both suites sweep.
const ONPAIR_DICT_BITS: [u8; 2] = [12, 16];

/// The configured benchmark inputs. Add a variant to extend the catalog.
enum Input {
    /// The `URL` column of one ClickBench `hits` shard.
    ClickBenchUrl(u32),
    /// The TPC-H SF1 `lineitem.l_comment` column.
    TpchLComment,
    /// A column of a pinned Hugging Face parquet file.
    Hf(&'static HfColumn),
}

impl Input {
    /// Name the `--columns` regex is matched against.
    fn name(&self) -> &'static str {
        match self {
            Self::ClickBenchUrl(_) => "URL",
            Self::TpchLComment => "l_comment",
            Self::Hf(source) => source.name,
        }
    }

    async fn load(&self, ctx: &mut ExecutionCtx) -> Result<StringColumn> {
        match *self {
            Self::ClickBenchUrl(shard) => load_clickbench_url(shard, ctx).await,
            Self::TpchLComment => load_tpch_l_comment(ctx).await,
            Self::Hf(source) => load_hf_column(source, ctx).await,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchmarkSuite {
    /// Run only the Vortex file write/read benchmark: the tracked suite.
    #[value(name = "vortex")]
    Vortex,
    /// Run only the direct whole-column codec microbenchmark: a local
    /// diagnostic for separating encoder cost from the Vortex file stack.
    #[value(name = "codec")]
    Codec,
    /// Run both benchmark paths.
    #[value(name = "both")]
    Both,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Benchmark path to run: the tracked Vortex file suite, the direct codec
    /// diagnostic, or both.
    #[arg(long, value_enum, default_value_t = BenchmarkSuite::Vortex)]
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
    /// Download or generate the selected input columns and exit without benchmarking.
    ///
    /// CI runs this as its own step so that preparing data is not interleaved with the timed
    /// runs in the logs.
    #[arg(long)]
    prepare_data: bool,
}

fn contains_duplicates<T: PartialEq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> Result<()> {
    vortex_bench::configure_allocator();
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
    let mut candidates: Vec<Candidate> = Vec::new();
    if args.encoders.contains(&StringEncoder::OnPair) {
        for bits in ONPAIR_DICT_BITS {
            candidates.push(Candidate::on_pair(bits)?);
        }
    }
    if args.encoders.contains(&StringEncoder::Fsst) {
        candidates.push(Candidate::Fsst);
    }

    let filter = args.columns.as_deref().map(Regex::new).transpose()?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut columns = Vec::new();
    for input in [
        Input::ClickBenchUrl(args.clickbench_shard),
        Input::TpchLComment,
    ]
    .into_iter()
    .chain(HF_COLUMNS.iter().map(Input::Hf))
    .filter(|input| filter.as_ref().is_none_or(|f| f.is_match(input.name())))
    {
        columns.push(input.load(&mut ctx).await?);
    }
    if columns.is_empty() {
        bail!("no input columns matched the --columns filter");
    }
    if args.prepare_data {
        for column in &columns {
            println!("{}", column.name);
        }
        return Ok(());
    }

    let verify = !args.no_verify;

    let steps_per_column = candidates.len() * (usize::from(run_codec) + usize::from(run_vortex));
    let progress = ProgressBar::new((columns.len() * steps_per_column) as u64);

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
                &candidates,
                &progress,
            )?);
        }
    }
    progress.finish();

    let mut writer = create_output_writer(&args.display_format, args.output_path, BENCHMARK_ID)?;
    match args.display_format {
        DisplayFormat::Table => {
            if !serialized_results.is_empty() {
                render_serialized_table(&mut writer, &serialized_results)?;
            }
            if !column_results.is_empty() {
                render_codec_table(&mut writer, &column_results)?;
            }
        }
        DisplayFormat::GhJson => {
            let mut measurements: Vec<CustomUnitMeasurement> = Vec::new();
            for result in &serialized_results {
                measurements.extend(result.measurements());
            }
            for result in &column_results {
                measurements.extend(result.measurements());
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
    candidates: &[Candidate],
    progress: &ProgressBar,
) -> Result<Vec<ColumnResult>> {
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let result = bench_column(column, iterations, warmup, candidate, verify)?;
        tracing::info!(
            "{} [{}]: {} rows, {:.3} GB canonical, size {:.2}%, compress {:.2} MB/s",
            result.name,
            result.encoder,
            result.rows,
            result.uncompressed_bytes as f64 / 1e9,
            result.encoded_size_pct(),
            result.compression_mbps(),
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
    candidates: &[Candidate],
    progress: &ProgressBar,
) -> Result<Vec<SerializedResult>> {
    let runtime = CurrentThreadRuntime::new();
    let session = VortexSession::default().with_handle(runtime.handle());
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let result = runtime.block_on(bench_serialized_with_session(
            &session, column, iterations, warmup, verify, candidate,
        ))?;
        tracing::info!(
            "{} [{}]: size {:.2}% | write {:.2} MB/s | read {:.2} MB/s",
            result.name,
            result.encoder,
            result.size_pct(),
            result.write_mbps(),
            result.read_mbps(),
        );
        results.push(result);
        progress.inc(1);
    }
    Ok(results)
}

/// Render one titled table, prefixed by a one-line note on how to read its
/// metrics.
fn render_table(
    writer: &mut dyn Write,
    title: &str,
    note: &str,
    header: &[&str],
    rows: Vec<Vec<String>>,
) -> Result<()> {
    let mut builder = Builder::default();
    builder.push_record(header.iter().copied());
    for row in rows {
        builder.push_record(row);
    }
    let mut table = builder.build();
    table.with(Style::modern());
    writeln!(writer, "{title}\n  {note}")?;
    writeln!(writer, "{table}")?;
    Ok(())
}

/// Render the three tracked metrics, one row per (column, encoder).
fn render_serialized_table(writer: &mut dyn Write, results: &[SerializedResult]) -> Result<()> {
    render_table(
        writer,
        "Vortex file: size, write, read",
        "Size = % of canonical uncompressed bytes; write = repartition + zone stats + compress \
         (string scheme and children) + layout + serialize; read = open + scan, decoding each row \
         split to canonical form in its own task. Single-threaded, in-memory; MB/s over canonical \
         uncompressed bytes.",
        &[
            "Column",
            "Encoder",
            "Size (%)",
            "Write (MB/s)",
            "Read (MB/s)",
        ],
        results
            .iter()
            .map(|r| {
                vec![
                    r.name.clone(),
                    r.encoder.clone(),
                    format!("{:.2}", r.size_pct()),
                    format!("{:.2}", r.write_mbps()),
                    format!("{:.2}", r.read_mbps()),
                ]
            })
            .collect(),
    )
}

/// Render the direct codec microbenchmark metrics.
fn render_codec_table(writer: &mut dyn Write, results: &[ColumnResult]) -> Result<()> {
    render_table(
        writer,
        "\nDirect codec microbenchmark (diagnostic)",
        "One whole-column codec state, no Vortex layout, child compression, or file I/O. \
         Not comparable with the file suite's size.",
        &["Column", "Encoder", "Size (%)", "Compress (MB/s)"],
        results
            .iter()
            .map(|r| {
                vec![
                    r.name.clone(),
                    r.encoder.clone(),
                    format!("{:.2}", r.encoded_size_pct()),
                    format!("{:.2}", r.compression_mbps()),
                ]
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_iterations() {
        assert!(Args::try_parse_from(["string-bench", "--iterations", "0"]).is_err());
    }

    /// CI passes no `--suite`, so the default is what gets tracked.
    #[test]
    fn suite_defaults_to_vortex() -> Result<()> {
        let args = Args::try_parse_from(["string-bench"])?;

        assert_eq!(args.suite, BenchmarkSuite::Vortex);
        Ok(())
    }
}
