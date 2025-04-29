use std::path::PathBuf;

use bench_vortex::display::DisplayFormat;
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::{IdempotentPath, Target, default_env_filter};
use clap::{Parser, value_parser};
use itertools::Itertools;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "datafusion:arrow",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,
    #[arg(long)]
    duckdb_path: Option<PathBuf>,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(short)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,
    #[arg(long)]
    export_spans: bool,
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = default_env_filter(args.verbose);
    #[cfg(not(feature = "tracing"))]
    bench_vortex::setup_logger(filter);

    #[cfg(feature = "tracing")]
    let _trace_guard = {
        use std::io::IsTerminal;

        use tracing_subscriber::prelude::*;

        let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
            .include_args(true)
            .trace_style(tracing_chrome::TraceStyle::Async)
            .file("tpcds.trace.json")
            .build();

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_level(true)
            .with_line_number(true)
            .with_ansi(std::io::stderr().is_terminal());

        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .with(fmt_layer)
            .init();
        _guard
    };

    let formats = args
        .targets
        .iter()
        .map(|t| t.format())
        .unique()
        .collect_vec();

    for format in formats {
        let opts = DuckdbTpcOptions::default()
            .with_scale_factor(1)
            .with_base_dir("tpcds".to_data_path())
            .with_dataset(TpcDataset::TpcDs)
            .with_format(format);
        generate_tpc(opts).expect("gen tpch-ds");
    }

    // Require trace guard lives until here
    #[cfg(feature = "tracing")]
    let _ = _trace_guard;
    Ok(())
}
