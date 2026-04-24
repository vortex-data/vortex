// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use clap::Parser;

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum Engine {
    #[clap(name = "datafusion")]
    DataFusion,
    #[clap(name = "duckdb")]
    DuckDB,
}

/// Binary args, including all flags that `cargo test` might pass.
#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short, long, value_enum, value_delimiter = ',')]
    pub engine: Option<Vec<Engine>>,
    #[arg(action)]
    pub filter: Option<String>,

    #[clap(
        long,
        help = "IGNORED (for compatibility with built in rust test runner)"
    )]
    pub format: Option<String>,

    #[clap(
        short = 'Z',
        long,
        help = "IGNORED (for compatibility with built in rust test runner)"
    )]
    pub z_options: Option<String>,

    #[clap(
        long,
        help = "IGNORED (for compatibility with built in rust test runner)"
    )]
    pub show_output: bool,

    #[clap(
        long,
        help = "Quits immediately, not listing anything (for compatibility with built-in rust test runner)"
    )]
    pub list: bool,

    #[clap(
        long,
        help = "IGNORED (for compatibility with built-in rust test runner)"
    )]
    pub ignored: bool,

    #[clap(
        long,
        help = "IGNORED (for compatibility with built-in rust test runner)"
    )]
    pub nocapture: bool,

    #[clap(
        long,
        help = "Number of threads used for running tests in parallel",
        default_value_t = 16
    )]
    pub test_threads: usize,
}
