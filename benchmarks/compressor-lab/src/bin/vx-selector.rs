// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vx-selector`: training rows, the sampling-share ceiling, and evaluation for the learned
//! scheme selector. See `compressor_lab::selector`.

use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use compressor_lab::selector;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_array::VortexSessionExecute;
use vortex_btrblocks::selector::INT_V1;

#[derive(Parser, Debug)]
#[command(name = "vx-selector")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// TPC-H scale factor.
    #[arg(long, default_value_t = 1.0, global = true)]
    scale_factor: f64,
    /// TPC-H tables to read integer columns from.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "lineitem,orders,partsupp,customer,part,supplier",
        global = true
    )]
    tables: Vec<String>,
    #[arg(long, default_value_t = 65_536, global = true)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 8, global = true)]
    max_chunks_per_column: usize,
    #[arg(long, default_value_t = 5, global = true)]
    compress_reps: u32,
    #[arg(long, default_value_t = 5, global = true)]
    decode_reps: u32,
    #[arg(long, global = true, default_value = "selector-out")]
    out: PathBuf,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Gather per-site candidate costs, features and the replay ceiling.
    Gather {
        /// Do not force candidates below this cascade depth.
        #[arg(long, default_value_t = 3)]
        max_depth: usize,
    },
    /// Compare production with the generated learned selector.
    Eval {
        #[arg(long, default_value_t = 0.9)]
        min_confidence: f64,
        #[arg(long, default_value_t = 3)]
        fallback_k: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();
    let tables: Vec<&str> = cli.tables.iter().map(String::as_str).collect();
    let chunks = selector::tpch_chunks(
        &tables,
        cli.scale_factor,
        cli.chunk_rows,
        cli.max_chunks_per_column,
        &mut ctx,
    )?;
    eprintln!("{} integer chunks", chunks.len());
    match cli.command {
        Command::Gather { max_depth } => {
            let options = selector::Options {
                compress_reps: cli.compress_reps,
                decode_reps: cli.decode_reps,
                max_depth,
            };
            selector::gather(&session, &chunks, &options, &cli.out)
        }
        Command::Eval {
            min_confidence,
            fallback_k,
        } => {
            let options = selector::Options {
                compress_reps: cli.compress_reps,
                decode_reps: cli.decode_reps,
                max_depth: 0,
            };
            eprintln!("model {}", INT_V1.name);
            selector::evaluate(
                &session,
                &chunks,
                &INT_V1,
                min_confidence,
                fallback_k,
                &options,
                &cli.out,
            )
        }
    }
}
