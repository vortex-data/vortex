// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bit_unpack;
mod indent;

use std::fs::{self};
use std::path::PathBuf;

use clap::Parser;

use crate::bit_unpack::generate_unpack;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let output_dir = args.output_dir.unwrap_or_else(|| PathBuf::from("kernels"));
    fs::create_dir_all(&output_dir)?;

    // Generate for all bit widths and both features
    generate_unpack::<u8>(&output_dir, 32)?;
    generate_unpack::<u16>(&output_dir, 32)?;
    generate_unpack::<u32>(&output_dir, 32)?;
    generate_unpack::<u64>(&output_dir, 16)?;
    Ok(())
}
