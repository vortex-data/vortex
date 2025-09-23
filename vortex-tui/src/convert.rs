// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use futures::StreamExt;
use indicatif::ProgressBar;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexExpect};
use vortex::file::{VortexWriteOptions, WriteStrategyBuilder};
use vortex::stream::ArrayStreamAdapter;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Strategy {
    Btrblocks,
    Compact,
}
#[derive(Debug, Clone, Parser)]
pub struct Flags {
    /// Path to the Parquet file on disk to convert to Vortex
    pub file: PathBuf,

    /// Execute quietly. No output will be printed.
    #[arg(short, long)]
    quiet: bool,

    /// Compression strategy.
    #[arg(short, long, default_value = "btrblocks")]
    strategy: Strategy,
}

const BATCH_SIZE: usize = 8192;

/// Convert Parquet files to Vortex.
pub async fn exec_convert(flags: Flags) -> anyhow::Result<()> {
    let input_path = flags.file.clone();
    if !flags.quiet {
        eprintln!("Converting input Parquet file: {}", input_path.display());
    }

    let output_path = input_path.with_extension("vortex");
    let file = File::open(input_path).await?;

    let parquet = ParquetRecordBatchStreamBuilder::new(file)
        .await?
        .with_batch_size(BATCH_SIZE);
    let num_rows = parquet.metadata().file_metadata().num_rows();

    let dtype = DType::from_arrow(parquet.schema().as_ref());
    let mut vortex_stream = parquet
        .build()?
        .map(|record_batch| {
            record_batch
                .map_err(|e| VortexError::generic(e.into()))
                .map(|rb| ArrayRef::from_arrow(rb, false))
        })
        .boxed();

    if !flags.quiet {
        // Parquet reader returns batches, rather than row groups. So make sure we correctly
        // configure the progress bar.
        let nbatches = u64::try_from(num_rows)
            .vortex_expect("negative row count?")
            .div_ceil(BATCH_SIZE as u64);
        vortex_stream = ProgressBar::new(nbatches)
            .wrap_stream(vortex_stream)
            .boxed();
    }

    let strategy = WriteStrategyBuilder::default();
    let strategy = match flags.strategy {
        Strategy::Btrblocks => strategy,
        Strategy::Compact => strategy.with_compressor(CompactCompressor::default()),
    };

    let mut file = File::create(output_path).await?;
    VortexWriteOptions::default()
        .with_strategy(strategy.build())
        .write(&mut file, ArrayStreamAdapter::new(dtype, vortex_stream))
        .await?;
    file.shutdown().await?;

    Ok(())
}
