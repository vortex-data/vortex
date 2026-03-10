// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(unused_imports)]

//! Bisect NVMe read throughput by testing each layer of the IO pipeline independently.
//!
//! Levels:
//!   1. raw-read:    pread into regular heap memory
//!   2. pinned-read: pread into CUDA pinned (page-locked) memory
//!   3. pinned-h2d:  pread into pinned memory + async H2D transfer
//!
//! All levels read the same file in fixed-size chunks with configurable concurrency.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use clap::ValueEnum;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_cuda::PinnedByteBufferPool;
use vortex_cuda::VortexCudaStreamPool;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

#[derive(Clone, Copy, ValueEnum)]
enum Level {
    /// pread into regular heap memory
    RawRead,
    /// pread into CUDA pinned (page-locked) memory
    PinnedRead,
    /// pread into pinned memory + async H2D DMA transfer
    PinnedH2d,
}

#[derive(Parser)]
#[command(
    name = "io-bisect",
    about = "Bisect NVMe read throughput across IO pipeline layers"
)]
struct Cli {
    /// Path to the file to read.
    file: PathBuf,

    /// Which pipeline level to test.
    #[arg(value_enum)]
    level: Level,

    /// Chunk size in bytes for each read.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    chunk_size: usize,

    /// Number of concurrent reads.
    #[arg(long, default_value_t = 32)]
    concurrency: usize,

    /// Number of iterations.
    #[arg(long, default_value_t = 1)]
    iterations: usize,
}

#[cuda_not_available]
fn main() {}

#[cuda_available]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let file = Arc::new(File::open(&cli.file)?);
    let file_size = file.metadata()?.len();
    let chunk_size = cli.chunk_size;
    let num_chunks = ((file_size + chunk_size as u64 - 1) / chunk_size as u64) as usize;

    eprintln!(
        "File: {} ({:.2} MB), chunk_size: {} bytes, chunks: {}, concurrency: {}",
        cli.file.display(),
        file_size as f64 / (1024.0 * 1024.0),
        chunk_size,
        num_chunks,
        cli.concurrency,
    );

    let ctx = cudarc::driver::CudaContext::new(0)?;
    let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(&ctx)));
    let cuda_streams = VortexCudaStreamPool::new(Arc::clone(&ctx), cli.concurrency);

    for iteration in 0..cli.iterations {
        let start = Instant::now();

        // Build (offset, length) pairs for each chunk
        let chunks: Vec<(u64, usize)> = (0..num_chunks)
            .map(|i| {
                let offset = i as u64 * chunk_size as u64;
                let len = std::cmp::min(chunk_size as u64, file_size - offset) as usize;
                (offset, len)
            })
            .collect();

        match cli.level {
            Level::RawRead => {
                let file = Arc::clone(&file);
                stream::iter(chunks)
                    .map(move |(offset, len)| {
                        let file = Arc::clone(&file);
                        tokio::task::spawn_blocking(move || {
                            let mut buf = vec![0u8; len];
                            file.read_exact_at(&mut buf, offset)?;
                            // Prevent optimization from eliding the read
                            std::hint::black_box(&buf);
                            Ok::<_, io::Error>(())
                        })
                    })
                    .buffer_unordered(cli.concurrency)
                    .try_collect::<Vec<_>>()
                    .await?;
            }
            Level::PinnedRead => {
                let file = Arc::clone(&file);
                let pool = Arc::clone(&pool);
                stream::iter(chunks)
                    .map(move |(offset, len)| {
                        let file = Arc::clone(&file);
                        let pool = Arc::clone(&pool);
                        tokio::task::spawn_blocking(move || {
                            let mut pinned = pool.get(len)?;
                            file.read_exact_at(pinned.as_mut_slice(), offset)?;
                            std::hint::black_box(&pinned);
                            Ok::<_, anyhow::Error>(())
                        })
                    })
                    .buffer_unordered(cli.concurrency)
                    .try_collect::<Vec<_>>()
                    .await?;
            }
            Level::PinnedH2d => {
                let file = Arc::clone(&file);
                let pool = Arc::clone(&pool);
                stream::iter(chunks.into_iter().enumerate().collect::<Vec<_>>())
                    .map(move |(i, (offset, len))| {
                        let file = Arc::clone(&file);
                        let pool = Arc::clone(&pool);
                        let stream =
                            cuda_streams.get_stream().expect("failed to get cuda stream");
                        async move {
                            let pinned = tokio::task::spawn_blocking(move || {
                                let mut pinned = pool.get(len)?;
                                file.read_exact_at(pinned.as_mut_slice(), offset)?;
                                Ok::<_, anyhow::Error>(pinned)
                            })
                            .await??;
                            let _device_buf = pinned.transfer_to_device(&stream)?;
                            std::hint::black_box(&_device_buf);
                            Ok::<_, anyhow::Error>(())
                        }
                    })
                    .buffer_unordered(cli.concurrency)
                    .try_collect::<Vec<_>>()
                    .await?;
            }
        }

        let elapsed = start.elapsed();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
        let throughput = file_size_mb / elapsed.as_secs_f64();
        eprintln!(
            "Iteration {}/{}: {:.3}s, {:.2} MB/s",
            iteration + 1,
            cli.iterations,
            elapsed.as_secs_f64(),
            throughput,
        );
    }

    Ok(())
}
