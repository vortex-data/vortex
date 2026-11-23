// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serve a local RAM snapshot through a learned S3 read endpoint.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use mimalloc::MiMalloc;
use serde_json::json;
use vortex_bench::s3::Profile;
use vortex_bench::s3::Server;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
struct Args {
    /// Local directory to preload; omit when benchmarks populate the endpoint using PUT.
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(
        long,
        required_unless_present = "zero_delay",
        conflicts_with = "zero_delay"
    )]
    profile: Option<PathBuf>,
    /// Explicitly measure endpoint/client overhead without service latency.
    #[arg(long)]
    zero_delay: bool,
    #[arg(long, default_value = "vortex-bench")]
    bucket: String,
    #[arg(long, default_value = "127.0.0.1:0")]
    bind: SocketAddr,
    /// Defaults to 40% of available RAM, reserving the rest for client/query work.
    #[arg(long)]
    max_resident_bytes: Option<u64>,
    #[arg(long)]
    ready_file: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    vortex_bench::configure_allocator();
    let args = Args::parse();
    anyhow::ensure!(
        !args.ready_file.exists(),
        "Readiness file already exists; choose a fresh file"
    );
    let profile = match &args.profile {
        Some(path) => Profile::from_path(path)?,
        None => Profile::zero_delay(),
    };
    let memory = sysinfo::System::new_all();
    let limit = args
        .max_resident_bytes
        .unwrap_or_else(|| memory.available_memory() / 5 * 2);
    let server = Server::start(
        args.root.as_deref(),
        profile,
        args.bucket.clone(),
        args.bind,
        limit,
    )
    .await?;
    let ready = json!({"endpoint": server.endpoint(), "bucket": args.bucket,
                      "resident_bytes": server.resident_bytes().await, "max_resident_bytes": limit,
                      "profile": args.profile, "zero_delay": args.zero_delay,
                      "scope": "loopback RAM-backed S3 read simulation; preparation writes are not latency-calibrated"});
    let temporary = args.ready_file.with_extension("tmp");
    std::fs::write(&temporary, serde_json::to_vec(&ready)?)?;
    std::fs::rename(temporary, &args.ready_file)?;
    println!("{ready}");
    tokio::signal::ctrl_c().await?;
    Ok(())
}
