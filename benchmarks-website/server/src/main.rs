// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary entrypoint for `vortex-bench-server`.
//!
//! Reads the following environment variables before handing off to
//! [`vortex_bench_server::app::router`]:
//!
//! - `INGEST_BEARER_TOKEN` — required. Token presented by ingest clients
//!   on `Authorization: Bearer <token>`. Compared in constant time.
//! - `ADMIN_BEARER_TOKEN` — optional. When set, mounts the
//!   `/api/admin/snapshot` and `/api/admin/sql` endpoints; both expect
//!   this token in the `Authorization: Bearer …` header. Without it the
//!   admin router is not mounted at all (404). The `INGEST_BEARER_TOKEN`
//!   does not work on admin routes — keep them separate so they rotate
//!   independently.
//! - `VORTEX_BENCH_DB` — DuckDB file path. Default: `bench.duckdb` in the
//!   working directory.
//! - `VORTEX_BENCH_SNAPSHOT_DIR` — directory `EXPORT DATABASE` writes to
//!   when an operator hits `/api/admin/snapshot`. Default:
//!   `<VORTEX_BENCH_DB parent>/snapshots`.
//! - `VORTEX_BENCH_BIND` — `host:port` to listen on. Default
//!   `127.0.0.1:3000`. Override to `0.0.0.0:3000` for container deploys.
//! - `VORTEX_BENCH_LOG` — `tracing-subscriber` env filter spec. Default
//!   `info`.

use std::env;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("VORTEX_BENCH_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let db_path: PathBuf = env::var("VORTEX_BENCH_DB")
        .unwrap_or_else(|_| "bench.duckdb".to_string())
        .into();
    let bearer_token =
        env::var("INGEST_BEARER_TOKEN").context("INGEST_BEARER_TOKEN env var must be set")?;
    let admin_bearer_token = env::var("ADMIN_BEARER_TOKEN").ok();
    let bind_addr = env::var("VORTEX_BENCH_BIND").unwrap_or_else(|_| "127.0.0.1:3000".to_string());

    let mut state = vortex_bench_server::app::AppState::open(&db_path, bearer_token)
        .with_context(|| format!("opening DuckDB at {}", db_path.display()))?;
    if let Some(token) = admin_bearer_token {
        state = state.with_admin(token);
    } else {
        tracing::warn!(
            "ADMIN_BEARER_TOKEN is unset — /api/admin/* will return 404 \
            (snapshot + read-only SQL disabled)"
        );
    }
    if let Ok(dir) = env::var("VORTEX_BENCH_SNAPSHOT_DIR") {
        state = state.with_snapshot_dir(PathBuf::from(dir));
    }
    let snapshot_dir = state.snapshot_dir.clone();
    let app = vortex_bench_server::app::router(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("binding to {bind_addr}"))?;
    tracing::info!(
        addr = %listener.local_addr()?,
        db = %db_path.display(),
        snapshot_dir = %snapshot_dir.display(),
        "bench server listening"
    );
    axum::serve(listener, app).await?;
    Ok(())
}
