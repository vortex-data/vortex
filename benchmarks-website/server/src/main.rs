// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary entrypoint for `vortex-bench-server`.
//!
//! Reads the following environment variables before handing off to
//! [`vortex_bench_server::app::router`]:
//!
//! - `INGEST_BEARER_TOKEN` — required and non-empty. Token presented by ingest clients
//!   on `Authorization: Bearer <token>`. Compared in constant time.
//! - `ADMIN_BEARER_TOKEN` — optional. When set to a non-empty value, mounts the
//!   `/api/admin/snapshot` and `/api/admin/sql` endpoints; both expect
//!   this token in the `Authorization: Bearer …` header. Without it the
//!   admin router is not mounted at all (404). The `INGEST_BEARER_TOKEN`
//!   does not work on admin routes — keep them separate so they rotate
//!   independently.
//! - `VORTEX_BENCH_DB` — DuckDB file path. Default: `bench.duckdb` in the
//!   working directory.
//! - `VORTEX_BENCH_SNAPSHOT_DIR` — directory `/api/admin/snapshot` writes
//!   per-table Vortex snapshots into (`schema.sql` plus one
//!   `<table>.vortex` file per table). Default:
//!   `<VORTEX_BENCH_DB parent>/snapshots`.
//! - `VORTEX_BENCH_BIND` — `host:port` to listen on. Highest priority. Default
//!   `127.0.0.1:3000` (after `PORT` fallback). Override to `0.0.0.0:3000` for
//!   container deploys.
//! - `PORT` — optional PaaS-conventional knob. When set and `VORTEX_BENCH_BIND`
//!   is not, the server binds to `0.0.0.0:$PORT`.
//! - `VORTEX_BENCH_LOG` — `tracing-subscriber` env filter spec. Default
//!   `info`.
//!
//! SIGTERM and SIGINT both trigger a graceful drain — in-flight requests
//! are allowed to finish before the process exits. systemd's
//! `TimeoutStopSec` (default 90s) bounds the grace window, which matters
//! because `systemctl restart` is what the deploy timer fires on every
//! new binary roll.

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
    let admin_bearer_token = env::var("ADMIN_BEARER_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty());
    // `VORTEX_BENCH_BIND` wins (full `host:port`). If unset, fall back to the
    // PaaS-conventional `PORT` env var (binds to `0.0.0.0:$PORT`). Otherwise
    // localhost-only on the default port.
    let bind_addr = env::var("VORTEX_BENCH_BIND")
        .ok()
        .or_else(|| env::var("PORT").ok().map(|p| format!("0.0.0.0:{p}")))
        .unwrap_or_else(|| "127.0.0.1:3000".to_string());

    let mut state = vortex_bench_server::app::AppState::open(&db_path, bearer_token)
        .with_context(|| format!("opening DuckDB at {}", db_path.display()))?;
    if let Some(token) = admin_bearer_token {
        state = state.with_admin(token);
    } else {
        tracing::warn!(
            "ADMIN_BEARER_TOKEN is unset or empty — /api/admin/* will return 404 \
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
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Resolves when the process receives SIGINT or SIGTERM. Used as the
/// graceful-shutdown future for `axum::serve` so a `systemctl restart`
/// (SIGTERM) lets in-flight requests finish before the process exits.
/// `systemd`'s `TimeoutStopSec` (default 90s) bounds the grace window —
/// nothing inside the process imposes its own timeout.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install ctrl_c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT — shutting down"),
        _ = terminate => tracing::info!("received SIGTERM — shutting down"),
    }
}
