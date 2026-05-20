// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary entrypoint for `vortex-bench-server`.
//!
//! Reads the following environment variables before constructing the two
//! routers it serves ([`vortex_bench_server::app::public_router`] and,
//! when `ADMIN_BEARER_TOKEN` is set, [`vortex_bench_server::app::admin_router`]):
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
//! - `VORTEX_BENCH_EXTENSION_DIR` — directory DuckDB installs extensions
//!   into. Default: `<VORTEX_BENCH_DB parent>/duckdb-extensions`. The
//!   default lives under `STATE_DIR`, which the systemd unit makes
//!   writable; this is what lets `INSTALL vortex` succeed under
//!   `ProtectHome=read-only`.
//! - `VORTEX_BENCH_BIND` — `host:port` the **public** listener binds to.
//!   Highest priority. Default `127.0.0.1:3000` (after `PORT` fallback).
//!   Override to `0.0.0.0:3000` for container deploys. Only ingest, read,
//!   HTML, and `/health` are served here — admin routes do not match.
//! - `PORT` — optional PaaS-conventional knob for the **public** listener
//!   only. When set and `VORTEX_BENCH_BIND` is not, the public listener
//!   binds `0.0.0.0:$PORT`. Does not affect the admin listener.
//! - `VORTEX_BENCH_ADMIN_BIND` — `host:port` the **admin** listener binds
//!   to when `ADMIN_BEARER_TOKEN` is set. Default `127.0.0.1:3001`. The
//!   address MUST resolve to a loopback IP (`127.0.0.0/8` or `::1`); the
//!   server refuses to start otherwise. This is the load-bearing guarantee
//!   that `/api/admin/*` never reaches the public network even when
//!   `VORTEX_BENCH_BIND=0.0.0.0:3000`. Must also resolve to a different
//!   address than the public bind.
//! - `VORTEX_BENCH_LOG` — `tracing-subscriber` env filter spec. Default
//!   `info`.
//!
//! On Unix, SIGTERM and SIGINT both trigger a graceful drain — in-flight
//! requests are allowed to finish before the process exits. systemd's
//! `TimeoutStopSec` (default 90s) bounds the grace window, which matters
//! because `systemctl restart` is what the deploy timer fires on every
//! new binary roll. On non-Unix targets only Ctrl-C/SIGINT is wired.

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::anyhow;
use futures::FutureExt;
use tokio::net::TcpListener;
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
    // localhost-only on the default port. `PORT` only affects the public
    // listener — the admin listener has its own env var below.
    let public_bind = env::var("VORTEX_BENCH_BIND")
        .ok()
        .or_else(|| env::var("PORT").ok().map(|p| format!("0.0.0.0:{p}")))
        .unwrap_or_else(|| "127.0.0.1:3000".to_string());
    let admin_bind =
        env::var("VORTEX_BENCH_ADMIN_BIND").unwrap_or_else(|_| "127.0.0.1:3001".to_string());

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
    if let Ok(dir) = env::var("VORTEX_BENCH_EXTENSION_DIR") {
        state = state
            .with_extension_dir(PathBuf::from(dir))
            .context("applying VORTEX_BENCH_EXTENSION_DIR")?;
    }
    let snapshot_dir = state.snapshot_dir.clone();
    let extension_dir = state.extension_dir.clone();
    let admin_app = vortex_bench_server::app::admin_router(state.clone());
    let public_app = vortex_bench_server::app::public_router(state);

    let public_listener = TcpListener::bind(&public_bind)
        .await
        .with_context(|| format!("binding public listener to {public_bind}"))?;
    let public_addr = public_listener.local_addr()?;

    let admin_listener = match admin_app.as_ref() {
        Some(_) => {
            let listener = TcpListener::bind(&admin_bind)
                .await
                .with_context(|| format!("binding admin listener to {admin_bind}"))?;
            let admin_addr = listener.local_addr()?;
            ensure_admin_is_loopback(&admin_bind, admin_addr)?;
            ensure_distinct_binds(public_addr, admin_addr)?;
            Some((listener, admin_addr))
        }
        None => None,
    };

    tracing::info!(
        public_addr = %public_addr,
        admin_addr = ?admin_listener.as_ref().map(|(_, a)| *a),
        db = %db_path.display(),
        snapshot_dir = %snapshot_dir.display(),
        extension_dir = %extension_dir.display(),
        "bench server listening"
    );

    // Both listeners share one shutdown trigger so a single SIGTERM drains
    // ingest, admin, and HTML in lockstep. `Shared` lets us hand the same
    // future to each `with_graceful_shutdown`.
    let shutdown = shutdown_signal().shared();
    match (admin_app, admin_listener) {
        (Some(admin_app), Some((admin_listener, _))) => {
            let public_fut =
                axum::serve(public_listener, public_app).with_graceful_shutdown(shutdown.clone());
            let admin_fut = axum::serve(admin_listener, admin_app).with_graceful_shutdown(shutdown);
            tokio::try_join!(public_fut, admin_fut)?;
        }
        _ => {
            axum::serve(public_listener, public_app)
                .with_graceful_shutdown(shutdown)
                .await?;
        }
    }
    Ok(())
}

/// Refuse to start if the public and admin listeners landed on the same
/// address. The admin listener exists specifically to keep `/api/admin/*`
/// off the public network, so the two collapsing back into one is a
/// silent rollback of that guarantee.
fn ensure_distinct_binds(public: SocketAddr, admin: SocketAddr) -> Result<()> {
    if public == admin {
        return Err(anyhow!(
            "public and admin listeners would bind the same address ({public}); \
             keep VORTEX_BENCH_ADMIN_BIND on a different port"
        ));
    }
    Ok(())
}

/// Refuse to start if the admin listener resolved to a non-loopback
/// address. Without this guard, `VORTEX_BENCH_ADMIN_BIND=0.0.0.0:3001`
/// (or any public IP / unspecified address / non-loopback hostname)
/// would silently expose `/api/admin/*` — the bearer-gated SQL and
/// snapshot endpoints — to the public network. The contract is that the
/// admin listener is loopback-only and the only way callers reach it is
/// from the same host. An operator who genuinely wants remote admin
/// access should put it behind an SSH tunnel rather than opening the
/// bind, so this check is intentionally strict.
fn ensure_admin_is_loopback(spec: &str, admin: SocketAddr) -> Result<()> {
    if !admin.ip().is_loopback() {
        return Err(anyhow!(
            "admin listener resolved to {admin} (from VORTEX_BENCH_ADMIN_BIND={spec:?}); \
             /api/admin/* must remain loopback-only. Use 127.0.0.1, ::1, or \
             a hostname that resolves to a loopback address; reach admin from \
             elsewhere via an SSH tunnel"
        ));
    }
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

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::ensure_admin_is_loopback;
    use super::ensure_distinct_binds;

    #[test]
    fn admin_loopback_v4_passes() {
        let addr: SocketAddr = "127.0.0.1:3001".parse().unwrap();
        ensure_admin_is_loopback("127.0.0.1:3001", addr).expect("127.0.0.1 is loopback");
    }

    #[test]
    fn admin_loopback_v6_passes() {
        let addr: SocketAddr = "[::1]:3001".parse().unwrap();
        ensure_admin_is_loopback("[::1]:3001", addr).expect("::1 is loopback");
    }

    #[test]
    fn admin_loopback_127_8_subnet_passes() {
        // The entire 127.0.0.0/8 block is loopback.
        let addr: SocketAddr = "127.1.2.3:3001".parse().unwrap();
        ensure_admin_is_loopback("127.1.2.3:3001", addr).expect("127.0.0.0/8 is loopback");
    }

    #[test]
    fn admin_zero_v4_rejected() {
        let addr: SocketAddr = "0.0.0.0:3001".parse().unwrap();
        let err = ensure_admin_is_loopback("0.0.0.0:3001", addr)
            .expect_err("0.0.0.0 must be rejected as non-loopback");
        let msg = err.to_string();
        assert!(msg.contains("loopback-only"), "{msg}");
    }

    #[test]
    fn admin_zero_v6_rejected() {
        let addr: SocketAddr = "[::]:3001".parse().unwrap();
        ensure_admin_is_loopback("[::]:3001", addr)
            .expect_err(":: must be rejected as non-loopback");
    }

    #[test]
    fn admin_public_ip_rejected() {
        let addr: SocketAddr = "10.0.0.5:3001".parse().unwrap();
        ensure_admin_is_loopback("10.0.0.5:3001", addr)
            .expect_err("private/public IP must be rejected as non-loopback");
    }

    #[test]
    fn distinct_binds_passes() {
        let p: SocketAddr = "127.0.0.1:3000".parse().unwrap();
        let a: SocketAddr = "127.0.0.1:3001".parse().unwrap();
        ensure_distinct_binds(p, a).expect("different ports are distinct");
    }

    #[test]
    fn same_bind_rejected() {
        let p: SocketAddr = "127.0.0.1:3000".parse().unwrap();
        let a: SocketAddr = "127.0.0.1:3000".parse().unwrap();
        ensure_distinct_binds(p, a).expect_err("identical binds must be rejected");
    }
}
