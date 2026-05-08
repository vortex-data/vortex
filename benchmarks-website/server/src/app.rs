// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Axum [`Router`] composition and shared [`AppState`].
//!
//! The router mounts:
//! - `/api/groups`, `/api/chart/{slug}`, `/api/group/{slug}`, `/health`
//!   (read API)
//! - `/api/ingest` (gated by [`crate::auth::require_bearer`])
//! - HTML routes contributed by [`crate::html::router`]
//!
//! All responses pass through [`CompressionLayer`] so HTML, JSON, and the
//! bundled `/static/*` JS/CSS are served gzipped or brotli-encoded when the
//! client advertises support. The landing page HTML alone is several
//! hundred KB uncompressed, so this is the single biggest cold-load win.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::routing::get;
use axum::routing::post;
use tokio::sync::Mutex;
use tower_http::compression::CompressionLayer;

use crate::api;
use crate::auth::require_bearer;
use crate::db::DbHandle;
use crate::db::{self};
use crate::html;
use crate::ingest;

/// Shared state for all handlers. Cheap to clone (everything is `Arc`-shaped
/// or a small `String`).
#[derive(Clone)]
pub struct AppState {
    /// r2d2 pool of DuckDB connections. See [`crate::db`].
    pub db: DbHandle,
    /// Serializes `/api/ingest` so concurrent ingests can't race on the same
    /// rows and trigger a DuckDB write-write conflict. Reads are unaffected.
    pub ingest_lock: Arc<Mutex<()>>,
    /// Bearer token expected on `/api/ingest`. Compared via constant-time eq.
    pub bearer_token: Arc<String>,
    /// On-disk path of the DuckDB file. Surfaced on `/health`.
    pub db_path: Arc<PathBuf>,
}

impl AppState {
    /// Open the DuckDB at `db_path`, apply the schema, and return shared state.
    pub fn open<P: AsRef<Path>>(db_path: P, bearer_token: String) -> Result<Self> {
        let path = db_path.as_ref().to_path_buf();
        let db = db::open(&path)?;
        Ok(Self {
            db,
            ingest_lock: Arc::new(Mutex::new(())),
            bearer_token: Arc::new(bearer_token),
            db_path: Arc::new(path),
        })
    }
}

/// Build the full Axum router for the bench server.
pub fn router(state: AppState) -> Router {
    let ingest_routes = Router::new()
        .route("/api/ingest", post(ingest::handle))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ));

    let read_routes = Router::new()
        .route("/api/groups", get(api::groups))
        .route("/api/chart/{slug}", get(api::chart))
        .route("/api/group/{slug}", get(api::group))
        .route("/health", get(api::health));

    Router::new()
        .merge(ingest_routes)
        .merge(read_routes)
        .merge(html::router())
        .layer(CompressionLayer::new())
        .with_state(state)
}
