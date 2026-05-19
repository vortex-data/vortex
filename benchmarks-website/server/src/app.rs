// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Axum [`Router`] composition and shared [`AppState`].
//!
//! The router mounts:
//! - `/api/groups`, `/api/chart/{slug}`, `/api/group/{slug}`, `/health`
//!   (read API)
//! - `/api/ingest` (gated by [`crate::auth::require_bearer`])
//! - `/api/admin/snapshot`, `/api/admin/sql` — only when
//!   [`AppState::with_admin`] has been called (gated by
//!   [`crate::admin::require_admin_bearer`])
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
use anyhow::ensure;
use axum::Router;
use axum::routing::get;
use axum::routing::post;
use tower_http::compression::CompressionLayer;

use crate::admin;
use crate::api;
use crate::auth::require_bearer;
use crate::db::DbHandle;
use crate::db::{self};
use crate::html;
use crate::ingest;
use crate::query_cache::QueryCache;
use crate::read_model::ReadStore;

/// Shared state for all handlers. Cheap to clone (everything is `Arc`-shaped
/// or a small `String`).
#[derive(Clone)]
pub struct AppState {
    /// Shared DuckDB handle. See [`crate::db`].
    pub db: DbHandle,
    /// Bearer token expected on `/api/ingest`. Compared via constant-time eq.
    pub bearer_token: Arc<String>,
    /// Bearer token expected on `/api/admin/*`. `None` disables the admin
    /// router entirely. Set via [`AppState::with_admin`].
    pub admin_bearer_token: Option<Arc<String>>,
    /// On-disk path of the DuckDB file. Surfaced on `/health`.
    pub db_path: Arc<PathBuf>,
    /// In-memory cache of every read-side query result. Cleared by
    /// [`crate::ingest`] after a successful commit. See [`crate::query_cache`].
    pub cache: Arc<QueryCache>,
    /// Materialized latest-100 read artifacts for the website hot path.
    pub read_store: Arc<ReadStore>,
    /// Directory `/api/admin/snapshot` writes per-table Vortex snapshots into
    /// (`schema.sql` plus one `<table>.vortex` file per table in
    /// [`crate::schema::TABLES`]). Defaults to `<db_path parent>/snapshots`.
    /// Override via [`AppState::with_snapshot_dir`].
    pub snapshot_dir: Arc<PathBuf>,
    /// Directory DuckDB installs extensions into. Surfaced on `AppState` so
    /// the operator can keep extensions on a writable filesystem path even
    /// when systemd's `ProtectHome=read-only` blocks the default
    /// `~/.duckdb/extensions/...`. Defaults to `<db_path parent>/duckdb-extensions`.
    /// Override via [`AppState::with_extension_dir`].
    pub extension_dir: Arc<PathBuf>,
}

impl AppState {
    /// Open the DuckDB at `db_path`, apply the schema, and return shared state.
    /// `bearer_token` must be non-empty. Admin endpoints are unmounted by
    /// default; call [`AppState::with_admin`] with a non-empty token to enable
    /// them.
    pub fn open<P: AsRef<Path>>(db_path: P, bearer_token: String) -> Result<Self> {
        validate_bearer_token("INGEST_BEARER_TOKEN", &bearer_token)?;
        let path = db_path.as_ref().to_path_buf();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let snapshot_dir = parent.join("snapshots");
        let extension_dir = parent.join("duckdb-extensions");
        let db = db::open(&path, &extension_dir)?;
        let read_store = ReadStore::build_initial(&db)?;
        Ok(Self {
            db,
            bearer_token: Arc::new(bearer_token),
            admin_bearer_token: None,
            db_path: Arc::new(path),
            cache: Arc::new(QueryCache::new()),
            read_store: Arc::new(read_store),
            snapshot_dir: Arc::new(snapshot_dir),
            extension_dir: Arc::new(extension_dir),
        })
    }

    /// Enable the `/api/admin/*` router, gated by `admin_bearer_token`.
    /// Without this call, the admin router is not mounted at all.
    pub fn with_admin(mut self, admin_bearer_token: String) -> Self {
        if !admin_bearer_token.trim().is_empty() {
            self.admin_bearer_token = Some(Arc::new(admin_bearer_token));
        }
        self
    }

    /// Override the directory `/api/admin/snapshot` writes per-table Vortex
    /// snapshots into (`schema.sql` plus one `<table>.vortex` file per table).
    /// Defaults to `<db_path parent>/snapshots`.
    pub fn with_snapshot_dir(mut self, dir: PathBuf) -> Self {
        self.snapshot_dir = Arc::new(dir);
        self
    }

    /// Override the directory DuckDB installs extensions into and re-run
    /// `SET GLOBAL extension_directory` on the root connection so the new
    /// value propagates to every cloned connection. Fallible because it
    /// touches the DB; the other builders are infallible because they only
    /// shuffle `AppState` fields.
    pub fn with_extension_dir(mut self, dir: PathBuf) -> Result<Self> {
        self.db.set_extension_directory(&dir)?;
        self.extension_dir = Arc::new(dir);
        Ok(self)
    }
}

fn validate_bearer_token(name: &str, token: &str) -> Result<()> {
    ensure!(!token.trim().is_empty(), "{name} must not be empty");
    Ok(())
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
        .route(
            "/api/artifacts/{generation}/groups/{group_slug}/shards/{index}",
            get(api::group_shard_artifact),
        )
        .route("/health", get(api::health));

    let mut router = Router::new()
        .merge(ingest_routes)
        .merge(read_routes)
        .merge(html::router());

    if state.admin_bearer_token.is_some() {
        let admin_routes = Router::new()
            .route("/api/admin/snapshot", post(admin::snapshot))
            .route("/api/admin/sql", post(admin::sql))
            .route_layer(axum::middleware::from_fn_with_state(
                state.clone(),
                admin::require_admin_bearer,
            ));
        router = router.merge(admin_routes);
    }

    router.layer(CompressionLayer::new()).with_state(state)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn open_rejects_empty_ingest_token() {
        let tmp = TempDir::new().unwrap();
        let result = AppState::open(tmp.path().join("bench.duckdb"), String::new());
        assert!(
            result.is_err(),
            "empty INGEST_BEARER_TOKEN should fail startup"
        );
    }

    #[test]
    fn empty_admin_token_leaves_admin_disabled() -> Result<()> {
        let tmp = TempDir::new()?;
        let state = AppState::open(tmp.path().join("bench.duckdb"), "ingest-token".to_string())?
            .with_admin(String::new());
        assert!(
            state.admin_bearer_token.is_none(),
            "empty ADMIN_BEARER_TOKEN should not mount admin routes"
        );
        Ok(())
    }
}
