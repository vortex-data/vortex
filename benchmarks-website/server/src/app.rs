// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Axum [`Router`] composition and shared [`AppState`].
//!
//! The server serves two routers on two listeners in production:
//!
//! - **Public** ([`public_router`]) - `/api/groups`, `/api/chart/{slug}`,
//!   `/api/group/{slug}`, `/api/ingest` (bearer-gated by
//!   [`crate::auth::require_bearer`]), `/health`, plus HTML routes
//!   contributed by [`crate::html::router`]. This is what
//!   `VORTEX_BENCH_BIND` exposes (typically `0.0.0.0:3000`).
//! - **Admin** ([`admin_router`]) - `/api/admin/snapshot`,
//!   `/api/admin/sql`, both gated by [`crate::admin::require_admin_bearer`].
//!   Only built when [`AppState::with_admin`] has been called. This is
//!   what `VORTEX_BENCH_ADMIN_BIND` exposes (typically `127.0.0.1:3001`),
//!   so admin endpoints never reach the public network even when the
//!   public listener binds `0.0.0.0`.
//!
//! [`router`] still exists as a convenience for tests that don't care
//! about the listener split: it merges public + admin onto a single
//! router.
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
use axum::extract::DefaultBodyLimit;
use axum::routing::get;
use axum::routing::post;
use parking_lot::Mutex;
use tower_http::compression::CompressionLayer;
use vortex_utils::aliases::hash_set::HashSet;

use crate::admin;
use crate::api;
use crate::auth::require_bearer;
use crate::db::DbHandle;
use crate::db::{self};
use crate::html;
use crate::ingest;
use crate::query_cache::QueryCache;
use crate::read_model::ReadStore;

/// Route-local JSON body limit for CI ingest envelopes.
///
/// Axum's `Json` extractor defaults to 2 MiB, which is too small for larger
/// benchmark matrices once `post-ingest.py` wraps a JSONL result file in one
/// envelope. Keep this bounded rather than disabling the extractor limit.
const INGEST_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

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
    /// Process-local reservation set for snapshot `ts` values currently being
    /// written.
    ///
    /// Two concurrent `/api/admin/snapshot?ts=X` calls in the same process race
    /// at the `tmp -> target` rename step; Linux `rename(2)` overwrites an
    /// existing destination, so without this reservation the second call would
    /// silently clobber the first. The set is consulted + populated atomically
    /// in the snapshot handler before any disk work, and the entry is removed
    /// whether the call succeeds or fails. Single `vortex-bench-server` process
    /// per host is the supported deployment; cross-process race is out of scope
    /// (`ensure_distinct_binds` already forbids two processes binding the same
    /// DB path concurrently in normal operation).
    pub pending_snapshots: Arc<Mutex<HashSet<String>>>,
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
            pending_snapshots: Arc::new(Mutex::new(HashSet::new())),
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

/// Build the public router - `/api/ingest`, the read API, `/health`,
/// and HTML routes. Does not include any `/api/admin/*` routes; those
/// live on a separate listener built by [`admin_router`].
pub fn public_router(state: AppState) -> Router {
    let ingest_routes = Router::new()
        .route("/api/ingest", post(ingest::handle))
        .layer(DefaultBodyLimit::max(INGEST_BODY_LIMIT_BYTES))
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

    Router::new()
        .merge(ingest_routes)
        .merge(read_routes)
        .merge(html::router())
        .layer(CompressionLayer::new())
        .with_state(state)
}

/// Build the admin router when [`AppState::with_admin`] has been called;
/// returns `None` otherwise so the caller skips binding the admin
/// listener entirely.
pub fn admin_router(state: AppState) -> Option<Router> {
    state.admin_bearer_token.as_ref()?;
    let admin_routes = Router::new()
        .route("/api/admin/snapshot", post(admin::snapshot))
        .route("/api/admin/sql", post(admin::sql))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            admin::require_admin_bearer,
        ));
    Some(
        admin_routes
            .layer(CompressionLayer::new())
            .with_state(state),
    )
}

/// Merge the public and admin routers into a single router. Used by
/// tests that don't care about the listener split; production runs the
/// two routers on separate listeners via [`public_router`] +
/// [`admin_router`].
///
/// Reuses [`public_router`] and [`admin_router`] so a route added to
/// either of those flows into the test surface here too. Previously this
/// re-spelled both router bodies inline, which silently drifted from
/// production whenever someone touched only `public_router`.
pub fn router(state: AppState) -> Router {
    let public = public_router(state.clone());
    match admin_router(state) {
        Some(admin) => public.merge(admin),
        None => public,
    }
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
