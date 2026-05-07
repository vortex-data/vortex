// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vortex-bench-server` — the v3 [`bench.vortex.dev`](https://bench.vortex.dev)
//! server.
//!
//! A single Rust binary that owns one DuckDB file on local disk, accepts
//! bearer-authenticated `/api/ingest` POSTs, and serves the JSON read API
//! plus every HTML page. All static assets (Chart.js + the zoom plugin +
//! `chart-init.js` + `style.css` + the two logos) are baked into the
//! binary via `include_bytes!` so a deploy is one binary plus a database
//! file. A `tower-http::CompressionLayer` wraps every response — the
//! landing page HTML alone is several hundred KB uncompressed, so this is
//! the single biggest cold-load win.
//!
//! ## Routes
//!
//! - `GET /` — landing page, every group rendered as a collapsed
//!   `<details>`. The first group's chart payloads are inlined into the
//!   HTML; the rest are shells fetched on first toggle.
//! - `GET /chart/{slug}` — single-chart permalink.
//! - `GET /group/{slug}` — every chart in one group on a single page.
//! - `GET /static/...` — the bundled JS / CSS / logos.
//! - `GET /api/groups` — flat list of every group with chart-link metadata.
//! - `GET /api/chart/{slug}` — one chart's payload (`commits`, `series`,
//!   `unit_kind`, ...).
//! - `GET /api/group/{slug}` — every chart in one group, payload-inlined.
//! - `GET /health` — liveness probe + per-table row counts.
//! - `POST /api/ingest` — bearer-gated ingest. See [`ingest`] for the HTTP
//!   matrix and [`auth`] for the bearer middleware.
//!
//! ## Module map
//!
//! | Module        | Role                                                                                        |
//! |---------------|---------------------------------------------------------------------------------------------|
//! | [`app`]       | [`app::AppState`] (DB handle + bearer + path) and the Axum router composition.              |
//! | [`auth`]      | Bearer-token middleware for `/api/ingest`.                                                  |
//! | [`db`]        | [`db::DbHandle`] connection wrapper + the per-fact-table `measurement_id_*` hash functions. |
//! | [`schema`]    | DuckDB DDL ([`schema::SCHEMA_DDL`]) and the wire schema version.                            |
//! | [`records`]   | Wire shapes for `POST /api/ingest`.                                                         |
//! | [`ingest`]    | `POST /api/ingest` handler — envelope validation, transaction, upsert dispatch.             |
//! | [`error`]     | [`error::IngestError`] and [`error::ApiError`] with their HTTP-status mapping.              |
//! | [`slug`]      | [`slug::ChartKey`] / [`slug::GroupKey`] enums + base64url round-trip.                       |
//! | [`api`]       | Read API. `mod.rs` mounts the handlers; submodules are listed on its module doc.            |
//! | [`html`]      | HTML pages — `mod.rs` mounts the routes; submodules render the body.                        |
//!
//! ## Request flow
//!
//! 1. Axum receives the request and routes by method + path.
//! 2. `/api/ingest` first passes through [`auth::require_bearer`]; other
//!    routes skip auth.
//! 3. The handler parses body / path / query into typed inputs (e.g.
//!    [`slug::ChartKey::from_slug`]).
//! 4. The handler hands a closure to [`db::run_blocking`], which acquires
//!    the connection mutex and runs the synchronous DuckDB call on
//!    `tokio::task::spawn_blocking` so the runtime stays free.
//! 5. The closure returns `Result<T, anyhow::Error>`. Errors are mapped
//!    into [`error::IngestError`] / [`error::ApiError`] with the right
//!    HTTP status.
//! 6. The response is rendered (JSON via [`axum::Json`], HTML via the
//!    `maud` router in [`html`]).
//! 7. Every response passes through [`tower_http::compression::CompressionLayer`]
//!    on the way out.

pub mod api;
pub mod app;
pub mod auth;
pub mod db;
pub mod error;
pub mod html;
pub mod ingest;
pub mod records;
pub mod schema;
pub mod slug;
