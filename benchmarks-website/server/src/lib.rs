// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex benchmarks website v3 (alpha) server.
//!
//! This crate is a leaf binary that owns a DuckDB file on local disk,
//! accepts authenticated `/api/ingest` POSTs, and serves a small read API
//! plus the HTML pages contributed by the web-ui component.

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
