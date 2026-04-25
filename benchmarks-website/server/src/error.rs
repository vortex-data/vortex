// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Error types for the bench server.
//!
//! [`IngestError`] models the HTTP matrix from `02-contracts.md` for the
//! `POST /api/ingest` route. [`ApiError`] is the catch-all for read routes.

use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use serde_json::json;
use thiserror::Error;

/// Errors surfaced by `POST /api/ingest`. Each variant maps to a specific
/// HTTP status per the contract.
#[derive(Debug, Error)]
pub enum IngestError {
    /// 400 - the request body wasn't valid JSON or didn't match the envelope
    /// schema (including unknown `kind` values and unknown fields).
    #[error("malformed request body: {0}")]
    Malformed(String),

    /// 400 - a per-record validation rule failed. Carries the offending
    /// record's index in `records` so emitters can pinpoint the problem.
    #[error("record at index {index} failed validation: {message}")]
    Record { index: usize, message: String },

    /// 401 - the bearer token was missing or did not match.
    #[error("missing or invalid bearer token")]
    Unauthorized,

    /// 409 - the envelope's `schema_version` is newer than this server expects.
    #[error("schema version {got} is newer than server's {expected}")]
    SchemaVersionTooNew { expected: i32, got: i32 },

    /// 500 - any other error. Avoid leaking internals to clients.
    #[error("internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IngestError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Malformed(_) | Self::Record { .. } => StatusCode::BAD_REQUEST,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::SchemaVersionTooNew { .. } => StatusCode::CONFLICT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for IngestError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = match &self {
            Self::Malformed(msg) => json!({ "error": "malformed", "message": msg }),
            Self::Record { index, message } => json!({
                "error": "record",
                "record_index": index,
                "message": message,
            }),
            Self::Unauthorized => json!({ "error": "unauthorized" }),
            Self::SchemaVersionTooNew { expected, got } => json!({
                "error": "schema_version_too_new",
                "expected": expected,
                "got": got,
            }),
            Self::Internal(err) => {
                tracing::error!(error = ?err, "ingest internal error");
                json!({ "error": "internal" })
            }
        };
        (status, Json(body)).into_response()
    }
}

/// Errors surfaced by the read API and `/health`.
#[derive(Debug, Error)]
pub enum ApiError {
    /// 404 - the slug supplied to `/api/chart/:slug` couldn't be parsed
    /// or matched no rows.
    #[error("not found: {0}")]
    NotFound(String),

    /// 400 - the slug or query parameters were syntactically invalid.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// 500 - any other error.
    #[error("internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, body) = match &self {
            Self::NotFound(msg) => (
                StatusCode::NOT_FOUND,
                json!({ "error": "not_found", "message": msg }),
            ),
            Self::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "bad_request", "message": msg }),
            ),
            Self::Internal(err) => {
                tracing::error!(error = ?err, "api internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({ "error": "internal" }),
                )
            }
        };
        (status, Json(body)).into_response()
    }
}
