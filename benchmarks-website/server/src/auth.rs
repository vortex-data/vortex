// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bearer-token authentication for `/api/ingest`.
//!
//! The token is read from `INGEST_BEARER_TOKEN` at startup and compared with
//! constant-time equality.

use axum::extract::Request;
use axum::extract::State;
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

use crate::app::AppState;
use crate::error::IngestError;

/// Axum middleware that enforces a `Bearer <token>` header on the route.
pub async fn require_bearer(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, IngestError> {
    let header = req
        .headers()
        .get(AUTHORIZATION)
        .ok_or(IngestError::Unauthorized)?
        .to_str()
        .map_err(|_| IngestError::Unauthorized)?;

    let presented = header
        .strip_prefix("Bearer ")
        .ok_or(IngestError::Unauthorized)?
        .as_bytes();
    let expected = state.bearer_token.as_bytes();

    if presented.ct_eq(expected).into() {
        Ok(next.run(req).await)
    } else {
        Err(IngestError::Unauthorized)
    }
}
