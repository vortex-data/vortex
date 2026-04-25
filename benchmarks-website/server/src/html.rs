// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! HTML routes.
//!
//! The web-ui component owns the actual landing-page and chart-page templates.
//! At alpha this module exposes a single placeholder route so the server can
//! be exercised end-to-end before web-ui lands; the web-ui PR replaces
//! [`router`] with the real Maud templates.

use axum::Router;
use axum::routing::get;
use maud::DOCTYPE;
use maud::Markup;
use maud::html;

use crate::app::AppState;

/// HTML routes mounted under `/`. Replaced by the web-ui component.
pub fn router() -> Router<AppState> {
    Router::new().route("/", get(placeholder))
}

async fn placeholder() -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                title { "bench.vortex.dev (v3 alpha)" }
            }
            body {
                h1 { "bench.vortex.dev (v3 alpha)" }
                p {
                    "Server is up. The landing page and chart page land with "
                    "the web-ui PR."
                }
                ul {
                    li { code { "GET /api/groups" } }
                    li { code { "GET /api/chart/:slug" } }
                    li { code { "GET /health" } }
                    li { code { "POST /api/ingest" } " (bearer auth)" }
                }
            }
        }
    }
}
