// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Static asset serving — bundled JS/CSS/PNG via [`include_bytes!`].
//!
//! Every static asset is baked into the binary at build time so the v3 server
//! is fully self-contained. Cache headers force the browser to revalidate on
//! every load, and the URL carries `?v={STATIC_ASSET_VERSION}` so a UI
//! release moves all callers to the new bytes.

use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;

const CHART_JS: &[u8] = include_bytes!("../../static/chart.umd.js");
const CHART_ZOOM_JS: &[u8] = include_bytes!("../../static/chartjs-plugin-zoom.umd.min.js");
const CHART_INIT_JS: &[u8] = include_bytes!("../../static/chart-init.js");
const STYLE_CSS: &[u8] = include_bytes!("../../static/style.css");
const VORTEX_BLACK_PNG: &[u8] = include_bytes!("../../../public/Vortex_Black_NoBG.png");
const VORTEX_WHITE_PNG: &[u8] = include_bytes!("../../../public/Vortex_White_NoBG.png");

/// Cache-busting suffix appended to every static asset URL. Bump on a UI
/// release so cached browsers see the new bytes.
pub(crate) const STATIC_ASSET_VERSION: &str = "bench-v3-ui-22";

/// Append the cache-bust query param to a static asset URL.
pub(crate) fn versioned_asset(path: &str) -> String {
    format!("{path}?v={STATIC_ASSET_VERSION}")
}

pub(crate) async fn serve_chart_js() -> impl IntoResponse {
    static_response(CHART_JS, "application/javascript; charset=utf-8")
}

pub(crate) async fn serve_chart_zoom_js() -> impl IntoResponse {
    static_response(CHART_ZOOM_JS, "application/javascript; charset=utf-8")
}

pub(crate) async fn serve_chart_init_js() -> impl IntoResponse {
    static_response(CHART_INIT_JS, "application/javascript; charset=utf-8")
}

pub(crate) async fn serve_style_css() -> impl IntoResponse {
    static_response(STYLE_CSS, "text/css; charset=utf-8")
}

pub(crate) async fn serve_vortex_black_png() -> impl IntoResponse {
    static_response(VORTEX_BLACK_PNG, "image/png")
}

pub(crate) async fn serve_vortex_white_png() -> impl IntoResponse {
    static_response(VORTEX_WHITE_PNG, "image/png")
}

fn static_response(bytes: &'static [u8], content_type: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CACHE_CONTROL,
                "no-cache, max-age=0, must-revalidate",
            ),
        ],
        bytes,
    )
        .into_response()
}
