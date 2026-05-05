// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for the bundled `/static/...` asset routes plus the
//! response compression layer.

mod common;

use std::io::Read as _;

use anyhow::Result;
use flate2::read::GzDecoder;

use self::common::Server;
use self::common::seed;

#[tokio::test]
async fn static_assets_are_served() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    for (path, ct_prefix) in [
        ("/static/chart.umd.js", "application/javascript"),
        (
            "/static/chartjs-plugin-zoom.umd.min.js",
            "application/javascript",
        ),
        ("/static/chart-init.js", "application/javascript"),
        ("/static/style.css", "text/css"),
        ("/Vortex_Black_NoBG.png", "image/png"),
        ("/Vortex_White_NoBG.png", "image/png"),
    ] {
        let resp = client.get(server.url(path)).send().await?;
        assert_eq!(resp.status(), 200, "GET {path} should be 200");
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            ct.starts_with(ct_prefix),
            "GET {path}: content-type {ct:?} should start with {ct_prefix:?}"
        );
        let cache_control = resp
            .headers()
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            cache_control.contains("no-cache"),
            "GET {path}: static assets should revalidate so UI CSS/JS changes are not stale"
        );
        let bytes = resp.bytes().await?;
        assert!(!bytes.is_empty(), "GET {path}: body must not be empty");
    }
    Ok(())
}

/// Every response — landing HTML, chart JSON, bundled JS — flows through
/// `tower-http`'s `CompressionLayer` so a client advertising
/// `Accept-Encoding: gzip` gets a gzipped (or brotli) body. The
/// reqwest dev-dependency is built without `gzip`/`brotli` features, so the
/// transport hands us the compressed bytes verbatim and we can both inspect
/// the `content-encoding` response header and decompress the body manually
/// to confirm it matches the uncompressed snapshot.
#[tokio::test]
async fn responses_are_compressed_when_client_accepts_gzip() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();

    // 1. Landing HTML.
    let plain_body = client.get(server.url("/")).send().await?.text().await?;
    let resp = client
        .get(server.url("/"))
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let encoding = resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        encoding, "gzip",
        "GET / with Accept-Encoding: gzip should respond with gzip"
    );
    let compressed = resp.bytes().await?;
    assert!(
        compressed.len() < plain_body.len(),
        "compressed body ({} B) should be smaller than plain body ({} B)",
        compressed.len(),
        plain_body.len(),
    );
    let mut decoded = String::new();
    GzDecoder::new(&compressed[..]).read_to_string(&mut decoded)?;
    assert_eq!(
        decoded, plain_body,
        "gzipped landing body should decompress to the uncompressed body"
    );

    // 2. Bundled JS — the heaviest static asset; gzip is the whole point.
    let plain_js = client
        .get(server.url("/static/chart.umd.js"))
        .send()
        .await?
        .bytes()
        .await?;
    let js_resp = client
        .get(server.url("/static/chart.umd.js"))
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    assert_eq!(js_resp.status(), 200);
    let js_encoding = js_resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        js_encoding, "gzip",
        "/static/chart.umd.js must compress so the cold load isn't dominated by ~200KB of JS"
    );
    let compressed_js = js_resp.bytes().await?;
    let mut decoded_js = Vec::new();
    GzDecoder::new(&compressed_js[..]).read_to_end(&mut decoded_js)?;
    assert_eq!(
        decoded_js,
        plain_js.as_ref(),
        "decompressed chart.umd.js should match the unencoded body byte-for-byte"
    );

    // 3. Brotli is also offered when the client prefers it.
    let br_resp = client
        .get(server.url("/"))
        .header(reqwest::header::ACCEPT_ENCODING, "br")
        .send()
        .await?;
    assert_eq!(br_resp.status(), 200);
    let br_encoding = br_resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        br_encoding, "br",
        "GET / with Accept-Encoding: br should respond with brotli"
    );

    Ok(())
}
